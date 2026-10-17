import random
import sys
import time
import psutil

import client
from engine import OffloadingEngine

# Read execution mode from command line arguments ('local', 'offload', or 'dynamic')
if len(sys.argv) < 2 or sys.argv[1] not in ["local", "offload", "dynamic"]:
    print("Usage: python peak_ram.py <local|offload|dynamic>")
    sys.exit(1)

mode = sys.argv[1]
N = 400
RUNS = 100

engine = OffloadingEngine(alpha=0.5)

# Warm-up run and local estimate
t_local = client.run_local_inference(N)
t_remote = t_local * 0.3

# Warm-up server connection if not strictly local
if mode != "local":
    client.run_remote_offload(N)

lat = []

for i in range(1, RUNS + 1):
    # Simulate fluctuating network and battery depletion over 100 runs
    rtt = (
        random.uniform(0.8, 1.8)
        if (i % 7 == 0 or i % 8 == 0)
        else random.uniform(0.01, 0.15)
    )
    battery = max(5.0, 100.0 - (i * 0.28))
    cpu = random.uniform(20.0, 75.0)

    # Determine execution path based on mode
    if mode == "dynamic":
        use_local = (
            engine.decide(battery, cpu, rtt, t_local, t_remote)["decision"]
            == "LOCAL"
        )
    else:
        use_local = mode == "local"

    # Measure task latency
    t0 = time.perf_counter()
    if use_local:
        client.run_local_inference(N)
    else:
        client.run_remote_offload(N)
    lat.append(time.perf_counter() - t0)

# Retrieve peak Working Set RAM (Windows) with fallback to RSS
mem_info = psutil.Process().memory_info()
if hasattr(mem_info, "peak_wset"):
    peak_mb = mem_info.peak_wset / (1024**2)
else:
    peak_mb = mem_info.rss / (1024**2)

mean_lat_ms = (sum(lat) / len(lat)) * 1000
throughput = len(lat) / sum(lat)

print(
    f"{mode.upper()}: mean latency {mean_lat_ms:.1f} ms | "
    f"throughput {throughput:.2f} tasks/s | peak RAM {peak_mb:.1f} MB"
)