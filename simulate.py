import random
import time
import matplotlib.pyplot as plt
import numpy as np

import client
from engine import OffloadingEngine


def run_simulation(iterations: int = 30, matrix_size: int = 400):
    """Runs a multi-step offloading simulation under fluctuating latency and battery levels."""
    engine = OffloadingEngine(alpha=0.5)

    # Logging structures
    rtt_history = []
    battery_history = []
    decision_history = []  # 0 for LOCAL, 1 for OFFLOAD

    # Cumulative energy tracking (in Joules)
    energy_dynamic = 0.0
    energy_always_local = 0.0
    energy_always_offload = 0.0

    # Cumulative latency tracking (in Seconds)
    latency_dynamic = 0.0
    latency_always_local = 0.0
    latency_always_offload = 0.0

    # Estimate benchmark base times
    print(f"Benchmarking baseline local task execution (N={matrix_size})...")
    t_local_est = client.run_local_inference(matrix_size)
    t_remote_est = t_local_est * 0.3  # Edge server is assumed ~3x faster
    print(
        f"Baseline Local Compute Time: {t_local_est:.4f}s | Est. Server Compute: {t_remote_est:.4f}s\n"
    )

    battery_level = 100.0

    for i in range(1, iterations + 1):
        # 1. Simulate fluctuating system state
        # Create periodic network latency spikes every 7-8 iterations
        if i % 7 == 0 or i % 8 == 0:
            simulated_rtt = random.uniform(0.8, 1.8)  # High latency / congestion
        else:
            simulated_rtt = random.uniform(0.01, 0.15)  # Fast edge connection

        # Battery slowly drains over iterations
        battery_level = max(5.0, 100.0 - (i * 2.8))
        cpu_load = random.uniform(20.0, 75.0)

        # 2. Decision Engine Evaluation
        decision_info = engine.decide(
            battery_pct=battery_level,
            cpu_load=cpu_load,
            rtt=simulated_rtt,
            t_local_est=t_local_est,
            t_remote_est=t_remote_est,
        )
        choice = decision_info["decision"]

        # Convert decision to binary metric (0 = Local, 1 = Offload)
        binary_decision = 1 if choice == "OFFLOAD" else 0

        # 3. Calculate metrics for all strategies
        # Local metrics
        e_loc, _ = engine.estimate_local_cost(t_local_est, cpu_load)
        t_loc = t_local_est

        # Offload metrics
        e_off, _ = engine.estimate_offload_cost(simulated_rtt, t_remote_est)
        t_off = simulated_rtt + t_remote_est

        # Update cumulative baselines
        energy_always_local += e_loc
        latency_always_local += t_loc

        energy_always_offload += e_off
        latency_always_offload += t_off

        # Update Dynamic strategy according to choice
        if choice == "LOCAL":
            energy_dynamic += e_loc
            latency_dynamic += t_loc
        else:
            energy_dynamic += e_off
            latency_dynamic += t_off

        # History recording
        rtt_history.append(simulated_rtt * 1000.0)  # convert to ms
        battery_history.append(battery_level)
        decision_history.append(binary_decision)

        print(
            f"Step {i:02d} | RTT: {simulated_rtt * 1000.0:6.1f} ms | Battery: {battery_level:4.1f}% | "
            f"Decision: {choice:<7} ({decision_info['reason']})"
        )

    # Plot results
    plot_decision_switches(
        rtt_history, battery_history, decision_history, iterations
    )
    plot_energy_comparison(
        energy_dynamic,
        energy_always_local,
        energy_always_offload,
        latency_dynamic,
        latency_always_local,
        latency_always_offload,
    )


def plot_decision_switches(rtt_hist, battery_hist, decisions, iterations):
    """Generates time-series plot of fluctuating metrics and offloading decisions."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    steps = list(range(1, iterations + 1))

    # Top Panel: System Telemetry
    color_rtt = "tab:red"
    ax1.set_ylabel("RTT (ms)", color=color_rtt, fontsize=11)
    line1 = ax1.plot(
        steps,
        rtt_hist,
        color=color_rtt,
        marker="o",
        linewidth=2,
        label="Network RTT (ms)",
    )
    ax1.tick_params(axis="y", labelcolor=color_rtt)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_bat = ax1.twinx()
    color_bat = "tab:green"
    ax1_bat.set_ylabel("Battery (%)", color=color_bat, fontsize=11)
    line2 = ax1_bat.plot(
        steps,
        battery_hist,
        color=color_bat,
        linestyle="--",
        linewidth=2,
        label="Battery Level (%)",
    )
    ax1_bat.tick_params(axis="y", labelcolor=color_bat)

    ax1.set_title(
        "Dynamic Offloading Engine: Network Conditions & Decision Switches",
        fontsize=13,
        pad=10,
    )

    # Bottom Panel: Step Chart of Binary Decisions
    ax2.step(
        steps, decisions, color="tab:blue", where="mid", linewidth=2.5
    )
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["LOCAL (0)", "OFFLOAD (1)"])
    ax2.set_ylabel("Execution Location", fontsize=11)
    ax2.set_xlabel("Task Iteration Step", fontsize=11)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("D:\\EDGE\\decision_plot.png", dpi=300)
    print("\nSaved plot: D:\\EDGE\\decision_plot.png")
    plt.show()


def plot_energy_comparison(e_dyn, e_loc, e_off, l_dyn, l_loc, l_off):
    """Generates comparative bar chart across offloading strategies."""
    strategies = ["Dynamic Engine\n(Proposed)", "Always-Local", "Always-Offload"]
    energies = [e_dyn, e_loc, e_off]
    latencies = [l_dyn, l_loc, l_off]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))

    rects1 = ax1.bar(
        x - width / 2, energies, width, label="Total Energy (Joules)", color="#2b5c8f"
    )
    ax1.set_ylabel("Energy Consumption (Joules)", color="#2b5c8f", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10, fontweight="bold")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2, latencies, width, label="Total Latency (Sec)", color="#d95f02"
    )
    ax2.set_ylabel("Total Latency (Seconds)", color="#d95f02", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    ax1.bar_label(rects1, fmt="%.1f J", padding=3)
    ax2.bar_label(rects2, fmt="%.2f s", padding=3)

    plt.title(
        "Overall System Energy & Latency Comparison", fontsize=13, pad=12
    )
    plt.tight_layout()
    plt.savefig("D:\\EDGE\\energy_comparison.png", dpi=300)
    print("Saved plot: D:\\EDGE\\energy_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_simulation(iterations=30, matrix_size=400)