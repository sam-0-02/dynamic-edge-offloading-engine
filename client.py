import time
import numpy as np
import psutil
import requests


def get_battery_level(default_val: float = 100.0) -> float:
    """Returns battery percentage.

    Falls back to default_val if no physical battery is present.
    """
    battery = psutil.sensors_battery()
    if battery is None:
        return default_val
    return float(battery.percent)


def get_cpu_utilization() -> float:
    """Returns current CPU utilization percentage."""
    return float(psutil.cpu_percent(interval=0.1))


def measure_rtt(server_url: str = "http://127.0.0.1:5000/ping") -> float:
    """Measures network Round-Trip Time (RTT) in seconds to the server endpoint.

    Returns float('inf') if server is unreachable.
    """
    try:
        start_time = time.perf_counter()
        response = requests.get(server_url, timeout=2.0)
        end_time = time.perf_counter()

        if response.status_code == 200:
            return end_time - start_time
        return float("inf")
    except requests.RequestException:
        return float("inf")


def run_local_inference(matrix_size: int) -> float:
    """Executes the task locally using NumPy and returns total compute duration in seconds."""
    start_time = time.perf_counter()
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    _ = np.dot(matrix_a, matrix_b)
    return time.perf_counter() - start_time


def run_remote_offload(
    matrix_size: int, server_url: str = "http://127.0.0.1:5000/offload"
) -> tuple[float, float]:
    """Sends task to edge server.

    Returns:
        (total_offload_time_sec, server_compute_time_sec)
    """
    start_time = time.perf_counter()
    payload = {"matrix_size": matrix_size}

    try:
        response = requests.post(server_url, json=payload, timeout=10.0)
        total_time = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            server_compute_time = data.get("server_compute_time", 0.0)
            return total_time, server_compute_time
        return float("inf"), 0.0
    except requests.RequestException:
        return float("inf"), 0.0