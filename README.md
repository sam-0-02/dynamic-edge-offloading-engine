# Dynamic Edge Computation Offloading Engine

A network- and energy-aware offloading decision engine that dynamically determines whether to execute computational workloads (e.g., machine learning inference) locally on an edge client or offload them to an Edge Data Center / Cloud.

## Architecture & Workflow

- **Edge Client (`client.py`)**: Monitors battery state (`psutil`), CPU load, and network Round-Trip Time (RTT).
- **Edge Data Center Simulator (`server.py`)**: Flask API providing `/ping` and `/offload` matrix multiplication endpoints.
- **Cost Engine (`engine.py`)**: Minimizes $C = \alpha \cdot \text{Latency} + \beta \cdot \text{Energy}$ with dynamic battery state weighting.
- **Simulator & Visualized Evaluation (`simulate.py`)**: Evaluates performance over 30 sequential iterations under fluctuating RTT latency and battery depletion.

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/dynamic-edge-offloading-engine.git](https://github.com/YOUR_USERNAME/dynamic-edge-offloading-engine.git)
   cd dynamic-edge-offloading-engine