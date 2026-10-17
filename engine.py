class OffloadingEngine:

    def __init__(
        self,
        alpha: float = 0.5,
        p_cpu_base: float = 2.0,
        p_cpu_max: float = 15.0,
        p_trans: float = 3.0,
    ):
        """Cost Decision Engine.

        Args:
            alpha: Weight for Latency (0.0 to 1.0). Beta = 1 - alpha (Energy weight).
            p_cpu_base: Base CPU idle power draw in Watts.
            p_cpu_max: Max CPU load power draw in Watts.
            p_trans: Network transmission power draw in Watts.
        """
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.p_cpu_base = p_cpu_base
        self.p_cpu_max = p_cpu_max
        self.p_trans = p_trans

    def estimate_local_cost(
        self, t_local: float, cpu_load: float
    ) -> tuple[float, float]:
        """Calculates estimated energy and cost for local execution."""
        # Power model: P_local = P_base + (P_max - P_base) * (cpu_load / 100)
        p_local = self.p_cpu_base + (
            self.p_cpu_max - self.p_cpu_base
        ) * (cpu_load / 100.0)
        e_local = p_local * t_local  # Energy in Joules (Watt * Seconds)

        c_local = self.alpha * t_local + self.beta * e_local
        return e_local, c_local

    def estimate_offload_cost(
        self, rtt: float, t_offload_est: float, data_size_mb: float = 0.5
    ) -> tuple[float, float]:
        """Calculates estimated latency, energy, and cost for offloaded execution."""
        # Transmission time estimate based on RTT
        t_trans = (data_size_mb * 0.01) + (rtt / 2.0)
        t_total_offload = rtt + t_offload_est

        # Energy spent powering Wi-Fi radio during transmission
        e_offload = self.p_trans * t_trans

        c_offload = self.alpha * t_total_offload + self.beta * e_offload
        return e_offload, c_offload

    def decide(
        self,
        battery_pct: float,
        cpu_load: float,
        rtt: float,
        t_local_est: float,
        t_remote_est: float,
    ) -> dict:
        """Determines whether to execute locally or offload.

        Returns decision dictionary with cost metrics.
        """
        # Hard heuristics overrides
        if rtt == float("inf") or rtt > 1.5:  # High network latency or offline
            decision = "LOCAL"
            reason = "Network unreachable or high latency spike"
        elif battery_pct < 15.0 and rtt < 0.2:  # Battery critical and fast connection
            decision = "OFFLOAD"
            reason = "Critical battery state; saving local power"
        else:
            e_local, c_local = self.estimate_local_cost(t_local_est, cpu_load)
            e_offload, c_offload = self.estimate_offload_cost(
                rtt, t_remote_est
            )

            # Weight bias adjustment based on battery level
            # Low battery increases energy penalty weight (beta)
            battery_factor = 1.0 + (100.0 - battery_pct) / 100.0
            c_local_adjusted = c_local * battery_factor

            if c_local_adjusted <= c_offload:
                decision = "LOCAL"
                reason = "Lower cost locally"
            else:
                decision = "OFFLOAD"
                reason = "Lower cost via offloading"

        return {
            "decision": decision,
            "reason": reason,
            "rtt": rtt,
            "battery": battery_pct,
        }