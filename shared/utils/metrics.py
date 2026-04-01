"""
H4wkQuant - Prometheus Metrics Exporter
Simple text-format Prometheus metrics (no external dependency).
"""
import time
from typing import Dict


class Counter:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value = 0.0

    def inc(self, amount: float = 1.0):
        self._value += amount

    def get(self) -> float:
        return self._value

    def render(self) -> str:
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} counter")
        lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help_text = help_text
        self._value = 0.0

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1.0):
        self._value += amount

    def dec(self, amount: float = 1.0):
        self._value -= amount

    def get(self) -> float:
        return self._value

    def render(self) -> str:
        lines = []
        if self.help_text:
            lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} gauge")
        lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class MetricsRegistry:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._metrics: Dict[str, object] = {}
        self._start_time = time.time()
        # Auto-add uptime gauge
        self.uptime = self.gauge("h4wkquant_uptime_seconds", "Service uptime in seconds")

    def counter(self, name: str, help_text: str = "") -> Counter:
        c = Counter(name, help_text)
        self._metrics[name] = c
        return c

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        g = Gauge(name, help_text)
        self._metrics[name] = g
        return g

    def render(self) -> str:
        self.uptime.set(round(time.time() - self._start_time, 1))
        parts = []
        for m in self._metrics.values():
            parts.append(m.render())
        return "\n\n".join(parts) + "\n"
