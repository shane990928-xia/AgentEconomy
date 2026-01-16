@dataclass
class SystemMetrics:
    """系统指标类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float