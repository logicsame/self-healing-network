import logging
from typing import Dict, Optional

class HealthLogger:
    """Custom logger for neuron health monitoring"""
    def __init__(self):
        self.logger = logging.getLogger("bioneuron")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log_health(self, health_data: Dict[str, float], prefix: Optional[str] = ""):
        """Log health-related information."""
        health = health_data.get("current_health", 0)
        stability = health_data.get("stability", 0)
        
        msg = f"{prefix}Health: {health:.3f} | Stability: {stability:.3f}"
        if health_data.get("repair_performed", False):
            msg += " | Repair performed!"
        
        self.logger.info(msg)