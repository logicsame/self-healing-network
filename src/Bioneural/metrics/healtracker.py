from typing import Dict, List, Optional
import time
from datetime import datetime
import torch

class HealthTracker:
    """Tracks and logs neuron health metrics during training"""
    def __init__(self):
        self.health_history = []
        self.repair_events = []
        self.start_time = time.time()

    def log_health(self, health_report: Dict[str, float], step: Optional[int] = None):
        """Log health metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Convert tensor values to Python floats
        processed_metrics = {}
        for key, value in health_report.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item() if value.numel() == 1 else value.mean().item()
            else:
                processed_metrics[key] = value

        entry = {
            "timestamp": timestamp,
            "step": step,
            "metrics": processed_metrics
        }
        self.health_history.append(entry)
        
        if processed_metrics.get("repair_performed", False):
            self.repair_events.append(entry)

    def get_summary(self) -> Dict:
        """Get a summary of health metrics."""
        if not self.health_history:
            return {}
            
        recent_health = [h["metrics"]["current_health"] for h in self.health_history[-100:]]
        return {
            "avg_recent_health": sum(recent_health) / len(recent_health),
            "total_repairs": len(self.repair_events),
            "uptime": time.time() - self.start_time
        }