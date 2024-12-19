from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
from Bioneural.core.homeostasis import HomeostaticRegulation
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from Bioneural.metrics.healtracker import HealthTracker
import logging
from Bioneural.visualization.biosysvisualization import BioNeuronVisualizer
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
from collections import defaultdict
import numpy as np
from Bioneural.core.homeostasis import HomeostaticRegulation
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import logging
from Bioneural.visualization.biosysvisualization import BioNeuronVisualizer

class BioLogicalNeuron(nn.Module):
    """Enhanced Biological Neuron with advanced repair mechanisms and multi-strategy monitoring"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        plasticity_rate: float = 0.008,
        repair_threshold: float = 0.5,
        repair_intensity: float = 0.08,
        enable_monitoring: bool = True,
        log_file: Optional[str] = "bioneuron_health.log",
        summary_interval: int = 100,  
        **kwargs
    ):
        super().__init__()
        self.linear = weight_norm(nn.Linear(in_features, out_features))
        self.η = plasticity_rate
        self.homeostasis = HomeostaticRegulation(**kwargs)
        self.repair_threshold = repair_threshold
        self.repair_intensity = repair_intensity
        self.repair_count = 0
        self.repair_cooldown = 0
        
        self.health_tracker = HealthTracker() if enable_monitoring else None
        self.logger = self._setup_logger(log_file) if enable_monitoring else None
        self.visualizer = BioNeuronVisualizer(save_dir="bio_vis") if enable_monitoring else None
        self.step_counter = 0
        self.summary_interval = summary_interval
        
        self.epoch_health_logs = defaultdict(list)
        self.current_epoch = 0
        self.epoch_metrics = defaultdict(list)
        
        # New advanced repair tracking
        self.repair_strategies = {
            'adaptive_noise': 0,
            'targeted_repair': 0,
            'gradient_aware_repair': 0,
            'periodic_reset': 0
        }
        
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"bioneuron_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear() 
            
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger

    def _update_epoch_metrics(self, health_report: Dict[str, Union[torch.Tensor, float, bool]]):
        """Update epoch-wise metrics"""
        self.epoch_metrics["current_health"].append(
            float(health_report['current_health'].mean() if isinstance(health_report['current_health'], torch.Tensor) 
            else health_report['current_health'])
        )
        self.epoch_metrics["stability"].append(
            float(health_report['stability'].mean() if isinstance(health_report['stability'], torch.Tensor) 
            else health_report['stability'])
        )
        self.epoch_metrics["base_health"].append(
            float(health_report['base_health'].mean() if isinstance(health_report['base_health'], torch.Tensor) 
            else health_report['base_health'])
        )
        self.epoch_metrics["calcium_trend"].append(
            float(health_report['calcium_trend'] if isinstance(health_report['calcium_trend'], (int, float)) 
            else health_report['calcium_trend'].item())
        )
        self.epoch_metrics["repair_performed"].append(health_report['repair_performed'])

    def _log_health_status(self, health_report: Dict[str, float]):
        """Log health status and updates"""
        if self.health_tracker and self.logger:
            self.health_tracker.log_health(health_report, self.step_counter)
            
            
            self._update_epoch_metrics(health_report)
            
  
            health = health_report["current_health"].mean().item() if isinstance(health_report["current_health"], torch.Tensor) else health_report["current_health"]
            stability = health_report["stability"].mean().item() if isinstance(health_report["stability"], torch.Tensor) else health_report["stability"]
            
            msg = f"Step {self.step_counter} | Health: {health:.3f} | Stability: {stability:.3f}"
            if health_report.get("repair_performed", False):
                msg += f" | Repair performed! (Count: {self.repair_count})"
            
            self.logger.info(msg)
            
            # Log summary at specified intervals
            if self.step_counter % self.summary_interval == 0:
                self._log_interval_summary()

    def _log_interval_summary(self):
        """Log summary statistics for the current interval"""
        if not self.epoch_metrics["current_health"]:
            return

        avg_health = np.mean(self.epoch_metrics["current_health"])
        avg_stability = np.mean(self.epoch_metrics["stability"])
        avg_base_health = np.mean(self.epoch_metrics["base_health"])
        total_repairs = sum(self.epoch_metrics["repair_performed"])
        
        summary = (
            f"\nHealth Summary (Steps {self.step_counter - self.summary_interval + 1} - {self.step_counter}):\n"
            f"  Average Health: {avg_health:.3f}\n"
            f"  Average Stability: {avg_stability:.3f}\n"
            f"  Average Base Health: {avg_base_health:.3f}\n"
            f"  Total Repairs: {total_repairs}\n"
            f"  Repair Rate: {(total_repairs/self.summary_interval)*100:.2f}%"
        )
        
        if self.logger:
            self.logger.info(summary)
        else:
            print(summary)

    def start_epoch(self, epoch_num: int):
        """Mark the start of a new epoch"""
        self.current_epoch = epoch_num
        self.epoch_metrics = defaultdict(list)
        if self.logger:
            self.logger.info(f"\nStarting Epoch {epoch_num}")

    def end_epoch(self):
        """Generate end-of-epoch summary and reset metrics"""
        if not self.epoch_metrics["current_health"]:
            return

        avg_health = np.mean(self.epoch_metrics["current_health"])
        avg_stability = np.mean(self.epoch_metrics["stability"])
        avg_base_health = np.mean(self.epoch_metrics["base_health"])
        total_repairs = sum(self.epoch_metrics["repair_performed"])
        repair_rate = (total_repairs / len(self.epoch_metrics["repair_performed"])) * 100
        
        summary = (
            f"\nEpoch {self.current_epoch} Summary:\n"
            f"  Average Health: {avg_health:.3f}\n"
            f"  Average Stability: {avg_stability:.3f}\n"
            f"  Average Base Health: {avg_base_health:.3f}\n"
            f"  Total Repairs: {total_repairs}\n"
            f"  Repair Rate: {repair_rate:.2f}%"
        )
        
        if self.logger:
            self.logger.info(summary)
        else:
            print(summary)
        
        # Store epoch summary
        self.epoch_health_logs[self.current_epoch] = {
            "avg_health": avg_health,
            "avg_stability": avg_stability,
            "avg_base_health": avg_base_health,
            "total_repairs": total_repairs,
            "repair_rate": repair_rate
        }
        
        # Reset metrics for next epoch
        self.epoch_metrics = defaultdict(list)

    def adaptive_repair(self, health_report: Dict[str, torch.Tensor], gradients: Optional[torch.Tensor] = None) -> bool:
        if self.repair_cooldown > 0:
            self.repair_cooldown -= 1
            return False 
        
        current_health = health_report['current_health'].mean().item()
        if current_health < self.repair_threshold:
            # 1. Adaptive Noise Injection
            noise_scale = 1.0 - health_report['stability'].item()
            training_progress = min(1.0, self.step_counter / 10000)
            adaptive_noise_intensity = self.repair_intensity * (1 - training_progress * 0.5)
        
            repair_noise = torch.randn_like(self.linear.weight, device=self.linear.weight.device)
            repair_noise = repair_noise * noise_scale * adaptive_noise_intensity
            self.repair_strategies['adaptive_noise'] += 1
        
            # 2. Targeted Repair Zones
            weight_variance = torch.var(self.linear.weight, dim=1)
            low_variance_mask = weight_variance < self.repair_threshold
            targeted_repair_noise = torch.randn_like(self.linear.weight)
            targeted_repair_noise[low_variance_mask] *= 0.5
            self.repair_strategies['targeted_repair'] += 1

            # Apply combined repairs
            self.linear.weight.data += repair_noise + targeted_repair_noise
        
            # Repair tracking
            self.repair_cooldown = 40
            self.repair_count += 1
        
            self.homeostasis.repair_history.append({
                'intensity': adaptive_noise_intensity,
                'health': current_health,
                'stability': health_report['stability'].item()
            })
            return True
        return False
    
    
    def get_adaptive_learning_rate(self, health_report):
        """Dynamically adjust learning rate based on neuron health"""
        base_lr = 0.001
        health_factor = health_report['current_health'].mean().item()
        stability_factor = health_report['stability'].mean().item()
        
        # Adaptive learning rate based on neuron health
        adaptive_lr = base_lr * health_factor * stability_factor
        return adaptive_lr

    def forward(self, x: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass with advanced health monitoring and multi-strategy repair"""
        pre_synaptic = x
        post_synaptic = F.gelu(self.linear(x))
        
        with torch.no_grad():
            calcium_level = self.homeostasis.update_calcium(post_synaptic)
            synaptic_strength = torch.norm(self.linear.weight, dim=1)
            self.homeostasis.synaptic_strengths.append(synaptic_strength)
            
            if len(self.homeostasis.synaptic_strengths) > self.homeostasis.window:
                self.homeostasis.synaptic_strengths = self.homeostasis.synaptic_strengths[-self.homeostasis.window:]
            
            health_report = self.homeostasis.predict_health()
            performed_repair = self.adaptive_repair(health_report, gradients)
            health_report['repair_performed'] = performed_repair
            
            # Compute adaptive learning rate
            adaptive_lr = self.get_adaptive_learning_rate(health_report)
            health_report['adaptive_lr'] = adaptive_lr
            
            self._log_health_status(health_report)
            self.step_counter += 1
            
            if self.visualizer:
                self.visualizer.update(
                    step=self.step_counter,
                    health_report=health_report,
                    calcium_level=calcium_level,
                    repair_strategies=self.repair_strategies
                )
                
                if self.step_counter % 400 == 0:
                    self.visualizer.save_all_plots()
            
        return post_synaptic, health_report
    
    def get_health_stats(self) -> Dict[str, float]:
        """Get comprehensive health statistics"""
        stats = {
            **self.homeostasis.get_health_summary(),
            "repair_count": self.repair_count,
            "cooldown": self.repair_cooldown,
            "repair_strategies": self.repair_strategies
        }
        
        if self.health_tracker:
            stats.update(self.health_tracker.get_summary())
            
        return stats