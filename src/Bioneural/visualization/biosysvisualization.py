import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List
import pandas as pd
from datetime import datetime

class BioNeuronVisualizer:
    """Advanced Visualizer for biological neuron metrics with repair strategy insights"""
    def __init__(self, save_dir: str = "bio_visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize history containers
        self.health_history = []
        self.stability_history = []
        self.calcium_history = []
        self.repair_events = []
        self.steps = []
        
        # New tracking for repair strategies
        self.repair_strategy_history = {
            'adaptive_noise': [],
            'targeted_repair': [],
            'gradient_aware_repair': [],
            'periodic_reset': []
        }
        
        # Set style properly
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
    def update(self, step: int, health_report: Dict, calcium_level: torch.Tensor, repair_strategies: Dict = None):
        """Update histories with new data including repair strategies"""
        self.steps.append(step)
        self.health_history.append(health_report['current_health'].mean().item())
        self.stability_history.append(health_report['stability'].mean().item())
        self.calcium_history.append(calcium_level.mean().item())
        
        # Track repair strategies
        if repair_strategies:
            for strategy, count in repair_strategies.items():
                if strategy in self.repair_strategy_history:
                    self.repair_strategy_history[strategy].append(count)
        
        if health_report.get('repair_performed', False):
            self.repair_events.append((step, health_report['current_health'].mean().item()))
            
    def plot_repair_strategies(self):
        """Visualize repair strategy distribution and evolution"""
        fig, ax = plt.subplots(figsize=(15, 7))
        
        strategies = list(self.repair_strategy_history.keys())
        strategy_data = [self.repair_strategy_history[strategy] for strategy in strategies]
        
        # Cumulative area plot
        ax.stackplot(self.steps, strategy_data, labels=strategies, alpha=0.7)
        
        ax.set_title('Repair Strategy Evolution', fontsize=15)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Cumulative Repair Strategies', fontsize=12)
        ax.legend(loc='upper left')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'repair_strategies_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
            
    def plot_health_metrics(self):
        """Plot health and stability trends"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot health trend
        ax.plot(self.steps, self.health_history, label='Health', linewidth=2)
        ax.plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        
        # Mark repair events
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            ax.scatter(repair_steps, repair_health, color='red', marker='*', 
                      s=100, label='Repair Events', zorder=5)
            
        ax.set_title('Neuron Health and Stability Over Time', pad=20)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Metric Value')
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'health_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_calcium_dynamics(self):
        """Plot calcium level trends"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot calcium levels
        ax.plot(self.steps, self.calcium_history, label='Calcium Level', 
                color='purple', linewidth=2)
        
        # Add rolling average
        window = min(50, len(self.calcium_history))
        if window > 1:
            rolling_avg = pd.Series(self.calcium_history).rolling(window=window).mean()
            ax.plot(self.steps, rolling_avg, label=f'{window}-step Average',
                   color='blue', linestyle='--', linewidth=1.5)
        
        ax.set_title('Calcium Dynamics Over Time', pad=20)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Calcium Level')
        ax.legend(frameon=True)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'calcium_dynamics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_phase_diagram(self):
        """Generate phase diagram showing relationship between health and calcium levels"""
        if not self.health_history or not self.calcium_history:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create phase plot
        plt.scatter(self.health_history, self.calcium_history, 
                   c=self.steps, cmap='viridis', 
                   alpha=0.6, s=50)
        
        # Add arrows to show direction of time
        step_interval = max(1, len(self.steps) // 20)  # Show arrows at 20 points
        for i in range(0, len(self.steps) - 1, step_interval):
            plt.arrow(self.health_history[i], self.calcium_history[i],
                     (self.health_history[i+1] - self.health_history[i]) * 0.2,
                     (self.calcium_history[i+1] - self.calcium_history[i]) * 0.2,
                     head_width=0.01, head_length=0.02, fc='gray', ec='gray',
                     alpha=0.5)
        
        plt.colorbar(label='Training Steps')
        plt.xlabel('Health Level')
        plt.ylabel('Calcium Level')
        plt.title('Health-Calcium Phase Diagram', pad=20)
        
        # Add repair event markers if they exist
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            repair_calcium = [self.calcium_history[self.steps.index(step)] 
                            for step in repair_steps]
            plt.scatter(repair_health, repair_calcium, 
                       color='red', marker='*', s=200,
                       label='Repair Events', zorder=5)
            plt.legend()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'phase_diagram_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_comprehensive_summary(self):
        """Generate a comprehensive summary plot with repair strategies"""
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Health and Stability
        axs[0, 0].plot(self.steps, self.health_history, label='Health', linewidth=2)
        axs[0, 0].plot(self.steps, self.stability_history, label='Stability', linewidth=2)
        if self.repair_events:
            repair_steps, repair_health = zip(*self.repair_events)
            axs[0, 0].scatter(repair_steps, repair_health, color='red', marker='*',
                               s=100, label='Repair Events', zorder=5)
        axs[0, 0].set_title('Neuron Health Metrics', fontsize=12)
        axs[0, 0].legend()
        
        # Calcium Dynamics
        axs[0, 1].plot(self.steps, self.calcium_history, label='Calcium Level', color='purple')
        window = min(50, len(self.calcium_history))
        if window > 1:
            rolling_avg = pd.Series(self.calcium_history).rolling(window=window).mean()
            axs[0, 1].plot(self.steps, rolling_avg, label=f'{window}-step Average', 
                           color='blue', linestyle='--')
        axs[0, 1].set_title('Calcium Level Dynamics', fontsize=12)
        axs[0, 1].legend()
        
        # Repair Strategies Cumulative Plot
        strategies = list(self.repair_strategy_history.keys())
        strategy_data = [self.repair_strategy_history[strategy] for strategy in strategies]
        axs[1, 0].stackplot(self.steps, strategy_data, labels=strategies, alpha=0.7)
        axs[1, 0].set_title('Repair Strategy Distribution', fontsize=12)
        axs[1, 0].legend(loc='upper left')
        
        # Health-Calcium Phase Space
        scatter = axs[1, 1].scatter(self.health_history, self.calcium_history, 
                                    c=self.steps, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=axs[1, 1], label='Training Steps')
        axs[1, 1].set_title('Health-Calcium Phase Space', fontsize=12)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.save_dir / f'comprehensive_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_all_plots(self):
        """Generate and save all visualization plots"""
        self.plot_health_metrics()
        self.plot_calcium_dynamics()
        self.plot_phase_diagram()
        self.plot_repair_strategies()  # New method
        self.generate_comprehensive_summary()  # Enhanced summary