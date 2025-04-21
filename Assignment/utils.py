import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EpsilonScheduler:
    """Linear epsilon decay scheduler"""
    def __init__(self, start_eps=1.0, end_eps=0.01, decay_steps=10000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps
        self.step = 0
        
    def get_epsilon(self):
        """Get current epsilon value"""
        if self.step >= self.decay_steps:
            return self.end_eps
        eps = self.start_eps - (self.start_eps - self.end_eps) * (self.step / self.decay_steps)
        self.step += 1
        return eps

class MetricLogger:
    """Log and plot training metrics"""
    def __init__(self):
        self.episode_rewards = []
        self.episode_collisions = []
        self.episode_deliveries = []
        self.episode_lengths = []
        
    def log_episode(self, reward, collisions, deliveries, length):
        """Log metrics for an episode"""
        self.episode_rewards.append(reward)
        self.episode_collisions.append(collisions)
        self.episode_deliveries.append(deliveries)
        self.episode_lengths.append(length)
        
    def plot_metrics(self, window=100):
        """Plot metrics with moving average"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        rewards_ma = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        axs[0,0].plot(rewards_ma)
        axs[0,0].set_title('Episode Rewards (Moving Average)')
        
        # Plot collisions
        collisions_ma = np.convolve(self.episode_collisions, np.ones(window)/window, mode='valid')
        axs[0,1].plot(collisions_ma)
        axs[0,1].set_title('Episode Collisions (Moving Average)')
        
        # Plot deliveries
        deliveries_ma = np.convolve(self.episode_deliveries, np.ones(window)/window, mode='valid')
        axs[1,0].plot(deliveries_ma)
        axs[1,0].set_title('Episode Deliveries (Moving Average)')
        
        # Plot episode lengths
        lengths_ma = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
        axs[1,1].plot(lengths_ma)
        axs[1,1].set_title('Episode Lengths (Moving Average)')
        
        plt.tight_layout()
        plt.show()
        
    def get_stats(self):
        """Get current statistics"""
        return {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_collisions': np.mean(self.episode_collisions[-100:]),
            'mean_deliveries': np.mean(self.episode_deliveries[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:])
        } 