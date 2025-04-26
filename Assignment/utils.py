import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import copy

# Global variables for the network
statespace_size = 15  # Our state space size
model = None
model2 = None
optimizer = None
loss_fn = None

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

def prepare_torch():
    """Initialize the neural networks and optimizer"""
    global statespace_size
    global model, model2
    global optimizer
    global loss_fn
    l1 = statespace_size  # 15 in our case
    l2 = 24
    l3 = 24
    l4 = 4
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4))
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model2

def update_target():
    """Copy weights from policy network to target network"""
    global model, model2
    model2.load_state_dict(model.state_dict())

def get_qvals(state):
    """Get Q-values for a state using policy network"""
    state1 = torch.from_numpy(state).float()
    qvals_torch = model(state1)
    qvals = qvals_torch.data.numpy()
    return qvals

def get_maxQ(s):
    """Get maximum Q-value for a state using target network"""
    return torch.max(model2(torch.from_numpy(s).float())).float().item()

def train_one_step(states, actions, targets, gamma):
    """Perform one training step"""
    # Convert states to tensor batch with proper shape [batch_size, state_dim]
    state1_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in states], dim=0)
    
    # Convert actions to tensor with proper shape [batch_size, 1]
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    
    # Get Q-values from policy network
    Q1 = model(state1_batch)
    
    # Gather Q-values for chosen actions and squeeze to [batch_size]
    X = Q1.gather(dim=1, index=action_batch).squeeze()
    
    # Convert targets to tensor with same shape as X [batch_size]
    Y = torch.tensor(targets, dtype=torch.float32)
    
    # Compute loss and update
    loss = loss_fn(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

class EpsilonScheduler:
    def __init__(self, start_eps=1.0, min_eps=0.1, decay_factor=0.997):
        self.start_eps = start_eps      # Starting epsilon (1.0)
        self.min_eps = min_eps          # Minimum epsilon (0.1)
        self.decay_factor = decay_factor # Decay factor (0.997)
        self.current_eps = start_eps     # Current epsilon value
        
    def get_epsilon(self):
        # Get current epsilon and decay it for next time
        current = self.current_eps
        self.current_eps = max(self.min_eps, self.current_eps * self.decay_factor)
        return current

class MetricLogger:
    """Log and plot training metrics"""
    def __init__(self):
        self.rewards = []
        self.collisions = []
        self.deliveries = []
        self.episode_lengths = []
        
    def log_episode(self, reward, collisions, deliveries, length):
        """Log metrics for an episode"""
        self.rewards.append(reward)
        self.collisions.append(collisions)
        self.deliveries.append(deliveries)
        self.episode_lengths.append(length)
        
    def plot_metrics(self):
        """Plot metrics with moving average"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.rewards)
        plt.title('Rewards per Episode')
        
        plt.subplot(2, 2, 2)
        plt.plot(self.collisions)
        plt.title('Collisions per Episode')
        
        plt.subplot(2, 2, 3)
        plt.plot(self.deliveries)
        plt.title('Deliveries per Episode')
        
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        
        plt.tight_layout()
        plt.show()
        
    def get_stats(self):
        """Get current statistics"""
        return {
            'mean_reward': np.mean(self.rewards),
            'mean_collisions': np.mean(self.collisions),
            'mean_deliveries': np.mean(self.deliveries),
            'mean_length': np.mean(self.episode_lengths)
        } 