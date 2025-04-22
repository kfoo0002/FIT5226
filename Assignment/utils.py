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
    return torch.max(model2(torch.from_numpy(s).float())).float()

def train_one_step(states, actions, targets, gamma):
    """Perform one training step"""
    state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
    action_batch = torch.Tensor(actions)
    Q1 = model(state1_batch)
    X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
    Y = torch.tensor(targets)
    loss = loss_fn(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

class EpsilonScheduler:
    """Linear epsilon decay scheduler"""
    def __init__(self, start_eps=1.0, end_eps=0.01, decay_steps=10000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps
        self.steps = 0
        
    def get_epsilon(self):
        """Get current epsilon value"""
        eps = self.start_eps - (self.start_eps - self.end_eps) * (self.steps / self.decay_steps)
        self.steps += 1
        return max(eps, self.end_eps)

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