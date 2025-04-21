import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
        
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN Agent with target network and experience replay"""
    def __init__(self, state_dim, action_dim, device, lr=1e-4, gamma=0.99, 
                 buffer_size=10000, batch_size=64, target_update=1000):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.steps = 0
        
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def update(self):
        """Update policy network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
            expected_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item() 