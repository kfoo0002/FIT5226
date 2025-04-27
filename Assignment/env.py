import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Define the Q-table agent with direction and item status
class QTableAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.has_item = 0
        self.direction = None  # True for A->B, False for B->A
        self.position = None
        self.previous_position = None  # Store previous position for collision detection
        self.local_mask = 0  # 8-bit mask for neighboring agents
        self.update_order = None  # Position in the update sequence
        self.collision_penalty = 0  # Track collision penalties
        self.num_actions = 4  # Number of possible actions (up, down, left, right)
        self.completed_round_trip = False  # Track if agent has completed their round trip

    def choose_action(self, state): 
        # hook for the policy
        return np.random.randint(self.num_actions)

# Define the grid world environment
class GridWorldEnvironment:
    def __init__(self, n, m, num_agents=4):
        self.n = n
        self.m = m
        self.num_agents = num_agents
        self.agents = [QTableAgent(i) for i in range(num_agents)]
        self.food_source_location = (0, 0)  # Location A
        self.nest_location = (n-1, m-1)     # Location B
        self.rewards = np.zeros((n, m))
        self.rewards[self.food_source_location[0], self.food_source_location[1]] = 10
        self.rewards[self.nest_location[0], self.nest_location[1]] = 50
        # Create figure once
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Define relative positions for 8 directions
        self.directions = [
            (-1, 0),   # N
            (-1, 1),   # NE
            (0, 1),    # E
            (1, 1),    # SE
            (1, 0),    # S
            (1, -1),   # SW
            (0, -1),   # W
            (-1, -1)   # NW
        ]
        
        # Initialize central clock
        self.clock = 0
        self.update_sequence = None
        self.set_update_sequence()  # Set initial update sequence
        
        # Collision tracking
        self.collision_count = 0
        self.collision_penalty = -20  # Penalty for head-on collision
        
        # Round trip tracking
        self.round_trip_count = 0  # Track number of completed round trips

    def set_update_sequence(self, sequence_type='round_robin'):
        """
        Set the update sequence for agents.
        sequence_type can be:
        - 'round_robin': agents take turns in order
        - 'random': random order each time
        - 'fixed': fixed order based on agent IDs
        """
        if sequence_type == 'round_robin':
            self.update_sequence = list(range(self.num_agents))
        elif sequence_type == 'random':
            self.update_sequence = np.random.permutation(self.num_agents).tolist()
        elif sequence_type == 'fixed':
            self.update_sequence = sorted(range(self.num_agents))
        
        # Assign update order to each agent
        for i, agent_id in enumerate(self.update_sequence):
            self.agents[agent_id].update_order = i
    
    def get_next_agent(self):
        """Get the next agent to update based on the current clock"""
        agent_id = self.update_sequence[self.clock % self.num_agents]
        self.clock += 1
        return agent_id

    def get_local_mask(self, agent_id):
        """
        Returns an 8-bit mask indicating presence of opposite-direction agents
        in the 8 neighboring cells. Each bit corresponds to a direction:
        [N, NE, E, SE, S, SW, W, NW]
        """
        current_agent = self.agents[agent_id]
        mask = 0
        
        for i, (dx, dy) in enumerate(self.directions):
            # Calculate neighboring cell position
            neighbor_row = current_agent.position[0] + dx
            neighbor_col = current_agent.position[1] + dy
            
            # Check if the position is within grid bounds
            if 0 <= neighbor_row < self.n and 0 <= neighbor_col < self.m:
                # Check all other agents
                for other_agent in self.agents:
                    if other_agent.agent_id != agent_id:  # Skip self
                        if (other_agent.position == (neighbor_row, neighbor_col) and 
                            other_agent.direction != current_agent.direction):
                            # Set the corresponding bit if opposite-direction agent found
                            mask |= (1 << i)
        
        return mask

    def _reset(self):
        # Reset all agents
        for agent in self.agents:
            # Randomly place agent at either A or B
            if np.random.random() < 0.5:
                agent.position = self.food_source_location
                agent.direction = True  # A->B
                agent.has_item = 1  # Start with item if at A
            else:
                agent.position = self.nest_location
                agent.direction = False  # B->A
                agent.has_item = 0  # No item if starting at B
            agent.local_mask = 0
            agent.previous_position = None
            agent.collision_penalty = 0
            agent.completed_round_trip = False  # Reset round trip completion
        
        # Reset clock and update sequence
        self.clock = 0
        self.set_update_sequence()
        self.collision_count = 0
        self.round_trip_count = 0  # Reset round trip count

    def get_state(self, agent_id):
        agent = self.agents[agent_id]
        # Get the 8-bit local mask for Moore neighbors
        local_mask = self.get_local_mask(agent_id)
        agent.local_mask = local_mask  # Store the mask back to the agent
        
        # Convert local mask to 8 binary flags
        neighbor_flags = [(local_mask >> i) & 1 for i in range(8)]
        
        # Construct the 15-dimensional state vector
        state = [
            agent.position[0],  # agent_row (0-4)
            agent.position[1],  # agent_col (0-4)
            self.food_source_location[0],  # A_row (pickup)
            self.food_source_location[1],  # A_col
            self.nest_location[0],  # B_row (drop-off)
            self.nest_location[1],  # B_col
            agent.has_item,  # carry_flag (0/1)
        ]
        # Add the 8 neighbor flags
        state.extend(neighbor_flags)
        
        return state

    def check_done(self):
        # Check if all agents have completed their round trips
        for agent in self.agents:
            if not agent.completed_round_trip:
                return False
        return True
    
    def check_collisions(self):
        """Check for collisions between agents with different item states"""
        # Create a dictionary to track positions and agents
        position_agents = {}
        
        # First pass: collect all agents at each position
        for agent in self.agents:
            if agent.position not in position_agents:
                position_agents[agent.position] = []
            position_agents[agent.position].append(agent)
        
        # Second pass: check for collisions
        for position, agents in position_agents.items():
            # Skip if only one agent at this position
            if len(agents) < 2:
                continue
                
            # Skip if position is A or B
            if position == self.food_source_location or position == self.nest_location:
                continue
                
            # Check if agents with different item states are colliding
            has_item = False
            no_item = False
            
            for agent in agents:
                if agent.has_item:
                    has_item = True
                else:
                    no_item = True
            
            # If both item states are present, it's a collision
            if has_item and no_item:
                self.collision_count += 1
                # Apply penalty to all agents involved
                for agent in agents:
                    agent.collision_penalty += self.collision_penalty

    def take_action(self, agent_id, action):
        agent = self.agents[agent_id]
        row, col = agent.position

        # Store previous position before moving
        agent.previous_position = agent.position

        # Perform the chosen action and observe the next state and reward
        if action == 0:  # Up
            next_position = (max(row - 1, 0), col)
        elif action == 1:  # Down
            next_position = (min(row + 1, self.n - 1), col)
        elif action == 2:  # Left
            next_position = (row, max(col - 1, 0))
        elif action == 3:  # Right
            next_position = (row, min(col + 1, self.m - 1))
       
        # Update agent's state and calculate reward
        reward = -1  # Default step penalty for any attempted move
        
        # Check for automatic pickup at A
        if next_position == self.food_source_location:
            agent.has_item = 1
            agent.direction = True  # Set direction to A->B
            reward = 10  # Direct pickup reward
        # Check for automatic dropoff at B
        elif next_position == self.nest_location and agent.has_item:
            agent.has_item = 0
            agent.direction = False  # Set direction to B->A
            reward = 50  # Direct dropoff reward
            
            # Check for round trip completion
            if agent.direction == True:  # A->B direction
                if agent.position == self.food_source_location and not agent.has_item:
                    agent.completed_round_trip = True  # Completed A->B->A
                    self.round_trip_count += 1  # Increment round trip count
            else:  # B->A direction
                if agent.position == self.nest_location and not agent.has_item:
                    agent.completed_round_trip = True  # Completed B->A->B
                    self.round_trip_count += 1  # Increment round trip count
            
        agent.position = next_position
        return reward

    def visualize(self, steps, total_reward):
        self.ax.clear()
        
        # Draw grid
        for i in range(self.n + 1):
            self.ax.axhline(y=i, color='k', linestyle='-')
            self.ax.axvline(x=i, color='k', linestyle='-')
        
        # Draw pickup and drop-off locations
        pickup_circle = patches.Circle((self.food_source_location[1] + 0.5, self.n - self.food_source_location[0] - 0.5), 
                                     0.3, facecolor='green')
        dropoff_circle = patches.Circle((self.nest_location[1] + 0.5, self.n - self.nest_location[0] - 0.5), 
                                      0.3, facecolor='red')
        self.ax.add_patch(pickup_circle)
        self.ax.add_patch(dropoff_circle)
        
        # Draw agents
        colors = ['blue', 'purple', 'orange', 'cyan']
        for i, agent in enumerate(self.agents):
            if agent.position:
                # Draw agent
                agent_circle = patches.Circle((agent.position[1] + 0.5, self.n - agent.position[0] - 0.5), 
                                            0.3, facecolor=colors[i])
                self.ax.add_patch(agent_circle)
                
                # Draw agent ID and update order
                self.ax.text(agent.position[1] + 0.5, self.n - agent.position[0] - 0.5, 
                           f'A{i}({agent.update_order})', ha='center', va='center', color='white')
                
                # Draw item indicator if carrying
                if agent.has_item:
                    self.ax.text(agent.position[1] + 0.5, self.n - agent.position[0] - 0.3, 
                               '●', ha='center', va='center', color='white')
                
                # Draw local mask visualization
                mask = agent.local_mask
                for j, (dx, dy) in enumerate(self.directions):
                    if mask & (1 << j):
                        # Draw a small indicator for occupied neighboring cells
                        neighbor_x = agent.position[1] + dy + 0.5
                        neighbor_y = self.n - (agent.position[0] + dx) - 0.5
                        self.ax.text(neighbor_x, neighbor_y, 'x', 
                                   ha='center', va='center', color='red', fontsize=8)
        
        # Add labels for pickup and drop-off
        self.ax.text(self.food_source_location[1] + 0.5, self.n - self.food_source_location[0] - 0.5, 
                    'A', ha='center', va='center', color='white')
        self.ax.text(self.nest_location[1] + 0.5, self.n - self.nest_location[0] - 0.5, 
                    'B', ha='center', va='center', color='white')
        
        # Set limits and remove ticks
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(0, self.n)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add status information
        title = f'Grid World - Step: {steps}/100 - Clock: {self.clock}\nTotal Reward: {total_reward} - Collisions: {self.collision_count} - Round Trips: {self.round_trip_count}'
        plt.title(title)
        
        plt.draw()
        plt.pause(0.1) 