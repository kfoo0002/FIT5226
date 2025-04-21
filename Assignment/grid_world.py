import numpy as np
from time import process_time
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

    def choose_action(self, state):
        # hook for the policy
        return np.random.randint(num_actions)

# Define the grid world environment
class GridWorldEnvironment:
    def __init__(self, n, m, num_agents=4):
        self.n = n
        self.m = m
        self.num_agents = num_agents
        self.agents = [QTableAgent(i) for i in range(num_agents)]
        self.food_source_location = (0, 0)  # Location A
        self.nest_location = (n-1, m-1)     # Location B
        self.rewards = np.zeros((grid_rows, grid_cols))
        self.rewards[self.food_source_location[0], self.food_source_location[1]] = 10
        self.rewards[self.nest_location[0], self.nest_location[1]] = 50
        # Create figure once
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
     
    def _reset(self):
        # Reset all agents
        for agent in self.agents:
            # Randomly place agent at either A or B
            if np.random.random() < 0.5:
                agent.position = self.food_source_location
                agent.direction = True  # A->B
            else:
                agent.position = self.nest_location
                agent.direction = False  # B->A
            agent.has_item = 0
      
    def get_state(self, agent_id):
        agent = self.agents[agent_id]
        return (agent.position[0], agent.position[1], 
                agent.direction, agent.has_item)

    def check_done(self):
        # Check if all agents have completed their tasks
        for agent in self.agents:
            if agent.direction and agent.position != self.nest_location:
                return False
            if not agent.direction and agent.position != self.food_source_location:
                return False
        return True
    
    def take_action(self, agent_id, action):
        agent = self.agents[agent_id]
        row, col = agent.position

        # Perform the chosen action and observe the next state and reward
        if action == 0:  # Up
            next_position = (max(row - 1, 0), col)
        elif action == 1:  # Down
            next_position = (min(row + 1, grid_rows - 1), col)
        elif action == 2:  # Left
            next_position = (row, max(col - 1, 0))
        elif action == 3:  # Right
            next_position = (row, min(col + 1, grid_cols - 1))
       
        # Update agent's state and calculate reward
        reward = -1  # Default step penalty
        
        if agent.direction:  # A->B
            if not agent.has_item and next_position == self.food_source_location:
                agent.has_item = 1
                reward = self.rewards[next_position]
            elif agent.has_item and next_position == self.nest_location:
                agent.has_item = 0
                reward = self.rewards[next_position]
        else:  # B->A
            if not agent.has_item and next_position == self.nest_location:
                agent.has_item = 1
                reward = self.rewards[next_position]
            elif agent.has_item and next_position == self.food_source_location:
                agent.has_item = 0
                reward = self.rewards[next_position]
            
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
                
                # Draw direction indicator
                direction = "→" if agent.direction else "←"
                self.ax.text(agent.position[1] + 0.5, self.n - agent.position[0] - 0.5, 
                           f'A{i}{direction}', ha='center', va='center', color='white')
                
                # Draw item indicator if carrying
                if agent.has_item:
                    self.ax.text(agent.position[1] + 0.5, self.n - agent.position[0] - 0.3, 
                               '●', ha='center', va='center', color='white')
        
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
        title = f'Grid World - Step: {steps}/100\nTotal Reward: {total_reward}'
        plt.title(title)
        
        plt.draw()
        plt.pause(0.1)

if __name__ == "__main__":
    t = process_time()

    num_episodes = 0  # Changed to 0 to run exactly 1 episode (episode 0)
    max_steps = 100 
    
    # Define the grid world dimensions
    grid_rows = 5
    grid_cols = 5
    
    # Define the number of actions (up, down, left, right)
    num_actions = 4
    
    # Create the grid world environment with 4 agents
    environment = GridWorldEnvironment(grid_rows, grid_cols, num_agents=4)
    
    reward_total = []
    
    print("Starting simulation...")
    plt.ion()  # Turn on interactive mode
    
    for episode in range(num_episodes+1):   # This will run only episode 0
        environment._reset()
        number_of_steps = 0
        reward_per_episode = 0
        
        while number_of_steps <= max_steps and not environment.check_done():
            # Each agent takes its turn in sequence
            for agent_id in range(environment.num_agents):
                state = environment.get_state(agent_id)
                action = environment.agents[agent_id].choose_action(state)
                reward = environment.take_action(agent_id, action)
                reward_per_episode += reward
            
            # Visualize after all agents have moved
            environment.visualize(number_of_steps, reward_per_episode)
            
            # Add a small delay to make the movement visible
            time.sleep(0.1)
            
            number_of_steps += 1
            
            if environment.check_done():
                print(f"Success! All agents completed their tasks in {number_of_steps} steps!")
                break
            elif number_of_steps >= max_steps:
                print(f"Failed to complete all tasks within {max_steps} steps.")
                break
            
        reward_total.append(reward_per_episode)
        print(f"Episode {episode} finished with total reward: {reward_per_episode}")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final visualization window open
    
    elapsed_time = process_time() - t
    
    print("Finished in", elapsed_time, "seconds.")
    print("Average reward:", np.mean(reward_total)) 