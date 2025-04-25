import numpy as np
import torch
import time
from time import process_time
import matplotlib.pyplot as plt
from env import GridWorldEnvironment
from dqn import ReplayBuffer
from utils import (set_seed, EpsilonScheduler, MetricLogger,
                  prepare_torch, update_target, get_qvals, get_maxQ, train_one_step)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    num_episodes = 1000
    max_steps = 100
    grid_rows = 5
    grid_cols = 5
    num_actions = 4  # up, down, left, right
    num_agents = 4
    gamma = 0.997    # Learning rate (discount factor)
    batch_size = 200 # Increased from 128
    buffer_size = 1000 # Decreased from 50000
    target_update = 500 # Update target network every 500 steps
    
    # Initialize environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = GridWorldEnvironment(grid_rows, grid_cols, num_agents=num_agents)
    
    # Initialize networks and optimizer
    prepare_torch()
    
    # Initialize replay buffers for each agent
    replay_buffers = [ReplayBuffer(buffer_size) for _ in range(num_agents)]
    
    # Initialize epsilon scheduler and metric logger
    epsilon_scheduler = EpsilonScheduler(start_eps=1.0, min_eps=0.1, decay_factor=0.997)
    metric_logger = MetricLogger()
    
    print("Starting training...")
    plt.ion()
    
    for episode in range(num_episodes):
        environment._reset()
        number_of_steps = 0
        reward_per_episode = 0
        episode_collisions = 0
        episode_deliveries = 0
        
        # Get epsilon for this episode
        epsilon = epsilon_scheduler.get_epsilon()
        
        while number_of_steps <= max_steps and not environment.check_done():
            # Store previous positions before any moves
            for agent in environment.agents:
                agent.previous_position = agent.position
            
            # Each agent takes its turn in sequence based on the clock
            for _ in range(environment.num_agents):
                agent_id = environment.get_next_agent()
                state = environment.get_state(agent_id)
                
                # Select action using epsilon-greedy policy
                if np.random.random() < epsilon:
                    action = np.random.randint(num_actions)
                else:
                    q_values = get_qvals(state)
                    action = np.argmax(q_values)
                
                # Take action and observe reward
                reward = environment.take_action(agent_id, action)
                reward_per_episode += reward
                
                # Get next state and store experience
                next_state = environment.get_state(agent_id)
                done = environment.check_done()
                replay_buffers[agent_id].push(state, action, reward, next_state, done)
                
                # Update network if enough experiences
                if len(replay_buffers[agent_id]) >= batch_size:
                    # Sample batch
                    states, actions, rewards, next_states, dones = replay_buffers[agent_id].sample(batch_size)
                    
                    # Compute targets
                    targets = []
                    for i in range(batch_size):
                        if dones[i]:
                            targets.append(rewards[i])
                        else:
                            next_q = get_maxQ(next_states[i])
                            targets.append(rewards[i] + gamma * next_q)
                    
                    # Train one step
                    loss = train_one_step(states, actions, targets, gamma)
            
            # Check for collisions after all agents have moved
            environment.check_collisions()
            
            # Add collision penalties to total reward
            for agent in environment.agents:
                reward_per_episode += agent.collision_penalty
                agent.collision_penalty = 0  # Reset collision penalty
            
            # Update metrics
            episode_collisions = environment.collision_count
            episode_deliveries = environment.delivery_count
            
            # Update target network periodically
            if number_of_steps % target_update == 0:
                update_target()
            
            # Visualize every 10 episodes
            if episode % 10 == 0:
                environment.visualize(number_of_steps, reward_per_episode)
                time.sleep(0.1)
            
            number_of_steps += 1
            
            if environment.check_done():
                print(f"Success! All agents completed their tasks in {number_of_steps} steps!")
                break
            elif number_of_steps >= max_steps:
                print(f"Failed to complete all tasks within {max_steps} steps.")
                break
        
        # Log episode metrics
        metric_logger.log_episode(reward_per_episode, episode_collisions, 
                                episode_deliveries, number_of_steps)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Total reward: {reward_per_episode}")
            print(f"Total collisions: {episode_collisions}")
            print(f"Total deliveries: {episode_deliveries}")
            print(f"Steps taken: {number_of_steps}")
            print(f"Epsilon: {epsilon_scheduler.get_epsilon():.2f}")
            print("---")
    
    # Plot final metrics
    plt.ioff()
    metric_logger.plot_metrics()
    
    # Print final statistics
    stats = metric_logger.get_stats()
    print("\nFinal Statistics:")
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Mean collisions: {stats['mean_collisions']:.2f}")
    print(f"Mean deliveries: {stats['mean_deliveries']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.2f}")

if __name__ == "__main__":
    t = process_time()
    main()
    elapsed_time = process_time() - t
    print(f"\nFinished in {elapsed_time:.2f} seconds.") 