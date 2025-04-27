import numpy as np
import torch
import time
from time import process_time
import matplotlib.pyplot as plt
from env import GridWorldEnvironment
from dqn import ReplayBuffer
from utils import (set_seed, EpsilonScheduler, MetricLogger,
                  prepare_torch, update_target, get_qvals, get_maxQ, train_one_step)
from collections import deque

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    num_episodes = 1000
    max_steps = 200
    grid_rows = 5
    grid_cols = 5
    num_actions = 4  # up, down, left, right
    num_agents = 4
    gamma = 0.997    # discount factor
    batch_size = 200 
    buffer_size = 1000 
    target_update = 500 # Update target network every 500 steps
    
    # Budgets
    step_budget = 1_500_000
    collision_budget = 4_000
    walltime_budget = 600  # 10 minutes in seconds
    success_threshold = 0.95  # 95% success rate
    
    # Track success rates
    success_rates = []
    
    # Initialize environment
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
    
    # Track total steps and collisions
    total_steps = 0
    total_collisions = 0
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):  # Start from 1 to num_episodes
        environment._reset()
        number_of_steps = 0
        reward_per_episode = 0
        episode_collisions = 0
        episode_round_trips = 0
        
        # Get epsilon for this episode
        epsilon = epsilon_scheduler.get_epsilon()
        
        while number_of_steps < max_steps and not environment.check_done():
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
                    state = np.array(state)  # Convert state to numpy array
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
                
                number_of_steps += 1
                total_steps += 1
                
                if number_of_steps >= max_steps or environment.check_done():
                    break
            
            # Check for collisions after all agents have moved in the cycle
            environment.check_collisions()
            total_collisions += environment.collision_count - episode_collisions
            episode_collisions = environment.collision_count
            
            # Add collision penalties to all agents involved
            for agent_id in range(environment.num_agents):
                reward_per_episode += environment.agents[agent_id].collision_penalty
                environment.agents[agent_id].collision_penalty = 0  # Reset collision penalty
            
            # Update metrics
            episode_round_trips = environment.round_trip_count
            
            # Update target network periodically
            if number_of_steps % target_update == 0:
                update_target()
            
            # Visualize every 10 episodes
            if episode % 10 == 0:
                environment.visualize(number_of_steps, reward_per_episode)
                time.sleep(0.1)
        
        # Log episode metrics
        metric_logger.log_episode(reward_per_episode, episode_collisions, 
                                episode_round_trips, number_of_steps)
        
        # Print progress and evaluate
        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Total reward: {reward_per_episode}")
            print(f"Total collisions: {episode_collisions}")
            print(f"Total round trips: {episode_round_trips}")
            print(f"Steps taken: {number_of_steps}")
            print(f"Epsilon: {epsilon:.2f}")
            print(f"Total steps so far: {total_steps}")
            print(f"Total collisions so far: {total_collisions}")
            print(f"Walltime: {(time.time() - start_time):.2f} seconds")
            
            # Evaluation for episodes >= 50
            if episode >= 50:
                success_rate = evaluate_success_rate(environment, num_agents)
                success_rates.append(success_rate)
                print(f"Success rate: {success_rate:.2%}")
                # Count evaluations above threshold
                above_threshold = sum(1 for rate in success_rates if rate >= success_threshold)
                print(f"Evaluations above {success_threshold:.2%} threshold: {above_threshold}/{len(success_rates)}")
            print("---")
        
        # Check early stopping conditions
        if total_steps >= step_budget:
            print(f"Stopping: Reached step budget of {step_budget}")
            break
            
        if total_collisions >= collision_budget:
            print(f"Stopping: Reached collision budget of {collision_budget}")
            break
            
        if (time.time() - start_time) >= walltime_budget:
            print(f"Stopping: Reached walltime budget of {walltime_budget} seconds")
            break
    
    # Plot final metrics
    plt.ioff()
    metric_logger.plot_metrics()
    
    # Print final statistics
    stats = metric_logger.get_stats()
    print("\nFinal Statistics:")
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Mean collisions: {stats['mean_collisions']:.2f}")
    print(f"Mean round trips: {stats['mean_round_trips']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total walltime: {(time.time() - start_time):.2f} seconds")
    # Calculate and print average success rate
    avg_success_rate = sum(success_rates) / len(success_rates)
    print(f"Average success rate: {avg_success_rate:.2%}")
    print(f"Number of evaluations: {len(success_rates)}")
    
    # Evaluate performance points
    if avg_success_rate > 0.95 and total_collisions < 500:
        print("2 performance points!")
    elif avg_success_rate > 0.85 and total_collisions < 1000:
        print("1 performance point!")
    else:
        print("No performance points earned.")

def evaluate_success_rate(environment, num_agents, eval_episodes=50):
    """Evaluate the current policy's success rate based on round trips using a rolling window"""
    # Track success/failure for the last 50 episodes
    success_history = deque(maxlen=eval_episodes)
    
    for _ in range(eval_episodes):
        environment._reset()
        steps = 0
        success = False
        
        # Track round trip progress for each agent
        # For A→B→A: 0: at A(no item), 1: going to B(with item), 2: at B(no item), 3: going back to A(no item)
        # For B→A→B: 0: at B(no item), 1: going to A(no item), 2: at A(with item), 3: going back to B(with item)
        agent_states = [0] * num_agents
        agent_round_trips = [False] * num_agents  # Track if each agent completed a round trip
        
        while steps < 20 and not success:  # Max 20 steps per evaluation
            # Reset collision count for this step
            environment.collision_count = 0
            
            # Each agent takes their turn
            for _ in range(num_agents):
                agent_id = environment.get_next_agent()
                state = np.array(environment.get_state(agent_id))
                q_values = get_qvals(state)
                action = np.argmax(q_values)  # Greedy action
                environment.take_action(agent_id, action)
                
                # Update agent's round trip state
                agent = environment.agents[agent_id]
                current_pos = agent.position
                
                # Check for A→B→A round trip
                if agent_states[agent_id] == 0 and current_pos == environment.food_source_location and agent.has_item:
                    agent_states[agent_id] = 1  # Picked up item at A, going to B
                elif agent_states[agent_id] == 1 and current_pos == environment.nest_location and not agent.has_item:
                    agent_states[agent_id] = 2  # Dropped off at B
                elif agent_states[agent_id] == 2 and current_pos == environment.food_source_location and not agent.has_item:
                    agent_states[agent_id] = 3  # Completed A→B→A round trip
                    agent_round_trips[agent_id] = True
                    success = True
                
                # Check for B→A→B round trip
                elif agent_states[agent_id] == 0 and current_pos == environment.nest_location and not agent.has_item:
                    agent_states[agent_id] = 1  # Going to A without item
                elif agent_states[agent_id] == 1 and current_pos == environment.food_source_location and agent.has_item:
                    agent_states[agent_id] = 2  # Picked up at A, going back to B
                elif agent_states[agent_id] == 2 and current_pos == environment.nest_location and not agent.has_item:
                    agent_states[agent_id] = 3  # Completed B→A→B round trip
                    agent_round_trips[agent_id] = True
                    success = True
            
            # Check for collisions after all agents move
            environment.check_collisions()
            if environment.collision_count > 0:
                break  # Failed if any collisions
            
            steps += 1
        
        # Record success/failure for this episode
        episode_success = success and steps < 20 and environment.collision_count == 0
        success_history.append(episode_success)
    
    # Calculate rolling success rate
    rolling_success_rate = sum(success_history) / len(success_history)
    return rolling_success_rate

if __name__ == "__main__":
    t = process_time()
    main()
    elapsed_time = process_time() - t
    print(f"\nFinished in {elapsed_time:.2f} seconds.") 