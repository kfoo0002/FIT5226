import numpy as np
import torch
import time
from time import process_time
import matplotlib.pyplot as plt
from env import GridWorldEnvironment
from dqn import ReplayBuffer
from utils import (set_seed, EpsilonScheduler, MetricLogger,
                  prepare_torch, update_target, get_qvals, get_maxQ, train_one_step,
                  model)  # Import model from utils
from collections import deque
import csv
from datetime import datetime

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize networks and optimizer first
    policy_model = prepare_torch()  # Store the returned model
    
    # Hyperparameters
    num_episodes = 1000
    max_steps = 120
    grid_rows = 5
    grid_cols = 5
    num_actions = 4  # up, down, left, right
    num_agents = 4
    gamma = 0.9    # discount factor 0.997
    batch_size = 200  # Increased batch size for more stable training
    buffer_size = 50000  # Increased buffer size for more diverse experiences
    target_update = 1000 # Update target network every 500 steps
    
    # Budgets
    step_budget = 1_500_000
    collision_budget = 4_000
    walltime_budget = 600  # 10 minutes in seconds
    
    # Initialize environment
    environment = GridWorldEnvironment(grid_rows, grid_cols, num_agents=num_agents)
    
    # Initialize replay buffers for each agent
    replay_buffers = [ReplayBuffer(10000) for _ in range(environment.num_agents)]
    
    # Initialize epsilon scheduler and metric logger
    epsilon_scheduler = EpsilonScheduler(start_eps=1.0, min_eps=0.1, decay_factor=0.997)
    metric_logger = MetricLogger()
    
    # plt.ion()  # Commented out to disable visualization during training
    
    # Track total steps and collisions
    total_steps = 0
    total_collisions = 0
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Generate random positions for A and B
        while True:
            a_pos = (np.random.randint(grid_rows), np.random.randint(grid_cols))
            b_pos = (np.random.randint(grid_rows), np.random.randint(grid_cols))
            if a_pos != b_pos:  # Ensure A and B are at different positions
                break
        
        # Update environment positions instead of creating new instance
        environment.food_source_location = a_pos
        environment.nest_location = b_pos
        environment._reset()
        
        number_of_steps = 0
        reward_per_episode = 0
        episode_collisions = 0
        episode_round_trips = 0
        
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
                
                # Check for collisions after this agent's move
                environment.check_collisions()
                if environment.collision_count > 0:
                    # Add collision penalties to all agents involved
                    for a_id in range(environment.num_agents):
                        if environment.agents[a_id].collision_penalty != 0:
                            if a_id == agent_id:  # Add penalty to current agent's reward
                                reward += environment.agents[a_id].collision_penalty
                            reward_per_episode += environment.agents[a_id].collision_penalty
                            environment.agents[a_id].collision_penalty = 0  # Reset collision penalty
                    
                    # Update collision counts
                    total_collisions += environment.collision_count - episode_collisions
                    episode_collisions = environment.collision_count
                
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
            
            # Update metrics
            episode_round_trips = environment.round_trip_count
            
            # Update target network less frequently
            if total_steps % target_update == 0:  # Update target network every 500 steps
                update_target()
        
        # Log episode metrics
        metric_logger.log_episode(reward_per_episode, episode_collisions, 
                                episode_round_trips, number_of_steps)
        
        # Print training progress
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
    
    # Save the trained model
    torch.save(policy_model.state_dict(), 'trained_model.pth')
    print("Model saved to trained_model.pth")
    
    # Print training statistics
    print("\nTraining Statistics:")
    stats = metric_logger.get_stats()
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Mean collisions: {stats['mean_collisions']:.2f}")
    print(f"Mean round trips: {stats['mean_round_trips']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total walltime: {(time.time() - start_time):.2f} seconds")
    
    return policy_model

def load_model(path='trained_model.pth'):
    """Load the trained model."""
    # Create a new sequential model with same architecture
    l1 = 15  # state space size
    l2 = 24
    l3 = 24
    l4 = 4
    loaded_model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4))
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()
    return loaded_model

def evaluate_model():
    """
    Evaluates the trained model across all possible configurations.
    Tests B→A→B path completion for one agent in each configuration.
    
    Returns:
        float: Success rate (between 0 and 1)
        dict: Detailed statistics about the evaluation
    """
    # Constants
    GRID_SIZE = 5
    MAX_STEPS = 25
    AGENT_DISTRIBUTIONS = [
        (1,3),  # (B agents, A agents)
        (2,2),
        (3,1),
        (4,0)
    ]
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'test_results_{timestamp}.csv'
    
    # CSV headers
    headers = ['Test #', 'A Position', 'B Position', 'B Agents', 'A Agents', 
               'Result', 'Steps Taken', 'Failure Reason']
    
    total_configs = 0
    successful_configs = 0
    stats = {
        'total_tests': 0,
        'successful_tests': 0,
        'failures': {
            'collisions': 0,
            'timeout': 0,
            'incomplete': 0
        },
        'avg_steps_successful': 0,
        'distribution_success': {
            '1,3': 0,
            '2,2': 0,
            '3,1': 0,
            '4,0': 0
        }
    }

    # Open CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Generate all possible A-B configurations
        for a_pos_idx in range(GRID_SIZE * GRID_SIZE):
            a_pos = (a_pos_idx // GRID_SIZE, a_pos_idx % GRID_SIZE)
            
            for b_pos_idx in range(GRID_SIZE * GRID_SIZE):
                b_pos = (b_pos_idx // GRID_SIZE, b_pos_idx % GRID_SIZE)
                
                # Skip if A and B are in same position
                if a_pos == b_pos:
                    continue
                    
                # Test each agent distribution for this A-B configuration
                for b_agents, a_agents in AGENT_DISTRIBUTIONS:
                    total_configs += 1
                    stats['total_tests'] = total_configs
                    
                    print(f"\nTest Configuration #{total_configs}:")
                    print(f"A position: {a_pos}, B position: {b_pos}")
                    print(f"Agents at B: {b_agents}, Agents at A: {a_agents}")
                    
                    # Initialize environment with this configuration
                    env = GridWorldEnvironment(
                        n=GRID_SIZE,
                        m=GRID_SIZE,
                        num_agents=4,
                        food_source_location=a_pos,
                        nest_location=b_pos
                    )
                    
                    # Set up agents at their starting positions
                    success, steps = test_single_configuration(
                        env, 
                        b_agents, 
                        a_agents, 
                        MAX_STEPS
                    )
                    
                    # Determine failure reason
                    failure_reason = ''
                    if not success:
                        if steps >= MAX_STEPS:
                            failure_reason = 'Timeout'
                        elif env.collision_count > 0:
                            failure_reason = 'Collision'
                        else:
                            failure_reason = 'Incomplete path'
                    
                    # Write test results to CSV
                    writer.writerow([
                        total_configs,
                        f"{a_pos}",
                        f"{b_pos}",
                        b_agents,
                        a_agents,
                        'Success' if success else 'Failure',
                        steps,
                        failure_reason
                    ])
                    
                    # Print results to terminal
                    print(f"Result: {'Success' if success else 'Failure'}")
                    if success:
                        print(f"Steps taken: {steps}")
                    else:
                        print(f"Failed due to: {failure_reason}")
                    
                    if success:
                        successful_configs += 1
                        stats['successful_tests'] += 1
                        stats['avg_steps_successful'] += steps
                        stats['distribution_success'][f'{b_agents},{a_agents}'] += 1
                    else:
                        if steps >= MAX_STEPS:
                            stats['failures']['timeout'] += 1
                        elif env.collision_count > 0:
                            stats['failures']['collisions'] += 1
                        else:
                            stats['failures']['incomplete'] += 1

    # Calculate final statistics
    success_rate = successful_configs / total_configs
    if stats['successful_tests'] > 0:
        stats['avg_steps_successful'] /= stats['successful_tests']
    
    print(f"\nTest results have been exported to {csv_filename}")
    
    return success_rate, stats

def test_single_configuration(env, b_agents, a_agents, max_steps):
    """
    Tests a single configuration with the specified distribution of agents.
    Args:
        env: GridWorldEnvironment instance
        b_agents: Number of agents at point B
        a_agents: Number of agents at point A
        max_steps: Maximum steps allowed per agent
    Returns:
        tuple: (success: bool, steps_taken: int)
    """
    # Initialize agents at their positions
    env._reset()
    
    # Place agents at their starting positions
    # First agent (being tested) always starts at B
    env.agents[0].position = env.nest_location
    env.agents[0].has_item = False
    
    # Place remaining agents at B (up to b_agents-1)
    for i in range(1, b_agents):
        env.agents[i].position = env.nest_location
        env.agents[i].has_item = False
    
    # Place remaining agents at A
    for i in range(b_agents, 4):
        env.agents[i].position = env.food_source_location
        env.agents[i].has_item = False
    
    # Track progress for all agents
    agent_states = [0] * env.num_agents  # Track state for each agent
    agent_collisions = [False] * env.num_agents  # Track if each agent was involved in a collision
    steps = 0
    done = False
    success = False
    successful_agent = None  # Track which agent succeeded
    
    # Allow each agent to make max_steps moves (total moves = max_steps * 4)
    while not done and steps < max_steps * env.num_agents:
        # Each agent takes their turn in round-robin order
        for _ in range(env.num_agents):
            agent_id = env.get_next_agent()  # Use the same round-robin order as training
            
            # Get state and select action (no exploration, pure exploitation)
            state = env.get_state(agent_id)
            state = np.array(state)  # Convert state to numpy array
            q_values = get_qvals(state)
            action = np.argmax(q_values)
            
            # Take action
            reward = env.take_action(agent_id, action)
            
            # Check for collisions after this agent's move
            env.check_collisions()
            if env.collision_count > 0:
                # Find which agents were involved in collisions
                for a_id in range(env.num_agents):
                    if env.agents[a_id].collision_penalty != 0:
                        agent_collisions[a_id] = True
            
            # Update agent's progress state
            agent = env.agents[agent_id]
            current_pos = agent.position
            
            # Update B→A→B state for this agent
            if agent_states[agent_id] == 0 and current_pos == env.food_source_location:
                agent_states[agent_id] = 1  # Reached A (item automatically given)
            elif agent_states[agent_id] == 1 and current_pos == env.nest_location and not agent.has_item:
                agent_states[agent_id] = 2  # Completed B→A→B
                successful_agent = agent_id
                done = True
                break  # Exit the agent loop if any agent succeeds
            
            # Count each agent's move
            steps += 1
            if steps >= max_steps * env.num_agents:  # Check total step limit
                break
        
        if done:  # Break outer loop only if task is completed
            break
    
    # Test is successful if:
    # 1. An agent completed their round trip
    # 2. That agent wasn't involved in any collisions
    if successful_agent is not None and not agent_collisions[successful_agent]:
        success = True
    
    return success, steps

if __name__ == "__main__":
    t = process_time()
    
    # Run training only
    
    print("\nStarting training...")
    trained_model = main()  # Store the returned model
    
    
    
    # Commented out evaluation code for later use
    
    print("\nLoading trained model...")
    # Load the trained model for evaluation
    loaded_model = load_model('trained_model.pth')
    # Update the global model variable in utils
    import utils
    utils.model = loaded_model
    
    print("\nStarting comprehensive evaluation...")
    success_rate, eval_stats = evaluate_model()  # Run evaluation
    print(f"\nEvaluation Results:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Required success rate: 75%")
    
    # Print detailed evaluation statistics
    print("\nDetailed Evaluation Statistics:")
    print(f"Total configurations tested: {eval_stats['total_tests']}")
    print(f"Successful configurations: {eval_stats['successful_tests']}")
    print(f"Average steps for successful tests: {eval_stats['avg_steps_successful']:.2f}")
    print("\nFailures breakdown:")
    print(f"Collisions: {eval_stats['failures']['collisions']}")
    print(f"Timeouts: {eval_stats['failures']['timeout']}")
    print(f"Incomplete paths: {eval_stats['failures']['incomplete']}")
    
    
    
    elapsed_time = process_time() - t
    print(f"\nFinished in {elapsed_time:.2f} seconds.") 