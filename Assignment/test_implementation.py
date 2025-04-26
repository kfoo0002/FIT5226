import numpy as np
import torch
from env import GridWorldEnvironment
from stage2 import evaluate_success_rate
from utils import set_seed, prepare_torch, get_qvals, get_maxQ, train_one_step

def test_environment():
    print("\n=== Testing Environment ===")
    # Create environment
    env = GridWorldEnvironment(5, 5, num_agents=4)
    
    # Test initialization
    env._reset()
    print("Initial state:")
    for i, agent in enumerate(env.agents):
        print(f"Agent {i}:")
        print(f"  Position: {agent.position}")
        print(f"  Has item: {agent.has_item}")
        print(f"  Direction: {agent.direction}")
    
    # Test movement and collisions
    print("\nTesting movement and collisions:")
    for step in range(5):
        print(f"\nStep {step + 1}:")
        for i in range(env.num_agents):
            agent_id = env.get_next_agent()
            state = env.get_state(agent_id)
            action = np.random.randint(4)
            reward = env.take_action(agent_id, action)
            print(f"Agent {agent_id}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward}")
            print(f"  New position: {env.agents[agent_id].position}")
            print(f"  Has item: {env.agents[agent_id].has_item}")
        
        env.check_collisions()
        print(f"Collisions this step: {env.collision_count}")

def test_learning():
    print("\n=== Testing Learning Components ===")
    # Initialize networks
    prepare_torch()
    
    # Test state to Q-value conversion
    test_state = np.random.rand(15)  # 15-dimensional state
    q_values = get_qvals(test_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    
    # Test target computation
    next_q = get_maxQ(test_state)
    print(f"Max Q-value: {next_q}")
    
    # Test training step
    batch_size = 32
    states = [np.random.rand(15) for _ in range(batch_size)]
    actions = np.random.randint(4, size=batch_size)
    targets = np.random.rand(batch_size)
    loss = train_one_step(states, actions, targets, 0.99)
    print(f"Training loss: {loss}")

def test_evaluation():
    print("\n=== Testing Evaluation ===")
    # Create environment
    env = GridWorldEnvironment(5, 5, num_agents=4)
    
    # Test success rate evaluation
    success_rate = evaluate_success_rate(env, 4, eval_episodes=10)
    print(f"Success rate: {success_rate:.2%}")
    
    # Test performance points
    if success_rate > 0.95 and env.collision_count < 500:
        print("2 performance points!")
    elif success_rate > 0.85 and env.collision_count < 1000:
        print("1 performance point!")
    else:
        print("No performance points earned.")

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run tests
    test_environment()
    test_learning()
    test_evaluation()

if __name__ == "__main__":
    main() 