import numpy as np
from env import GridWorldEnvironment
import matplotlib.pyplot as plt

def test_environment():
    # Create environment with 4 agents (matching our implementation)
    env = GridWorldEnvironment(n=5, m=5, num_agents=4)
    
    # Reset the environment
    env._reset()
    
    # Print initial state
    print("Initial state:")
    for i, agent in enumerate(env.agents):
        print(f"Agent {i}:")
        print(f"  Position: {agent.position}")
        print(f"  Has item: {agent.has_item}")
        print(f"  Direction: {agent.direction}")
    
    # Take a few random steps
    print("\nTaking some random steps...")
    for step in range(5):
        print(f"\nStep {step + 1}:")
        
        # Each agent takes a random action
        for i in range(env.num_agents):
            agent_id = env.get_next_agent()
            state = env.get_state(agent_id)
            action = np.random.randint(4)  # Random action
            reward = env.take_action(agent_id, action)
            
            print(f"Agent {agent_id}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward}")
            print(f"  New position: {env.agents[agent_id].position}")
            print(f"  Has item: {env.agents[agent_id].has_item}")
        
        # Check for collisions
        env.check_collisions()
        print(f"Collisions this step: {env.collision_count}")
        print(f"Total deliveries: {env.delivery_count}")
        
        # Visualize
        env.visualize(step, 0)
        plt.pause(0.5)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_environment()
    plt.show()  # Keep the window open 