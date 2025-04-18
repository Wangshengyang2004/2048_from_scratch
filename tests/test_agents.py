import torch
import numpy as np
import pytest
import os
import tempfile
from env import Env2048
from agent import (
    RandomAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent,
    REINFORCEAgent, PPOAgent
)


def test_random_agent():
    """Test random agent"""
    # Create agent
    agent = RandomAgent()
    
    # Create environment
    env = Env2048()
    
    # Reset environment
    state, _ = env.reset()
    
    # Get valid actions
    valid_actions = env.get_valid_actions()
    
    # Select action with agent
    action = agent.select_action(state, valid_actions)
    
    # Action should be a tensor with a single value
    assert isinstance(action, torch.Tensor)
    assert action.numel() == 1
    
    # Action value should be in range [0, 3]
    assert 0 <= action.item() < 4
    
    # If valid actions exist, action should be valid
    if valid_actions.sum() > 0:
        assert valid_actions[action.item()]
    
    # Batch action selection
    batch_size = 3
    states = torch.stack([state] * batch_size)
    batch_valid_actions = torch.stack([valid_actions] * batch_size)
    
    batch_actions = agent.batch_act(states, batch_valid_actions)
    
    # Should return a tensor of shape [batch_size]
    assert batch_actions.shape == (batch_size,)
    
    # Close environment
    env.close()


def test_dqn_agent():
    """Test DQN agent"""
    # Create agent
    agent = DQNAgent()
    
    # Create environment
    env = Env2048()
    
    # Reset environment
    state, _ = env.reset()
    
    # Get valid actions
    valid_actions = env.get_valid_actions()
    
    # Select action with agent
    action = agent.select_action(state, valid_actions)
    
    # Action should be a tensor with a single value
    assert isinstance(action, torch.Tensor)
    assert action.numel() == 1
    
    # Action value should be in range [0, 3]
    assert 0 <= action.item() < 4
    
    # Take a step in the environment
    next_state, reward, done, truncated, info = env.step(action)
    
    # Store experience
    agent.store_experience(state, action, reward, next_state, done)
    
    # Update agent
    loss = agent.update()
    
    # Saving and loading
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "dqn_test.pt")
        
        # Save model
        agent.save(model_path)
        
        # Load model
        new_agent = DQNAgent()
        new_agent.load(model_path)
    
    # Close environment
    env.close()


def test_policy_agents():
    """Test policy-based agents"""
    # Create agents
    agents = [
        REINFORCEAgent(),
        PPOAgent()
    ]
    
    for agent in agents:
        # Create environment
        env = Env2048()
        
        # Reset environment
        state, _ = env.reset()
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Select action with agent
        action = agent.select_action(state, valid_actions)
        
        # Action should be a tensor with a single value
        assert isinstance(action, torch.Tensor)
        assert action.numel() == 1
        
        # Action value should be in range [0, 3]
        assert 0 <= action.item() < 4
        
        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store reward
        agent.store_reward(reward, done)
        
        # Collect a few more experiences
        for _ in range(3):
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_reward(reward, done)
            state = next_state
        
        # Update agent
        update_info = agent.update()
        
        # Should return a dictionary with metrics
        assert isinstance(update_info, dict)
        
        # Saving and loading
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "policy_test.pt")
            
            # Save model
            agent.save(model_path)
            
            # Load model
            new_agent = agent.__class__()
            new_agent.load(model_path)
        
        # Close environment
        env.close()


if __name__ == "__main__":
    # Run tests
    test_random_agent()
    test_dqn_agent()
    test_policy_agents()
    print("All tests passed!") 