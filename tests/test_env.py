import torch
import numpy as np
import pytest
from env import Game2048, Env2048, make_vec_env


def test_game_init():
    """Test game initialization"""
    game = Game2048()
    assert game.board.shape == (4, 4)
    assert torch.sum(game.board > 0) == 2  # Two initial tiles
    assert game.score == 0
    assert game.done is False


def test_game_actions():
    """Test game actions"""
    game = Game2048()
    
    # Get initial state
    initial_board = game.board.clone()
    
    # Try all actions
    for action in range(4):
        # Reset the game
        game.reset()
        
        # Take action
        next_board, reward, done, info = game.step(action)
        
        # Check that the board is a 4x4 tensor
        assert next_board.shape == (4, 4)
        
        # Check that info contains expected keys
        assert "score" in info
        assert "max_tile" in info
        assert "valid_move" in info
        assert "empty_cells" in info


def test_valid_actions():
    """Test valid actions detection"""
    game = Game2048()
    
    # Get valid actions
    valid_actions = game.get_valid_actions()
    
    # Should be a tensor of shape [4] with boolean values
    assert valid_actions.shape == (4,)
    assert valid_actions.dtype == torch.bool


def test_game_state_tensor():
    """Test state tensor conversion"""
    game = Game2048()
    
    # Get state tensor
    state_tensor = game.get_state_tensor()
    
    # Should be a tensor of shape [12, 4, 4] with float values
    assert state_tensor.shape == (12, 4, 4)
    assert state_tensor.dtype == torch.float32


def test_env2048():
    """Test gym environment wrapper"""
    env = Env2048()
    
    # Reset the environment
    state, _ = env.reset()
    
    # State should be a tensor of shape [12, 4, 4]
    assert state.shape == (12, 4, 4)
    
    # Take a step
    action = 0  # UP
    next_state, reward, done, truncated, info = env.step(action)
    
    # Check that next_state is a tensor of shape [12, 4, 4]
    assert next_state.shape == (12, 4, 4)
    
    # Check that info contains expected keys
    assert "score" in info
    assert "max_tile" in info
    
    # Close the environment
    env.close()


def test_vec_env():
    """Test vectorized environment"""
    num_envs = 2
    vec_env = make_vec_env(num_envs=num_envs)
    
    # Reset the environment
    states = vec_env.reset()
    
    # States should be a tensor of shape [num_envs, 12, 4, 4]
    assert states.shape == (num_envs, 12, 4, 4)
    
    # Get valid actions
    valid_actions = vec_env.get_valid_actions()
    
    # Should be a tensor of shape [num_envs, 4] with boolean values
    assert valid_actions.shape == (num_envs, 4)
    
    # Take a step with random actions
    actions = torch.randint(0, 4, (num_envs,), device=states.device)
    next_states, rewards, dones, info = vec_env.step(actions)
    
    # Check that next_states is a tensor of shape [num_envs, 12, 4, 4]
    assert next_states.shape == (num_envs, 12, 4, 4)
    
    # Check that rewards is a tensor of shape [num_envs]
    assert rewards.shape == (num_envs,)
    
    # Check that dones is a tensor of shape [num_envs]
    assert dones.shape == (num_envs,)
    
    # Close the environment
    vec_env.close()


if __name__ == "__main__":
    # Run tests
    test_game_init()
    test_game_actions()
    test_valid_actions()
    test_game_state_tensor()
    test_env2048()
    test_vec_env()
    print("All tests passed!") 