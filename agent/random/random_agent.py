import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple


class RandomAgent:
    """
    Random policy agent for 2048 game.
    Takes random valid actions from the environment.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def select_action(self, 
                      state: torch.Tensor, 
                      valid_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Select a random valid action based on the current state
        
        Args:
            state: Current state tensor [channels, height, width]
            valid_actions: Tensor of valid actions [action_space]
            
        Returns:
            Selected action as a tensor
        """
        # If valid actions mask is provided, only sample from valid actions
        if valid_actions is not None:
            if valid_actions.sum() == 0:
                # If no valid actions, return a random action (will be invalid)
                return torch.randint(0, 4, (1,), device=self.device)
            
            # Sample uniformly from valid actions
            valid_indices = torch.nonzero(valid_actions, as_tuple=True)[0]
            random_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
            return valid_indices[random_idx]
        
        # If valid actions not provided, sample uniformly from all actions
        return torch.randint(0, 4, (1,), device=self.device)
    
    @torch.no_grad()
    def batch_act(self, 
                  states: torch.Tensor, 
                  valid_actions_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Select random valid actions for a batch of states
        
        Args:
            states: Batch of states [batch_size, channels, height, width]
            valid_actions_batch: Batch of valid action masks [batch_size, action_space]
            
        Returns:
            Batch of selected actions [batch_size]
        """
        batch_size = states.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # If valid actions batch is provided, use it
        if valid_actions_batch is not None:
            for i in range(batch_size):
                valid_actions = valid_actions_batch[i]
                if valid_actions.sum() == 0:
                    # If no valid actions, choose randomly (will be invalid)
                    actions[i] = torch.randint(0, 4, (1,), device=self.device)
                else:
                    # Sample uniformly from valid actions
                    valid_indices = torch.nonzero(valid_actions, as_tuple=True)[0]
                    random_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
                    actions[i] = valid_indices[random_idx]
        else:
            # Sample uniformly from all actions
            actions = torch.randint(0, 4, (batch_size,), device=self.device)
        
        return actions
    
    def save(self, path: str) -> None:
        """
        Save the agent (no-op for random agent)
        """
        pass
    
    def load(self, path: str) -> None:
        """
        Load the agent (no-op for random agent)
        """
        pass 