import torch
import numpy as np
import os
import argparse
import time
import json
from typing import Dict, List, Tuple, Any

from env import Env2048
from agent import (
    RandomAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent,
    REINFORCEAgent, PPOAgent
)


def evaluate_agent(
    env: Env2048,
    agent: Any,
    num_episodes: int = 10,
    max_steps: int = 10000,
    render: bool = False,
    render_delay: float = 0.1
) -> Tuple[float, float, Dict[int, int], List[float], List[int]]:
    """
    Evaluate an agent
    
    Args:
        env: The environment
        agent: The agent to evaluate
        num_episodes: Number of episodes to evaluate for
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        render_delay: Delay between frames when rendering
        
    Returns:
        avg_score: Average score
        avg_max_tile: Average max tile
        tile_counts: Dictionary mapping tile values to counts
        scores: List of scores
        max_tiles: List of max tiles
    """
    scores = []
    max_tiles = []
    tile_counts = {}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action using agent (in evaluation mode)
            action = agent.select_action(state, valid_actions, training=False)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Render if needed
            if render:
                env.render()
                time.sleep(render_delay)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        # Log episode metrics
        scores.append(episode_reward)
        max_tile = info["max_tile"]
        max_tiles.append(max_tile)
        
        # Count tiles
        tile_counts[max_tile] = tile_counts.get(max_tile, 0) + 1
        
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Score: {episode_reward:.2f}, "
              f"Max Tile: {max_tile}, Steps: {step_count}")
    
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    
    print(f"\nEvaluation Results:")
    print(f"Avg Score: {avg_score:.2f}")
    print(f"Avg Max Tile: {avg_max_tile:.2f}")
    print(f"Tile Distribution: {tile_counts}")
    
    return avg_score, avg_max_tile, tile_counts, scores, max_tiles


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate RL agents for 2048 game")
    
    parser.add_argument("--agent", type=str, default="dqn", 
                        choices=["random", "dqn", "ddqn", "dueling_dqn", "reinforce", "ppo"],
                        help="Agent to evaluate")
    
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to the model file (not needed for random agent)")
    
    parser.add_argument("--num_episodes", type=int, default=10, 
                        help="Number of episodes to evaluate for")
    
    parser.add_argument("--max_steps", type=int, default=10000, 
                        help="Maximum steps per episode")
    
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during evaluation")
    
    parser.add_argument("--render_delay", type=float, default=0.1, 
                        help="Delay between frames when rendering")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for evaluation")
    
    parser.add_argument("--output_dir", type=str, default="eval_results", 
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    agent_name = os.path.basename(args.model_path).split(".")[0] if args.model_path else args.agent
    output_dir = os.path.join(args.output_dir, agent_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create environment
    env = Env2048(render_mode="human" if args.render else None, device=args.device)
    
    # Create agent
    if args.agent == "random":
        agent = RandomAgent(device=args.device)
    elif args.agent == "dqn":
        agent = DQNAgent(device=args.device)
        if args.model_path:
            agent.load(args.model_path)
    elif args.agent == "ddqn":
        agent = DoubleDQNAgent(device=args.device)
        if args.model_path:
            agent.load(args.model_path)
    elif args.agent == "dueling_dqn":
        agent = DuelingDQNAgent(device=args.device)
        if args.model_path:
            agent.load(args.model_path)
    elif args.agent == "reinforce":
        agent = REINFORCEAgent(device=args.device)
        if args.model_path:
            agent.load(args.model_path)
    elif args.agent == "ppo":
        agent = PPOAgent(device=args.device)
        if args.model_path:
            agent.load(args.model_path)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
    
    # Evaluate agent
    print(f"Evaluating {args.agent} agent for {args.num_episodes} episodes...")
    
    avg_score, avg_max_tile, tile_counts, scores, max_tiles = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        render_delay=args.render_delay
    )
    
    # Save evaluation results
    eval_results = {
        "avg_score": float(avg_score),
        "avg_max_tile": float(avg_max_tile),
        "tile_counts": {str(k): v for k, v in tile_counts.items()},  # Convert keys to strings for JSON
        "scores": [float(s) for s in scores],
        "max_tiles": [int(t) for t in max_tiles]
    }
    
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 