import torch
import numpy as np
import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from env import Env2048
from agent import (
    RandomAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent,
    REINFORCEAgent, PPOAgent
)


def evaluate_agent(
    env: Env2048,
    agent: Any,
    agent_name: str,
    num_episodes: int = 10,
    max_steps: int = 10000,
    render: bool = False,
    render_delay: float = 0.1
) -> Dict[str, Any]:
    """
    Evaluate an agent
    
    Args:
        env: The environment
        agent: The agent to evaluate
        agent_name: Name of the agent for display
        num_episodes: Number of episodes to evaluate for
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        render_delay: Delay between frames when rendering
        
    Returns:
        results: Dictionary containing evaluation results
    """
    scores = []
    max_tiles = []
    tile_counts = {}
    steps = []
    
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
        steps.append(step_count)
        
        # Count tiles
        tile_counts[max_tile] = tile_counts.get(max_tile, 0) + 1
        
        print(f"{agent_name} - Episode {episode + 1}/{num_episodes}, Score: {episode_reward:.2f}, "
              f"Max Tile: {max_tile}, Steps: {step_count}")
    
    avg_score = np.mean(scores)
    avg_max_tile = np.mean(max_tiles)
    avg_steps = np.mean(steps)
    
    print(f"\n{agent_name} Results:")
    print(f"Avg Score: {avg_score:.2f}")
    print(f"Avg Max Tile: {avg_max_tile:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Tile Distribution: {tile_counts}")
    
    return {
        "agent": agent_name,
        "avg_score": float(avg_score),
        "avg_max_tile": float(avg_max_tile),
        "avg_steps": float(avg_steps),
        "tile_counts": {str(k): v for k, v in tile_counts.items()},
        "scores": [float(s) for s in scores],
        "max_tiles": [int(t) for t in max_tiles],
        "steps": [int(s) for s in steps]
    }


def plot_comparison(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create plots comparing the agents
    
    Args:
        results: List of results for each agent
        output_dir: Directory to save the plots
    """
    # Extract data
    agents = [r["agent"] for r in results]
    avg_scores = [r["avg_score"] for r in results]
    avg_max_tiles = [r["avg_max_tile"] for r in results]
    avg_steps = [r["avg_steps"] for r in results]
    
    # Set up colors and bar width
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
    bar_width = 0.25
    
    # Plot average scores
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(agents)), avg_scores, width=bar_width, color=colors)
    plt.xlabel("Agent")
    plt.ylabel("Average Score")
    plt.title("Average Score by Agent")
    plt.xticks(np.arange(len(agents)), agents, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_scores.png"))
    
    # Plot average max tiles
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(agents)), avg_max_tiles, width=bar_width, color=colors)
    plt.xlabel("Agent")
    plt.ylabel("Average Max Tile")
    plt.title("Average Max Tile by Agent")
    plt.xticks(np.arange(len(agents)), agents, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_max_tiles.png"))
    
    # Plot average steps
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(agents)), avg_steps, width=bar_width, color=colors)
    plt.xlabel("Agent")
    plt.ylabel("Average Steps")
    plt.title("Average Steps by Agent")
    plt.xticks(np.arange(len(agents)), agents, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_steps.png"))
    
    # Plot tile distribution
    plt.figure(figsize=(15, 8))
    
    # Combine all tile counts
    all_tiles = set()
    for r in results:
        all_tiles.update(map(int, r["tile_counts"].keys()))
    
    all_tiles = sorted(list(all_tiles))
    x = np.arange(len(all_tiles))
    width = 0.8 / len(agents)
    offsets = np.linspace(-(0.8/2), (0.8/2), len(agents))
    
    # Plot tile distribution for each agent
    for i, (agent, offset, color) in enumerate(zip(results, offsets, colors)):
        tiles = all_tiles.copy()
        counts = []
        
        for tile in tiles:
            counts.append(agent["tile_counts"].get(str(tile), 0))
        
        plt.bar(x + offset, counts, width, label=agent["agent"], color=color, alpha=0.8)
    
    plt.xlabel("Tile Value")
    plt.ylabel("Count")
    plt.title("Distribution of Max Tiles by Agent")
    plt.xticks(x, [str(t) for t in all_tiles], rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tile_distribution.png"))
    
    # Plot boxplot of scores
    plt.figure(figsize=(12, 6))
    score_data = [r["scores"] for r in results]
    plt.boxplot(score_data, labels=agents)
    plt.xlabel("Agent")
    plt.ylabel("Score")
    plt.title("Score Distribution by Agent")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_boxplot.png"))
    
    # Plot boxplot of max tiles
    plt.figure(figsize=(12, 6))
    tile_data = [r["max_tiles"] for r in results]
    plt.boxplot(tile_data, labels=agents)
    plt.xlabel("Agent")
    plt.ylabel("Max Tile")
    plt.title("Max Tile Distribution by Agent")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_tile_boxplot.png"))


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate and compare all 2048 agents")
    
    parser.add_argument("--model_dir", type=str, default="results",
                        help="Directory containing trained models")
    
    parser.add_argument("--num_episodes", type=int, default=20, 
                        help="Number of episodes to evaluate for each agent")
    
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
    
    parser.add_argument("--output_dir", type=str, default="comparison_results", 
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, "comparison_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create environment
    env = Env2048(render_mode="human" if args.render else None, device=args.device)
    
    # Define agents to evaluate
    agent_configs = [
        {
            "name": "Random",
            "class": RandomAgent,
            "model_path": None
        },
        {
            "name": "DQN",
            "class": DQNAgent,
            "model_path": os.path.join(args.model_dir, "dqn", "dqn_model_final.pt")
        },
        {
            "name": "Double DQN",
            "class": DoubleDQNAgent,
            "model_path": os.path.join(args.model_dir, "ddqn", "dqn_model_final.pt")
        },
        {
            "name": "Dueling DQN",
            "class": DuelingDQNAgent,
            "model_path": os.path.join(args.model_dir, "dueling_dqn", "dqn_model_final.pt")
        },
        {
            "name": "REINFORCE",
            "class": REINFORCEAgent,
            "model_path": os.path.join(args.model_dir, "reinforce", "policy_model_final.pt")
        },
        {
            "name": "PPO",
            "class": PPOAgent,
            "model_path": os.path.join(args.model_dir, "ppo", "policy_model_final.pt")
        }
    ]
    
    # Validate agent configs
    valid_agents = []
    for config in agent_configs:
        if config["model_path"] is None or os.path.exists(config["model_path"]):
            valid_agents.append(config)
        else:
            print(f"Warning: Model path {config['model_path']} for {config['name']} not found. Skipping.")
    
    # Evaluate agents
    results = []
    
    for agent_config in valid_agents:
        print(f"\nEvaluating {agent_config['name']} agent...")
        
        # Create agent
        agent = agent_config["class"](device=args.device)
        
        # Load model if available
        if agent_config["model_path"] and os.path.exists(agent_config["model_path"]):
            try:
                agent.load(agent_config["model_path"])
                print(f"Loaded model from {agent_config['model_path']}")
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
        
        # Evaluate agent
        result = evaluate_agent(
            env=env,
            agent=agent,
            agent_name=agent_config["name"],
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
            render_delay=args.render_delay
        )
        
        results.append(result)
    
    # Save all results
    with open(os.path.join(args.output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot comparison
    if len(results) > 1:
        plot_comparison(results, args.output_dir)
    
    # Print summary
    print("\nSummary:")
    print("=" * 80)
    print(f"{'Agent':<15} {'Avg Score':<15} {'Avg Max Tile':<15} {'Avg Steps':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['agent']:<15} {result['avg_score']:<15.2f} {result['avg_max_tile']:<15.2f} {result['avg_steps']:<15.2f}")
    
    print("=" * 80)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 