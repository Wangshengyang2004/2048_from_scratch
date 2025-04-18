import torch
import numpy as np
import os
import argparse
import time
import random
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from env import Env2048, make_vec_env
from agent import (
    RandomAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent,
    REINFORCEAgent, PPOAgent
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_dqn(
    env: Env2048,
    agent: DQNAgent,
    num_episodes: int,
    max_steps: int,
    save_path: str,
    eval_interval: int = 10,
    log_interval: int = 1,
    vector_env: bool = False,
    num_envs: int = 4,
    render: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train a DQN-based agent
    
    Args:
        env: The environment
        agent: The DQN agent
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        save_path: Path to save the agent
        eval_interval: Evaluate the agent every n episodes
        log_interval: Log metrics every n episodes
        vector_env: Whether to use vectorized environment
        num_envs: Number of environments in vectorized env
        render: Whether to render the environment
        
    Returns:
        scores: List of scores
        max_tiles: List of max tiles
        losses: List of losses
    """
    # Import required modules
    import time
    
    scores = []
    max_tiles = []
    losses = []
    
    # Initialize GPU monitoring if available
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        try:
            import psutil
            import GPUtil
            gpu_monitoring = True
            print("GPU monitoring enabled")
        except ImportError:
            gpu_monitoring = False
            print("GPU monitoring disabled (install psutil and GPUtil packages to enable)")
    else:
        gpu_monitoring = False
        print("GPU not available - running on CPU")
    
    # Initialize pygame if rendering
    if render:
        import pygame
        import time
        # Note about vector env rendering
        if vector_env:
            print("Warning: Rendering with vector environments will only show the first environment")
            print("Vector environments don't support multiple rendering windows simultaneously")
    
    if vector_env:
        # Use vectorized environment for faster training
        vec_env = make_vec_env(num_envs=num_envs, render_mode="human" if render else None)
        observations = vec_env.reset()
        
        # Display environment and agent configuration
        print(f"\nTraining Configuration:")
        print(f"  Agent: {agent.__class__.__name__}")
        print(f"  Vector Environment: True, Num Envs: {num_envs}")
        print(f"  State Shape: {observations[0].shape}")
        print(f"  Device: {agent.device}")
        print(f"  Replay Buffer Size: {agent.replay_buffer.capacity}")
        print(f"  Batch Size: {agent.batch_size}")
        print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"  Network Architecture: {agent.q_network}")
        print("")
        
        # Training loop
        for episode in range(num_episodes):
            episode_start_time = time.time()
            episode_losses = []
            episode_rewards = [0] * num_envs
            max_episode_tiles = [0] * num_envs
            step_times = []
            update_times = []
            
            # Run multiple steps
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Get valid actions for each environment
                valid_actions = vec_env.get_valid_actions()
                
                # Get actions from agent
                actions = agent.batch_act(observations, valid_actions)
                
                # Take step in environment
                next_observations, rewards, dones, infos = vec_env.step(actions)
                
                # Render first environment if render flag is set
                if render and step % 5 == 0:  # Only render every 5th step to speed up training
                    # We can only render the first environment in vector mode
                    if hasattr(vec_env, 'render'):
                        vec_env.render()
                    else:
                        print("Vector environment doesn't support rendering")
                    pygame.event.pump()
                    time.sleep(0.05)
                
                # Store experiences in replay buffer
                for i in range(num_envs):
                    agent.store_experience(
                        observations[i], 
                        actions[i], 
                        rewards[i].item(), 
                        next_observations[i], 
                        dones[i].item()
                    )
                    
                    # Update episode metrics
                    episode_rewards[i] += rewards[i].item()
                    max_episode_tiles[i] = max(max_episode_tiles[i], infos["max_tiles"][i])
                
                # Update observations
                observations = next_observations
                
                # Record step time
                step_times.append(time.time() - step_start_time)
                
                # Update agent
                update_start_time = time.time()
                loss = agent.update()
                update_times.append(time.time() - update_start_time)
                episode_losses.append(loss)
                
                # Reset environments that are done
                if any(dones):
                    for i in range(num_envs):
                        if dones[i]:
                            scores.append(episode_rewards[i])
                            max_tiles.append(max_episode_tiles[i])
                            episode_rewards[i] = 0
                            max_episode_tiles[i] = 0
            
            # Calculate episode metrics
            episode_duration = time.time() - episode_start_time
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_step_time = np.mean(step_times) if step_times else 0
            avg_update_time = np.mean(update_times) if update_times else 0
            steps_per_second = len(step_times) / episode_duration if episode_duration > 0 else 0
            
            # Get GPU metrics if available
            if gpu_monitoring and gpu_available and episode % log_interval == 0:
                gpu = GPUtil.getGPUs()[0]
                gpu_util = gpu.load * 100
                gpu_mem_used = gpu.memoryUsed
                gpu_mem_total = gpu.memoryTotal
                gpu_temp = gpu.temperature
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                ram_used = psutil.virtual_memory().used / (1024**3)  # GB
                ram_total = psutil.virtual_memory().total / (1024**3)  # GB
                
                # Log detailed metrics
                losses.append(avg_loss)
                print(f"Episode {episode}/{num_episodes}:")
                print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s")
                print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {avg_update_time*1000:.2f}ms")
                print(f"  GPU Util: {gpu_util:.1f}%, Mem: {gpu_mem_used}/{gpu_mem_total}MB, Temp: {gpu_temp}°C")
                print(f"  CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}% ({ram_used:.1f}/{ram_total:.1f}GB)")
                print(f"  Training: Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                print(f"  Rewards: {np.mean(episode_rewards):.2f}, Max Tiles: {np.max(max_episode_tiles)}")
                print(f"  Replay Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
                print("")
            elif episode % log_interval == 0:
                # Simpler output if GPU monitoring not available
                losses.append(avg_loss)
                print(f"Episode {episode}/{num_episodes}:")
                print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s")
                print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {avg_update_time*1000:.2f}ms")
                print(f"  Training: Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                print(f"  Rewards: {np.mean(episode_rewards):.2f}, Max Tiles: {np.max(max_episode_tiles)}")
                print(f"  Replay Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
                print("")
            
            # Save model
            if episode % eval_interval == 0:
                agent.save(os.path.join(save_path, f"dqn_model_ep{episode}.pt"))
    
    else:
        # Single environment training
        # Display environment and agent configuration
        state, _ = env.reset()
        print(f"\nTraining Configuration:")
        print(f"  Agent: {agent.__class__.__name__}")
        print(f"  Vector Environment: False")
        print(f"  State Shape: {state.shape}")
        print(f"  Device: {agent.device}")
        print(f"  Replay Buffer Size: {agent.replay_buffer.capacity}")
        print(f"  Batch Size: {agent.batch_size}")
        print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"  Network Architecture: {agent.q_network}")
        print("")
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_losses = []
            step_times = []
            update_times = []
            step_count = 0
            
            for step in range(max_steps):
                step_start_time = time.time()
                step_count += 1
                
                # Render the environment if render flag is set
                if render:
                    env.render()
                    # Process pygame events to keep the window responsive
                    pygame.event.pump()
                    # Add a small delay to make the rendering visible
                    time.sleep(0.05)
                    
                # Get valid actions
                valid_actions = env.get_valid_actions()
                
                # Select action using agent
                action = agent.select_action(state, valid_actions)
                
                # Take action in environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Update agent
                update_start_time = time.time()
                loss = agent.update()
                update_times.append(time.time() - update_start_time)
                if loss > 0:
                    episode_losses.append(loss)
                
                # Calculate step time
                step_times.append(time.time() - step_start_time)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    # Show the final state before ending the episode
                    if render:
                        env.render()
                        pygame.event.pump()
                        time.sleep(0.5)  # Slightly longer pause for the final state
                    break
            
            # Log episode metrics
            scores.append(episode_reward)
            max_tiles.append(info["max_tile"])
            
            # Calculate episode metrics
            episode_duration = time.time() - episode_start_time
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_step_time = np.mean(step_times) if step_times else 0
            avg_update_time = np.mean(update_times) if update_times else 0
            steps_per_second = step_count / episode_duration if episode_duration > 0 else 0
            
            if episode % log_interval == 0:
                # Get GPU metrics if available
                if gpu_monitoring and gpu_available:
                    gpu = GPUtil.getGPUs()[0]
                    gpu_util = gpu.load * 100
                    gpu_mem_used = gpu.memoryUsed
                    gpu_mem_total = gpu.memoryTotal
                    gpu_temp = gpu.temperature
                    cpu_percent = psutil.cpu_percent()
                    ram_percent = psutil.virtual_memory().percent
                    ram_used = psutil.virtual_memory().used / (1024**3)  # GB
                    ram_total = psutil.virtual_memory().total / (1024**3)  # GB
                    
                    # Log detailed metrics
                    losses.append(avg_loss)
                    print(f"Episode {episode}/{num_episodes}:")
                    print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s, Steps: {step_count}")
                    print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {avg_update_time*1000:.2f}ms")
                    print(f"  GPU Util: {gpu_util:.1f}%, Mem: {gpu_mem_used}/{gpu_mem_total}MB, Temp: {gpu_temp}°C")
                    print(f"  CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}% ({ram_used:.1f}/{ram_total:.1f}GB)")
                    print(f"  Score: {episode_reward:.2f}, Max Tile: {info['max_tile']}")
                    print(f"  Training: Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                    print(f"  Replay Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
                    print("")
                else:
                    # Simpler output if GPU monitoring not available
                    losses.append(avg_loss)
                    print(f"Episode {episode}/{num_episodes}:")
                    print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s, Steps: {step_count}")
                    print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {avg_update_time*1000:.2f}ms")
                    print(f"  Score: {episode_reward:.2f}, Max Tile: {info['max_tile']}")
                    print(f"  Training: Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                    print(f"  Replay Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity}")
                    print("")
            
            # Save model
            if episode % eval_interval == 0:
                agent.save(os.path.join(save_path, f"dqn_model_ep{episode}.pt"))
    
    # Save final model
    agent.save(os.path.join(save_path, "dqn_model_final.pt"))
    
    return scores, max_tiles, losses


def train_policy_based(
    env: Env2048,
    agent: Any,  # REINFORCEAgent or PPOAgent
    num_episodes: int,
    max_steps: int,
    save_path: str,
    eval_interval: int = 10,
    log_interval: int = 1,
    render: bool = False
) -> Tuple[List[float], List[float], Dict[str, List[float]]]:
    """
    Train a policy-based agent (REINFORCE, PPO)
    
    Args:
        env: The environment
        agent: The policy-based agent
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        save_path: Path to save the agent
        eval_interval: Evaluate the agent every n episodes
        log_interval: Log metrics every n episodes
        render: Whether to render the environment
        
    Returns:
        scores: List of scores
        max_tiles: List of max tiles
        metrics: Dictionary of metrics
    """
    # Import required modules
    import time
    
    scores = []
    max_tiles = []
    metrics = {"policy_loss": [], "value_loss": [], "entropy": []}
    
    # Initialize GPU monitoring if available
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        try:
            import psutil
            import GPUtil
            gpu_monitoring = True
            print("GPU monitoring enabled")
        except ImportError:
            gpu_monitoring = False
            print("GPU monitoring disabled (install psutil and GPUtil packages to enable)")
    else:
        gpu_monitoring = False
        print("GPU not available - running on CPU")
    
    # Initialize pygame if rendering
    if render:
        import pygame
        import time
    
    # Display environment and agent configuration
    state, _ = env.reset()
    print(f"\nTraining Configuration:")
    print(f"  Agent: {agent.__class__.__name__}")
    print(f"  State Shape: {state.shape}")
    print(f"  Device: {agent.device}")
    if hasattr(agent, 'policy_network'):
        print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"  Network Architecture: {agent.policy_network}")
    print("")
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        step_times = []
        
        # Clear episode buffer
        agent.reset_episode_buffer()
        
        for step in range(max_steps):
            step_start_time = time.time()
            step_count += 1
            
            # Render the environment if render flag is set
            if render:
                env.render()
                # Process pygame events to keep the window responsive
                pygame.event.pump()
                # Add a small delay to make the rendering visible
                time.sleep(0.05)
                
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action using agent
            action = agent.select_action(state, valid_actions)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store reward and done flag
            agent.store_reward(reward, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            
            # Record step time
            step_times.append(time.time() - step_start_time)
            
            if done or truncated:
                # Show the final state before ending the episode
                if render:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.5)  # Slightly longer pause for the final state
                break
        
        # Update agent after episode
        update_start_time = time.time()
        update_info = agent.update()
        update_time = time.time() - update_start_time
        
        # Log episode metrics
        scores.append(episode_reward)
        max_tiles.append(info["max_tile"])
        
        for key, value in update_info.items():
            if key in metrics:
                metrics[key].append(value)
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start_time
        avg_step_time = np.mean(step_times) if step_times else 0
        steps_per_second = step_count / episode_duration if episode_duration > 0 else 0
        
        if episode % log_interval == 0:
            # Get GPU metrics if available
            if gpu_monitoring and gpu_available:
                gpu = GPUtil.getGPUs()[0]
                gpu_util = gpu.load * 100
                gpu_mem_used = gpu.memoryUsed
                gpu_mem_total = gpu.memoryTotal
                gpu_temp = gpu.temperature
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                ram_used = psutil.virtual_memory().used / (1024**3)  # GB
                ram_total = psutil.virtual_memory().total / (1024**3)  # GB
                
                # Log detailed metrics
                print(f"Episode {episode}/{num_episodes}:")
                print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s, Steps: {step_count}")
                print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {update_time*1000:.2f}ms")
                print(f"  GPU Util: {gpu_util:.1f}%, Mem: {gpu_mem_used}/{gpu_mem_total}MB, Temp: {gpu_temp}°C")
                print(f"  CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}% ({ram_used:.1f}/{ram_total:.1f}GB)")
                print(f"  Score: {episode_reward:.2f}, Max Tile: {info['max_tile']}")
                print(f"  Policy Loss: {update_info.get('policy_loss', 0):.4f}, Value Loss: {update_info.get('value_loss', 0):.4f}")
                print(f"  Entropy: {update_info.get('entropy', 0):.4f}, Approx KL: {update_info.get('approx_kl', 0):.4f}")
                print("")
            else:
                # Simpler output if GPU monitoring not available
                print(f"Episode {episode}/{num_episodes}:")
                print(f"  Performance: {steps_per_second:.2f} steps/s, Duration: {episode_duration:.2f}s, Steps: {step_count}")
                print(f"  Step Time: {avg_step_time*1000:.2f}ms, Update Time: {update_time*1000:.2f}ms")
                print(f"  Score: {episode_reward:.2f}, Max Tile: {info['max_tile']}")
                print(f"  Policy Loss: {update_info.get('policy_loss', 0):.4f}, Value Loss: {update_info.get('value_loss', 0):.4f}")
                print("")
        
        # Save model
        if episode % eval_interval == 0:
            agent.save(os.path.join(save_path, f"policy_model_ep{episode}.pt"))
    
    # Save final model
    agent.save(os.path.join(save_path, "policy_model_final.pt"))
    
    return scores, max_tiles, metrics


def evaluate_agent(
    env: Env2048,
    agent: Any,
    num_episodes: int = 10,
    render: bool = False
) -> Tuple[float, float, Dict[int, int]]:
    """
    Evaluate an agent
    
    Args:
        env: The environment
        agent: The agent to evaluate
        num_episodes: Number of episodes to evaluate for
        render: Whether to render the environment
        
    Returns:
        avg_score: Average score
        avg_max_tile: Average max tile
        tile_counts: Dictionary mapping tile values to counts
    """
    # Import required modules
    import time
    
    scores = []
    max_tiles = []
    tile_counts = {}
    
    # Initialize pygame if rendering
    if render:
        import pygame
        import time
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Render the environment if requested
            if render:
                env.render()
                pygame.event.pump()
                time.sleep(0.1)  # Slightly longer delay for evaluation to better view actions
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action using agent (in evaluation mode)
            action = agent.select_action(state, valid_actions, training=False)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                # Show final state with a longer pause
                if render:
                    env.render()
                    pygame.event.pump()
                    time.sleep(0.5)
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
    
    print(f"Evaluation Results - Avg Score: {avg_score:.2f}, Avg Max Tile: {avg_max_tile:.2f}")
    
    return avg_score, avg_max_tile, tile_counts


def plot_metrics(metrics: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot training metrics
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, (key, values) in enumerate(metrics.items()):
        plt.subplot(len(metrics), 1, i + 1)
        plt.plot(values)
        plt.title(key)
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train RL agents for 2048 game")
    
    parser.add_argument("--agent", type=str, default="dqn", 
                        choices=["random", "dqn", "ddqn", "dueling_dqn", "reinforce", "ppo"],
                        help="Agent to train")
    
    parser.add_argument("--num_episodes", type=int, default=1000, 
                        help="Number of episodes to train for")
    
    parser.add_argument("--max_steps", type=int, default=10000, 
                        help="Maximum steps per episode")
    
    parser.add_argument("--log_interval", type=int, default=1, 
                        help="Log metrics every n episodes")
    
    parser.add_argument("--eval_interval", type=int, default=50, 
                        help="Evaluate and save the agent every n episodes")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during training")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    
    parser.add_argument("--vector_env", action="store_true", 
                        help="Use vectorized environment for faster training (DQN only)")
    
    parser.add_argument("--num_envs", type=int, default=4, 
                        help="Number of environments in vectorized env")
    
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results")
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pretrained model to continue training from")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.agent)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create environment
    env = Env2048(render_mode="human" if args.render else None, device=args.device)
    
    # Create agent
    if args.agent == "random":
        agent = RandomAgent(device=args.device)
    elif args.agent == "dqn":
        agent = DQNAgent(device=args.device, jit_compile=False)
    elif args.agent == "ddqn":
        agent = DoubleDQNAgent(device=args.device, jit_compile=False)
    elif args.agent == "dueling_dqn":
        agent = DuelingDQNAgent(device=args.device, jit_compile=False)
    elif args.agent == "reinforce":
        agent = REINFORCEAgent(device=args.device, jit_compile=False)
    elif args.agent == "ppo":
        agent = PPOAgent(device=args.device, jit_compile=False)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
    
    # Load pretrained model if provided
    if args.model_path is not None:
        print(f"Loading pretrained model from {args.model_path}")
        agent.load(args.model_path)
    
    # Train agent
    print(f"Training {args.agent} agent for {args.num_episodes} episodes...")
    
    start_time = time.time()
    
    if isinstance(agent, (DQNAgent, DoubleDQNAgent, DuelingDQNAgent)):
        scores, max_tiles, losses = train_dqn(
            env=env,
            agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            save_path=output_dir,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            vector_env=args.vector_env,
            num_envs=args.num_envs,
            render=args.render
        )
        
        # Save metrics
        metrics = {
            "scores": scores,
            "max_tiles": max_tiles,
            "losses": losses
        }
        
    elif isinstance(agent, (REINFORCEAgent, PPOAgent)):
        scores, max_tiles, metrics_dict = train_policy_based(
            env=env,
            agent=agent,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            save_path=output_dir,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            render=args.render
        )
        
        # Save metrics
        metrics = {
            "scores": scores,
            "max_tiles": max_tiles,
            **metrics_dict
        }
    else:
        # For random agent, just run evaluation
        avg_score, avg_max_tile, tile_counts = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=args.num_episodes,
            render=args.render
        )
        
        # Save metrics
        metrics = {
            "avg_score": [avg_score],
            "avg_max_tile": [avg_max_tile],
            "tile_counts": tile_counts
        }
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Plot metrics
    plot_metrics(metrics, os.path.join(output_dir, "metrics.png"))
    
    # Final evaluation
    print("\nFinal Evaluation:")
    avg_score, avg_max_tile, tile_counts = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=10,
        render=args.render
    )
    
    # Save evaluation results
    eval_results = {
        "avg_score": avg_score,
        "avg_max_tile": avg_max_tile,
        "tile_counts": tile_counts
    }
    
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 