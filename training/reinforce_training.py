"""
Enhanced REINFORCE Training Script with Fixed Issues
This script implements an improved REINFORCE agent with baseline reduction, entropy regularization,
gradient clipping, learning rate scheduling, and early stopping with proper environment handling.
"""
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from stable_baselines3 import PPO  # Added for commented PPO section
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Add environment to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import EnhancedAgrikaTractorFleetEnv  # Updated to Enhanced environment

class ImprovedREINFORCEAgent:
    """
    Enhanced REINFORCE agent with baseline reduction for optimal stability
    Features: Value baseline, entropy regularization, gradient clipping, LR scheduling, early stopping
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.0002, gamma: float = 0.99,
                 entropy_coef: float = 0.1, baseline_coef: float = 0.5,
                 max_grad_norm: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        
        # Policy network with dropout for regularization
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value network for baseline
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Optimizers with different learning rates
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate * 0.5)
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=500, gamma=0.95)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=500, gamma=0.95)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [], 
            'value_losses': [], 
            'entropies': [], 
            'advantages': [],
            'returns': [],
            'best_reward': -float('inf'), 
            'no_improvement': 0, 
            'patience': 20  # Increased patience
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if training:
            # Get action probabilities
            logits = self.policy_net(state_tensor)
            action_probs = F.softmax(logits, dim=-1)
            
            # Create distribution and sample
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Store for training
            self.episode_states.append(state)
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(log_prob)
            
            return action.item()
        else:
            # Deterministic action for evaluation
            with torch.no_grad():
                logits = self.policy_net(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                return torch.argmax(action_probs).item()
    
    def store_reward(self, reward: float):
        """Store reward for current timestep"""
        self.episode_rewards.append(reward)
    
    def compute_returns_and_advantages(self):
        """Compute discounted returns and advantages with baseline"""
        returns = []
        advantages = []
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states))
        rewards = torch.FloatTensor(self.episode_rewards)
        
        # Compute returns (discounted cumulative rewards)
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Compute baselines (value function estimates)
        with torch.no_grad():
            baselines = self.value_net(states).squeeze()
        
        # Compute advantages
        advantages = returns - baselines
        
        # Normalize advantages if we have multiple steps
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages, baselines
    
    def update_policy(self):
        """Update policy and value networks using collected episode data"""
        if not self.episode_rewards:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'mean_return': 0.0}, 0.0
        
        # Compute returns and advantages
        returns, advantages, baselines = self.compute_returns_and_advantages()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states))
        log_probs = torch.stack(self.episode_log_probs)
        
        # Policy loss with advantage weighting
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Entropy bonus for exploration
        logits = self.policy_net(states)
        action_probs = F.softmax(logits, dim=-1)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        # Total policy loss
        total_policy_loss = policy_loss - self.entropy_coef * entropy
        
        # Value loss (baseline training)
        current_values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(current_values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        self.value_scheduler.step()
        
        # Collect statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }
        
        # Update training stats
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        # Early stopping logic
        current_reward = returns.mean().item()
        if current_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = current_reward
            self.training_stats['no_improvement'] = 0
        else:
            self.training_stats['no_improvement'] += 1
        
        # Clear episode data
        self._clear_episode_data()
        
        return stats, current_reward
    
    def _clear_episode_data(self):
        """Clear episode data after training update"""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    def save(self, filepath: str):
        """Save model and training statistics"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'state_dim': self.state_dim, 
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate, 
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef, 
                'baseline_coef': self.baseline_coef
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training statistics"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        print(f"Model loaded from {filepath}")

def train_reinforce_optimized(episodes=2000, curriculum_stages=4):
    """Train REINFORCE with curriculum learning and proper error handling"""
    print("🌾 Starting Enhanced REINFORCE Training")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Episodes: {episodes}, Curriculum stages: {curriculum_stages}")
    print()

    # Create necessary directories
    os.makedirs("models/pg/reinforce", exist_ok=True)
    os.makedirs("logs/reinforce", exist_ok=True)

    # Initialize environment and agent
    env = EnhancedAgrikaTractorFleetEnv()  # Updated to Enhanced environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: State dim={state_dim}, Action dim={action_dim}")

    # Create agent with optimized hyperparameters from your report
    agent = ImprovedREINFORCEAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        learning_rate=0.0002,  # From your report
        gamma=0.99,            # From your report
        entropy_coef=0.1,      # From your report
        baseline_coef=0.5,
        max_grad_norm=0.5
    )
    
    start_time = time.time()
    episodes_per_stage = episodes // curriculum_stages
    
    # Curriculum parameters for progressive difficulty
    curriculum_params = [
        {'max_weather_changes': 2, 'breakdown_multiplier': 0.5},
        {'max_weather_changes': 4, 'breakdown_multiplier': 0.7},
        {'max_weather_changes': 6, 'breakdown_multiplier': 1.0},
        {'max_weather_changes': 8, 'breakdown_multiplier': 1.2}
    ]
    
    # Training history tracking
    training_history = {
        'episode_rewards': [], 
        'episode_lengths': [], 
        'policy_losses': [],
        'value_losses': [], 
        'entropies': [],
        'returns': []
    }
    
    print("🎓 Starting curriculum learning...")
    
    # Train through curriculum stages
    for stage, stage_params in enumerate(curriculum_params):
        print(f"\n📚 Curriculum Stage {stage + 1}/{curriculum_stages}")
        print(f"   Parameters: {stage_params}")
        stage_rewards = []
        
        for episode in range(episodes_per_stage):
            global_episode = stage * episodes_per_stage + episode
            
            # Reset environment with curriculum parameters
            try:
                # First try with options (newer gymnasium style)
                obs, info = env.reset(options=stage_params)
            except TypeError:
                try:
                    # Fallback: try with seed parameter (older style)
                    obs, info = env.reset(seed=None)
                    # Manually set curriculum parameters on environment
                    if hasattr(env, 'curriculum_params'):
                        env.curriculum_params = stage_params
                except Exception as e:
                    print(f"⚠️  Environment reset failed: {e}")
                    # Final fallback: standard reset
                    obs, info = env.reset()
            
            episode_reward = 0
            episode_length = 0
            
            # Run episode
            while True:
                action = agent.select_action(obs, training=True)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.store_reward(reward)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Update policy after episode
            stats, current_reward = agent.update_policy()
            
            # Store training history
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_lengths'].append(episode_length)
            training_history['policy_losses'].append(stats['policy_loss'])
            training_history['value_losses'].append(stats['value_loss'])
            training_history['entropies'].append(stats['entropy'])
            training_history['returns'].append(stats['mean_return'])
            
            stage_rewards.append(episode_reward)
            
            # Early stopping check
            if agent.training_stats['no_improvement'] >= agent.training_stats['patience']:
                print(f"   Early stopping triggered at episode {global_episode}")
                break
            
            # Progress reporting
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(stage_rewards[-100:])
                avg_entropy = np.mean(training_history['entropies'][-100:])
                print(f"   Episode {global_episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Entropy: {avg_entropy:.3f} | "
                      f"LR: {agent.policy_optimizer.param_groups[0]['lr']:.6f}")
        
        # Stage summary
        stage_avg = np.mean(stage_rewards)
        stage_std = np.std(stage_rewards)
        print(f"   Stage {stage + 1} completed - Reward: {stage_avg:.2f} ± {stage_std:.2f}")
        
        # Save intermediate model
        agent.save(f'models/pg/reinforce/reinforce_stage_{stage + 1}.pt')
    
    # Final evaluation
    training_time = time.time() - start_time
    print(f"\n🧪 Final evaluation...")
    
    eval_rewards = []
    eval_episodes = 50
    
    for eval_ep in range(eval_episodes):
        try:
            # Use most challenging curriculum for evaluation
            obs, info = env.reset(options=curriculum_params[-1])
        except TypeError:
            obs, info = env.reset()
        
        episode_reward = 0
        while True:
            action = agent.select_action(obs, training=False)  # Deterministic
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(episode_reward)
    
    # Calculate final metrics
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    # Save final model
    agent.save('models/pg/reinforce/final_reinforce_model.pt')
    
    # Save training history
    training_summary = {
        'training_summary': {
            'total_episodes': len(training_history['episode_rewards']),
            'training_time_minutes': training_time / 60,
            'final_mean_reward': float(mean_reward),
            'final_std_reward': float(std_reward),
            'curriculum_stages': curriculum_stages,
            'best_reward': float(agent.training_stats['best_reward'])
        },
        'hyperparameters': {
            'learning_rate': agent.learning_rate,
            'gamma': agent.gamma,
            'entropy_coef': agent.entropy_coef,
            'baseline_coef': agent.baseline_coef
        },
        'training_history': {k: [float(x) for x in v] for k, v in training_history.items()}
    }
    
    with open('logs/reinforce/reinforce_training_history.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    # Generate training plots
    create_training_plots(training_history, 'logs/reinforce/reinforce_analysis.png')
    
    # Final report
    print(f"\n✅ Enhanced REINFORCE Training Completed!")
    print(f"   Final performance: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Training time: {training_time / 60:.1f} minutes")
    print(f"   Total episodes: {len(training_history['episode_rewards'])}")
    print(f"   Best reward achieved: {agent.training_stats['best_reward']:.2f}")
    print(f"   Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results saved to 'logs/reinforce/' and 'models/pg/reinforce/'")
    
    env.close()
    return agent, training_summary

def create_training_plots(training_history, save_path):
    """Create comprehensive training analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('REINFORCE Training Analysis', fontsize=16)
    
    episodes = len(training_history['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    # Plot 1: Episode rewards with moving average
    ax1 = axes[0, 0]
    ax1.plot(episode_range, training_history['episode_rewards'], alpha=0.3, color='blue', linewidth=0.5)
    if episodes > 50:
        window_size = min(100, episodes // 10)
        moving_avg = np.convolve(training_history['episode_rewards'], 
                                np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size, episodes + 1), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Policy and value losses
    ax2 = axes[0, 1]
    ax2.plot(episode_range, training_history['policy_losses'], 
            label='Policy Loss', color='orange', alpha=0.7)
    ax2.plot(episode_range, training_history['value_losses'], 
            label='Value Loss', color='green', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy over time
    ax3 = axes[0, 2]
    ax3.plot(episode_range, training_history['entropies'], 
            color='purple', alpha=0.7)
    if episodes > 50:
        window_size = min(50, episodes // 10)
        entropy_avg = np.convolve(training_history['entropies'], 
                                 np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size, episodes + 1), entropy_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Policy Entropy')
    ax3.set_title('Exploration (Entropy)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Episode lengths
    ax4 = axes[1, 0]
    ax4.plot(episode_range, training_history['episode_lengths'], 
            color='brown', alpha=0.6)
    if episodes > 50:
        window_size = min(50, episodes // 10)
        length_avg = np.convolve(training_history['episode_lengths'], 
                                np.ones(window_size)/window_size, mode='valid')
        ax4.plot(range(window_size, episodes + 1), length_avg, 
                color='red', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Length')
    ax4.set_title('Episode Lengths')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Reward distribution
    ax5 = axes[1, 1]
    ax5.hist(training_history['episode_rewards'], bins=50, alpha=0.7, 
            color='skyblue', edgecolor='black')
    ax5.axvline(np.mean(training_history['episode_rewards']), 
               color='red', linestyle='--', linewidth=2, label='Mean')
    ax5.set_xlabel('Episode Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Returns over time
    ax6 = axes[1, 2]
    ax6.plot(episode_range, training_history['returns'], 
            color='teal', alpha=0.7)
    if episodes > 50:
        window_size = min(50, episodes // 10)
        returns_avg = np.convolve(training_history['returns'], 
                                 np.ones(window_size)/window_size, mode='valid')
        ax6.plot(range(window_size, episodes + 1), returns_avg, 
                color='red', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Mean Return')
    ax6.set_title('Returns (Discounted Rewards)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {save_path}")

if __name__ == "__main__":
    # Add command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train REINFORCE agent')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--stages', type=int, default=4, help='Number of curriculum stages')
    args = parser.parse_args()
    
    agent, summary = train_reinforce_optimized(episodes=args.episodes, curriculum_stages=args.stages)