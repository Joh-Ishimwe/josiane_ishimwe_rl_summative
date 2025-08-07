#reinforce_training.py
"""
Enhanced REINFORCE Training Script
This script implements an improved REINFORCE agent with baseline reduction, entropy regularization,
gradient clipping, learning rate scheduling, and early stopping.
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

# Add environment to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import AgrikaTractorFleetEnv

class ImprovedREINFORCEAgent:
    """
    Enhanced REINFORCE agent with baseline reduction for optimal stability
    Features: Value baseline, entropy regularization, gradient clipping, LR scheduling, early stopping
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.0007, gamma: float = 0.996,
                 entropy_coef: float = 0.025, baseline_coef: float = 0.6,
                 max_grad_norm: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate * 0.5)
        
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=500, gamma=0.95)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=500, gamma=0.95)
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        self.training_stats = {
            'policy_losses': [], 'value_losses': [], 'entropies': [], 'advantages': [],
            'best_reward': -float('inf'), 'no_improvement': 0, 'patience': 15
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_net(state_tensor)
        if training:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            self.episode_states.append(state)
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(log_prob)
            return action.item()
        return torch.argmax(action_probs).item()
    
    def store_reward(self, reward: float):
        self.episode_rewards.append(reward)
    
    def compute_returns_and_advantages(self):
        returns = []
        advantages = []
        states = torch.FloatTensor(np.array(self.episode_states))
        rewards = torch.FloatTensor(self.episode_rewards)
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        with torch.no_grad():
            baselines = self.value_net(states).squeeze()
        advantages = returns - baselines
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages, baselines
    
    def update_policy(self):
        if not self.episode_rewards:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}, 0.0
        
        returns, advantages, baselines = self.compute_returns_and_advantages()
        states = torch.FloatTensor(np.array(self.episode_states))
        log_probs = torch.stack(self.episode_log_probs)
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        action_probs = self.policy_net(states)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        total_policy_loss = policy_loss - self.entropy_coef * entropy
        
        current_values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(current_values, returns)
        
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        self.value_scheduler.step()
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        current_reward = returns.mean().item()
        if current_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = current_reward
            self.training_stats['no_improvement'] = 0
        else:
            self.training_stats['no_improvement'] += 1
        
        self._clear_episode_data()
        return stats, current_reward
    
    def _clear_episode_data(self):
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    def save(self, filepath: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'hyperparameters': {
                'state_dim': self.state_dim, 'action_dim': self.action_dim,
                'learning_rate': self.learning_rate, 'gamma': self.gamma,
                'entropy_coef': self.entropy_coef, 'baseline_coef': self.baseline_coef
            }
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

def train_reinforce_separately(episodes=2000, curriculum_stages=4):
    print("🌾 Starting Standalone Enhanced REINFORCE Training")
    print("=" * 50)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')} CAT")
    print()

    # Create necessary directories
    os.makedirs("models/optimized_reinforce", exist_ok=True)
    os.makedirs("results/optimized_training", exist_ok=True)

    env = AgrikaTractorFleetEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ImprovedREINFORCEAgent(
        state_dim=state_dim, action_dim=action_dim,
        learning_rate=0.0007, gamma=0.996,
        entropy_coef=0.025, baseline_coef=0.6,
        max_grad_norm=0.5
    )
    
    start_time = time.time()
    episodes_per_stage = episodes // curriculum_stages
    
    curriculum_params = [
        {'max_weather_changes': 2, 'breakdown_multiplier': 0.4},
        {'max_weather_changes': 4, 'breakdown_multiplier': 0.7},
        {'max_weather_changes': 6, 'breakdown_multiplier': 1.0},
        {'max_weather_changes': 8, 'breakdown_multiplier': 1.3}
    ]
    
    training_history = {
        'episode_rewards': [], 'episode_lengths': [], 'policy_losses': [],
        'value_losses': [], 'entropies': []
    }
    
    for stage, stage_params in enumerate(curriculum_params):
        print(f"\n📚 Curriculum Stage {stage + 1}/{curriculum_stages}")
        stage_rewards = []
        
        for episode in range(episodes_per_stage):
            global_episode = stage * episodes_per_stage + episode
            # Use options parameter to pass curriculum settings
            try:
                obs, info = env.reset(options=stage_params)  # Pass as options dictionary
            except TypeError as e:
                print(f"❌ Reset failed with options: {e}")
                print(f"Stage params: {stage_params}")
                return
            except Exception as e:
                print(f"❌ Unexpected error during reset: {e}")
                print(f"Stage params: {stage_params}")
                return
                
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = agent.select_action(obs, training=True)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.store_reward(reward)
                episode_reward += reward
                episode_length += 1
                if terminated or truncated:
                    break
            
            stats, current_reward = agent.update_policy()
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_lengths'].append(episode_length)
            training_history['policy_losses'].append(stats['policy_loss'])
            training_history['value_losses'].append(stats['value_loss'])
            training_history['entropies'].append(stats['entropy'])
            
            stage_rewards.append(episode_reward)
            
            if agent.training_stats['no_improvement'] >= agent.training_stats['patience']:
                print(f"Early stopping triggered at episode {global_episode}")
                break
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(stage_rewards[-100:])
                print(f"   Episode {global_episode + 1} | Avg Reward: {avg_reward:.2f}")
        
        stage_avg = np.mean(stage_rewards)
        print(f"   Stage {stage + 1} completed - Average reward: {stage_avg:.2f}")
        agent.save(f'models/optimized_reinforce/reinforce_stage_{stage + 1}.pt')
    
    training_time = time.time() - start_time
    eval_rewards = []
    for _ in range(50):
        try:
            obs, info = env.reset(options=curriculum_params[-1])  # Use last stage for evaluation
        except TypeError:
            obs, info = env.reset()  # Fallback to default reset if options fail
        episode_reward = 0
        while True:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(episode_reward)
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    agent.save('models/optimized_reinforce/final_improved_reinforce.pt')
    
    with open('results/optimized_training/reinforce_training_history.json', 'w') as f:
        json.dump({
            'training_summary': {
                'episodes': episodes,
                'training_time_minutes': training_time / 60,
                'final_mean_reward': mean_reward,
                'final_std_reward': std_reward
            },
            'training_history': {k: v for k, v in training_history.items()}
        }, f, indent=2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Enhanced REINFORCE Training Analysis')
    
    episodes = len(training_history['episode_rewards'])
    episode_range = range(1, episodes + 1)
    
    ax1.plot(episode_range, training_history['episode_rewards'], alpha=0.3, color='blue')
    window_size = min(100, episodes // 10)
    if window_size > 1:
        moving_avg = np.convolve(training_history['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size, episodes + 1), moving_avg, color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Progress')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(episode_range, training_history['policy_losses'], label='Policy Loss')
    ax2.plot(episode_range, training_history['value_losses'], label='Value Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/optimized_training/reinforce_analysis.png', dpi=300)
    plt.close()
    
    print(f"\n✅ Enhanced REINFORCE completed! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Training time: {training_time / 60:.1f} minutes")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')} CAT")
    print("📁 Results saved to 'results/optimized_training/' and 'models/optimized_reinforce/'")

if __name__ == "__main__":
    train_reinforce_separately()