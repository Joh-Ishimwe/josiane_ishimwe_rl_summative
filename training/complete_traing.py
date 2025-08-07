"""
Complete Agrika RL Training Suite
Combines fully optimized training with enhanced REINFORCE implementation
Includes all algorithms: DQN, PPO, A2C, and Enhanced REINFORCE with baseline
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
from collections import deque
import random
from typing import List, Tuple, Dict, Any


# Add environment to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import AgrikaTractorFleetEnv


# Stable Baselines3 imports
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy


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


class CompleteAgrikaTrainer:
    def __init__(self, load_optimized_params=True):
        self.results = {}
        self.optimized_params = {}
        self.load_optimized_params = load_optimized_params
        self.training_history = {
            'episode_rewards': [], 'episode_lengths': [], 'policy_losses': [],
            'value_losses': [], 'entropies': [], 'learning_rates': []
        }
       
        self.create_directories()
        if load_optimized_params:
            self.load_hyperparameters()
        else:
            self.use_default_params()
   
    def create_directories(self):
        dirs = [
            "models/optimized_dqn", "models/optimized_ppo", "models/optimized_a2c", "models/optimized_reinforce",
            "logs/optimized_dqn", "logs/optimized_ppo", "logs/optimized_a2c", "logs/optimized_reinforce",
            "results/optimized_training", "results/comparisons", "videos/optimized_demos"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
   
    def load_hyperparameters(self):
        try:
            for algo in ['DQN', 'PPO', 'A2C', 'REINFORCE']:
                file_path = f'results/hyperparameter_studies/{algo.lower()}_optimization.json'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.optimized_params[algo] = json.load(f)['best_params']
                    print(f"✅ Loaded optimized {algo} hyperparameters")
            if not self.optimized_params:
                print("⚠️ No optimized hyperparameters found, using enhanced defaults")
                self.use_default_params()
        except Exception as e:
            print(f"❌ Error loading optimized hyperparameters: {e}")
            self.use_default_params()
   
    def use_default_params(self):
        self.optimized_params = {
            'DQN': {
                'learning_rate': 0.0004, 'buffer_size': 100000, 'batch_size': 64,
                'gamma': 0.997, 'exploration_fraction': 0.2, 'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.03, 'target_update_interval': 800, 'train_freq': 4,
                'net_arch': 'deep_large'
            },
            'PPO': {
                'learning_rate': 0.00025, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10,
                'gamma': 0.997, 'gae_lambda': 0.95, 'clip_range': 0.18, 'ent_coef': 0.015,
                'vf_coef': 0.5, 'max_grad_norm': 0.5, 'net_arch_pi': 'deep_large',
                'net_arch_vf': 'deep_large'
            },
            'A2C': {
                'learning_rate': 0.0006, 'n_steps': 10, 'gamma': 0.997, 'gae_lambda': 0.96,
                'ent_coef': 0.015, 'vf_coef': 0.25, 'max_grad_norm': 0.5, 'rms_prop_eps': 1e-5,
                'net_arch': 'deep_large'
            },
            'REINFORCE': {
                'learning_rate': 0.0007, 'gamma': 0.996, 'entropy_coef': 0.025,
                'baseline_coef': 0.6, 'max_grad_norm': 0.5
            }
        }
   
    def _convert_net_arch(self, arch_key):
        arch_map = {
            'small': [64, 64], 'medium': [128, 128], 'large': [256, 256],
            'deep_medium': [128, 256, 128], 'deep_large': [256, 256, 128],
            'very_large': [256, 512, 256]
        }
        return arch_map.get(arch_key, [256, 256])
   
    def train_optimized_dqn(self, timesteps=200000):
        print("🤖 Training Optimized DQN...")
        env = Monitor(AgrikaTractorFleetEnv(), "logs/optimized_dqn")
        params = self.optimized_params['DQN'].copy()
        net_arch = self._convert_net_arch(params.pop('net_arch', 'deep_large'))
       
        model = DQN(
            "MlpPolicy", env, **params,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log="logs/optimized_dqn",
            verbose=1, device="auto"
        )
       
        eval_env = Monitor(AgrikaTractorFleetEnv(), "logs/optimized_dqn/eval")
        eval_callback = EvalCallback(
            eval_env, best_model_save_path="models/optimized_dqn/",
            log_path="logs/optimized_dqn/", eval_freq=5000,
            deterministic=True, n_eval_episodes=15,
            callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=2200, verbose=1)
        )
       
        start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
        training_time = time.time() - start_time
       
        model.save("models/optimized_dqn/final_optimized_dqn")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)
       
        results = {
            'algorithm': 'Optimized_DQN',
            'training_time': training_time,
            'timesteps': timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparameters': self.optimized_params['DQN'],
            'model': model
        }
       
        self.plot_learning_curve("logs/optimized_dqn", "Optimized DQN Learning",
                                "results/optimized_training/optimized_dqn_learning.png")
        print(f"✅ Optimized DQN completed! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return results
   
    def train_optimized_ppo(self, timesteps=250000):
        print("🎯 Training Optimized PPO...")
       
        def make_env():
            return Monitor(AgrikaTractorFleetEnv(), "logs/optimized_ppo")
       
        vec_env = make_vec_env(make_env, n_envs=4)
        params = self.optimized_params['PPO'].copy()
        net_arch_pi = self._convert_net_arch(params.pop('net_arch_pi', 'deep_large'))
        net_arch_vf = self._convert_net_arch(params.pop('net_arch_vf', 'deep_large'))
       
        model = PPO(
            "MlpPolicy", vec_env, **params,
            policy_kwargs=dict(net_arch=dict(pi=net_arch_pi, vf=net_arch_vf), activation_fn=torch.nn.ReLU),
            tensorboard_log="logs/optimized_ppo",
            verbose=1, device="auto"
        )
       
        eval_env = Monitor(AgrikaTractorFleetEnv(), "logs/optimized_ppo/eval")
        eval_callback = EvalCallback(
            eval_env, best_model_save_path="models/optimized_ppo/",
            log_path="logs/optimized_ppo/", eval_freq=10000,
            deterministic=True, n_eval_episodes=15,
            callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=2400, verbose=1)
        )
       
        start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
        training_time = time.time() - start_time
       
        model.save("models/optimized_ppo/final_optimized_ppo")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)
       
        results = {
            'algorithm': 'Optimized_PPO',
            'training_time': training_time,
            'timesteps': timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparameters': self.optimized_params['PPO'],
            'model': model
        }
        print(f"✅ Optimized PPO completed! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return results
   
    def train_optimized_a2c(self, timesteps=180000):
        print("⚡ Training Optimized A2C...")
       
        def make_env():
            return Monitor(AgrikaTractorFleetEnv(), "logs/optimized_a2c")
       
        vec_env = make_vec_env(make_env, n_envs=8)
        params = self.optimized_params['A2C'].copy()
        net_arch = self._convert_net_arch(params.pop('net_arch', 'deep_large'))
       
        model = A2C(
            "MlpPolicy", vec_env, **params,
            use_rms_prop=True,
            policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
            tensorboard_log="logs/optimized_a2c",
            verbose=1, device="auto"
        )
       
        eval_env = Monitor(AgrikaTractorFleetEnv(), "logs/optimized_a2c/eval")
        eval_callback = EvalCallback(
            eval_env, best_model_save_path="models/optimized_a2c/",
            log_path="logs/optimized_a2c/", eval_freq=5000,
            deterministic=True, n_eval_episodes=15,
            callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=2100, verbose=1)
        )
       
        start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
        training_time = time.time() - start_time
       
        model.save("models/optimized_a2c/final_optimized_a2c")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=30)
       
        results = {
            'algorithm': 'Optimized_A2C',
            'training_time': training_time,
            'timesteps': timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparameters': self.optimized_params['A2C'],
            'model': model
        }
        print(f"✅ Optimized A2C completed! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return results
     
   
   
    def train_all_algorithms(self):
        print("🚀 COMPLETE AGRIKA RL TRAINING SUITE")
        print("="*70)
       
        algorithms = [
            ("Optimized_DQN", self.train_optimized_dqn),
            ("Optimized_PPO", self.train_optimized_ppo),
            ("Optimized_A2C", self.train_optimized_a2c),
           
        ]
       
        all_results = {}
        training_summary = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameter_source': 'optimized' if self.load_optimized_params else 'enhanced_defaults',
            'results': {}
        }
       
        for name, train_func in algorithms:
            print(f"\n{'='*20} Training {name} {'='*20}")
            try:
                results = train_func()
                all_results[name] = results
                training_summary['results'][name] = {
                    'mean_reward': results['mean_reward'],
                    'std_reward': results['std_reward'],
                    'training_time': results['training_time']
                }
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                all_results[name] = {'status': 'failed', 'error': str(e)}
       
        training_summary['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open('results/optimized_training/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
       
        self.results = all_results
        self.generate_complete_comparison_report()
        return all_results
   
    def plot_learning_curve(self, log_dir: str, title: str, save_path: str):
        try:
            results = load_results(log_dir)
            x, y = ts2xy(results, 'timesteps')
            plt.figure(figsize=(12, 6))
            plt.plot(x, y, linewidth=2, alpha=0.8)
            window_size = len(y) // 20
            if window_size > 1:
                moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                plt.plot(x[window_size-1:], moving_avg, linewidth=3, color='red', label=f'Moving Avg ({window_size})')
                plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Learning curve saved to {save_path}")
        except Exception as e:
            print(f"Could not generate learning curve: {e}")
   
    def create_enhanced_reinforce_visualizations(self, results):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced REINFORCE Training Analysis', fontsize=16)
       
        episodes = len(self.training_history['episode_rewards'])
        episode_range = range(1, episodes + 1)
       
        ax1.plot(episode_range, self.training_history['episode_rewards'], alpha=0.3, color='blue')
        window_size = min(100, episodes // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.training_history['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size, episodes + 1), moving_avg, color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Learning Progress')
        ax1.grid(True, alpha=0.3)
       
        if self.training_history['policy_losses']:
            ax2.plot(episode_range, self.training_history['policy_losses'], label='Policy Loss')
            ax2.plot(episode_range, self.training_history['value_losses'], label='Value Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Losses')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
       
        if self.training_history['entropies']:
            ax3.plot(episode_range, self.training_history['entropies'], color='green')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Policy Entropy')
            ax3.set_title('Exploration (Entropy) Over Time')
            ax3.grid(True, alpha=0.3)
       
        final_rewards = self.training_history['episode_rewards'][-500:] if len(self.training_history['episode_rewards']) >= 500 else self.training_history['episode_rewards']
        ax4.hist(final_rewards, bins=30, alpha=0.7, color='purple')
        ax4.axvline(results['mean_reward'], color='red', linestyle='--', label=f'Mean: {results["mean_reward"]:.1f}')
        ax4.set_xlabel('Episode Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Final Performance Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.savefig('results/optimized_training/enhanced_reinforce_analysis.png', dpi=300)
        plt.close()
        print("📊 Enhanced REINFORCE visualizations saved!")
   
    def generate_complete_comparison_report(self):
        print("\n" + "="*70)
        print("📊 COMPLETE TRAINING RESULTS SUMMARY")
        print("="*70)
       
        print(f"{'Algorithm':<20} {'Mean Reward':<20} {'Training Time':<15} {'Algorithm Type':<15}")
        print("-" * 70)
       
        for name, results in self.results.items():
            if 'mean_reward' in results:
                algo_type = "Policy Gradient" if "REINFORCE" in name else "Value-Based" if "DQN" in name else "Actor-Critic"
                print(f"{name:<20} {results['mean_reward']:.2f} ± {results['std_reward']:.2f}     "
                      f"{results['training_time']:.1f}s        {algo_type:<15}")
       
        self.create_complete_comparison_plots()
        best_algo = self._get_best_algorithm()
        print(f"\n🏆 Best performing algorithm: {best_algo}")
        self._save_complete_detailed_results()
        print(f"\n📈 All results saved to 'results/optimized_training/'")
   
    def create_complete_comparison_plots(self):
        algorithms = []
        mean_rewards = []
        std_rewards = []
        training_times = []
       
        for name, results in self.results.items():
            if 'mean_reward' in results:
                algorithms.append(name.replace('Optimized_', '').replace('Enhanced_', ''))
                mean_rewards.append(results['mean_reward'])
                std_rewards.append(results['std_reward'])
                training_times.append(results['training_time'])
       
        if not algorithms:
            return
       
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agrika RL: Complete Algorithm Performance Analysis', fontsize=16)
       
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
       
        ax1.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=8, color=colors[:len(algorithms)], alpha=0.8)
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Algorithm Performance (with Std Dev)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
       
        efficiency = [r/t*60 for r, t in zip(mean_rewards, training_times)]
        ax2.bar(algorithms, efficiency, color=colors[:len(algorithms)], alpha=0.8)
        ax2.set_ylabel('Reward per Minute')
        ax2.set_title('Training Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
       
        ax3.bar(algorithms, [t/60 for t in training_times], color=colors[:len(algorithms)], alpha=0.8)
        ax3.set_ylabel('Training Time (minutes)')
        ax3.set_title('Training Duration')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
       
        plt.tight_layout()
        plt.savefig('results/optimized_training/complete_algorithm_comparison.png', dpi=300)
        plt.close()
        print("📊 Complete algorithm comparison plots saved!")
   
    def _get_best_algorithm(self) -> str:
        best_reward = float('-inf')
        best_algo = None
        for name, results in self.results.items():
            if 'mean_reward' in results and results['mean_reward'] > best_reward:
                best_reward = results['mean_reward']
                best_algo = name
        return best_algo or "None"
   
    def _save_complete_detailed_results(self):
        detailed_results = {
            'training_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameter_optimization_used': self.load_optimized_params,
                'total_algorithms_trained': len([r for r in self.results.values() if 'mean_reward' in r])
            },
            'algorithm_results': {name: {'mean_reward': res['mean_reward'], 'std_reward': res['std_reward'],
                                       'training_time': res['training_time']} for name, res in self.results.items()
                                 if 'mean_reward' in res}
        }
        with open('results/optimized_training/complete_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
   
    def generate_final_report(self):
        print("\n" + "🎉" * 30)
        print("FINAL AGRIKA RL TRAINING REPORT")
        print("🎉" * 30)
       
        successful_algorithms = [name for name, res in self.results.items() if 'mean_reward' in res]
        if not successful_algorithms:
            print("❌ No algorithms completed successfully!")
            return
       
        print(f"\n✅ Successfully trained {len(successful_algorithms)} algorithms:")
        for algo in successful_algorithms:
            results = self.results[algo]
            print(f"   • {algo}: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
       
        best_algo = self._get_best_algorithm()
        best_results = self.results[best_algo]
        print(f"\n🏆 CHAMPION ALGORITHM: {best_algo}")
        print(f"   Performance: {best_results['mean_reward']:.2f} ± {best_results['std_reward']:.2f}")
        print(f"   Training Time: {best_results['training_time']/60:.1f} minutes")
       
        total_time = sum(res['training_time'] for res in self.results.values() if 'training_time' in res)
        print(f"\n⏱️  Total Training Time: {total_time/3600:.1f} hours")
        print(f"\n📁 All results, models, and visualizations saved to:")
        print(f"   • Models: models/optimized_*/")
        print(f"   • Logs: logs/optimized_*/")
        print(f"   • Results: results/optimized_training/")


def main():
    print("🌾 COMPLETE AGRIKA RL TRAINING SUITE")
    print("="*50)
    print("Training all algorithms: DQN, PPO, A2C, Enhanced REINFORCE")
    print("With fully optimized hyperparameters and advanced features")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')} CAT")
    print()
   
    trainer = CompleteAgrikaTrainer(load_optimized_params=True)
    results = trainer.train_all_algorithms()
    trainer.generate_final_report()
    print("\n🎉 COMPLETE TRAINING SUITE FINISHED!")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')} CAT")
    print("Check 'results/optimized_training/' for comprehensive analysis")
    return results


if __name__ == "__main__":
    main()