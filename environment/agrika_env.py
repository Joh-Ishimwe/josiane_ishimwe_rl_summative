#agrika_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import random

class AgrikaTractorFleetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, normalize_rewards=True):
        super().__init__()
        self.season_length = 60
        self.num_tractors = 3
        self.max_hours_per_day = 12
        self.breakdown_threshold = 80
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([100, 100, 30, 100, 100, 30, 100, 100, 30, 3, 3, 3, 3, 60, 100]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)
        self.weather_types = ['Sunny', 'Rainy', 'Stormy', 'Dry']
        self.render_mode = render_mode
        self.normalize_rewards = normalize_rewards
        self.reward_buffer = []  # For running mean/std
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.tractors = np.array([
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0]
        ])
        self.weather = np.random.randint(0, 4, size=4)
        self.current_day = 0
        self.crop_demand = self._calculate_crop_demand()
        self.total_productivity = 0.0
        self.total_maintenance_cost = 0.0
        self.breakdown_count = 0
        self.reward_buffer = []  # Reset buffer
        self.reward_mean = 0.0
        self.reward_std = 1.0
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        tractor_actions = self.decode_action(action)
        daily_productivity = 0.0
        daily_maintenance_cost = 0.0
        daily_breakdown_penalty = 0.0
        
        for i, tractor_action in enumerate(tractor_actions):
            if tractor_action == 0:  # OPERATE
                productivity, breakdown = self._operate_tractor(i)
                daily_productivity += productivity
                if breakdown:
                    daily_breakdown_penalty += 200
                    self.breakdown_count += 1
            elif tractor_action == 1:  # MAINTAIN
                cost = self._maintain_tractor(i)
                daily_maintenance_cost += cost
            elif tractor_action == 2:  # REST
                self._rest_tractor(i)
        
        self._update_environment()
        raw_reward = self._calculate_reward(
            daily_productivity, 
            daily_maintenance_cost, 
            daily_breakdown_penalty
        )
        
        # Normalize reward
        if self.normalize_rewards:
            self.reward_buffer.append(raw_reward)
            if len(self.reward_buffer) > 1000:
                self.reward_buffer.pop(0)
            self.reward_mean = np.mean(self.reward_buffer)
            self.reward_std = np.std(self.reward_buffer) + 1e-8
            reward = (raw_reward - self.reward_mean) / self.reward_std
        else:
            reward = raw_reward
        
        terminated = self.current_day >= self.season_length
        info = {
            'day': self.current_day,
            'productivity': daily_productivity,
            'maintenance_cost': daily_maintenance_cost,
            'breakdowns': self.breakdown_count,
            'weather': self.weather_types[self.weather[0]],
            'raw_reward': raw_reward  # Log raw reward for analysis
        }
        
        return self._get_observation(), reward, terminated, False, info
        
    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """Convert discrete action to individual tractor actions"""
        # Convert base-10 to base-3 representation
        tractor3_action = action % 3
        tractor2_action = (action // 3) % 3
        tractor1_action = (action // 9) % 3
        return tractor1_action, tractor2_action, tractor3_action
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Initialize tractors: [hours_used, condition, days_since_maintenance]
        self.tractors = np.array([
            [0.0, 100.0, 0.0],  # Tractor 1
            [0.0, 100.0, 0.0],  # Tractor 2  
            [0.0, 100.0, 0.0]   # Tractor 3
        ])
        
        # Initialize weather (4 days: current + 3-day forecast)
        self.weather = np.random.randint(0, 4, size=4)
        
        # Season progress
        self.current_day = 0
        self.crop_demand = self._calculate_crop_demand()
        
        # Performance tracking
        self.total_productivity = 0.0
        self.total_maintenance_cost = 0.0
        self.breakdown_count = 0
        
        return self._get_observation(), {}
    
    def _calculate_crop_demand(self) -> float:
        """Calculate crop demand based on season day (realistic farming patterns)"""
        # Higher demand during planting (days 1-20) and harvest (days 40-60)
        if self.current_day <= 20:
            return 70 + 30 * np.sin(np.pi * self.current_day / 20)
        elif 20 < self.current_day <= 40:
            return 30 + 20 * np.sin(np.pi * (self.current_day - 20) / 20)
        else:  # Harvest season
            return 80 + 20 * np.sin(np.pi * (self.current_day - 40) / 20)
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = np.concatenate([
            self.tractors.flatten(),  # 9 values (3 tractors × 3 attributes)
            self.weather.astype(np.float32),  # 4 values (current + 3-day forecast)
            [self.current_day, self.crop_demand]  # 2 values
        ])
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Decode action for each tractor
        tractor_actions = self.decode_action(action)
        
        # Initialize daily metrics
        daily_productivity = 0.0
        daily_maintenance_cost = 0.0
        daily_breakdown_penalty = 0.0
        
        # Process each tractor
        for i, tractor_action in enumerate(tractor_actions):
            if tractor_action == 0:  # OPERATE
                productivity, breakdown = self._operate_tractor(i)
                daily_productivity += productivity
                if breakdown:
                    daily_breakdown_penalty += 200
                    self.breakdown_count += 1
                    
            elif tractor_action == 1:  # MAINTAIN
                cost = self._maintain_tractor(i)
                daily_maintenance_cost += cost
                
            elif tractor_action == 2:  # REST
                self._rest_tractor(i)
        
        # Update environment state
        self._update_environment()
        
        # Calculate reward
        reward = self._calculate_reward(
            daily_productivity, 
            daily_maintenance_cost, 
            daily_breakdown_penalty
        )
        
        # Check termination
        terminated = self.current_day >= self.season_length
        
        # Additional info
        info = {
            'day': self.current_day,
            'productivity': daily_productivity,
            'maintenance_cost': daily_maintenance_cost,
            'breakdowns': self.breakdown_count,
            'weather': self.weather_types[self.weather[0]]
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def _operate_tractor(self, tractor_id: int) -> Tuple[float, bool]:
        """Operate a tractor and return productivity and breakdown status"""
        
        # Weather impact on operation
        weather_multiplier = {0: 1.0, 1: 0.7, 2: 0.3, 3: 1.2}[self.weather[0]]
        
        # Calculate base productivity
        base_hours = min(self.max_hours_per_day, 
                        self.tractors[tractor_id, 1] / 10)  # Condition affects hours
        
        actual_productivity = base_hours * weather_multiplier
        
        # Update tractor state
        self.tractors[tractor_id, 0] += base_hours  # Add hours
        self.tractors[tractor_id, 1] -= random.uniform(1, 3)  # Condition degrades
        self.tractors[tractor_id, 2] += 1  # Days since maintenance
        
        # Check for breakdown
        breakdown_probability = self._calculate_breakdown_probability(tractor_id)
        breakdown = random.random() < breakdown_probability
        
        if breakdown:
            self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - 20)
            actual_productivity *= 0.5  # Reduced productivity due to breakdown
        
        return actual_productivity, breakdown
    
    def _maintain_tractor(self, tractor_id: int) -> float:
        """Perform maintenance on a tractor"""
        
        # Calculate maintenance cost based on current condition
        condition = self.tractors[tractor_id, 1]
        base_cost = 50 + (100 - condition) * 2  # More expensive for worse condition
        
        # Restore tractor condition
        self.tractors[tractor_id, 1] = min(100, condition + 30)
        self.tractors[tractor_id, 2] = 0  # Reset days since maintenance
        
        return base_cost
    
    def _rest_tractor(self, tractor_id: int):
        """Rest a tractor (no operation, minimal degradation)"""
        self.tractors[tractor_id, 2] += 1  # Still counts as a day
        # Minimal condition loss during rest
        if random.random() < 0.1:
            self.tractors[tractor_id, 1] -= 0.5
    
    def _calculate_breakdown_probability(self, tractor_id: int) -> float:
        """Calculate probability of breakdown based on tractor state"""
        hours_used = self.tractors[tractor_id, 0]
        condition = self.tractors[tractor_id, 1]
        days_since_maintenance = self.tractors[tractor_id, 2]
        
        # Base probability increases with usage and poor maintenance
        prob = 0.01  # Base 1% chance
        
        if hours_used > self.breakdown_threshold:
            prob += 0.05 * (hours_used - self.breakdown_threshold) / 20
        
        if condition < 50:
            prob += 0.10 * (50 - condition) / 50
            
        if days_since_maintenance > 10:
            prob += 0.03 * (days_since_maintenance - 10) / 20
        
        # Weather impact
        if self.weather[0] == 2:  # Stormy weather increases breakdown risk
            prob *= 2.0
        
        return min(prob, 0.5)  # Cap at 50%
    
    def _update_environment(self):
        """Update environment state (weather, season)"""
        self.current_day += 1
        
        # Update weather (shift forecast and add new day)
        self.weather[:-1] = self.weather[1:]
        self.weather[-1] = random.randint(0, 3)
        
        # Update crop demand
        self.crop_demand = self._calculate_crop_demand()
    
    def _calculate_reward(self, productivity: float, maintenance_cost: float, 
                         breakdown_penalty: float) -> float:
        """Calculate reward based on multiple factors"""
        
        # Base productivity reward
        productivity_reward = productivity * 10  # 10 points per productive hour
        
        # Demand fulfillment bonus
        demand_fulfillment = min(productivity / (self.crop_demand / 10), 1.0)
        demand_bonus = demand_fulfillment * 50
        
        # Maintenance cost penalty
        maintenance_penalty = maintenance_cost
        
        # Weather adaptation bonus
        weather_bonus = 0
        if self.weather[0] == 1 and productivity > 0:  # Working in rain
            weather_bonus = 20
        elif self.weather[0] == 2 and productivity == 0:  # Resting during storm
            weather_bonus = 10
        
        # Efficiency bonus for balanced operations
        active_tractors = sum([1 for i in range(3) if self._is_tractor_active(i)])
        if 1 <= active_tractors <= 2:  # Not overworking or underutilizing
            efficiency_bonus = 15
        else:
            efficiency_bonus = 0
        
        total_reward = (productivity_reward + demand_bonus + weather_bonus + 
                       efficiency_bonus - maintenance_penalty - breakdown_penalty)
        
        return total_reward
    
    def _is_tractor_active(self, tractor_id: int) -> bool:
        """Check if tractor was used today (simplified check)"""
        return self.tractors[tractor_id, 1] < 100  # Condition decreased = was used
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Day {self.current_day}/{self.season_length} ===")
            print(f"Weather: {self.weather_types[self.weather[0]]} (Forecast: {[self.weather_types[w] for w in self.weather[1:]]})")
            print(f"Crop Demand: {self.crop_demand:.1f}")
            print("\nTractor Status:")
            for i in range(3):
                hours, condition, days_maint = self.tractors[i]
                print(f"  Tractor {i+1}: {hours:.1f}h used, {condition:.1f}% condition, {days_maint:.0f} days since maintenance")
            print(f"\nTotal Breakdowns: {self.breakdown_count}")

# Test the environment
if __name__ == "__main__":
    # Create environment
    env = AgrikaTractorFleetEnv()
    
    # Test with random actions
    obs, info = env.reset()
    print("Initial State Shape:", obs.shape)
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    # Run a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: {action} -> {env.decode_action(action)}")
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")
        
        env.render()
        
        if terminated:
            break