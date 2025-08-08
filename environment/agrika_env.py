import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class EnhancedAgrikaTractorFleetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self):
        super().__init__()
        self.season_length = 7
        self.num_tractors = 3
        self.max_hours_per_day = 8
        self.breakdown_threshold = 100
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([150, 100, 30, 150, 100, 30, 150, 100, 30, 1, 1, 60, 2]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)
        self.weather_types = ['Rainy', 'Dry']
        self.demand_types = ['Low', 'Moderate', 'High']
        self.curriculum_params = {
            'max_weather_changes': 6,
            'breakdown_multiplier': 1.0
        }
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_rewards = []
        self.reset()
    
    def decode_action(self, action: int) -> tuple[int, int, int]:
        tractor3_action = action % 3
        tractor2_action = (action // 3) % 3
        tractor1_action = (action // 9) % 3
        return tractor1_action, tractor2_action, tractor3_action
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.episode_reward = 0.0
        if options:
            self.curriculum_params.update(options)
        self.tractors = np.array([
            [random.uniform(0, 5), random.uniform(85, 100), 0.0],
            [random.uniform(0, 5), random.uniform(85, 100), 0.0],
            [random.uniform(0, 5), random.uniform(85, 100), 0.0]
        ])
        self.weather = np.random.randint(0, 2, size=2)
        self.current_day = 0
        self.working_demand = self._calculate_working_demand()
        self.episode_stats = {
            'total_productivity': 0.0,
            'total_maintenance_cost': 0.0,
            'breakdown_count': 0,
            'efficiency_score': 0.0
        }
        info = {
            'day': self.current_day,
            'weather': self.weather_types[self.weather[0]],
            'demand': self.demand_types[int(self.working_demand)],
            'productivity': 0.0,
            'maintenance_cost': 0.0,
            'breakdowns': 0,
            'episode_stats': None,
            'episode_reward': 0.0
        }
        return self._get_observation(), info
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        tractor_actions = self.decode_action(action)
        daily_productivity = 0.0
        daily_maintenance_cost = 0.0
        daily_breakdown_penalty = 0.0
        for i, tractor_action in enumerate(tractor_actions):
            if tractor_action == 0:  # OPERATE
                productivity, breakdown = self._operate_tractor(i)
                daily_productivity += productivity
                if breakdown:
                    daily_breakdown_penalty += 200 * self.curriculum_params['breakdown_multiplier']
                    self.episode_stats['breakdown_count'] += 1
            elif tractor_action == 1:  # MAINTAIN
                cost = self._maintain_tractor(i)
                daily_maintenance_cost += cost
            elif tractor_action == 2:  # REST
                self._rest_tractor(i)
        self._update_environment()
        self.episode_stats['total_productivity'] += daily_productivity
        self.episode_stats['total_maintenance_cost'] += daily_maintenance_cost
        reward = self._calculate_reward(daily_productivity, daily_maintenance_cost, daily_breakdown_penalty)
        self.episode_reward += reward
        terminated = self.current_day >= self.season_length
        info = {
            'day': self.current_day,
            'productivity': daily_productivity,
            'maintenance_cost': daily_maintenance_cost,
            'breakdowns': self.episode_stats['breakdown_count'],
            'weather': self.weather_types[self.weather[0]],
            'demand': self.demand_types[int(self.working_demand)],
            'episode_stats': self.episode_stats.copy() if terminated else None,
            'episode_reward': self.episode_reward
        }
        if terminated:
            self.episode_rewards.append(self.episode_reward)
            print(f"Episode {self.episode_count - 1} Cumulative Reward: {self.episode_reward:.2f}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/ppo_3d_rewards.txt", "a") as f:
                f.write(f"Episode {self.episode_count - 1}: Cumulative Reward = {self.episode_reward:.2f}\n")
        return self._get_observation(), reward, terminated, False, info

    def _calculate_working_demand(self) -> float:
        if self.current_day <= 2:
            return 2
        elif 2 < self.current_day <= 5:
            return 1
        else:
            return 0
    
    def _get_observation(self) -> np.ndarray:
        obs = np.concatenate([
            self.tractors.flatten(),
            self.weather.astype(np.float32),
            [self.current_day, self.working_demand]
        ])
        return obs.astype(np.float32)
    
    def _operate_tractor(self, tractor_id: int) -> tuple[float, bool]:
        weather_multiplier = 0.3 if self.weather[0] == 0 else 1.2
        condition_factor = max(0.3, self.tractors[tractor_id, 1] / 100)
        base_hours = min(self.max_hours_per_day, 6 + condition_factor * 2)
        actual_productivity = base_hours * weather_multiplier * condition_factor
        self.tractors[tractor_id, 0] += base_hours
        degradation = random.uniform(1.5, 3.0) * (1 + 0.5 * (self.weather[0] == 0))
        self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - degradation)
        self.tractors[tractor_id, 2] += 1
        breakdown_prob = self._calculate_breakdown_probability(tractor_id)
        breakdown = random.random() < breakdown_prob
        if breakdown:
            self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - 25)
            actual_productivity *= 0.4
        return actual_productivity, breakdown
    
    def _maintain_tractor(self, tractor_id: int) -> float:
        condition = self.tractors[tractor_id, 1]
        base_cost = 50 + (100 - condition) * 1.0
        condition_improvement = min(30, 100 - condition)
        self.tractors[tractor_id, 1] = min(100, condition + condition_improvement)
        self.tractors[tractor_id, 2] = 0
        return base_cost
    
    def _rest_tractor(self, tractor_id: int):
        self.tractors[tractor_id, 2] += 1
        if random.random() < 0.05:
            self.tractors[tractor_id, 1] = max(0, self.tractors[tractor_id, 1] - 0.5)
    
    def _calculate_breakdown_probability(self, tractor_id: int) -> float:
        hours_used = self.tractors[tractor_id, 0]
        condition = self.tractors[tractor_id, 1]
        days_since_maintenance = self.tractors[tractor_id, 2]
        prob = 0.005
        if hours_used > self.breakdown_threshold:
            prob += 0.02 * (hours_used - self.breakdown_threshold) / 20
        if condition < 50:
            prob += 0.06 * (50 - condition) / 50
        if days_since_maintenance > 10:
            prob += 0.01 * (days_since_maintenance - 10) / 10
        prob *= 1.3 if self.weather[0] == 0 else 0.8
        prob *= self.curriculum_params['breakdown_multiplier']
        return min(prob, 0.3)
    
    def _update_environment(self):
        self.current_day += 1
        if random.random() < 0.7:
            self.weather[0] = self.weather[1]
            self.weather[1] = random.randint(0, 1)
        else:
            self.weather = np.random.randint(0, 2, size=2)
        self.working_demand = self._calculate_working_demand()
    
    def _calculate_reward(self, productivity: float, maintenance_cost: float, breakdown_penalty: float) -> float:
        productivity_reward = productivity * 15
        target_productivity = {0: 8, 1: 15, 2: 25}[int(self.working_demand)]
        if productivity >= target_productivity:
            demand_bonus = 50 * (productivity / target_productivity)
        else:
            demand_bonus = -20 * (1 - productivity / target_productivity)
        weather_bonus = 20 if self.weather[0] == 1 and productivity > 10 else 0
        maintenance_bonus = 0
        for i in range(self.num_tractors):
            if self.tractors[i, 1] > 70 and self.tractors[i, 2] < 5:
                maintenance_bonus += 10
        total_reward = (productivity_reward + demand_bonus + weather_bonus +
                        maintenance_bonus - maintenance_cost - breakdown_penalty)
        return total_reward
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n{'='*50}")
            print(f"Day {self.current_day}/{self.season_length}")
            print(f"Weather: {self.weather_types[self.weather[0]]}")
            print(f"Demand: {self.demand_types[int(self.working_demand)]}")
            print(f"{'='*50}")
            print("TRACTOR FLEET STATUS:")
            for i in range(3):
                hours, condition, days_maint = self.tractors[i]
                status = self._get_tractor_status(condition)
                print(f"  🚜 Tractor {i+1}: {hours:.1f}h | {condition:.1f}% ({status}) | {days_maint:.0f} days since maintenance")
    
    def _get_tractor_status(self, condition: float) -> str:
        if condition > 80:
            return "Excellent"
        elif condition > 60:
            return "Good"
        elif condition > 40:
            return "Fair"
        elif condition > 20:
            return "Poor"
        else:
            return "Critical"