import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import matplotlib.pyplot as plt
import random   
import sys
import importlib
import gymnasium as gym
from stable_baselines3 import PPO

print("Visualization started — PPO MODEL VERSION")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environment')))
from agrika_env import EnhancedAgrikaTractorFleetEnv

class Agrika3DVisualizer:
    """
    Agrika Tractor Fleet Management
    Using PyBullet for intuitive rendering tailored for farmers
    """
    
    def __init__(self, env):
        self.env = env
        self.physics_client = None
        self.tractor_ids = []
        self.tractor_wheel_ids = []
        self.tractor_tag_ids = []  # Store IDs for tractor tags
        self.rest_park_id = None
        self.maintenance_area_id = None
        self.ui_elements = {}
        self.weather_particles = []
        self.maintenance_signs = []
        self.sun_particles = []
        
        # Enhanced tractor movement tracking for straight-line field work
        self.tractor_positions = [(0, -8), (0, 0), (0, 8)]  # Start at different rows
        self.tractor_orientations = [0, 0, 0]  # Yaw angles in radians
        
        # Progress tracking system
        self.field_grid = {}  # Track worked areas
        self.grid_size = 0.5  # Size of each grid cell
        self.worked_area_ids = []  # Visual indicators for worked areas
        self.progress_markers = []  # Small markers showing tractor paths
        
        # Row-based progress tracking
        self.row_progress = {}  # Track progress on each row
        self.row_visual_ids = {}  # Visual elements for each row
        self.total_rows = list(range(-8, 9, 2))  # All available rows
        
        # Initialize progress tracking
        for row in self.total_rows:
            self.row_progress[row] = {
                'worked_sections': set(),  # Set of x-coordinates that have been worked
                'current_tractors': set(),  # Which tractors are currently on this row
                'completion_percentage': 0.0
            }
        
        # Straight-line field work patterns
        self.field_work_patterns = [
            {
                'type': 'straight_line',
                'row': -8,          # Y coordinate of the row
                'direction': 1,     # 1 for right, -1 for left
                'x_position': -9,   # Start at the left edge
                'row_spacing': 2.5,  # Distance between rows
                'last_worked_x': None  # Track last position for progress
            },
            {
                'type': 'straight_line', 
                'row': 0,
                'direction': 1,
                'x_position': -9,
                'row_spacing': 2.5,
                'last_worked_x': None
            },
            {
                'type': 'straight_line',
                'row': 8, 
                'direction': 1,
                'x_position': -9,
                'row_spacing': 2.5,
                'last_worked_x': None
            }
        ]
        
        # Field boundaries for tractor movement (inside the fenced area)
        self.field_boundaries = {
            'min_x': -9, 'max_x': 9,
            'min_y': -9, 'max_y': 9
        }
        
        # Speed settings based on condition (increased for visibility)
        self.speed_settings = {
            'excellent': {'base_speed': 2.0, 'description': 'Fast & Smooth'},  # Doubled speed
            'good': {'base_speed': 1.2, 'description': 'Moderate Speed'},      # Doubled speed
            'poor': {'base_speed': 0.4, 'description': 'Barely Crawling'}      # Doubled speed
        }
        
        # Colors for different states
        self.colors = {
            'healthy': [0, 0.8, 0, 1],        # Green
            'warning': [1, 1, 0, 1],          # Yellow
            'critical': [1, 0, 0, 1],         # Red
            'maintenance': [0, 0, 1, 1],      # Blue
            'resting': [0.5, 0.5, 0.5, 1],   # Gray
            'field': [0.4, 0.6, 0.2, 1],     # Green field
            'worked_field': [0.6, 0.4, 0.2, 1],  # Brown worked field
            'unworked_row': [0.2, 0.5, 0.1, 0.8],  # Dark green unworked
            'partially_worked_row': [0.7, 0.5, 0.2, 0.8],  # Brown partially worked
            'completed_row': [0.4, 0.2, 0.1, 0.8],  # Dark brown completed
            'current_work': [1, 0.8, 0, 0.9],  # Bright yellow for current work
            'rest_park': [0.5, 0.5, 0.5, 1], # Gray rest park
            'maintenance_area': [0.2, 0.2, 0.8, 1], # Blue maintenance area
            'rain': [0.6, 0.8, 1, 0.8],      # Light blue rain
            'sun': [1, 1, 0, 0.9],            # Yellow sun
            'maintenance_sign': [1, 0.5, 0, 1], # Orange maintenance sign
            'tractor_tag': [0, 0, 0]          # Black for tractor tags (RGB, no alpha)
        }
    
    def initialize_visualization(self):
        """Initialize PyBullet 3D environment without video recording"""
        print("🎬 Initializing PyBullet 3D visualization...")
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        self._create_farm_environment()
        self._create_tractors()
        self._create_maintenance_signs()
        self._create_field_rows()
        self._setup_camera()
        self._create_ui_elements()
        self._create_tractor_tags()
        print("✅ 3D visualization initialized successfully!")
        
    def _create_farm_environment(self):
        """Create a limited 3D farm environment with defined boundaries"""
        field_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.05])
        field_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.05],
                                          rgbaColor=self.colors['field'])
        self.field_id = p.createMultiBody(0, field_shape, field_visual, [0, 0, 0.05])
        
        soil_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[15, 15, 0.02])
        soil_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[15, 15, 0.02],
                                         rgbaColor=[0.4, 0.2, 0.1, 1])
        self.soil_id = p.createMultiBody(0, soil_shape, soil_visual, [0, 0, 0.02])
        
        self._create_field_boundaries()
        
        rest_park_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 0.1])
        rest_park_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.1],
                                              rgbaColor=self.colors['rest_park'])
        self.rest_park_id = p.createMultiBody(0, rest_park_shape, rest_park_visual, [10, 10, 0.1])
        
        maintenance_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3, 3, 0.1])
        maintenance_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.1],
                                                rgbaColor=self.colors['maintenance_area'])
        self.maintenance_area_id = p.createMultiBody(0, maintenance_shape, maintenance_visual, [-10, -10, 0.1])
    
    def _create_field_boundaries(self):
        """Create wooden fence boundaries around the operating field"""
        fence_height = 1.0
        fence_thickness = 0.2
        fence_color = [0.6, 0.3, 0.1, 1]
        
        fence_positions = [
            [0, 10, fence_height/2], [0, -10, fence_height/2],
            [10, 0, fence_height/2], [-10, 0, fence_height/2]
        ]
        fence_dimensions = [
            [10, fence_thickness/2, fence_height/2],
            [10, fence_thickness/2, fence_height/2],
            [fence_thickness/2, 10, fence_height/2],
            [fence_thickness/2, 10, fence_height/2]
        ]
        self.fence_ids = []
        for pos, dims in zip(fence_positions, fence_dimensions):
            fence_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            fence_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=dims,
                                             rgbaColor=fence_color)
            self.fence_ids.append(p.createMultiBody(0, fence_shape, fence_visual, pos))
        
        corner_positions = [
            [10, 10, fence_height/2], [10, -10, fence_height/2],
            [-10, 10, fence_height/2], [-10, -10, fence_height/2]
        ]
        for pos in corner_positions:
            post_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=fence_height)
            post_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=fence_height,
                                            rgbaColor=fence_color)
            self.fence_ids.append(p.createMultiBody(0, post_shape, post_visual, pos))
    
    def _create_field_rows(self):
        """Create visual indicators for field rows with progress tracking"""
        self.field_row_ids = []
        
        for y in range(-8, 9, 2):
            if abs(y) <= 8:
                # Create row segments for progress tracking
                segments_per_row = 18  # Divide each row into segments
                segment_width = 18 / segments_per_row  # Total width is 18 (-9 to 9)
                
                row_segments = []
                for i in range(segments_per_row):
                    x_start = -9 + (i * segment_width)
                    x_center = x_start + (segment_width / 2)
                    
                    row_shape = p.createCollisionShape(p.GEOM_BOX, 
                                                     halfExtents=[segment_width/2, 0.15, 0.02])
                    row_visual = p.createVisualShape(p.GEOM_BOX, 
                                                   halfExtents=[segment_width/2, 0.15, 0.02],
                                                   rgbaColor=self.colors['unworked_row'])
                    row_id = p.createMultiBody(0, row_shape, row_visual, [x_center, y, 0.12])
                    row_segments.append(row_id)
                
                self.row_visual_ids[y] = row_segments
                self.field_row_ids.extend(row_segments)
    
    def _create_tractors(self):
        """Create 3D tractor models with proper orientation"""
        self.tractor_wheel_ids = []
        self.tractor_tag_ids = []
        for i in range(3):
            chassis_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.2, 0.6, 0.4])
            chassis_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.2, 0.6, 0.4],
                                                rgbaColor=self.colors['healthy'])
            x_pos, y_pos = self.tractor_positions[i]
            tractor_id = p.createMultiBody(1000, chassis_shape, chassis_visual,
                                          [x_pos, y_pos, 0.4])
            
            wheel_ids = self._add_wheels_to_tractor(tractor_id, x_pos, y_pos)
            self.tractor_wheel_ids.append(wheel_ids)
            self.tractor_ids.append(tractor_id)
    
    def _add_wheels_to_tractor(self, tractor_id, x_pos, y_pos):
        """Add wheels to tractor for realistic appearance"""
        wheel_positions = [
            [1.0, 0.7, 0], [1.0, -0.7, 0], [-1.0, 0.7, 0], [-1.0, -0.7, 0]
        ]
        wheel_ids = []
        for dx, dy, dz in wheel_positions:
            wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.4, height=0.3)
            wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.4, length=0.3,
                                              rgbaColor=[0.1, 0.1, 0.1, 1])
            wheel_id = p.createMultiBody(10, wheel_shape, wheel_visual,
                                        [x_pos + dx, y_pos + dy, 0.4])
            p.createConstraint(tractor_id, -1, wheel_id, -1, p.JOINT_FIXED,
                              [0, 0, 0], [dx, dy, dz], [0, 0, 0])
            wheel_ids.append(wheel_id)
        return wheel_ids
    
    def _create_maintenance_signs(self):
        """Create maintenance signs at the maintenance area"""
        sign_positions = [
            [-10, -7, 2], [-7, -10, 2], [-13, -10, 2], [-10, -13, 2]
        ]
        for pos in sign_positions:
            post_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=2)
            post_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length=2,
                                            rgbaColor=[0.4, 0.2, 0, 1])
            post_id = p.createMultiBody(0, post_shape, post_visual, [pos[0], pos[1], 1])
            sign_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 0.3])
            sign_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 0.3],
                                            rgbaColor=self.colors['maintenance_sign'])
            sign_id = p.createMultiBody(0, sign_shape, sign_visual, pos)
            self.maintenance_signs.extend([post_id, sign_id])
    
    def _create_tractor_tags(self):
        """Create text tags (T1, T2, T3) above each tractor"""
        self.tractor_tag_ids = []
        for i in range(3):
            x_pos, y_pos = self.tractor_positions[i]
            tag_id = p.addUserDebugText(
                text=f"T{i+1}",
                textPosition=[x_pos, y_pos, 1.5],  # 1.5 units above tractor
                textColorRGB=self.colors['tractor_tag'],
                textSize=1.2,
                lifeTime=0  # Persistent text
            )
            self.tractor_tag_ids.append(tag_id)
    
    def _setup_camera(self):
        """Setup camera for optimal viewing"""
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0]
        )
    
    def _create_ui_elements(self):
        """Create UI elements for displaying information"""
        self.ui_elements = {
            'day': None,
            'weather': None,
            'demand': None,
            'tractors': [None, None, None]
        }
    
    def _update_progress_tracking(self, tractor_idx, x_pos, y_pos, is_operating):
        """Update progress tracking for a tractor"""
        if not is_operating:
            return
        
        pattern = self.field_work_patterns[tractor_idx]
        current_row = pattern['row']
        
        # Find the closest actual row
        closest_row = min(self.total_rows, key=lambda r: abs(r - current_row))
        
        if closest_row in self.row_progress:
            # Mark this x-position as worked (rounded to nearest grid)
            worked_x = round(x_pos / self.grid_size) * self.grid_size
            self.row_progress[closest_row]['worked_sections'].add(worked_x)
            self.row_progress[closest_row]['current_tractors'].add(tractor_idx)
            
            # Calculate completion percentage
            total_width = self.field_boundaries['max_x'] - self.field_boundaries['min_x']
            worked_width = len(self.row_progress[closest_row]['worked_sections']) * self.grid_size
            self.row_progress[closest_row]['completion_percentage'] = min(100, 
                (worked_width / total_width) * 100)
            
            # Update visual representation
            self._update_row_visuals(closest_row)
            
            # Create progress markers (small visual indicators of tractor path)
            if pattern['last_worked_x'] is not None:
                self._create_progress_marker(pattern['last_worked_x'], current_row, x_pos, current_row)
            
            pattern['last_worked_x'] = x_pos
    
    def _update_row_visuals(self, row):
        """Update the visual appearance of a row based on progress"""
        if row not in self.row_visual_ids:
            return
        
        progress_data = self.row_progress[row]
        completion_percentage = progress_data['completion_percentage']
        worked_sections = progress_data['worked_sections']
        current_tractors = progress_data['current_tractors']
        
        # Determine row color based on completion
        if completion_percentage >= 90:
            row_color = self.colors['completed_row']
        elif completion_percentage > 10:
            row_color = self.colors['partially_worked_row']
        else:
            row_color = self.colors['unworked_row']
        
        # If tractors are currently working on this row, highlight it
        if current_tractors:
            row_color = self.colors['current_work']
        
        # Update each segment of the row
        segments = self.row_visual_ids[row]
        segments_per_row = len(segments)
        segment_width = 18 / segments_per_row
        
        for i, segment_id in enumerate(segments):
            x_start = -9 + (i * segment_width)
            x_end = x_start + segment_width
            
            # Check if this segment has been worked
            segment_worked = any(x_start <= worked_x <= x_end for worked_x in worked_sections)
            
            if segment_worked:
                color = self.colors['completed_row']
            elif current_tractors:
                # Check if any tractor is currently in this segment
                for tractor_idx in current_tractors:
                    tractor_x = self.tractor_positions[tractor_idx][0]
                    if x_start <= tractor_x <= x_end:
                        color = self.colors['current_work']
                        break
                else:
                    color = row_color
            else:
                color = row_color
            
            p.changeVisualShape(segment_id, -1, rgbaColor=color)
    
    def _create_progress_marker(self, x1, y1, x2, y2):
        """Create a small visual marker showing tractor progress"""
        if len(self.progress_markers) > 100:  # Limit number of markers
            old_marker = self.progress_markers.pop(0)
            try:
                p.removeBody(old_marker)
            except:
                pass
        
        # Create a small marker showing the work path
        marker_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                          rgbaColor=[0.8, 0.6, 0.2, 0.8])
        marker_id = p.createMultiBody(0, marker_shape, marker_visual, 
                                    [(x1 + x2) / 2, (y1 + y2) / 2, 0.2])
        self.progress_markers.append(marker_id)
    
    def _create_weather_particles(self, weather):
        """Create weather particles (rain or sun) within field boundaries"""
        for particle_id in self.weather_particles + self.sun_particles:
            try:
                p.removeBody(particle_id)
            except:
                pass
        self.weather_particles = []
        self.sun_particles = []
        
        if weather == 'Rainy':
            for _ in range(30):
                x = np.random.uniform(-12, 12)
                y = np.random.uniform(-12, 12)
                z = np.random.uniform(8, 15)
                drop_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
                drop_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                                rgbaColor=self.colors['rain'])
                self.weather_particles.append(p.createMultiBody(0, drop_shape, drop_visual, [x, y, z]))
        elif weather == 'Dry':
            for _ in range(15):
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                z = np.random.uniform(10, 20)
                sun_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
                sun_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                               rgbaColor=self.colors['sun'])
                self.sun_particles.append(p.createMultiBody(0, sun_shape, sun_visual, [x, y, z]))
    
    def _update_weather_particles(self, weather):
        """Update weather particle positions within field area"""
        if weather == 'Rainy':
            for particle_id in self.weather_particles:
                try:
                    pos, orn = p.getBasePositionAndOrientation(particle_id)
                    new_z = pos[2] - 0.2
                    if new_z < 0:
                        new_z = 15
                        new_x = np.random.uniform(-12, 12)
                        new_y = np.random.uniform(-12, 12)
                        p.resetBasePositionAndOrientation(particle_id, [new_x, new_y, new_z], orn)
                    else:
                        p.resetBasePositionAndOrientation(particle_id, [pos[0], pos[1], new_z], orn)
                except:
                    pass
        elif weather == 'Dry':
            for i, particle_id in enumerate(self.sun_particles):
                try:
                    pos, orn = p.getBasePositionAndOrientation(particle_id)
                    time_offset = time.time() + i * 0.5
                    new_x = pos[0] + 0.05 * math.sin(time_offset)
                    new_y = pos[1] + 0.05 * math.cos(time_offset)
                    new_z = pos[2] + 0.02 * math.sin(time_offset * 2)
                    new_x = max(-10, min(10, new_x))
                    new_y = max(-10, min(10, new_y))
                    p.resetBasePositionAndOrientation(particle_id, [new_x, new_y, new_z], orn)
                except:
                    pass
    
    def _get_condition_category(self, condition):
        """Categorize tractor condition for speed calculation"""
        if condition > 70:
            return 'excellent'
        elif condition > 40:
            return 'good'
        else:
            return 'poor'
    
    def _calculate_straight_line_movement(self, tractor_idx, speed, weather):
        """Calculate straight-line field work movement (like plowing rows)"""
        pattern = self.field_work_patterns[tractor_idx]
        current_x, current_y = self.tractor_positions[tractor_idx]
        
        # Adjust speed based on weather
        weather_modifier = 0.6 if weather == 'Rainy' else 1.0
        effective_speed = speed * weather_modifier
        
        # Move along the current row
        new_x = pattern['x_position'] + (pattern['direction'] * effective_speed)
        new_y = pattern['row']
        
        # Check if reached the end of the row
        if new_x >= self.field_boundaries['max_x']:
            pattern['direction'] = -1  # Go left
            pattern['row'] += pattern['row_spacing']
            if pattern['row'] > self.field_boundaries['max_y']:
                pattern['row'] = self.field_boundaries['min_y'] + 1
            new_x = self.field_boundaries['min_x'] + 0.1
            new_y = pattern['row']
        elif new_x <= self.field_boundaries['min_x']:
            pattern['direction'] = 1  # Go right
            pattern['row'] += pattern['row_spacing']
            if pattern['row'] > self.field_boundaries['max_y']:
                pattern['row'] = self.field_boundaries['min_y'] + 1
            new_x = self.field_boundaries['max_x'] - 0.1
            new_y = pattern['row']
        
        pattern['x_position'] = new_x
        orientation = 0 if pattern['direction'] > 0 else math.pi
        
        return new_x, new_y, orientation
    
    def _get_tractor_status(self, condition: float) -> str:
        """Get tractor status based on condition"""
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
    
    def _get_season_phase(self) -> str:
        """Get current season phase"""
        if self.env.current_day <= 2:
            return "Planting"
        elif self.env.current_day <= 5:
            return "Growing"
        else:
            return "Harvest"
    
    def update_visualization(self, obs, action, reward, info):
        """Update the 3D visualization based on current state"""
        weather = info.get('weather', 'Dry')
        
        if not self.weather_particles and not self.sun_particles:
            self._create_weather_particles(weather)
        
        self._update_weather_effects(weather)
        self._update_weather_particles(weather)
        self._update_tractor_states(obs, action, info, weather)
        self._update_tractor_tags()
        self._update_text_display(obs, reward, info)
        self._render_terminal(obs, action, reward, info)
        
        # Clear current tractors from all rows
        for row_data in self.row_progress.values():
            row_data['current_tractors'].clear()
        
        p.stepSimulation()
    
    def _update_tractor_states(self, obs, action, info, weather):
        """Update tractor visual states and positions with straight-line field work"""
        tractor_actions = self.env.decode_action(action)
        
        for i in range(3):
            tractor_start_idx = i * 3
            condition = obs[tractor_start_idx + 1]
            condition_category = self._get_condition_category(condition)
            
            is_operating = tractor_actions[i] == 0
            
            if tractor_actions[i] == 1:  # MAINTAIN
                color = self.colors['maintenance']
            elif tractor_actions[i] == 2:  # REST
                color = self.colors['resting']
            elif tractor_actions[i] == 0:  # OPERATE
                if condition > 70:
                    color = self.colors['healthy']
                elif condition > 40:
                    color = self.colors['warning']
                else:
                    color = self.colors['critical']
            
            p.changeVisualShape(self.tractor_ids[i], -1, rgbaColor=color)
            
            current_pos, current_orn = p.getBasePositionAndOrientation(self.tractor_ids[i])
            
            if tractor_actions[i] == 0:  # OPERATE - Straight-line field work
                speed_info = self.speed_settings[condition_category]
                speed = speed_info['base_speed']
                
                new_x, new_y, orientation = self._calculate_straight_line_movement(i, speed, weather)
                self.tractor_positions[i] = (new_x, new_y)
                self.tractor_orientations[i] = orientation
                quaternion = p.getQuaternionFromEuler([0, 0, orientation])
                
                # Update progress tracking
                self._update_progress_tracking(i, new_x, new_y, True)
                
                p.resetBasePositionAndOrientation(self.tractor_ids[i],
                                                 [new_x, new_y, current_pos[2]],
                                                 quaternion)
                
                for j, wheel_id in enumerate(self.tractor_wheel_ids[i]):
                    wheel_positions = [
                        [1.0, 0.7, 0], [1.0, -0.7, 0], [-1.0, 0.7, 0], [-1.0, -0.7, 0]
                    ]
                    dx, dy, dz = wheel_positions[j]
                    rotated_dx = dx * math.cos(orientation) - dy * math.sin(orientation)
                    rotated_dy = dx * math.sin(orientation) + dy * math.cos(orientation)
                    wheel_x = new_x + rotated_dx
                    wheel_y = new_y + rotated_dy
                    p.resetBasePositionAndOrientation(wheel_id,
                                                     [wheel_x, wheel_y, current_pos[2]],
                                                     quaternion)
            
            elif tractor_actions[i] == 1:  # MAINTAIN
                target_pos = [-10, -10, current_pos[2]]
                self.tractor_positions[i] = (-10, -10)
                p.resetBasePositionAndOrientation(self.tractor_ids[i], target_pos, current_orn)
                for j, wheel_id in enumerate(self.tractor_wheel_ids[i]):
                    wheel_positions = [
                        [1.0, 0.7, 0], [1.0, -0.7, 0], [-1.0, 0.7, 0], [-1.0, -0.7, 0]
                    ]
                    dx, dy, dz = wheel_positions[j]
                    p.resetBasePositionAndOrientation(wheel_id,
                                                     [-10 + dx, -10 + dy, current_pos[2]],
                                                     current_orn)
            
            elif tractor_actions[i] == 2:  # REST
                target_pos = [10, 10, current_pos[2]]
                self.tractor_positions[i] = (10, 10)
                p.resetBasePositionAndOrientation(self.tractor_ids[i], target_pos, current_orn)
                for j, wheel_id in enumerate(self.tractor_wheel_ids[i]):
                    wheel_positions = [
                        [1.0, 0.7, 0], [1.0, -0.7, 0], [-1.0, 0.7, 0], [-1.0, -0.7, 0]
                    ]
                    dx, dy, dz = wheel_positions[j]
                    p.resetBasePositionAndOrientation(wheel_id,
                                                     [10 + dx, 10 + dy, current_pos[2]],
                                                     current_orn)
    
    def _update_tractor_tags(self):
        """Update positions of tractor tags to follow tractors"""
        for i in range(3):
            x_pos, y_pos = self.tractor_positions[i]
            tag_id = self.tractor_tag_ids[i]
            p.addUserDebugText(
                text=f"T{i+1}",
                textPosition=[x_pos, y_pos, 1.5],  # 1.5 units above tractor
                textColorRGB=self.colors['tractor_tag'],
                textSize=1.2,
                lifeTime=0,
                replaceItemUniqueId=tag_id
            )
            self.tractor_tag_ids[i] = tag_id
    
    def _update_weather_effects(self, weather):
        """Update lighting and environment based on weather"""
        if weather == 'Dry':
            p.changeVisualShape(self.field_id, -1, rgbaColor=self.colors['field'])
        else:  # Rainy
            p.changeVisualShape(self.field_id, -1, rgbaColor=[0.3, 0.4, 0.2, 1])
    
    def _get_total_field_progress(self):
        """Calculate overall field completion percentage"""
        total_progress = 0
        for row_data in self.row_progress.values():
            total_progress += row_data['completion_percentage']
        return total_progress / len(self.row_progress) if self.row_progress else 0
    
    def _update_text_display(self, obs, reward, info):
        """Update text overlay with essential info including speed details and progress"""
        def get_health_and_speed(condition):
            category = self._get_condition_category(condition)
            speed_info = self.speed_settings[category]
            if condition > 70:
                return f"Excellent ({condition:.1f}%) - {speed_info['description']}"
            elif condition > 40:
                return f"Good ({condition:.1f}%) - {speed_info['description']}"
            else:
                return f"Poor ({condition:.1f}%) - {speed_info['description']}"
        
        # Calculate progress statistics
        total_progress = self._get_total_field_progress()
        rows_completed = sum(1 for row_data in self.row_progress.values() 
                           if row_data['completion_percentage'] >= 90)
        rows_in_progress = sum(1 for row_data in self.row_progress.values() 
                             if 10 < row_data['completion_percentage'] < 90)
        
        text_lines = [
            f"Day: {info.get('day', 0)}/7",
            f"Weather: {info.get('weather', 'Dry')}",
            f"Demand: {info.get('demand', 'Moderate')}",
            f"Reward: {reward:.2f}",
            "",
            "FIELD PROGRESS:",
            f"Overall Completion: {total_progress:.1f}%",
            f"Rows Completed: {rows_completed}/{len(self.total_rows)}",
            f"Rows In Progress: {rows_in_progress}",
            "",
            "Tractor Status & Speed:"
        ]
        
        for i in range(3):
            tractor_start_idx = i * 3
            condition = obs[tractor_start_idx + 1]
            current_row = self.field_work_patterns[i]['row']
            text_lines.append(f"T{i+1}: {get_health_and_speed(condition)} (Row: {current_row})")
        
        text_lines.extend([
            "",
            "ROW PROGRESS DETAILS:",
        ])
        
        # Show progress for each row
        for row in sorted(self.total_rows):
            if row in self.row_progress:
                progress = self.row_progress[row]['completion_percentage']
                active_tractors = len(self.row_progress[row]['current_tractors'])
                status = "✓" if progress >= 90 else "→" if progress > 10 else "○"
                active_indicator = f" [{active_tractors} working]" if active_tractors > 0 else ""
                text_lines.append(f"  Row {row:2d}: {status} {progress:5.1f}%{active_indicator}")
        
        text_lines.extend([
            "",
            "VISUAL LEGEND:",
            "Row Colors:",
            "• Dark Green: Unworked rows",
            "• Brown: Partially worked rows", 
            "• Dark Brown: Completed rows (≥90%)",
            "• Bright Yellow: Currently being worked",
            "• Small Orange Dots: Tractor path markers",
            "",
            "Areas:",
            "• Green: Operating Field",
            "• Gray: Rest Park", 
            "• Blue: Maintenance Zone",
            "• Brown Fence: Field Boundaries",
            "• Black Text (T1, T2, T3): Tractor Labels"
        ])
        
        # Clear old text elements
        for key in list(self.ui_elements.keys()):
            if key.startswith('line_'):
                del self.ui_elements[key]
        
        # Create new text elements
        for i, line in enumerate(text_lines):
            text_id = p.addUserDebugText(
                text=line,
                textPosition=[-22, -15 + i * 0.5, 8],
                textColorRGB=[0, 0, 0],
                textSize=1.0,
                replaceItemUniqueId=self.ui_elements.get(f'line_{i}', -1)
            )
            self.ui_elements[f'line_{i}'] = text_id
    
    def _render_terminal(self, obs, action, reward, info):
        """Render environment status to terminal, mirroring EnhancedAgrikaTractorFleetEnv.render"""
        print(f"\n{'='*50}")
        print(f"Day {info['day']}/7 - Season Progress: {info['day']/self.env.season_length*100:.1f}%")
        print(f"Weather: {info['weather']} | Next Day: {self.env.weather_types[self.env.weather[1]]}")
        print(f"Working Demand: {info['demand']} | Season Phase: {self._get_season_phase()}")
        print(f"Action: {action} -> {self.env.decode_action(action)}")
        print(f"Reward: {reward:.2f}")
        print(f"{'='*50}")
        
        print("TRACTOR FLEET STATUS:")
        for i in range(3):
            hours, condition, days_maint = self.env.tractors[i]
            status = self._get_tractor_status(condition)
            print(f"  🚜 Tractor {i+1}: {hours:.1f}h | {condition:.1f}% condition ({status}) | {days_maint:.0f} days since maintenance")
        
        print(f"\nEPISODE STATISTICS:")
        print(f"  Total Productivity: {info['productivity']:.1f}")
        print(f"  Maintenance Costs: ${info['maintenance_cost']:.2f}")
        print(f"  Breakdowns: {info['breakdowns']}")
        
        if info['day'] >= self.env.season_length:
            print(f"  Efficiency Score: {self.env.episode_stats['efficiency_score']:.2f}")
            print("🏁 SEASON COMPLETED!")
    
    def render_frame(self, obs, action, reward, info):
        """Render a single frame of the visualization"""
        self.update_visualization(obs, action, reward, info)
        time.sleep(0.01)  # Reduced sleep time for faster, smoother movement
    
    def close(self):
        """Clean up visualization"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

def test_visualization():
    """
    Test function for the 3D visualization with TRAINED PPO MODEL
    """
    print("🔧 RUNNING PPO MODEL VERSION")
    print("=" * 60)
    
    env = EnhancedAgrikaTractorFleetEnv()
    visualizer = Agrika3DVisualizer(env)
    
    # Load the trained PPO model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pg', 'ppo', 'final_ppo_model')
    try:
        model = PPO.load(model_path, env=env, device='cpu')
        print(f"✅ Loaded PPO model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load PPO model: {e}")
        raise
    
    try:
        visualizer.initialize_visualization()
        
        if not p.isConnected():
            raise RuntimeError("Failed to initialize PyBullet GUI")
        
        print("🎮 Running with TRAINED PPO MODEL")
        
        episode_rewards = []
        episode_count = 0
        
        for episode in range(3):  # Run exactly 3 episodes
            episode_count += 1
            cumulative_reward = 0.0
            print(f"\n🎯 Starting Episode {episode_count}")
            
            obs, info = env.reset()
            visualizer.update_visualization(obs, 0, 0.0, info)  # Initial render
            
            while True:
                # Use PPO model predictions instead of random actions
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                
                # Render the current frame
                visualizer.render_frame(obs, action, reward, info)
                
                if terminated or truncated:
                    episode_rewards.append(cumulative_reward)
                    print(f"Episode {episode_count} Cumulative Reward: {cumulative_reward:.2f}")
                    
                    # Log results
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/ppo_3d_rewards.txt", "a") as f:
                        f.write(f"Episode {episode_count}: Cumulative Reward = {cumulative_reward:.2f}\n")
                    break
        
        # Update UI to show episode rewards
        print("\n📊 Episode Summary:")
        for i, reward in enumerate(episode_rewards):
            print(f"  Episode {i+1}: {reward:.2f}")
            text_id = p.addUserDebugText(
                text=f"Episode {i+1}: Reward = {reward:.2f}",
                textPosition=[-22, -15 + (len(visualizer.ui_elements) + i) * 0.5, 8],
                textColorRGB=[0, 0, 0],
                textSize=1.0,
                lifeTime=0
            )
            visualizer.ui_elements[f'episode_reward_{i}'] = text_id
        
        # Create a reward plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=np.mean(episode_rewards), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(episode_rewards):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('PPO Policy Performance (3D Visualization Test)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        os.makedirs('results/ppo', exist_ok=True)
        plt.savefig('results/ppo/ppo_reward_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Reward plot saved to 'results/ppo/ppo_reward_plot.png'")
        
        print("\n🎉 Visualization test completed successfully!")
        print("📋 Summary:")
        print(f"   - Total Episodes: {len(episode_rewards)}")
        print(f"   - Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"   - Best Episode: {max(episode_rewards):.2f}")
        print(f"   - Worst Episode: {min(episode_rewards):.2f}")
        print("\n💡 Close the PyBullet window when you're done exploring the visualization.")
        
        # Keep the visualization running until user closes it
        while p.isConnected():
            time.sleep(0.1)
    
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            visualizer.close()
            env.close()
            print("🧹 Resources cleaned up successfully.")
        except Exception as e:
            print(f"🧹 Cleanup error: {e}")

if __name__ == "__main__":
    test_visualization()