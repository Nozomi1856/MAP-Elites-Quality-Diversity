"""
DQN Agent for creative architecture search
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple
from tqdm import tqdm

from architecture import ArchitectureState, ActionSpace
from gnn_models import DQNetwork
from novelty import RewardFunction
from evaluation import train_architecture


class CreativityDQN:
    """
    DQN agent for discovering creative architectures
    """
    def __init__(self, 
                 device='cuda',
                 gamma=0.99,
                 lr=0.0003,
                 buffer_size=5000,
                 batch_size=32,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 operation_strategy='diverse'):
        
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.operation_strategy = operation_strategy
        
        # Networks
        self.q_network = DQNetwork().to(device)
        self.target_network = DQNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Reward function
        self.reward_fn = RewardFunction(alpha=0.5, beta=0.35, gamma=0.15)
        
        # Stats
        self.episode = 0
        self.total_steps = 0
        
    def select_action(self, state: ArchitectureState, explore=True) -> Tuple[int, any]:
        """
        Select action using epsilon-greedy policy
        
        Returns:
            (action_type, target)
        """
        valid_actions = state.get_valid_actions()
        
        if not valid_actions:
            return (ActionSpace.STOP_BUILDING, None)
        
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            # Random action
            return random.choice(valid_actions)
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                data = state.to_pyg_data().to(self.device)
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                
                q_values = self.q_network(data.x, data.edge_index, batch)
                q_values = q_values[0]  # Unbatch
                
                # Find best valid action type
                best_action = None
                best_q = float('-inf')
                
                # Group actions by type
                actions_by_type = {}
                for action_type, target in valid_actions:
                    if action_type not in actions_by_type:
                        actions_by_type[action_type] = []
                    actions_by_type[action_type].append(target)
                
                # Choose best action type
                for action_type, targets in actions_by_type.items():
                    if q_values[action_type] > best_q:
                        best_q = q_values[action_type]
                        best_action = (action_type, random.choice(targets) if targets else None)
                
                return best_action if best_action else valid_actions[0]
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> float:
        """
        Sample from replay buffer and update Q-network
        
        Returns:
            loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Process batch
        losses = []
        
        for state, action, reward, next_state, done in batch:
            # Current Q-value
            data = state.to_pyg_data().to(self.device)
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            current_q = self.q_network(data.x, data.edge_index, batch_idx)[0]
            
            action_type, _ = action
            current_q_value = current_q[action_type]
            
            # Target Q-value
            if done:
                target_q_value = reward
            else:
                with torch.no_grad():
                    next_data = next_state.to_pyg_data().to(self.device)
                    next_batch = torch.zeros(next_data.x.size(0), dtype=torch.long, 
                                            device=self.device)
                    next_q = self.target_network(next_data.x, next_data.edge_index, 
                                                 next_batch)[0]
                    
                    # Get max Q for valid actions
                    valid_actions = next_state.get_valid_actions()
                    valid_types = set(a[0] for a in valid_actions)
                    max_next_q = max(next_q[t].item() for t in valid_types)
                    
                    target_q_value = reward + self.gamma * max_next_q
            
            # Loss
            loss = F.mse_loss(current_q_value, 
                            torch.tensor(target_q_value, device=self.device))
            losses.append(loss)
        
        # Optimize
        if losses:
            total_loss = torch.stack(losses).mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            return total_loss.item()
        
        return 0.0
    
    def generate_architecture(self, max_steps=20, eval_mode=False):
        """
        Generate one architecture through iterative building
        
        Returns:
            (final_arch, episode_reward, trajectory)
        """
        state = ArchitectureState.initialize_starter(self.operation_strategy)
        
        episode_reward = 0
        trajectory = []
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, explore=not eval_mode)
            action_type, target = action
            
            # Apply action
            next_state = state.apply_action(action_type, target, self.operation_strategy)
            
            # Check if done
            done = (action_type == ActionSpace.STOP_BUILDING) or (step == max_steps - 1)
            
            if done:
                # Evaluate final architecture
                performance = train_architecture(
                    next_state, 
                    epochs=3,
                    device=self.device,
                    subset_size=10000
                )
                
                reward, components = self.reward_fn.compute_reward(next_state, performance)
            else:
                # Small penalty for each step
                reward = -0.01
                components = {}
            
            # Store experience
            if not eval_mode:
                self.store_experience(state, action, reward, next_state, done)
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'components': components
            })
            
            episode_reward += reward
            state = next_state
            self.total_steps += 1
            
            if done:
                break
        
        return next_state, episode_reward, trajectory
    
    def train(self, num_episodes=1000, update_freq=10, eval_freq=50):
        """
        Main training loop
        
        Returns:
            List of best architectures
        """
        best_archs = []
        stats = {
            'episode_rewards': [],
            'losses': [],
            'epsilons': []
        }
        
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            self.episode = episode
            
            # Generate architecture
            arch, episode_reward, trajectory = self.generate_architecture()
            
            # Train Q-network
            if episode > 10:  # Start after some exploration
                for _ in range(5):
                    loss = self.train_step()
                    stats['losses'].append(loss)
            
            # Update target network
            if episode % update_freq == 0 and episode > 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Record stats
            stats['episode_rewards'].append(episode_reward)
            stats['epsilons'].append(self.epsilon)
            
            # Update progress bar
            if len(trajectory) > 0 and trajectory[-1]['components']:
                components = trajectory[-1]['components']
                pbar.set_postfix({
                    'reward': f"{episode_reward:.3f}",
                    'perf': f"{components.get('performance', 0):.3f}",
                    'topo': f"{components.get('topological_novelty', 0):.3f}",
                    'scale': f"{components.get('scale_novelty', 0):.3f}",
                    'eps': f"{self.epsilon:.3f}"
                })
            
            # Save good architectures
            if episode_reward > 0.3:  # Threshold for "interesting"
                best_archs.append({
                    'architecture': arch,
                    'reward': episode_reward,
                    'trajectory': trajectory,
                    'episode': episode
                })
                
                # Keep top 100
                best_archs.sort(key=lambda x: x['reward'], reverse=True)
                best_archs = best_archs[:100]
        
        return best_archs, stats
    
    def save(self, path):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode = checkpoint['episode']
        self.epsilon = checkpoint['epsilon']
