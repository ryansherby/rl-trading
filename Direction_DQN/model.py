from .utils import *

import os
import random
import logging
import argparse
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    """
    A simple Feed-Forward Neural Network for DQN.
    Architecture similar to:
      Input -> [128] -> [256] -> [256] -> [128] -> Output (action scores)
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, action_size)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
         
        self.relu = nn.ReLU()

    def forward(self, x):
        y = nn.Sequential(
            self.fc1(x),
            self.bn1(self.fc1(x)),
            self.relu(),
            self.fc2(x),
            self.bn2(self.fc2(x)),
            self.relu(),
            self.fc3(x),
            self.bn3(self.fc3(x)),
            self.relu(),
            self.fc4(x),
            self.bn4(self.fc4(x)),
            self.relu(),
            self.fc5(x)
        )(x)
        
        # Argmax over actions to get the best action
        return y
    


# --- Agent Definition ---

class Agent:
    """
    Stock Trading Bot Agent using Deep Q-Learning with experience replay.
    Supports "dqn", "t-dqn" (fixed target) and "double-dqn" strategies.
    """
    def __init__(self, state_size, action_size=2, strategy="double-dqn", update_target_every=50,update_after_actions=5,memory_size=1000,
                 pretrained=False, model_name="model_debug", device=None, hyperparams=None):
        
        
        # Device configuration (CPU/GPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent configuration
        self.strategy = strategy
        self.state_size = state_size  # length of the state vector (window)
        self.action_size = action_size        # -1: Fall; 1: Rise      
        self.model_name = model_name
        self.memory_size = memory_size
        self.memory = []
        self.priorities = torch.zeros(self.memory_size, dtype=torch.float32).to(self.device)
        self.update_after_actions = update_after_actions
        self.update_target_every = update_target_every
        self.n_iter = 0
        self.action_iter = 0

        # Hyperparameters for Q-Learning
        
        if hyperparams is None:
            hyperparams = {}
            
        self.gamma = hyperparams.get("gamma", 0.95)
        self.epsilon = hyperparams.get("epsilon", 1.0)
        self.epsilon_min = hyperparams.get("epsilon_min", 0.005)
        self.epsilon_decay = hyperparams.get("epsilon_decay", 0.995)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        self.alpha = hyperparams.get("alpha", 0.6)
        self.beta = hyperparams.get("beta", 0.4)
        self.beta_max = hyperparams.get("beta_max", 1.0)
        self.beta_decay = hyperparams.get("beta_decay", 0.995)



        # Initialize the primary Q-network
        self.model = DQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            print("Using Pretrained")
            model_path = os.path.join("models", self.model_name + ".pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set up target network for "t-dqn" or "double-dqn" strategies.
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.target_model = DQN(state_size, self.action_size).to(self.device)
            # Initialize target network with same weights
            self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuple in memory."""
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError(f"NaN or Inf in state: {state}")
        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            raise ValueError(f"NaN or Inf in next_state: {next_state}")
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError(f"NaN or Inf in reward: {reward}")
        
        if len(self.memory) >= self.memory_size:
            idx = torch.argmin(self.priorities)
            self.memory[idx] = (state, action, reward, next_state, done)
            self.priorities[idx] = max(self.priorities.max(),1)
        else:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities[len(self.memory)-1] = max(self.priorities.max(),1)  # Initialize with max priority

    def act(self, state, is_eval=False):
        """
        Returns an action for a given state.
        Uses epsilon-greedy exploration when not evaluating.
        """
        
        self.action_iter += 1
        
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convert state (np.array) to PyTorch tensor.
        state_tensor = torch.FloatTensor(state).view(len(state), -1).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().data.numpy()[0])

    def train_experience_replay(self, batch_size):
        """
        Trains the DQN using a mini-batch sampled from experience replay memory.
        Uses Double DQN if enabled, and Huber loss for stability.
        """
        if len(self.memory) < batch_size or self.action_iter % self.update_after_actions != 0:
            return 0.0  # not enough data
        
        
        #TODO: Normalize again per batch

        # Sample a mini-batch from memory using prioritized sampling
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        mini_batch = [self.memory[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.priorities) * probs[indices]).reciprocal().pow(self.beta)
        weights /= weights.max()  

        # Split the batch into components
        states = np.array([experience[0][0] for experience in mini_batch])  # shape: (batch, window, features)
        actions = np.array([experience[1] for experience in mini_batch])
        rewards = np.array([experience[2] for experience in mini_batch])
        next_states = np.array([experience[3][0] for experience in mini_batch])
        dones = np.array([1.0 if experience[4] else 0.0 for experience in mini_batch])
        

        # Flatten states: (batch_size, window_size * feature_dim)
        states = torch.FloatTensor(states.reshape(batch_size, -1)).to(self.device)
        next_states = torch.FloatTensor(next_states.reshape(batch_size, -1)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        assert not torch.isnan(states).any(), "NaNs in states"
        assert not torch.isnan(next_states).any(), "NaNs in next_states"
        assert not torch.isnan(rewards).any(), "NaNs in rewards"
        assert not torch.isnan(dones).any(), "NaNs in dones"
        
        # Predict Q-values for current states and gather the ones for taken actions
        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.strategy == "double-dqn":
                if self.n_iter % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # Online network selects action
                online_next_q = self.model(next_states)
                best_actions = torch.argmax(online_next_q, dim=1)

                # Target network evaluates value of selected action
                target_next_q = self.target_model(next_states)
                target_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                

                assert not torch.isnan(online_next_q).any(), "NaNs in online_next_q"
                assert not torch.isnan(target_next_q).any(), "NaNs in target_next_q"
            elif self.strategy == "t-dqn":
                # fixed-target DQN: use target network only
                if self.n_iter % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                target_q = self.target_model(next_states).max(1)[0]
            elif self.strategy == "dqn":
                target_q = self.model(next_states).max(1)[0]

            else:
                raise NotImplementedError(f"Unknown strategy: {self.strategy}")

            # Compute final target values
            targets = rewards + self.gamma * target_q * (1 - dones)
            
            self.priorities[[indices]] = (torch.abs(current_q - targets) + 1e-6)
        
        # Huber loss (Smooth L1)
        huber_each = F.smooth_l1_loss(current_q, targets, reduction='none')
        huber_each = huber_each.squeeze(-1)
        loss = (huber_each * weights).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.beta < self.beta_max:
            self.beta /= self.beta_decay

        # Update iteration counter
        self.n_iter += 1

        return loss.item()


    def save(self, episode):
        """Saves the current model parameters."""
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), "models/{}_{}.pth".format(self.model_name, episode))

    
# --- Training and Evaluation Functions ---

def train_model(agent, episode, X, y, ep_count=100, batch_size=256, window_size=0):
    """
    Runs one training episode on the provided data.
    
    Returns:
        tuple: (episode, ep_count, total_profit, average_loss)
    """
    #agent.first_iter = True
    total_returns = 0
    data_length = len(X) - 1
    
    avg_loss = []

    state = get_state(X, 0, window_size + 1)
    
    for t in range(data_length):
        reward = 0
        next_state = get_state(X, t + 1, window_size + 1)
        y_val = get_state(y, t + 1, window_size + 1)
        
        # select an action using the agent's policy
        action = agent.act(state)

        # FALL
        if action == 0:
            reward = -1*y_val[0][0]
            
        # RISE
        elif action == 1:
            reward = 1*y_val[0][0]


        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            if loss != 0:
                avg_loss.append(loss)

        state = next_state
        
        total_returns += reward
        
    if episode % 100 == 0:
        agent.save(episode)
        
    

    avg_loss_value = np.mean(avg_loss) if avg_loss else 0.0
    return (episode, ep_count, total_returns, avg_loss_value)


def evaluate_model(agent, episode, X, y, window_size=0, debug=False):
    """
    Evaluates the agent over the validation data.
    
    Returns:
        tuple: (total_profit, history) where history is a list of (price, action_str)
    """
    total_profit = 0
    data_length = len(X) - 1

    history = []

    state = get_state(X, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(X, t + 1, window_size + 1)
        y_val = get_state(y, t + 1, window_size + 1)

        action = agent.act(state, is_eval=True)

        # BUY
        if action == 0:
            reward = -1*y_val[0][0]


        # SELL
        elif action == 1:
            reward = 1*y_val[0][0]
      
        if debug:
            print("Earned return: ", reward)
            
        history.append({'true':1 if reward >= 0 else 0,'pred':action,'reward':reward})
        


        done = (t == data_length - 1)
        # Optionally, record experiences for evaluation.
        #agent.memory.append((state, action, reward, next_state, done))

        
        state = next_state
        total_profit += reward
        
        
    return (total_profit, history)

