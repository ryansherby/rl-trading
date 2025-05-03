from utils import *

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
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# --- Agent Definition ---

class Agent:
    """
    Stock Trading Bot Agent using Deep Q-Learning with experience replay.
    Supports "dqn", "t-dqn" (fixed target) and "double-dqn" strategies.
    """
    def __init__(self, state_size, strategy="t-dqn", reset_every=1000,
                 pretrained=False, model_name="model_debug", device=None):
        self.strategy = strategy
        self.state_size = state_size  # length of the state vector (window)
        self.action_size = 3          # [HOLD, BUY, SELL]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # Hyperparameters for Q-Learning
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        # Device configuration (CPU/GPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the primary Q-network
        self.model = DQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            print("Using Pretrained")
            model_path = os.path.join("models", self.model_name + ".pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set up target network for "t-dqn" or "double-dqn" strategies.
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every
            self.target_model = DQN(state_size, self.action_size).to(self.device)
            # Initialize target network with same weights
            self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """
        Returns an action for a given state.
        Uses epsilon-greedy exploration when not evaluating.
        """
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if self.first_iter:
            # Force a BUY on the first iteration to ensure an initial position.
            self.first_iter = False
            return 1

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
        if len(self.memory) < batch_size:
            return 0.0  # not enough data

        # Sample a mini-batch from memory
        mini_batch = random.sample(self.memory, batch_size)

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
                if self.n_iter % self.reset_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # Online network selects action
                online_next_q = self.model(next_states)
                best_actions = torch.argmax(online_next_q, dim=1)

                # Target network evaluates value of selected action
                target_next_q = self.target_model(next_states)
                target_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)

                assert not torch.isnan(online_next_q).any(), "NaNs in online_next_q"
                assert not torch.isnan(target_next_q).any(), "NaNs in target_next_q"
            # elif self.strategy == "t-dqn":
            #     if self.n_iter % self.reset_every == 0:
            #         self.target_model.load_state_dict(self.model.state_dict())
            #     target_q = self.target_model(next_states).max(1)[0]
            # elif self.strategy == "dqn":
            #     target_q = self.model(next_states).max(1)[0]
            else:
                raise NotImplementedError(f"Unknown strategy: {self.strategy}")

            # Compute final target values
            targets = rewards + self.gamma * target_q * (1 - dones)
        
        # Huber loss (Smooth L1)
        loss = F.smooth_l1_loss(current_q, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update iteration counter
        self.n_iter += 1

        return loss.item()


    def save(self, episode):
        """Saves the current model parameters."""
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), "models/{}_{}.pth".format(self.model_name, episode))

    


# --- Training and Evaluation Functions ---

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    """
    Runs one training episode on the provided data.
    
    Returns:
        tuple: (episode, ep_count, total_profit, average_loss)
    """
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)
    
    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action using the agent's policy
        action = agent.act(state)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            
            delta = data[t][-1] - bought_price[-1]
            reward = delta
            total_profit += delta

        # HOLD (action == 0) does nothing

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 100 == 0:
        agent.save(episode)

    avg_loss_value = np.mean(avg_loss) if avg_loss else 0.0
    return (episode, ep_count, total_profit, avg_loss_value)


def evaluate_model(agent, data, window_size, debug=False):
    """
    Evaluates the agent over the validation data.
    
    Returns:
        tuple: (total_profit, history) where history is a list of (price, action_str)
    """
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []

    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            history.append((data[t][-1], "BUY"))
            if debug:
                print("Buy at: {}".format(format_currency(data[t][-1])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t][-1] - bought_price[-1]
            reward = delta
            total_profit += delta
            history.append((data[t][-1], "SELL"))
            if debug:
                print("Sell at: {} | Position: {}".format(
                    format_currency(data[t][-1]), format_position(data[t] - bought_price)))
        else:
            history.append((data[t][-1], "HOLD"))

        done = (t == data_length - 1)
        # Optionally, record experiences for evaluation.
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history


