from datetime import datetime
from copy import copy
import csv
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DDPG:
    def __init__(self, buffer_size=500, batch_size=32, gamma=0.9, tau=0.9):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
    
    def _init_nn(self, env):
        self.critic = CriticNetwork(input_dim=env.num_tickers+env.state_dim)
        self.actor = ActorNetwork(input_dim=env.state_dim, action_dim=env.num_tickers)
        self.target_critic = copy(self.critic)
        self.target_actor = copy(self.actor)
        self.replay_buffer = []
        
    def _select_action(self, state):

        self.actor.eval()
        
        action = self.actor.forward(state).detach().numpy()
        action = action + np.random.normal(scale=0.1, size=len(action))
        action = np.where(action > 0.1, 2, action)
        action = np.where(abs(action) < 0.1, 1, action)
        action = np.where(action < -0.1, 0, action)
        
        self.actor.train()

        return action
    
    def _update_replay_buffer(self, curr_state, action, reward, new_state, done):
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append((curr_state, action, reward, new_state, done))
        else:
            idx = np.random.randint(0, self.buffer_size)
            self.replay_buffer[idx] = (curr_state, action, reward, new_state, done)
        
    def train(self, env, num_episodes=50, num_timesteps=2000):
        
        os.makedirs("exp", exist_ok=True)
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.mkdir(f"exp/exp_{exp_name}")
        with open(f'exp/exp_{exp_name}/log.csv', 'w') as csvfile:
            logwriter = csv.writer(csvfile)
            logwriter.writerow(['ep', 't'] + list(env.historical_data.columns[1:]) + ['balance', 'loss'])
            
        self._init_nn(env)
        env.n_step = num_timesteps
        
        for ep in range(num_episodes):
            env.reset()
            for t in range(num_timesteps):
                curr_state = copy(env.state)
                action = self._select_action(curr_state)
                new_state, reward, done, info = env.step(action)
                self._update_replay_buffer(curr_state, action, reward, new_state, done)
                if len(self.replay_buffer) >= self.batch_size:
                    target_values = []
                    transitions = []
                    for k in range(self.batch_size):
                        idx = np.random.randint(0, len(self.replay_buffer))
                        state_k, action_k, reward_k, new_state_k, done_k = self.replay_buffer[idx]
                        transitions.append((state_k, action_k, reward_k, new_state_k, done_k))
                        
                        target_action = self.target_actor.forward(new_state_k).detach()
                        target_q = self.target_critic.forward(new_state_k, target_action).item()
                        target_values.append(reward_k + (1 - done_k) * self.gamma * target_q)
                    target_values = np.array(target_values, dtype=np.float32)[:, None]
                    target_values = torch.from_numpy(target_values)

                    critic_values = []
                    states = []
                    actions = []
                    for k, sample in enumerate(transitions):
                        state_k, action_k, reward_k, new_state_k, done_k = sample
                        states.append(state_k)
                        actions.append(action_k)
                    states = np.array(states)
                    actions = np.array(actions)
                    critic_values = self.critic.forward(states, actions)
                    
                    self.critic.optimizer.zero_grad()
                    critic_loss = F.mse_loss(target_values, critic_values)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                    self.critic.optimizer.step()
                                             
                    self.critic.eval()
                    self.actor.optimizer.zero_grad()
                    p = self.actor.forward(curr_state).detach()
                    actor_loss = -self.critic.forward(curr_state, p)
                    actor_loss = torch.mean(actor_loss)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                    self.actor.optimizer.step()
                    self.critic.train()

                    self._update_target_networks()
                    
                    with open(f'exp/exp_{exp_name}/actions_log.csv', 'a') as csvfile:
                        logwriter = csv.writer(csvfile)
                        logwriter.writerow([f'{ep}', f'{t}'] + list(action.astype(np.int)) + [new_state[-1], critic_loss.item()])
                    
                    print(t, critic_loss.item(), curr_state[-1])
                else:
                    with open(f'exp/exp_{exp_name}/actions_log.csv', 'a') as csvfile:
                        logwriter = csv.writer(csvfile)
                        logwriter.writerow([f'{ep}', f'{t}'] + list(env.valid_action.astype(np.int)) + [new_state[-1], 0])
        
        
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim=32, alpha=1e-3):
        super(CriticNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.activation2 = nn.Tanh()
        self.output = nn.Linear(hid_dim, 1)

        self.optimizer = optim.SGD(self.parameters(), lr=alpha)

    def forward(self, action, state):
        x = np.concatenate((action, state), axis=-1)
        x = torch.from_numpy(x.astype(np.float32))
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.output(x)

        return x


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hid_dim=32, alpha=1e-4):
        super(ActorNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.activation2 = nn.Tanh()
        self.output = nn.Linear(hid_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.output(x)

        return torch.tanh(x)
