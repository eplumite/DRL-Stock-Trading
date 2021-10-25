from copy import copy 
import numpy as np
import gym

class StockEnv(gym.Env):
    def __init__(self, data, savings=10000, n_step=60):
        self.metadata = {"render.modes": ['human']}
        self.reward_range = (-float("inf"), float("inf"))

        self.action_space = (0,1,2)
        # 0 -> hold
        # 1 -> buy
        # 2 -> sell
        self.num_tickers = data.shape[1] - 1
        self.state_dim = self.num_tickers * 2 + 1
        # num_tickers -> number of shares in portfolio
        # num_tickers -> prices of shares 
        # 1 -> remaining savings or, just, balance
        self.state = None
        self.i_step = 0
        
        # smth to remember
        self.historical_data = data
        self.savings = savings
        self.n_step = n_step
    
    def reset(self):
        self.balance = self.savings
        self.i_step = 0
        
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.state[self.num_tickers:-1] = self.historical_data.values[self.i_step, 1:]
        self.state[-1] = self.balance
        
    def _portfolio_value(self, state=None):
        if state is None:
            return np.dot(self.state[:self.num_tickers], self.state[self.num_tickers:-1]) + self.state[-1]
        else:
            return np.dot(state[:self.num_tickers], state[self.num_tickers:-1]) + state[-1]
        
    def _reward(self, state, new_state):
        return self._portfolio_value(new_state) - self._portfolio_value(state)
    
    def step(self, action):
        # remember previous state
        old_state = copy(self.state)
        self.valid_action = action
        
        # update step number
        self.i_step += 1
        
        # update prices
        self.state[self.num_tickers:-1] = self.historical_data.values[self.i_step, 1:]
        
        # sell shares
        for i in np.argwhere(action == 2):
            if self.state[i] > 0:
                self.state[i] -= 1
                self.balance += self.state[self.num_tickers + i]
            else:
                self.valid_action[i] = 0
        
        # buy shares
        for i in np.argwhere(action == 1):
            if self.balance >= self.state[self.num_tickers + i]:
                self.state[i] += 1
                self.balance -= self.state[self.num_tickers + i]
            else:
                self.valid_action[i] = 0
                
        # hold other shares
        self.state[-1] = self.balance
        
        # calculate reward 
        reward = self._reward(old_state, self.state)
        
        # check if episode is over
        done = (self.i_step == self.n_step)
        
        # calculate current portfolio value
        info = self._portfolio_value()
        
        return self.state, reward, done, info

    