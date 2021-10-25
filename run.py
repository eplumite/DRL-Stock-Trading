import numpy as np
import pandas as pd

from utils.loader import download_data, load_historical_data
from env.stock_env import StockEnv
from model.ddpg import DDPG

############ Data #################
download_data()
data = load_historical_data()
data = data.replace([np.inf], np.nan).dropna(axis=1)

############ Environment #################
env = StockEnv(data, savings=500000)

############ Agent #################
agent = DDPG()

############ Train #################
agent.train(env)

