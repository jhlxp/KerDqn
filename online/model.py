from re import X
from tkinter.tix import MAX
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import torch.multiprocessing as mp
import math
import random

class ReplayMemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a

        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.all_a = np.random.randint(low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.count = 0
        self.t = 0

    def add_memo(self, s, a, r, s_):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_ = []
        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s__tensor

class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()  
        in_features = n_input  

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, n_output))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_tensor.unsqueeze(0))  
        max_q_index = torch.argmax(q_values)
        action = max_q_index.detach().item()  
        return action

class Agent(object):
    def __init__(self, idx, n_input, n_output, mode="train"):
        self.filepath = './para'
        self.state_num = n_input
        self.action = 1
        self.prev_state = np.zeros(n_input, dtype=int)
        self.current_state = np.zeros(n_input, dtype=int)

        self.receive_cwnd = 0
        self.delay_time = 0
        self.timestamp = 0
        self.time_release_start = time.time()
        self.time_release_end = time.time()
        self.rtt = 0
        self.bw = 0
        self.min_rtt = 0
        self.max_bw = 0
        self.rtt_last = 0
        self.rtt_last_last = 0
        self.rtt_last_last_last = 0
        self.bw_last = 0
        self.bw_last_last = 0
        self.bw_last_last_last = 0
        self.utility = 0
        self.utility_last = 0
        self.utility_diff = 0
        self.reward = 0
        self.cwnd = 10
        self.count = 0
        self.updated_Qvalue = 0
        self.port = 0

        self.send_msg_select = 10
        self.lr_discount = 0.999

        self.idx = idx
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.9
        self.lr_discount = 0.999
        self.episode_reward = 0
        self.up_keep_down = 1
        self.memo = ReplayMemory(n_s=self.n_input, n_a=self.n_output)

        self.EPSILON = 0.9
        self.learning_rate = 0.001

        if self.mode == "train":
            self.online_net = DQN(self.n_input, self.n_output)
            self.target_net = DQN(self.n_input, self.n_output)

            self.target_net.load_state_dict(self.online_net.state_dict())  

            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

    def ql_bictcp_update_getenv(self, rtt, bw, max_bw, min_rtt, receive_cwnd):
        self.rtt_last_last = self.rtt_last
        self.rtt_last = self.rtt
        self.bw_last_last = self.bw_last
        self.bw_last = self.bw

        for i in range(self.state_num):
            self.prev_state[i] = self.current_state[i]

        self.rtt = rtt
        self.bw = bw
        self.max_bw = max_bw
        self.min_rtt = min_rtt
        self.receive_cwnd = receive_cwnd

        if (self.learning_rate > 0.0001) & ((self.count % 10) == 0):
            self.learning_rate = self.learning_rate * self.lr_discount

        if self.rtt <= 0:
            self.current_state[0] = 0
        elif self.rtt >= 500000:
            self.current_state[0] = 49
        else:
            self.current_state[0] = int(self.rtt / 10000)

        if self.min_rtt <= 0:
            self.current_state[1] = 0
        elif self.min_rtt >= 500000:
            self.current_state[1] = 49
        else:
            self.current_state[1] = int(self.min_rtt / 10000)

        if self.bw <= 0:
            self.current_state[2] = 0
        elif self.bw >= 300000:
            self.current_state[2] = 59
        else:
            self.current_state[2] = int(self.bw / 5000)

        if self.max_bw <= 0:
            self.current_state[3] = 0
        elif self.max_bw >= 300000:
            self.current_state[3] = 59
        else:
            self.current_state[3] = int(self.max_bw / 5000)

        self.current_state[4] = self.cwnd

    def getRewardFromEnvironment(self):

        state_max = 60

        alpha0_reduc = math.pow(self.current_state[2] / state_max, 0.5)
        alpha3_reduc = math.pow((state_max - self.current_state[2]) / state_max, 3)

        if self.min_rtt < 25000:
            k = 1.4
        elif self.min_rtt < 35000:
            k = 1.3
        elif self.min_rtt < 45000:
            k = 1.25
        elif self.min_rtt < 85000:
            k = 1.2
        else:
            k = 1.15

        self.reward = 0

        if (self.rtt > (k * self.min_rtt)) & (self.current_state[0] > self.current_state[1]):
            if self.up_keep_down != 0:
                self.reward = 1 * (self.current_state[1] - self.current_state[0]) * alpha0_reduc
            elif self.up_keep_down == 0:
                self.reward = 1 * (self.current_state[0] - self.current_state[1]) * alpha0_reduc
        else:
            if self.current_state[2] > self.prev_state[2]:
                if self.up_keep_down == 2:
                    self.reward = 2 * alpha3_reduc
                if self.up_keep_down == 1:
                    self.reward = -1 * alpha3_reduc
                elif self.up_keep_down == 0:
                    self.reward = -1 * alpha3_reduc
            elif self.current_state[2] == self.prev_state[2]:
                if self.up_keep_down == 2:
                    self.reward = 2 * alpha3_reduc
                if self.up_keep_down == 1:
                    self.reward = 0 * alpha3_reduc
                elif self.up_keep_down == 0:
                    self.reward = -1 * alpha3_reduc
            else:
                if self.up_keep_down == 2:
                    self.reward = 2 * alpha3_reduc
                if self.up_keep_down == 1:
                    self.reward = -1 * alpha3_reduc
                elif self.up_keep_down == 0:
                    self.reward = -2 * alpha3_reduc

        self.rtt_last_last = self.rtt_last
        self.rtt_last = self.rtt

        self.bw_last_last = self.bw_last
        self.bw_last = self.bw

        self.count = self.count + 1

        return self.reward

    def executeAction(self):

        if self.action == 0:
            self.up_keep_down = 1
            self.cwnd = self.cwnd
        elif self.action == 1:
            self.up_keep_down = 2
            self.cwnd = self.cwnd + 1
        elif self.action == 2:
            self.up_keep_down = 0
            self.cwnd = max(self.cwnd - 1, 10)
        elif self.action == 3:
            self.up_keep_down = 2
            self.cwnd = self.cwnd + 2
        elif self.action == 4:
            self.up_keep_down = 0
            self.cwnd = max(self.cwnd - 2, 10)
        elif self.action == 5:
            self.up_keep_down = 2
            self.cwnd = self.cwnd + 3
        elif self.action == 6:
            self.up_keep_down = 0
            self.cwnd = max(self.cwnd - 3, 10)
        elif self.action == 7:
            self.up_keep_down = 2
            self.cwnd = self.cwnd + 4
        elif self.action == 8:
            self.up_keep_down = 0
            self.cwnd = max(self.cwnd - 4, 10)
        elif self.action == 9:
            self.up_keep_down = 2
            self.cwnd = self.cwnd + 5
        elif self.action == 10:
            self.up_keep_down = 0
            self.cwnd = max(self.cwnd - 5, 10)
        else:
            print("error!")

        self.send_msg_select = str(int(self.cwnd))

    def print_value(self):
        print("-----------------------------------------")
        print("-", "count:", self.count, "-", "lr:", '%.6f' % self.learning_rate, "-", "time:",
              '%.2f' % self.timestamp, "-", 's',)
        print("rtt:", '%.2f' % (self.rtt/1000), 'ms', " minrtt:", '%.2f' % (self.min_rtt/1000),'ms', " bw:", '%.2f' % (self.bw/1000), 'Mbps', " maxbw:", '%.2f' % (self.max_bw/1000), 'Mbps')
        print("cwnd:", self.cwnd, " episode_reward:", '%.2f' % self.episode_reward, "reward:", '%.2f' % self.reward)
        print("-----------------------------------------")


