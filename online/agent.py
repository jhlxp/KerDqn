from re import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque
import torch.optim as optim
import argparse
import os
import torch.multiprocessing as mp
import math
import random
import socket
from itertools import groupby
from model import Agent
import datetime
import csv
import argparse

STATE0_MAX = 50
STATE1_MAX = 50
STATE2_MAX = 60
STATE3_MAX = 60
NUM_OF_ACTION = 11
NUM_OF_STATE = 5
data_dir = "./state_now"

def make_data_file(file_name, data_np):
    f = open(file_name, 'a', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(data_np.tolist())
    f.close()

def train(port_num, protocol_num, release_time):
    os.makedirs(data_dir, exist_ok=True)
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = data_dir + '/' + now_time + '_state_now.csv'
    f = open(file_name, 'w', encoding='utf-8')
    f.close()

    IS_DATA = 1
    NO_DATA = 0
    bw_list = []
    rtt_list = []
    max_bw_list = []
    min_rtt_list = []
    receive_cwnd_list = []

    sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, protocol_num)

    sock.bind((port_num, 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

    nlmsg_len = 16 + len("10")
    nlmsg_flags = 0
    nlmsg_type = 0
    nlmsg_seq = 0
    nlmsg_pid = port_num

    nlmsg_len = '0x{:04X}'.format(nlmsg_len)[2:]
    nlmsg_flags = '0x{:02X}'.format(nlmsg_flags)[2:]
    nlmsg_type = '0x{:02X}'.format(nlmsg_type)[2:]
    nlmsg_seq = '0x{:04X}'.format(nlmsg_seq)[2:]
    nlmsg_pid = '0x{:04X}'.format(nlmsg_pid)[2:]

    nlmsg_header = str(nlmsg_len + nlmsg_flags + nlmsg_type + nlmsg_seq + nlmsg_pid)
    fault_cwnd = str(10)
    send_nlmsg = bytes(nlmsg_header + fault_cwnd, encoding="utf8")

    sock.sendto(send_nlmsg, (0, 0))
    print("connecting kernel is normal!")

    agent = Agent(idx=0,
                  n_input=NUM_OF_STATE,
                  n_output=NUM_OF_ACTION,
                  mode='train',
                  )
    model = agent.online_net

    model.load_state_dict(model.state_dict())

    optimizer = agent.optimizer

    agent.port = port_num

    while True:
        flag = NO_DATA
        try:
            sock.recv(1024, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            flag = IS_DATA
        except BlockingIOError as err:
            pass

        while flag == IS_DATA:
            if flag == IS_DATA:
                msg, ancdata, flags, addr = sock.recvmsg(64)
                msg = msg[16:]
                msg = msg.decode("utf-8")
                msg_split = [''.join(list(g)) for k, g in groupby(msg, key=lambda x: x.isdigit())]
                bw = int(msg_split[1])
                rtt = int(msg_split[3])
                max_bw = int(msg_split[5])
                min_rtt = int(msg_split[7])
                receive_cwnd = int(msg_split[9])

                bw_list.append(bw)
                rtt_list.append(rtt)
                max_bw_list.append(max_bw)
                min_rtt_list.append(min_rtt)
                receive_cwnd_list.append(receive_cwnd)


                flag = NO_DATA
                try:
                    sock.recv(1024, socket.MSG_DONTWAIT | socket.MSG_PEEK)
                    flag = IS_DATA
                except BlockingIOError as err:
                    pass

        if (len(bw_list) == len(rtt_list) >= 1) & (bw_list != []) & (rtt_list != []):

            rtt = sum(rtt_list) / len(rtt_list)
            bw = sum(bw_list) / len(bw_list)
            max_bw = max(max_bw_list)
            min_rtt = min(min_rtt_list)
            receive_cwnd = receive_cwnd_list[-1]
            agent.ql_bictcp_update_getenv(rtt, bw, max_bw, min_rtt, receive_cwnd)
            agent.action = np.random.randint(0, NUM_OF_ACTION)
            agent.executeAction()
            send_nlmsg = bytes(nlmsg_header + agent.send_msg_select, encoding="utf8")
            sock.sendto(send_nlmsg, (0, 0))
            bw_list = []
            rtt_list = []
            max_bw_list = []
            min_rtt_list = []
            receive_cwnd_list = []
            break

    agent.time_release_start = time.time()
    while True:
        agent.time_release_end = time.time()
        agent.timestamp = agent.time_release_end - agent.time_release_start

        if agent.timestamp > release_time:
            print(f"timestamp > release time {release_time}, release")
            break

        flag = NO_DATA
        try:
            sock.recv(1024, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            flag = IS_DATA
        except BlockingIOError as err:
            pass

        while flag == IS_DATA:
            if flag == IS_DATA:
                msg, ancdata, flags, addr = sock.recvmsg(1024)
                msg = msg[16:]
                msg = msg.decode("utf-8")
                msg_split = [''.join(list(g)) for k, g in groupby(msg, key=lambda x: x.isdigit())]
                bw = int(msg_split[1])
                rtt = int(msg_split[3])
                max_bw = int(msg_split[5])
                min_rtt = int(msg_split[7])
                receive_cwnd = int(msg_split[9])
                bw_list.append(bw)
                rtt_list.append(rtt)
                max_bw_list.append(max_bw)
                min_rtt_list.append(min_rtt)
                receive_cwnd_list.append(receive_cwnd)

                flag = NO_DATA
                try:
                    sock.recv(1024, socket.MSG_DONTWAIT | socket.MSG_PEEK)
                    flag = IS_DATA
                except BlockingIOError as err:
                    pass

        if (len(bw_list) == len(rtt_list) >= 1) & (bw_list != []) & (rtt_list != []):

            time_start = time.time()

            rtt = sum(rtt_list) / len(rtt_list)
            bw = sum(bw_list) / len(bw_list)
            max_bw = max(max_bw_list)
            min_rtt = min(min_rtt_list)
            receive_cwnd = receive_cwnd_list[-1]

            agent.ql_bictcp_update_getenv(rtt, bw, max_bw, min_rtt, receive_cwnd)
            agent.getRewardFromEnvironment()

            s = np.array([agent.prev_state[0], agent.prev_state[1], agent.prev_state[2], agent.prev_state[3], agent.prev_state[4]])
            a = agent.action
            r = agent.reward
            s_ = np.array([agent.current_state[0], agent.current_state[1], agent.current_state[2], agent.current_state[3], agent.current_state[4]])

            agent.memo.add_memo(s, a, r, s_)
            agent.episode_reward = agent.episode_reward + r

            batch_s, batch_a, batch_r, batch_s_ = agent.memo.sample()

            target_q_values = agent.target_net(batch_s_)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_r + agent.GAMMA * max_target_q_values

            q_values = agent.online_net(batch_s)
            a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

            loss = nn.functional.smooth_l1_loss(a_q_values, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if (agent.count % 500) == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())

            if (agent.count % 1000) == 0:
                if not os.path.exists(agent.filepath):
                    os.makedirs(agent.filepath)

                now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  

                str_name = now_time + '_agent_{num}.pth'.format(num=agent.count)
                torch.save(agent.online_net.state_dict(), os.path.join(agent.filepath, str_name))

            if np.random.uniform() <= agent.EPSILON:  
                a = agent.online_net.act(s_)
            else:  
                a = np.random.randint(0, NUM_OF_ACTION)

            agent.action = a
            agent.executeAction()

            send_nlmsg = bytes(nlmsg_header + agent.send_msg_select, encoding="utf8")
            sock.sendto(send_nlmsg, (0, 0))

            time_end = time.time()
            time_c = time_end - time_start
            print('This iteration time is: ', time_c, 's')

            bw_list = []
            rtt_list = []
            max_bw_list = []
            min_rtt_list = []
            receive_cwnd_list = []

            agent.print_value()

            make_data_file(file_name, s_)

    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    print("child process socket close")

def parser():
    parser = argparse.ArgumentParser(description='local agent parser')
    parser.add_argument('--port', default=8023, type=int, help='Port num')
    parser.add_argument('--protocol', default=23, type=int, help='Protocol num')
    parser.add_argument('--release_time', default=86400, type=int, help='Release time')
    return parser

if __name__ == "__main__" :
    os.environ['OMP_NUM_THREADS'] = '4'
    args = parser().parse_args()
    print("-----------Local agent starting-----------")
    print("port:",args.port)
    print("protocol:", args.protocol)
    print("release time:", args.release_time,"s")
    train(args.port, args.protocol, args.release_time)