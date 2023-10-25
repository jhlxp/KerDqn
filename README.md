# KerDqn

Deep Reinforcement Learning Plugin for Kernel Congestion

## Source Code

```
├── online
│   ├── agent.py
│   ├── model.py
├── source
│   ├── tcp_kerdqn.c
│   ├── Makefile
```

## Usage

Linux kernel 5.4.233
Insert kernel modules:

```shell
cd src
make
sudo insmod tcp_kerdqn.ko
```

Start agent, default port 8023:

```shell
cd online
python ./agent.py --port 8023 --protocol 23
```

## Future Work

The distributed congestion control scheme combined with edge computing is being tested...
