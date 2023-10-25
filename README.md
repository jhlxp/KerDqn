# KerDqn

Deep Reinforcement Learning Plugin for Kernel Congestion

## Source Code

```
├── online
│   ├── agent.py
│   ├── model.py
├── source
│   ├── tcp_kerdqn.py
│   ├── Makefile
```

## Usage

Insert kernel modules:

```shell
cd source
make
sudo insmod tcp_kerdqn.ko
```

Start agent, default port 8023:

```shell
python ./agent.py --port 8023 --protocol 23
```

## Future Work

The distributed congestion control scheme combined with edge computing is being tested...
