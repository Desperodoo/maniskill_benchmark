# RLFT: Reinforcement Learning and Flow-based Training

<p align="center">
  <b>A unified framework for robot learning with diffusion/flow policies</b>
</p>

RLFT 是一个统一的机器人学习框架，支持从模仿学习到在线强化学习的完整流水线。所有算法的关键超参数均经过 **级联超参数扫描 (Cascade Sweep)** 系统化调优。

- **Imitation Learning (IL)**: Diffusion Policy, Flow Matching, ShortCut Flow, Consistency Flow, Reflected Flow
- **Offline RL**: CPQL, AWCP, AW-ShortCut Flow, DQC, Offline SAC
- **Online RL**: RLPD (SAC/AWSC), DSRL-SAC, PLD-SAC, ReinFlow

## 📁 项目结构

```
rlft/
├── algorithms/              # 策略学习算法
│   ├── il/                  # 模仿学习算法
│   │   ├── diffusion_policy.py      # Diffusion Policy (DDPM)
│   │   ├── flow_matching.py         # Flow Matching (ODE-based)
│   │   ├── shortcut_flow.py         # ShortCut Flow (few-step sampling)
│   │   ├── consistency_flow.py      # Consistency Flow
│   │   └── reflected_flow.py        # Reflected Flow (bounded actions)
│   ├── offline_rl/          # 离线强化学习算法
│   │   ├── cpql.py                  # CPQL (Conservative Policy Q-Learning)
│   │   ├── awcp.py                  # AWCP (Advantage-Weighted Conservative Policy)
│   │   ├── aw_shortcut_flow.py      # AW-ShortCut Flow
│   │   ├── dqc.py                   # DQC (Decoupled Q-Chunking)
│   │   └── sac.py                   # Offline SAC (多正则化: td3bc/awr/iql/cql)
│   └── online_rl/           # 在线强化学习算法
│       ├── sac.py                   # SAC + Action Chunking (RLPD-style)
│       ├── awsc.py                  # AWSC (Advantage-Weighted ShortCut Flow)
│       ├── reinflow.py              # ReinFlow (PPO + Flow Matching)
│       ├── dsrl_sac.py              # DSRL-SAC (SAC in flow noise space)
│       └── pld_sac.py               # PLD-SAC (SAC in residual action space)
│
├── networks/                # 神经网络架构
│   ├── unet.py              # Conditional 1D U-Net
│   ├── velocity.py          # Velocity networks (VelocityUNet1D, ShortCutVelocityUNet1D)
│   ├── q_networks.py        # Q-networks (DoubleQ, EnsembleQ, SigmoidQ)
│   ├── actors.py            # Actor networks (Gaussian, Temperature)
│   └── encoders.py          # Visual/State encoders (PlainConv, ResNet)
│
├── buffers/                 # 数据缓冲区
│   ├── replay_buffer.py     # Off-policy replay buffers (RLPD)
│   ├── dsrl_buffer.py       # DSRL 标准 MDP buffer
│   ├── pld_buffer.py        # PLD online/offline 混合 buffer
│   ├── success_buffer.py    # Success-filtered replay buffer (AWSC)
│   ├── rollout_buffer.py    # On-policy rollout buffer (PPO/ReinFlow)
│   └── smdp.py              # SMDP cumulative reward computation
│
├── datasets/                # 数据集加载
│   ├── maniskill_dataset.py # ManiSkill3 HDF5 demo loading
│   ├── carm_dataset.py      # CARM real robot demo loading
│   └── data_utils.py        # Data utilities & observation encoding
│
├── envs/                    # 环境工具
│   ├── make_env.py          # Environment factory
│   ├── evaluate.py          # Evaluation utilities
│   ├── base_flow_env.py     # Flow 环境基类 (共享 obs 编码逻辑)
│   ├── dsrl_env.py          # DSRL noise-space 环境包装器
│   └── pld_env.py           # PLD residual-action 环境包装器
│
├── offline/                 # 离线训练脚本
│   ├── train_maniskill.py   # ManiSkill 仿真训练 (IL + Offline RL)
│   └── train_carm.py        # CARM 真实机器人训练
│
├── online/                  # 在线训练脚本
│   ├── train_rlpd.py        # RLPD/AWSC 训练 (Off-policy + demo mixing)
│   ├── train_dsrl.py        # DSRL-SAC 训练 (noise-space RL)
│   ├── train_pld.py         # PLD-SAC 训练 (residual RL + Cal-QL)
│   ├── train_reinflow.py    # ReinFlow 训练 (On-policy PPO)
│   └── _flow_helpers.py     # Flow 训练共享工具函数
│
├── scripts/                 # 自动化脚本
│   ├── run_full_pipeline.sh     # 一键环境+数据+训练
│   ├── run_all_algorithms.sh    # 多 GPU 批量训练
│   ├── download_demos.sh        # 演示数据下载
│   ├── replay_demos.sh          # 数据 replay 预处理
│   ├── setup_maniskill_env.sh   # Conda 环境配置
│   ├── monitor_training.sh      # 训练监控
│   └── sweep/                   # 级联超参数扫描系统
│       ├── common.sh                # 通用配置与工具函数
│       ├── run_cascade_sweep.sh     # 三阶段级联扫描主控
│       ├── sweep_*.sh               # 各算法扫描脚本
│       └── fine/                    # 精细化扫描脚本
│
└── utils/                   # 通用工具
    ├── checkpoint.py        # 检查点保存/加载
    ├── ema.py               # EMA (Exponential Moving Average)
    ├── flow_wrapper.py      # ShortCut Flow 策略加载/推理包装
    ├── model_factory.py     # 模型构建工厂
    └── schedulers.py        # 学习率调度器
```

## 🚀 快速开始

### 环境配置

```bash
bash rlft/scripts/setup_maniskill_env.sh
conda activate maniskill
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 数据准备

```bash
bash rlft/scripts/download_demos.sh LiftPegUpright-v1
bash rlft/scripts/replay_demos.sh LiftPegUpright-v1
```

### 离线训练 (IL / Offline RL)

```bash
# Flow Matching (基础 IL)
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm flow_matching \
    --obs_mode rgb

# ShortCut Flow
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm shortcut_flow \
    --obs_mode rgb

# AW-ShortCut Flow (Offline RL, 需要先训练 ShortCut Flow checkpoint)
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm aw_shortcut_flow \
    --obs_mode rgb
```

### 在线训练 (Online RL)

以下三种在线 RL 方法均需要一个预训练的 ShortCut Flow checkpoint：

```bash
# DSRL-SAC: 在 flow noise space 中跑 SAC
python -m rlft.online.train_dsrl \
    --env_id LiftPegUpright-v1 \
    --checkpoint /path/to/shortcut_flow_best.pt

# PLD-SAC: 在 residual action space 中跑 SAC + Cal-QL 预训练
python -m rlft.online.train_pld \
    --env_id LiftPegUpright-v1 \
    --checkpoint /path/to/shortcut_flow_best.pt

# RLPD (SAC + demo mixing)
python -m rlft.online.train_rlpd \
    --env_id LiftPegUpright-v1 \
    --algorithm sac
```

### 批量训练与监控

```bash
# 批量训练所有离线算法
bash rlft/scripts/run_all_algorithms.sh --full

# 级联超参数扫描
bash rlft/scripts/sweep/run_cascade_sweep.sh

# 监控训练
tensorboard --logdir runs/
```

---

## 📖 使用指南

### 1. 离线模仿学习 (Imitation Learning)

通过 `train_maniskill.py` 统一入口，支持 5 种 IL 算法：

```bash
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --demo_path ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --algorithm <algorithm> \
    --obs_mode rgb \
    --total_iters 25000
```

| 算法 | `--algorithm` | 描述 | Sweep 最优 success_once |
|------|---------------|------|------------------------|
| Flow Matching | `flow_matching` | ODE-based 连续时间流 | 28% |
| ShortCut Flow | `shortcut_flow` | 自适应步长快速采样 (1-8步) | 34% |
| Consistency Flow | `consistency_flow` | 一致性正则化流 | **48%** |
| Diffusion Policy | `diffusion_policy` | DDPM-based 多步去噪 | 5% |
| Reflected Flow | `reflected_flow` | 边界反射处理 | — |

#### CARM 真实机器人

```bash
python -m rlft.offline.train_carm \
    --demo_path ~/recorded_data/pick_place \
    --algorithm flow_matching \
    --total_iters 100000
```

---

### 2. 离线强化学习 (Offline RL)

同样通过 `train_maniskill.py` 入口：

```bash
python -m rlft.offline.train_maniskill \
    --env_id LiftPegUpright-v1 \
    --algorithm <algorithm> \
    --obs_mode rgb \
    --total_iters 25000
```

| 算法 | `--algorithm` | 描述 | Sweep 最优 success_once |
|------|---------------|------|------------------------|
| AW-ShortCut Flow | `aw_shortcut_flow` | Q-weighted ShortCut Flow | **80%** |
| CPQL | `cpql` | Conservative Q-Learning + Flow Policy | 22% |
| AWCP | `awcp` | Advantage-Weighted Consistency Policy | 7% |
| DQC | `dqc` | Decoupled Q-Chunking (dual sigmoid critic) | — |
| Offline SAC | `sac` | 多正则化 SAC (td3bc/awr/iql/cql) | — |

---

### 3. 在线强化学习 (Online RL)

#### DSRL-SAC (noise-space RL)

在冻结 ShortCut Flow 策略的 **noise space** 中运行 SAC。环境包装器内部将 noise 解码为真实动作。

```bash
python -m rlft.online.train_dsrl \
    --env_id LiftPegUpright-v1 \
    --checkpoint /path/to/shortcut_flow_best.pt \
    --total_timesteps 1000000
```

核心设计（经 sweep 调优）：3×2048 MLP + Tanh，10 Q-networks，UTD=60，`action_magnitude=2.5`，`gamma=0.95`，`target_entropy=-3.5`。

#### PLD-SAC (residual RL)

在冻结 ShortCut Flow 策略的 **residual action space** 中运行 SAC，附带 Cal-QL critic 预训练。

```bash
python -m rlft.online.train_pld \
    --env_id LiftPegUpright-v1 \
    --checkpoint /path/to/shortcut_flow_best.pt \
    --total_timesteps 500000
```

核心设计（经 sweep 调优）：3×1024 MLP，5 Q-networks，UTD=60，`action_scale=0.3`，`gamma=0.99`，`target_entropy=-3.5`，`init_temperature=0.1`，Cal-QL 预训练 1000 步。

#### RLPD (demo mixing RL)

在线 SAC/AWSC 与离线演示数据混合训练 (RLPD-style)。

```bash
# SAC
python -m rlft.online.train_rlpd \
    --env_id LiftPegUpright-v1 \
    --algorithm sac

# AWSC (需要预训练 ShortCut Flow checkpoint)
python -m rlft.online.train_rlpd \
    --env_id LiftPegUpright-v1 \
    --algorithm awsc \
    --pretrain_path runs/shortcut_bc/best.pt
```

核心设计（经 sweep 调优）：`online_ratio=0.15`，`awsc_beta=50`，`awsc_bc_weight=2.0`，`lr_actor=1e-4`。

#### ReinFlow (on-policy PPO + Flow)

从预训练 Flow Matching 模型出发，用 PPO 进行 on-policy 微调。

```bash
python -m rlft.online.train_reinflow \
    --env_id PushCube-v1 \
    --pretrained_path runs/flow_matching/checkpoint.pt \
    --total_updates 10000
```

支持的 Online RL 算法总览：

| 算法 | 训练脚本 | Action Space | 基座策略 | 描述 |
|------|---------|-------------|---------|------|
| DSRL-SAC | `train_dsrl.py` | noise space | ShortCut Flow | SAC in flow noise space |
| PLD-SAC | `train_pld.py` | residual action | ShortCut Flow | Residual SAC + Cal-QL |
| SAC | `train_rlpd.py --algorithm sac` | raw action | — | SAC + Action Chunking + Demo Mixing |
| AWSC | `train_rlpd.py --algorithm awsc` | raw action | ShortCut Flow | Q-weighted ShortCut Flow (RLPD) |
| ReinFlow | `train_reinflow.py` | raw action | Flow Matching | PPO + Flow Matching |

---

## 🔧 关键参数说明

> **注**：以下默认值均为经过超参数扫描后的最优值，一般无需手动覆盖。

### 通用参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--seed` | 1 (offline) / 42 (online) | 随机种子 |
| `--cuda` | True | 是否使用 GPU |
| `--track` | True | 是否使用 WandB 记录 |
| `--capture_video` | False | 是否录制评估视频 |

### Action Chunking 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--obs_horizon` | 2 | 观测历史长度 |
| `--act_horizon` | 8 | 执行动作长度 |
| `--pred_horizon` | 8 | 预测动作长度 (sweep 最优: 8 > 16) |

### Flow / ShortCut 参数 (Offline)

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--num_flow_steps` | 20 | Flow Matching ODE 步数 (sweep: 20 > 10 > 5) |
| `--ema_decay` | 0.9995 | EMA 衰减率 (sweep wave3 最优) |
| `--sc_self_consistency_k` | 0.25 | ShortCut consistency batch 比例 |
| `--sc_step_size_mode` | fixed | 步长采样模式 (fixed > power2 > uniform) |
| `--sc_fixed_step_size` | 0.15 | 固定步长值 (sweep wave3 最优) |
| `--sc_num_inference_steps` | 8 | 推理采样步数 |

### Offline RL 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--alpha` | 0.0005 | CPQL 熵系数 (sweep wave3 最优) |
| `--beta` | 10.0 | AWR/AWAC 温度 |
| `--reward_scale` | 0.05 | 奖励缩放因子 (sweep wave3 最优) |
| `--weight_clip` | 200.0 | AWR 权重裁剪 (sweep wave3 最优) |
| `--consistency_weight` | 0.3 | 一致性正则权重 |

### DSRL-SAC 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--action_magnitude` | 2.5 | Noise space 边界 $[-\text{mag}, +\text{mag}]$ |
| `--utd_ratio` | 60 | Update-to-Data ratio (sweep 最关键参数) |
| `--num_qs` | 10 | Ensemble Q-network 数量 |
| `--gamma` | 0.95 | 折扣因子 (匹配 ~12 RL steps/episode) |
| `--target_entropy` | -3.5 | 目标熵 (sweep: -3.5 >> auto -56) |
| `--log_std_init` | -5.0 | Actor 初始 log-std (保守探索) |
| `--layer_size` | 2048 | MLP 层宽 |

### PLD-SAC 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--action_scale` | 0.3 | Residual action 边界 $[-\xi, +\xi]$ |
| `--utd_ratio` | 60 | Update-to-Data ratio |
| `--num_qs` | 5 | Ensemble Q-network 数量 |
| `--gamma` | 0.99 | 折扣因子 |
| `--target_entropy` | -3.5 | 目标熵 |
| `--init_temperature` | 0.1 | 初始温度 (近确定性启动) |
| `--learning_rate` | 1e-4 | 学习率 (防止高 UTD 下 Q 发散) |
| `--layer_size` | 1024 | MLP 层宽 |
| `--calql_pretrain_steps` | 1000 | Cal-QL critic 预训练步数 |
| `--calql_alpha` | 0.0 | Cal-QL 保守系数 (sweep: 0 最优) |
| `--online_ratio` | 1.0 | 在线数据比例 (1.0=纯在线) |

### RLPD / AWSC 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--online_ratio` | 0.15 | 在线数据比例 (sweep v3/v4 最优) |
| `--lr_actor` | 1e-4 | Actor 学习率 (sweep: 1e-4 防止灾难遗忘) |
| `--awsc_beta` | 50.0 | Advantage weighting 温度 (sweep v2-v4 最优) |
| `--awsc_bc_weight` | 2.0 | Flow BC 损失权重 (sweep v2: 锚定预训练策略) |
| `--awsc_advantage_mode` | per_state_v | Advantage 计算模式 (sweep: 优于 batch_mean) |
| `--num_qs` | 10 | Ensemble Q-network 数量 |
| `--num_min_qs` | 2 | Min-Q 采样数 |

### DQC 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--backup_horizon` | 16 | Chunk critic horizon $h$ |
| `--kappa_b` | 0.9 | V-network backup expectile |
| `--kappa_d` | 0.8 | Action critic distillation expectile |
| `--best_of_n` | 32 | Best-of-N 推理候选数 |

---

## 🏗️ 架构设计

### 训练流水线 (三阶段)

```
Stage 1: IL (纯 BC)           Stage 2: Offline RL (Q-weighted BC)     Stage 3: Online RL (微调)
─────────────────────         ──────────────────────────────────       ──────────────────────────
Flow Matching ──────────┐     AW-ShortCut Flow                        DSRL-SAC  (noise space)
ShortCut Flow ──────────┤     CPQL                                    PLD-SAC   (residual space)
Consistency Flow ───────┼──▶  AWCP                                ──▶ RLPD/AWSC (demo mixing)
Diffusion Policy ───────┤     DQC                                     ReinFlow  (PPO)
Reflected Flow ─────────┘     Offline SAC
```

### 算法继承关系

```
nn.Module
├── DiffusionPolicyAgent
├── FlowMatchingAgent
│   └── ShortCutFlowAgent
│       ├── ConsistencyFlowAgent
│       └── ReflectedFlowAgent
├── CPQLAgent
│   └── AWCPAgent
│       └── AWShortCutFlowAgent
├── DQCAgent                    # Dual sigmoid critic + flow actor
├── OfflineSACAgent             # Multi-regularization (td3bc/awr/iql/cql)
├── SACAgent                    # RLPD-style online SAC
├── AWSCAgent                   # Online Q-weighted ShortCut Flow
├── DSRLSACAgent                # Noise-space SAC (MLP actor/critic)
├── PLDSACAgent                 # Residual-space SAC (MLP + Cal-QL)
└── ReinFlowAgent               # PPO + Flow Matching
```

### 网络架构

```
Visual Encoder (PlainConv/ResNet)
        │
        ▼
    obs_features (B, T * feature_dim)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
VelocityUNet1D / ShortCutVelocityUNet1D   Q-Networks (Double/Ensemble/Sigmoid)
        │                                              │
        ▼                                              ▼
   actions (B, pred_horizon, act_dim)            Q-values (B, 1)


DSRL-SAC / PLD-SAC 架构 (MLP):
        obs_features
        │          │
        ▼          ▼
   DSRLActor   DSRLCritic / PLDCritic
   (MLP+Tanh)  (Ensemble MLP, 10/5 Q-nets)
        │          │
        ▼          ▼
   noise/residual  Q(obs, noise/residual)
        │
        ▼
   Flow Policy (frozen) → real actions
```

### SMDP (Semi-Markov Decision Process) 公式

对于 action chunk 长度 $\tau$：
- **累积奖励**: $R_t^{(\tau)} = \sum_{i=0}^{\tau-1} \gamma^i r_{t+i}$
- **折扣因子**: $\gamma^\tau$
- **Bellman 方程**: $Q(s_t, a_{t:t+\tau}) = R_t^{(\tau)} + \gamma^\tau (1 - d) Q(s_{t+\tau}, a')$

### DQC 公式 (Decoupled Q-Chunking)

四网络分阶段训练:
1. **Value**: $L_V = \mathbb{E}[|\kappa_b - \mathbf{1}(Q^{h_a}_\text{target} - V < 0)| \cdot (Q^{h_a}_\text{target} - V)^2]$
2. **Chunk Critic**: $L_{Q^h} = \text{BCE}(Q^h_\text{logit}, \sigma(\text{target}))$, target from $V$-bootstrap
3. **Action Critic**: $L_{Q^{h_a}} = \mathbb{E}[|\kappa_d - \mathbf{1}(Q^h - Q^{h_a} < 0)| \cdot (Q^h - Q^{h_a})^2]$
4. **Actor**: $L_\pi = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$ (纯 Flow Matching BC, best-of-N 推理)

---

## 🔍 超参数扫描系统

本项目内置了**三阶段级联超参数扫描系统** (`scripts/sweep/`)，按依赖顺序自动调优：

```
阶段 1 (基础 IL):      flow_matching, diffusion_policy
         ↓ 继承最优参数
阶段 2 (依赖 IL):      consistency_flow, shortcut_flow, reflected_flow
         ↓ 继承最优参数
阶段 3 (Offline RL):   cpql, awcp, aw_shortcut_flow
```

```bash
# 运行完整级联扫描
bash rlft/scripts/sweep/run_cascade_sweep.sh

# 只运行某一阶段
bash rlft/scripts/sweep/run_cascade_sweep.sh --stage 1

# 只运行某个算法
bash rlft/scripts/sweep/run_cascade_sweep.sh --algorithm awcp

# 精细化扫描 (在第一轮最优点附近加密搜索)
bash rlft/scripts/sweep/run_cascade_sweep.sh --fine-sweep

# 查看进度
bash rlft/scripts/sweep/run_cascade_sweep.sh --status

# 重跑失败的实验
bash rlft/scripts/sweep/run_cascade_sweep.sh --retry-failed
```

支持多 GPU 并行/串行模式、CUDA 错误自动重试、WandB 日志集成。

---

## 📊 实验结果记录

训练日志保存在 `runs/` 目录：

```
runs/
├── {exp_name}__{timestamp}/
│   ├── config.json          # 训练配置
│   ├── events.out.tfevents  # TensorBoard 日志
│   ├── checkpoints/
│   │   ├── best.pt          # 最佳模型
│   │   ├── step_*.pt        # 定期保存的检查点
│   │   └── final.pt         # 最终模型
│   └── videos/              # 评估视频
```

```bash
tensorboard --logdir runs/
```

---

## 📚 参考文献

- **Diffusion Policy**: [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)
- **Flow Matching**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)
- **ShortCut Flow**: [Frans et al., 2024](https://arxiv.org/abs/2410.12557)
- **RLPD**: [Ball et al., ICML 2023](https://arxiv.org/abs/2302.02948)
- **ReinFlow**: [Ding et al., 2024](https://arxiv.org/abs/2402.14262)
- **CPQL**: [Nakamoto et al., ICLR 2024](https://arxiv.org/abs/2310.07297)
- **DSRL**: [Wagen et al., 2024](https://github.com/ajwagen/dsrl)
- **PLD**: [Xiao et al., 2024](https://arxiv.org/abs/2511.00091)
- **DQC**: [Li, Park, Levine, 2025](https://arxiv.org/abs/2512.10926)
- **Cal-QL**: [Nakamoto et al., NeurIPS 2024](https://arxiv.org/abs/2303.05479)
- **IQL**: [Kostrikov et al., ICLR 2022](https://arxiv.org/abs/2110.06169)

---

## 📝 License

MIT License
