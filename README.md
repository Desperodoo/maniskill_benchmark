# RLFT: Reinforcement Learning and Flow-based Training

<p align="center">
  <b>A robot learning workspace for ManiSkill simulation, real-robot data, and flow/diffusion policies</b>
</p>

RLFT 是一个面向机器人学习实验的工作区，围绕 ManiSkill 仿真、CARM 真机数据与部署流程，提供从模仿学习、离线强化学习到在线强化学习的一套统一训练入口和实验脚本。

当前仓库除了核心 `rlft/` 训练框架外，还包含：

- 真机/ROS 部署代码 (`carm_ros_deploy/`)
- 机械臂控制 SDK (`arm_control_sdk/`)
- 数据采集、benchmark、扫参与分析脚本 (`scripts/`)
- 一个正在集成中的 reward model 子项目 (`rlft/robometer/`)

## 仓库能做什么

支持的算法族包括：

- **Imitation Learning (IL)**
  - Diffusion Policy
  - Flow Matching
  - ShortCut Flow
  - Consistency Flow
  - Reflected Flow
- **Offline RL**
  - CPQL
  - AWCP
  - AW-ShortCut Flow
  - DQC
  - Offline SAC
- **Online RL**
  - RLPD / SAC
  - AWSC
  - DSRL-SAC
  - PLD-SAC
  - ReinFlow

## 仓库结构

```text
.
├── rlft/                  # 核心训练框架：算法、网络、buffer、dataset、env、训练入口
├── scripts/               # 环境安装、扫参、分析、数据处理、benchmark、专题实验脚本
├── carm_ros_deploy/       # ROS/catkin 工作区，含 CARM 部署与 RealSense 相关包
├── arm_control_sdk/       # 机械臂控制 SDK 与 Python 绑定
├── recorded_data/         # 录制数据
├── runs/                  # 训练输出与 checkpoint
├── inference_logs/        # 推理/评测日志
└── README.md
```

### `rlft/` 重点目录

```text
rlft/
├── algorithms/            # IL / Offline RL / Online RL 算法实现
├── networks/              # 编码器、U-Net、Q 网络、Actor
├── buffers/               # replay / rollout / success / SMDP buffer
├── datasets/              # ManiSkill / CARM 数据集加载
├── envs/                  # 环境构建与 wrapper
├── offline/               # 离线训练入口
├── online/                # 在线训练入口
├── utils/                 # checkpoint / scheduler / model factory 等
└── robometer/             # reward model 子项目（独立性较强，仍在集成中）
```

## 关键入口

### 离线训练

- `python -m rlft.offline.train_maniskill`
- `python -m rlft.offline.train_carm`

### 在线训练

- `python -m rlft.online.train_rlpd`
- `python -m rlft.online.train_dsrl`
- `python -m rlft.online.train_pld`
- `python -m rlft.online.train_reinflow`

### 典型脚本

- `scripts/setup/setup_maniskill_env.sh`：安装 ManiSkill 训练环境
- `scripts/setup/download_demos.sh`：下载 demo 数据
- `rlft/scripts/replay_demos.sh`：将原始轨迹 replay 成训练用 RGB / state 数据集
- `scripts/sweep/sweep.sh`：批量 sweep
- `rlft/scripts/run_cascade_sweep.sh`：按依赖阶段执行级联超参数扫描
- `rlft/scripts/run_full_pipeline.sh`：一键执行环境、下载、replay 与训练
- `rlft/scripts/run_all_algorithms.sh`：多 GPU 批量训练 ManiSkill 离线算法
- `rlft/scripts/monitor_training.sh`：监控批量训练日志与 GPU 状态
- `scripts/run_dual_camera_benchmark.sh`：双相机采集 benchmark

## 快速开始

### 1. 配置环境

```bash
bash scripts/setup/setup_maniskill_env.sh
conda activate maniskill
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

`scripts/setup/setup_maniskill_env.sh` 默认会创建 `maniskill` conda 环境，并安装：

- Python 3.10
- PyTorch + CUDA 12.1
- ManiSkill
- diffusers
- wandb
- tensorboard
- h5py / einops / opencv-python / matplotlib 等常用依赖

### 2. 准备数据

对于 ManiSkill demo，通常需要先下载，再进行 replay 预处理：

```bash
bash scripts/setup/download_demos.sh
bash rlft/scripts/replay_demos.sh LiftPegUpright-v1
```

如果你使用的是默认 ManiSkill demo 路径，也可以直接在训练命令中使用默认值；但在多数离线训练流程里，仍建议先完成 replay 预处理。若使用真机数据，则通常使用 `recorded_data/` 下的数据或自定义路径。

## 训练使用说明

### 1. ManiSkill 离线训练

统一入口：

```bash
python -m rlft.offline.train_maniskill \
  --env_id LiftPegUpright-v1 \
  --algorithm flow_matching \
  --obs_mode rgb
```

常见算法示例：

```bash
# Flow Matching
python -m rlft.offline.train_maniskill \
  --env_id LiftPegUpright-v1 \
  --algorithm flow_matching \
  --obs_mode rgb

# ShortCut Flow
python -m rlft.offline.train_maniskill \
  --env_id LiftPegUpright-v1 \
  --algorithm shortcut_flow \
  --obs_mode rgb

# AW-ShortCut Flow
python -m rlft.offline.train_maniskill \
  --env_id LiftPegUpright-v1 \
  --algorithm aw_shortcut_flow \
  --obs_mode rgb
```

`train_maniskill` 支持的算法包括：

- `diffusion_policy`
- `flow_matching`
- `consistency_flow`
- `reflected_flow`
- `shortcut_flow`
- `cpql`
- `awcp`
- `aw_shortcut_flow`
- `sac`
- `dqc`

### 2. CARM 真机离线训练

```bash
python -m rlft.offline.train_carm \
  --demo_path ~/recorded_data/pick_place \
  --algorithm flow_matching
```

适合直接在真机录制数据上训练策略。

### 3. 在线强化学习

#### RLPD / SAC

```bash
python -m rlft.online.train_rlpd \
  --env_id LiftPegUpright-v1 \
  --algorithm sac
```

#### AWSC

```bash
python -m rlft.online.train_rlpd \
  --env_id LiftPegUpright-v1 \
  --algorithm awsc \
  --pretrain_path runs/shortcut_bc/best.pt
```

#### DSRL-SAC

需要一个预训练的 ShortCut Flow checkpoint。DSRL 在冻结策略的 noise space 中运行 SAC，再由 flow policy 解码为真实动作。

```bash
python -m rlft.online.train_dsrl \
  --env_id LiftPegUpright-v1 \
  --checkpoint /path/to/shortcut_flow_best.pt
```

#### PLD-SAC

需要一个预训练的 ShortCut Flow checkpoint。PLD 在 residual action space 中运行 SAC，并支持 Cal-QL critic 预训练。

```bash
python -m rlft.online.train_pld \
  --env_id LiftPegUpright-v1 \
  --checkpoint /path/to/shortcut_flow_best.pt
```

#### ReinFlow

通常从预训练的 Flow Matching checkpoint 出发继续做 on-policy 微调。

```bash
python -m rlft.online.train_reinflow \
  --env_id PushCube-v1 \
  --pretrained_path runs/flow_matching/checkpoint.pt
```

支持的 Online RL 算法总览：

| 算法 | 训练脚本 | Action space | 基座策略 | 说明 |
|------|---------|-------------|---------|------|
| DSRL-SAC | `train_dsrl.py` | noise space | ShortCut Flow | SAC in flow noise space |
| PLD-SAC | `train_pld.py` | residual action | ShortCut Flow | Residual SAC + Cal-QL |
| SAC | `train_rlpd.py --algorithm sac` | raw action | — | SAC + action chunking + demo mixing |
| AWSC | `train_rlpd.py --algorithm awsc` | raw action | ShortCut Flow | Q-weighted ShortCut Flow (RLPD) |
| ReinFlow | `train_reinflow.py` | raw action | Flow Matching | PPO + Flow Matching |

## 训练参数怎么看

最直接的方式是看训练脚本里的 `Args` 定义：

- `rlft/offline/train_maniskill.py`
- `rlft/offline/train_carm.py`
- `rlft/online/train_rlpd.py`
- `rlft/online/train_dsrl.py`
- `rlft/online/train_pld.py`
- `rlft/online/train_reinflow.py`

这些入口基本都采用命令行参数驱动，常见参数包括：

- 环境：`--env_id`, `--obs_mode`, `--control_mode`
- 数据：`--demo_path`, `--num_demos`
- 训练：`--total_iters` / `--total_timesteps`, `--batch_size`
- 记录：`--track`, `--wandb_project_name`, `--capture_video`

### 当前常用默认值

以下数值已经在当前代码里可以直接对应到默认参数，适合作为阅读入口：

- `train_maniskill.py`
  - `--obs_horizon=2`
  - `--act_horizon=8`
  - `--pred_horizon=8`
  - `--num_flow_steps=20`
  - `--ema_decay=0.9995`
  - `--sc_fixed_step_size=0.15`
  - `--sc_num_inference_steps=8`
- `train_rlpd.py`
  - `--online_ratio=0.15`
  - `--lr_actor=1e-4`
  - `--num_qs=10`
  - `--num_min_qs=2`
  - `--awsc_beta=50.0`
  - `--awsc_bc_weight=4.0`
- `train_dsrl.py`
  - `--action_magnitude=2.5`
  - `--gamma=0.95`
  - `--utd_ratio=60`
  - `--target_entropy=-3.5`
  - `--layer_size=2048`
  - `--num_qs=10`
- `train_pld.py`
  - `--action_scale=0.3`
  - `--gamma=0.99`
  - `--utd_ratio=60`
  - `--init_temperature=0.5`
  - `--target_entropy=-3.5`
  - `--online_ratio=1.0`
  - `--layer_size=1024`
  - `--num_qs=5`
  - `--calql_pretrain_steps=1000`

## 实验脚本与工作流

`scripts/` 目录承担了大量实验 orchestration 工作，常见用途包括：

- 环境安装与依赖准备
- 大规模 sweep / ablation 启动
- 数据处理、统计与可视化
- 真机数据采集与诊断
- 相机 benchmark 与同步测试

如果你是第一次看这个仓库，建议优先关注：

- `scripts/setup/`
- `scripts/sweep/`
- `scripts/run_dual_camera_benchmark.sh`

### 级联超参数扫描

仓库中保留了 `rlft/scripts/sweep/run_cascade_sweep.sh` 这套级联 sweep 工作流，用于按算法依赖关系分阶段扫描参数：

```text
阶段 1：flow_matching, diffusion_policy
阶段 2：consistency_flow, shortcut_flow, reflected_flow
阶段 3：cpql, awcp, aw_shortcut_flow
```

常用命令示例：

```bash
# 运行完整级联扫描
bash rlft/scripts/sweep/run_cascade_sweep.sh

# 只运行某一阶段
bash rlft/scripts/sweep/run_cascade_sweep.sh --stage 1

# 只运行某个算法
bash rlft/scripts/sweep/run_cascade_sweep.sh --algorithm awcp

# 只做分析 / 查看状态 / 精细化扫描
bash rlft/scripts/sweep/run_cascade_sweep.sh --analyze
bash rlft/scripts/sweep/run_cascade_sweep.sh --status
bash rlft/scripts/sweep/run_cascade_sweep.sh --fine-sweep
```

脚本还支持 `--parallel` 并行分配 GPU，以及 `--retry-failed` 重跑失败实验。

### 一键全流程与批量训练

如果你想从环境、数据到训练一把跑通，可以使用：

```bash
bash rlft/scripts/run_full_pipeline.sh --quick
```

常见参数包括：

- `--task <TASK_ID>`：指定任务
- `--skip-env`：跳过环境配置
- `--skip-download`：跳过数据下载
- `--skip-replay`：跳过 replay
- `--quick` / `--full`：快速验证或完整训练

如果你已经完成数据准备，也可以直接批量启动 ManiSkill 离线训练：

```bash
bash rlft/scripts/run_all_algorithms.sh --quick --obs-mode rgb --gpus 0,1,2,3
```

`run_all_algorithms.sh` 会按 `trajectory.{obs_mode}.pd_ee_delta_pose.physx_cuda.h5` 自动寻找 replay 后的数据，并把日志写到 `logs/training_*`。

查看批量训练状态：

```bash
bash rlft/scripts/monitor_training.sh logs/training_latest
```

### 双相机 benchmark

如果你在调试真机采集链路，可以直接运行：

```bash
bash scripts/run_dual_camera_benchmark.sh
```

这个脚本会：

- 读取 wrist / third-person 相机序列号
- 扫描多个 FPS 配置
- 设置同步窗口与压力录制频率
- 调用 `scripts/benchmark_dual_camera_fps.py` 输出推荐的稳定采集设置

可通过环境变量覆盖默认参数：

```bash
WRIST_SERIAL=218622279840 \
THIRD_SERIAL=037522250003 \
FPS_LIST=15,30,45,60 \
SYNC_SLOP=0.05 \
RECORD_FREQ=120 \
DURATION=20 \
SETTLE=6 \
bash scripts/run_dual_camera_benchmark.sh
```

## 日志与输出

仓库中的运行产物主要在以下目录：

- `runs/`：训练日志、checkpoint、TensorBoard 输出
- `recorded_data/`：录制数据
- `inference_logs/`：推理或评测结果

查看训练日志：

```bash
tensorboard --logdir runs/
```

## 真机/ROS 相关部分

如果你关注真实机器人链路，建议重点看：

- `carm_ros_deploy/`：ROS 工作区
- `arm_control_sdk/`：机械臂控制 SDK
- `scripts/setup/setup_carm_env.sh`：CARM 环境配置脚本
- `scripts/setup/build_catkin.sh`：catkin 构建脚本

## Robometer 子项目

`rlft/robometer/` 是一个相对独立的 reward model 项目，提供：

- reward / progress / preference 建模
- 独立训练入口 `rlft/robometer/train.py`
- 数据转换与上传工具
- benchmark eval 流程

它和主 RLFT 训练框架不完全耦合，当前更适合作为单独子模块理解。

## 建议的阅读顺序

如果你想快速理解仓库，建议按这个顺序：

1. `rlft/offline/train_maniskill.py`
2. `rlft/online/train_rlpd.py`
3. `rlft/datasets/` 和 `rlft/envs/`
4. `scripts/setup/` 和 `scripts/sweep/`
5. 如果涉及真机，再看 `carm_ros_deploy/` 和 `arm_control_sdk/`

## License

MIT License
