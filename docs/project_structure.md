# PRCF 项目目录与实验组织

本文参考 `/home/dai/E1/E1_parkour_mjlab` 的任务组织方式：每个实验变体集中
管理环境配置、算法配置、MDP 和 RL 扩展，通过任务 ID 选择运行。下面保留
PRCF 已创建的目录命名。Mimic 一阶段已实现，操作见 [preprocessing.md](preprocessing.md)。Parkour 的 TIMP 基线已接入训练和回放，操作见 [TIMP 说明](../src/prcf/tasks/parkour/config/TIMP/README.md)。

## 1. 当前目录

```text
PRCF/
├── scripts/
│   ├── train.py                 # 训练、准备环境、创建 runner
│   ├── play.py                  # 加载 checkpoint 看策略
│   ├── replay_motion.py         # 回放输入动作、编辑障碍物
│   ├── export_reference.py      # 执行 checkpoint，写出单个 AMP NPZ
│   ├── csv_to_npz_e1.py          # 重定向 CSV 转 mjlab NPZ
│   └── list_envs.py              # 列出任务 ID
├── src/prcf/
│   ├── asset/
│   │   ├── scene.py              # 脚本共用的场景构建与输入读取、对齐
│   │   ├── motion/e1/            # raw、csv、npz 和配套场景 JSON
│   │   └── robot/e1/             # E1 模型和执行器配置
│   └── tasks/
│       ├── mimic/config/
│       │   ├── e1/               # 基线环境、PPO、MDP 和任务注册
│       │   └── e1_reference/     # 参考残差动作和脚部跟踪奖励
│       └── parkour/config/TIMP/  # 地形索引 AMP 与几何支撑适应基线
├── logs/
│   ├── <task>/<run>/             # 模型、TensorBoard、训练配置
│   └── exports/                 # 导出的 AMP NPZ
├── tests/
└── docs/
```

Mimic 的任务注册、环境和奖励继续放在 `src/prcf/tasks/mimic/config/`。
`play.py` 与 `export_reference.py` 复用 `train.py` 的环境准备和 runner 创建，
三者共用 `asset/scene.py` 加载动作和场景。数据采集和列拼接直接在
`export_reference.py` 完成。

## 2. 单个任务变体的内部结构

TIMP 基线集中在 [`parkour/config/TIMP/`](../src/prcf/tasks/parkour/config/TIMP/README.md)。
`env_cfgs.py`、`rl_cfg.py` 和 `terrains.py` 组装实验，`mdp/` 管理环境逻辑，`rl/`
管理 AMP 和 PPO 扩展。`reference_pools.json` 维护各 AMP 头对应的物理参考数据。

以完整 PRCF 实验为例，任务相关内容集中在一个目录：

```text
tasks/parkour/config/e1_25dof_prcf/
├── __init__.py                         # 注册任务 ID、环境、算法配置与 runner
├── README.md                           # 方法说明、相对基线的改动及运行方法
├── env_cfgs.py                         # 环境、观测、动作、奖励、事件和终止配置
├── rl_cfg.py                           # Actor/Critic、PPO、AMP、辅助学习与 runner 配置
├── terrains.py                         # 地形组合、障碍采样与课程参数
├── sensors.py                          # 深度、局部几何与接触传感器配置
├── symmetry.py                         # 需要对称增强时才增加
├── env.py                              # 标准 manager 环境不能满足需求时才增加
│
├── mdp/
│   ├── __init__.py
│   ├── commands.py                     # 速度/目标命令、任务进度与训练期路由
│   ├── observations.py                 # Actor/Critic/AMP 观测与监督信号
│   ├── rewards.py                      # 任务、接触与可靠性相关环境奖励
│   ├── terminations.py                 # 失败、成功和超时条件
│   ├── events.py                       # 初始化、扰动与随机化
│   ├── curricula.py                    # 地形、遮挡、支撑区域等课程
│   └── metrics.py                      # 通过效率、接触质量与失败原因
│
└── rl/
    ├── __init__.py
    ├── algorithm.py                    # 继承 PPO，增加 AMP 与辅助优化逻辑
    ├── runner.py                       # 采样、训练调度、记录与 checkpoint 扩展
    ├── models.py                       # Actor/Critic 组合与模型接口
    ├── discriminator.py                # 多头 AMP 判别器及损失
    ├── replay_buffer.py                # AMP 与辅助任务所需回放数据
    ├── motion_loader.py                # 参考池采样与运动窗口构造
    ├── normalizer.py                   # 需要自定义时才增加
    ├── geometry_memory.py              # 运动补偿、历史几何与可靠性表示
    ├── contact_predictor.py            # 统一足、手、膝接触计划
    └── contact_feasibility.py          # 接触执行结果/可行性评价
```

这是完整方法的展开示例。每个变体只创建实际需要的文件；普通 PPO Mimic
任务可以直接使用官方算法与网络，不要求先实现自定义判别器或 runner。

### 配置与实现的职责

| 位置 | 负责内容 |
|---|---|
| `env_cfgs.py` | 组装仿真场景和 MDP，决定策略可见信息与训练期特权信息 |
| `rl_cfg.py` | 选择模型、算法和 runner，配置优化参数及辅助损失权重 |
| `mdp/` | 与环境状态、奖励、命令、接触测量和终止条件相关的逻辑 |
| `rl/` | 网络前向计算、损失、优化、参考采样与学习过程 |
| `scripts/` | 解析参数、选任务、启动运行、保存或读取结果 |
| `asset/scene.py` | 从场景 JSON 构建几何并读取、对齐动作 |

例如，接触标签从仿真状态中生成的逻辑属于 `mdp/`；接触预测网络和监督损失
属于 `rl/`。AMP 参考池窗口采样可放在后续任务的 `rl/motion_loader.py`，复用导出的字段与列顺序。

算法优先继承当前安装版本 `rsl_rl` 的 `PPO`，runner 和模型优先复用 mjlab /
RSL-RL 已有接口，仅覆盖项目需要的部分。旧项目提供组织参考，具体 API 在
实现时按 PRCF 的锁定依赖适配，不直接复制整套旧 PPO 或修改官方安装包。

## 3. 消融与实验对照

每个正式对照实验使用独立目录和任务 ID。Mimic 和 TIMP 已注册，PRCF 完整方法与消融为拟定项。

| 实验目录 | 拟定任务 ID | 用途 |
|---|---|---|
| `mimic/config/e1` | `PRCF-Mimic-E1-25DoF` | 物理参考准备 |
| `parkour/config/TIMP` | `PRCF-Parkour-E1-25DoF-TIMP` | 地形索引 AMP 与几何支撑适应基线 |
| `parkour/config/e1_25dof_prcf` | `PRCF-Parkour-E1-21DoF-Full` | 完整可靠感知与接触方法 |
| `parkour/config/e1_25dof_no_memory` | `PRCF-Parkour-E1-21DoF-NoMemory` | 移除历史几何，保留明确的当前帧输入方案 |
| `parkour/config/e1_25dof_no_reliability` | `PRCF-Parkour-E1-21DoF-NoReliability` | 保留记忆，移除可靠性信息及其决策作用 |
| `parkour/config/e1_25dof_no_contact_feasibility` | `PRCF-Parkour-E1-21DoF-NoContactFeasibility` | 保留接触预测，移除执行结果可行性评价 |

任务 ID 同时绑定环境配置、算法配置和 runner，`train.py` 与 `play.py` 共用
注册入口，避免训练和评估误选不同的实现。前述消融是实验组织示例，具体
对照定义随最终算法明确后固定，并记录于每个变体的 `README.md`。

### 复用方式

- 基线保留通用 MDP 和 PPO/AMP 实现，完整方法在其上扩展。
- 消融从完整方法配置工厂获取新的配置实例，只修改相关观测、奖励、模型、
  损失和权重；运行配置不得直接修改被其他任务共享的可变对象。
- 只改变参数时，变体保留 `__init__.py`、`env_cfgs.py`、`rl_cfg.py` 和说明；
  改变奖励或算法时，再增加本地 `mdp/rewards.py` 或 `rl/algorithm.py` 等。
- 本地实现通过显式导入、重导出或继承复用基线，避免每项消融复制整套代码。
  修改观测或网络时，同时核对维度、归一化和 checkpoint 兼容性。
- 开发期的小范围参数试验使用命令行覆盖；固定为正式对照后再注册任务变体。

这样可以直接查看实验目录理解差异，也能通过配置与少量覆盖文件检查改动。
后续如多个任务确有稳定共用实现，再提取共享模块，当前不预先建立庞大的
顶层 `learning/`、`perception/`、`contacts/`；这些方法先由任务内 `rl/` 管理。

### 对照结果保存

训练结果位于 `logs/<task_id>/<run_id>/`，包含模型权重、TensorBoard 和
`config/`。导出 NPZ 位于 `logs/exports/`，也可用 `--output-file` 指定路径。
各对照应记录运行配置、参考池划分及训练预算。

## 4. 动作数据与参考池

原始、重定向和仿真输入统一放在 `src/prcf/asset/motion/e1/`。使用
`export_reference.py` 让 checkpoint 从第 0 帧执行并提取八项 AMP 特征。
每次执行得到一个 NPZ，可在后续训练的参考池清单里按动作族或地形引用。
详细操作见 [预处理说明](preprocessing.md)，字段见 [数据格式](motion_format.md)。
