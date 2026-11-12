# PRCF 项目目录与实验组织

本文参考 `/home/dai/E1/E1_parkour_mjlab` 的任务组织方式：每个实验变体集中
管理环境配置、算法配置、MDP 和 RL 扩展，通过任务 ID 选择运行。下面保留
PRCF 已创建的目录命名。Mimic 一阶段已实现，操作见 [preprocessing.md](preprocessing.md)；Parkour 和消融目录仍为设计提案。

## 1. 整体目录

```text
PRCF/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .venv/                              # uv 环境，显示名称 PRCF
├── README.md
│
├── src/prcf/
│   ├── __init__.py
│   ├── asset/                          # 沿用现有命名
│   │   ├── robot/
│   │   │   └── e1/
│   │   │       ├── e1_constants.py
│   │   │       └── xmls/               # 现有 MJCF、场景和网格资源
│   │   ├── motion/e1/                  # 当前 E1 动作资源
│   │   │   ├── raw/
│   │   │   ├── csv/
│   │   │   └── npz/                   # NPZ + stairs/up_box/down_box.json
│   │   └── objects/                    # 后续按需增加障碍物资源
│   │
│   ├── motion/                         # 跨任务复用的数据处理逻辑
│   │   ├── schema.py
│   │   ├── validation.py
│   │   ├── recording.py
│   │   ├── export.py
│   │   └── quality.py
│   │
│   └── tasks/
│       ├── __init__.py                 # 加载项目任务注册
│       ├── mimic/                      # 已建目录：物理参考动作跟踪
│       │   ├── __init__.py
│       │   ├── workflow.py             # Mimic 数据准备、runner 与评估导出
│       │   ├── replay.py               # CPU 动作回放、网页场景编辑与保存
│       │   └── config/
│       │       ├── e1/                 # 首个 Mimic 任务配置
│       │       │   ├── __init__.py      # 注册任务
│       │       │   ├── env_cfgs.py      # 环境、跟踪/接触身体集合
│       │       │   ├── rl_cfg.py        # PPO 配置
│       │       │   └── mdp/            # 命令、终止、指标和轨迹记录
│       │       └── e1_reference/       # 参考残差动作 + 双脚世界位置跟踪
│       └── parkour/                    # 已建目录：统一视觉全身越障
│           ├── __init__.py
│           └── config/
│               ├── e1_25dof_baseline/
│               ├── e1_25dof_prcf/
│               ├── e1_25dof_no_memory/
│               ├── e1_25dof_no_reliability/
│               └── e1_25dof_no_contact_feasibility/
│
├── scripts/                            # 统一命令行入口，沿用已建目录
│   ├── train.py
│   ├── play.py
│   ├── evaluate.py
│   ├── list_envs.py
│   ├── validate_motion.py
│   ├── export_reference.py
│   ├── replay_motion.py
│   ├── csv_to_npz_e1.py
│   └── export_policy.py
│
├── datasets/                           # 沿用现有目录，不再另建 data/
│   └── e1/
│       ├── locomotion/                 # 已建动作类别目录
│       ├── stairs/
│       ├── up_box/
│       ├── down_box/
│       └── manifests/                  # 后续：动作、场景、数据划分、参考池清单
│
├── logs/                               # 训练、评估和导出结果，不纳入 Git
│   └── <task_id>/<run_id>/
│       ├── config/
│       ├── checkpoints/
│       ├── logs/
│       ├── evaluation/
│       └── videos/
│
├── deploy/                             # 已建目录，后续部署配置与运行入口
├── tests/                              # 数据约定与关键逻辑验证
└── docs/
    ├── project_structure.md
    ├── motion_format.md                # 已实现的数据约定
    ├── preprocessing.md                # 已实现的 Mimic 流程
    └── experiments.md                  # 后续编写
```

当前已有 E1 资产与数据类别，因此提案以 `e1_25dof` 举例；这不将项目限制为
E1。后续如使用 G1，增加相应资产、数据和任务配置目录，明确其自由度与
关节/身体映射，不混用机器人数据。当前命令行参数解析直接位于 `scripts/`；Mimic 的共用运行流程集中在
`tasks/mimic/workflow.py`，项目根包下不再设置 `cli.py` 或 `mimic.py`。

## 2. 单个任务变体的内部结构

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
| `motion/` | 输入校验、记录、导出与数据质量处理，供多个任务调用 |

例如，接触标签从仿真状态中生成的逻辑属于 `mdp/`；接触预测网络和监督损失
属于 `rl/`。通用 NPZ 校验属于 `motion/validation.py`；AMP 参考池窗口采样
属于任务的 `rl/motion_loader.py`，二者复用格式约定。

算法优先继承当前安装版本 `rsl_rl` 的 `PPO`，runner 和模型优先复用 mjlab /
RSL-RL 已有接口，仅覆盖项目需要的部分。旧项目提供组织参考，具体 API 在
实现时按 PRCF 的锁定依赖适配，不直接复制整套旧 PPO 或修改官方安装包。

## 3. 消融与实验对照

每个正式对照实验使用独立目录和任务 ID。下列名称为拟定名称，尚未注册：

| 实验目录 | 拟定任务 ID | 用途 |
|---|---|---|
| `mimic/config/e1` | `PRCF-Mimic-E1-25DoF` | 物理参考准备 |
| `parkour/config/e1_25dof_baseline` | `PRCF-Parkour-E1-21DoF-Baseline` | 基础多头 AMP 系统 |
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

每次运行将最终生效的环境与算法配置、任务 ID、随机种子、训练步数、数据
清单/划分、机器人与场景版本、依赖锁定信息、代码版本和初始化 checkpoint
写入 `logs/<task_id>/<run_id>/config/`。各对照沿用相同评估场景、指标和
数据划分；训练预算和额外数据使用量也要记录，以便判断方法本身的作用。

## 4. scripts 的使用方式

`train.py` 和 `play.py` 通过任务 ID 处理 Mimic 与 Parkour，通常不再分别创建
`train_mimic.py`、`train_parkour.py`。`evaluate.py` 负责固定场景和随机种子的
批量评估；`play.py` 负责交互查看与录像。

数据处理入口也放在 `scripts/`，调用 `prcf.motion` 中的处理函数。命令行参数与启动逻辑放在对应脚本内；Mimic 数据准备和 runner 等共用流程
位于 `tasks/mimic/workflow.py`，通用数据校验、场景描述和导出仍在 `motion/`。策略导出使用
`export_policy.py`，物理参考轨迹导出使用 `export_reference.py`，明确区分
两类产物。后续形成稳定流程后，如确有需要，再增加批处理入口。

项目打包配置完善后，通过安装的 `prcf` 包导入任务，不依赖其他本地项目的
源码目录。具体命令行参数在实现时确定，本文不提供尚不能运行的命令。

## 5. datasets 与物理参考流程

保留已建的 `locomotion`、`stairs`、`up_box`、`down_box` 类别。每类数据后续
可以按处理阶段细分，例如：

```text
datasets/e1/up_box/
├── retargeted/                         # 原始重定向 NPZ，保留不覆盖
├── rollouts/                           # 实际执行记录，包含成功与失败
└── references/                         # 通过验收的物理参考
```

动作类别目录用于管理文件，多判别器的参考池由 `datasets/e1/manifests/` 中的
清单组织，不要求“一个目录对应一个判别器”。例如，`stairs/` 内可以同时
存在上台阶和下台阶动作，通过元数据分别分配。新增沟壑、撑越等数据时再
增加对应类别。同一物理轨迹可以由多个参考池引用，文件只保存一份。

清单记录输入相对路径、动作族、原始动作来源、机器人配置、采样约定、身体/
关节映射、障碍几何与位姿、坐标对齐、起止状态和成功条件。大体积数据与
运行产物不纳入 Git，清单及实验配置保留版本管理；实施时同步补齐忽略规则。

当前预处理链路保持为：

1. 接收重定向 NPZ，记录来源与校验和。
2. 校验字段、维度、有限值、采样率、身体/关节顺序和四元数/速度约定。
3. 对齐机器人、初始状态及障碍场景，攀箱和跨沟壑分别配置实际支撑几何。
4. 用 Mimic 训练跟踪策略，在独立评估中检验完整动作。
5. 使用选定 checkpoint 实际执行并记录状态、动作、接触与其他物理信息。
6. 按通过结果、跟踪误差、接触有效性、穿透、滑移和执行约束筛选。
7. 导出物理参考，回放复核，并保留来源与质量报告。

“物理参考”表示实际执行并通过明确验收的轨迹，不等于无误差轨迹。
运动学回放或前向运动学计算本身不构成物理跟踪验证。

当前安装版 `MotionLoader` 读取 `joint_pos`、`joint_vel`、`body_pos_w`、
`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`；官方转换器另外写入 `fps`。
加载器按控制步推进数组，未自动依据 `fps` 重采样，因此必须检查参考周期
与环境控制周期一致。

### 导出约定

每条成功轨迹建议保存三个关联文件：

- `<clip>.npz`：mjlab 标准运动字段，保持正确顺序。
- `<clip>.extras.npz`：额外物理状态、接触与局部地形数据。
- `<clip>.json`：格式版本、时间/坐标约定、身体与关节名称、机器人和场景
  校验和、原始动作来源、checkpoint、随机种子及质量指标。

动作、关节目标和实际执行力矩分别保存；原始接触与有效支撑标记分别定义。
各字段明确单位、坐标系与时间戳，并区分控制周期和物理子步。接触力依据
模拟器可读接口记录，不能通过运动学结果冒充。深度图按后续视觉实验需求
独立采样保存，避免默认写入所有参考文件。

## 6. 当前实施范围

已实现 `mimic/config/e1`（E1 25 自由度）、`motion/` 数据处理，以及训练、查看、
评估、导出和运动学回放入口。实际环境文件沿用 `env_cfgs.py` 命名。当前支持
单动作与配套静态 box/stairs 场景、单 GPU PPO。额外记录通过
`mdp/recorders.py` 接入，输入示例位于 `examples/mimic/`。CPU 校验与测试已具备；
真实动作和 GPU 跟踪仍需用实际数据验证。完整方法和消融目录按进度建立。
