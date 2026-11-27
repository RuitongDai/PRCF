# APEX 爬箱技能 · E1 25DoF

这两个任务将 [APEX-humanoid](https://github.com/apex-humanoid/APEX-humanoid) 的单技能训练迁移到 PRCF 的 E1 25DoF 和 mjlab 1.6.0。两个任务分别用 PPO 训练，从仿真场景和 E1 姿态配置初始化。

| 任务 ID | 起点 | 末段目标 |
| --- | --- | --- |
| `PRCF-Parkour-E1-25DoF-ApexClimbUp` | 箱前地面站立，朝向平台 | 全身上箱，在箱顶保持肘膝支撑的低姿态 |
| `PRCF-Parkour-E1-25DoF-ApexClimbDown` | 同一侧箱沿内的箱顶俯卧姿态，头朝箱内 | 倒退下箱，双脚落地后恢复站姿 |

## 配置

- 上箱训练采用实心箱与浮空薄平台两列。下箱采用实心箱。箱顶高 0.55–0.84 m，分十级。训练初始等级为 0–5。回放从第 0 级开始，上箱默认显示两种地形。
- 策略以 50 Hz 输出 25 个关节位置指令，物理步长 5 ms，每回合 7 秒。重置后前 15 个控制步，上箱维持随机化默认站姿，下箱维持本次重置的俯卧关节姿态。两个任务共用 E1 的 MJCF、PD 参数和关节动作尺度。
- Actor 读取六帧本体历史和一帧 26×26 的躯干 yaw 对齐高度图，共 1,162 维。Critic 按原版顺序读取躯干状态历史、关节历史、箱高、回合进度、回合时间及高度图，上箱 1,232 维，下箱 1,226 维。上箱原版 Critic 另有六个保持重置值的进度字段，E1 版保留这些字段。Actor 使用原版逐项均匀噪声，Critic 不加噪声。
- 高度图的射线起点抬高 2 m，观测仍以躯干高度为基准。上箱支持逐回合采样恒定 XYZ 漂移，沿用原版默认零幅度，可通过 `ApexHeightScanCfg.drift_range` 配置。下箱沿用原版关闭漂移。
- 上箱追踪全身高度和向箱内的质心进度，下箱追踪全身降低和向箱外的质心进度。尚未达到对应阶段目标时，当前进度未刷新本回合纪录会受罚。第 4 秒后启用对应的俯卧或站立姿态奖励。
- 上箱成功检查全身越沿、肘膝支撑和低线速度。下箱成功检查双脚在箱外地面承重、骨盆高度、身体直立和低速度。成功用于回合末统计。地形课程另用原版的越沿高度或落地站高条件，成功时以 80% 概率升一级，否则抽取当前及较低等级。满足失败降级条件时降一级。
- 训练启用根姿态和关节重置扰动、躯干质量与质心扰动、静态与动态摩擦、恢复系数和默认关节位置随机化。躯干质量改变后按质量比例调整惯量。下箱保留 1–3 秒间隔的随机质心速度扰动。使用标准 mjlab PPO 与 E1 左右镜像增广。 [APEX 方法](https://arxiv.org/abs/2602.11143)

## 奖励与物理适配

上箱 25 项、下箱 27 项，权重逐项沿用源码。两项技能保留独立的高度与水平进度奖励，上箱保留俯卧关节、肘膝支撑和分组腾空奖励，下箱分别记录站立关节、直立、躯干线速度、躯干角速度和肩部高度五项。

共享项包括存活、失败、停滞、足底压强、接触力与头部碰撞、滑移、关节目标变化、髋和腰偏离、关节限位、关节速度与超速、关节加速度、力矩、功率、横滚与俯仰角速度、躯干加速度及全身刚体线加速度。足底压强以 E1 足底碰撞体的水平截面估算，接触力读取 MuJoCo 合力，刚体加速度取 MuJoCo 空间加速度并转换到各刚体原点。

接触材料保留原版的 64 组随机材料和系数范围。MuJoCo 使用一个切向摩擦系数，这里按几何体的水平速度在静摩擦和动摩擦之间插值，并把恢复系数映射到接触阻尼比。E1 的关节限位、站高与俯卧对齐使用 E1 标定值。

## 训练

从 PRCF 仓库根目录执行。切换 `TASK` 训练另一项技能。任务 ID 自带对应的地形和姿态。

```bash
TASK=PRCF-Parkour-E1-25DoF-ApexClimbUp
# TASK=PRCF-Parkour-E1-25DoF-ApexClimbDown

uv run python scripts/list_envs.py
uv run python scripts/train.py "$TASK" --dry-run --num-envs 1
uv run python scripts/train.py "$TASK" --device cuda:0 --num-envs 4096
```

短训练可用 `--num-envs 2 --max-iterations 1 --device cpu` 检查 PPO 更新。Warp 首次使用某一设备时会编译核心；可用 `WARP_CACHE_PATH` 指定可写缓存目录。训练日志与 checkpoint 保存在 `logs/<任务ID>/<运行目录>/`。TensorBoard 中可查看 `Episode_Metrics/success`、`Episode_Metrics/peak_contact_force` 和课程箱高等级。

## 回放

使用与你训练时相同的任务 ID。

```bash
uv run python scripts/play.py "$TASK" --dry-run
uv run python scripts/play.py "$TASK" --preview-terrain --viewer viser
uv run python scripts/play.py "$TASK" \
  --checkpoint "logs/$TASK/<运行目录>/model_1000.pt" \
  --device cuda:0 --viewer viser
```

回放配置关闭训练扰动，固定初始箱高和起始姿态。可用 `--num-envs 1` 单独查看一个环境。使用当前 Critic 布局保存的 checkpoint 可通过 `scripts/train.py "$TASK" --checkpoint <路径>` 续训。早期 APEX checkpoint 的 Actor 仍可回放，训练请从当前布局重新开始。

## 源码

- `env_cfgs.py` 和 `rl_cfg.py` 组装任务与 PPO。
- `terrains.py`、`poses.py`、`sensors.py` 定义箱体、E1 姿态和测量。
- `mdp/` 实现动作、初始化、观测、进度、奖励、终止和课程。
- `symmetry.py` 负责 E1 关节目标、本体历史和高度图的左右镜像。
