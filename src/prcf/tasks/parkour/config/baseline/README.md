# E1 25DoF Baseline

任务 ID 为 `PRCF-Parkour-E1-25DoF-Baseline`。参考 `E1_parkour_mjlab` 的 `e1_21dof_amp_vdwl_periodic_camera_10hz_2f_hybrid_nav`，适配 PRCF 的 E1 25DoF、现有物理参考和三类 AMP 头。

## 训练与回放

在 PRCF 根目录执行。

```bash
uv run python scripts/train_baseline.py --device cuda:0 --num-envs 4096
```

训练保存到 `logs/PRCF-Parkour-E1-25DoF-Baseline/<run>/`，包含配置快照、参考池清单、TensorBoard 和 checkpoint。`--max-iterations` 设置本次追加更新次数，默认 50,000。续训会加载 Actor、Critic、优化器和 AMP 判别器状态。

```bash
uv run python scripts/train_baseline.py --device cuda:0 --num-envs 4096 \
  --checkpoint logs/PRCF-Parkour-E1-25DoF-Baseline/<run>/model_1000.pt \
  --max-iterations 5000
```

```bash
uv run python scripts/play_baseline.py --device cuda:0 \
  --checkpoint logs/PRCF-Parkour-E1-25DoF-Baseline/<run>/model_1000.pt
```

`--viewer viser` 使用网页 viewer。回放默认八个环境，支持 `--num-envs`。`--terrains high_box long_high_box` 可选择地形子集，训练与回放都支持。回放和 dry-run 使用临时仿真状态。

```bash
uv run python scripts/play_baseline.py --preview-terrain --viewer viser
uv run python scripts/train_baseline.py --dry-run --num-envs 8
```

## 文件组织

- `env_cfgs.py` 定义机器人、相机、观测、奖励、随机化和课程。
- `env.py`、`sensors.py` 管理真实10 Hz相机采集，控制频率50 Hz。
- `terrains.py` 是从当前 PRCF 配置复制的独立 `e1_amp_parkour_terrain_cfg`，含全部八类地形和十行难度。
- `mdp/commands.py`、`stage2_commands.py` 保留参考任务的导航与台阶 zero-stop。
- `mdp/observations.py` 处理本体、深度历史和估计器监督目标。
- `mdp/amp.py` 计算与 PRCF NPZ 对齐的357维 AMP 特征。
- `rl/models.py` 为25关节适配 VDWL，`rl/algorithm.py` 扩展官方 PPO。
- `reference_pools.json` 配置各头的专家文件和采样权重。

## 观测和网络

| 模块 | 当前配置 |
|---|---|
| 本体输入 | 步态时钟2、IMU角速度3、重力方向3、命令3、关节位置25、关节速度25、上一动作25，共86维 |
| 历史 | Actor 15帧，1290维。策略使用最近5帧，加51维融合特征，最终481维 |
| 深度 | 64×36采集，裁剪为32×18。相邻两次真实采集，2×18×32，间隔0.1秒 |
| 深度编码 | 1152→512→256→128 |
| 本体编码 | 1290→256→128→64 |
| 融合编码 | 192→256→128→51，包含速度3、观测重建隐变量16、高度重建隐变量32 |
| 策略 | 481→512→256→128→25 |
| Critic | 五帧93维观测和273维高度扫描，共738维，隐藏层512、256、128 |
| AMP | 每帧357维，连续四帧，1428→1024→512→256共享主干和三个独立输出头 |

AMP 使用现有八组特征，依次为 `dof_pos`、`body_pos_local`、`root_lin_vel`、`root_ang_vel`、`dof_vel`、`contact_state`、`dof_torque`、`height_scan`。高度扫描21×11，接触力阈值5 N，驱动力矩取四个物理子步平均。

| 地形 | AMP头 | 参考池 |
|---|---|---|
| perlin_rough、square_gaps、stairs_up | locomotion | locomotion.npz |
| stairs_up_high、stairs_down、stairs_down_high | stairs | stairs.npz 第[326,477)行 |
| high_box、long_high_box | box | up_box.npz、down_box.npz 等概率采样 |

所有专家窗口均在单条文件内连续采样。策略窗口在回合结束、重置和头变化时重新积累。两个箱体全程使用box头。判别器沿用参考任务的最小二乘损失、专家梯度惩罚10、权重衰减和风格奖励，任务与风格奖励按0.9和0.1混合。

## 对参考任务的适配

保留原任务的10 Hz相机、两帧深度、15帧本体历史、VDWL辅助监督、左右镜像损失0.3、初始学习率1e-5和自适应KL。动作初始标准差1.0、下限0.05，沿用参考任务的动作接口。机器人使用 PRCF 的全身碰撞和执行器参数。

多头AMP替换原来的66维两帧条件判别器，使用现有的357维四帧参考。每头回放分别收集样本，总容量100,000。每次判别器更新对活跃头的损失取均值，每头采样256个窗口。

台阶保留原来的斜向、下楼后退、25%回合zero-stop和对应课程。箱体导航目标在各自后沿后0.5 m，采用正向中央路线。箱体列关闭直立双足专用的姿态、步态和落脚引导奖励，允许手膝等肢体接触，骨盆和躯干接触仍触发失败，箱体姿态终止角为1.77 rad。其他地形使用原来的1.0 rad。没有参考状态初始化事件。

地形初始等级训练为0～5，回放从0开始。课程会根据跟踪、行进距离和路线完成情况升降等级。改动地形时请检查箱体目标区域和地图尺寸，默认普通箱和长箱后沿分别为2.73 m和3.93 m。

## 验证范围

CPU上验证原生场景编译、八列仿真、一次完整PPO/AMP/VDWL更新、checkpoint续训和推理。相机sample_id每五个策略步更新，环境局部reset保持其他环境路线，课程能够提升全部八类地形。正式训练的站立及越障成功率需要另外评估。
