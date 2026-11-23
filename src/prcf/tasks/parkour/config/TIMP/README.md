# TIMP 越障基线

任务 ID 为 `PRCF-Parkour-E1-25DoF-TIMP`。现有 Mimic 负责 Stage 1 的物理参考准备。TIMP 在 E1 25DoF 和 mjlab 上实现 Stage 2 地形索引 AMP，以及 Stage 3 几何支撑适应。训练循环沿用 mjlab Runner 和 RSL-RL PPO。

## 训练与回放

在 PRCF 项目根目录执行。训练、策略回放和静态预览共用 `e1_amp_parkour_terrain_cfg()`，默认启用配置里的全部八种地形、十行难度。

```bash
uv run python scripts/train.py PRCF-Parkour-E1-25DoF-TIMP --stage 2 --device cuda:0
```

用 Stage 2 checkpoint 开始支撑适应。Stage 3 默认 10,000 次更新。

```bash
uv run python scripts/train.py PRCF-Parkour-E1-25DoF-TIMP --stage 3 --device cuda:0 \
  --checkpoint logs/PRCF-Parkour-E1-25DoF-TIMP/<run>/model_<iteration>.pt
```

```bash
uv run python scripts/play.py PRCF-Parkour-E1-25DoF-TIMP --checkpoint logs/PRCF-Parkour-E1-25DoF-TIMP/<run>/model_<iteration>.pt
```

`--terrains` 可选择要运行的地形，例如 `--terrains high_box long_high_box`。`--num-envs` 和 `--max-iterations` 可覆盖默认规模。`--reference-pools` 可替换清单，训练目录中的 `config/reference_pools.json` 保存本次配置，`source.json` 记录阶段、地形和启用的判别头。`--dry-run` 检查参考数据并编译场景。回放使用正常起点，关闭参考状态初始化。

### 静态地形预览

用以下命令查看与训练相同的地形网格。加 `--terrains` 可选择要看的地形，加 `--viewer viser` 可用浏览器打开 `http://127.0.0.1:8080`。预览只编译静态地形，直接在 CPU 上运行。

```bash
uv run python scripts/play.py PRCF-Parkour-E1-25DoF-TIMP --preview-terrain
```

两个高箱共用 `box_height_range=(0.70, 1.0)`。`long_high_box` 沿 +X 长 `2.40 m`、横向宽 `1.20 m`，与 `high_box` 的前缘对齐。原 `--terrain-layout copied` 预览命令也指向同一布局。

地形生成器的 `curriculum=True` 用于把地形类型排列成列、把几何难度排列成行。并行环境从这些拼块中分配出生点，采样汇总到同一个策略。目前地形等级在初始化时分配，训练还没有根据通过率升降难度的课程逻辑。

## 地形与参考池

参考池为每个判别头提供正例窗口。在线策略窗口按训练期地形路由。Actor 只使用部署侧观测。地形标签、高度扫描真值、力矩和接触通道用于训练。

| 场景 | AMP 头 | 物理参考 |
|---|---|---|
| `perlin_rough`、`square_gaps`、`stairs_up` | `locomotion` | `locomotion.npz` |
| `stairs_up_high`、`stairs_down`、`stairs_down_high` | `stairs` | 共享 `stairs.npz` 第 `[326, 477)` 行 |
| `high_box`、`long_high_box` | `box` | 两种地形全程共用一个头。`up_box.npz`、`down_box.npz` 各占一半采样概率。 |

`motion_dir` 相对项目根目录，`file` 相对 `motion_dir`。按条目权重选轨迹，再在同一轨迹的左闭右开 `frame_range` 内采连续四帧。高箱的策略窗口可以连续覆盖上下箱过渡，回合重置时清空。四份参考来自原采集场景，作为动作先验使用。上、下箱参考状态初始化限于箱高与参考相差不超过 `1.5 cm` 的难度行，按清单里的 `reset_box_edge` 分别对齐前缘 (`front`) 和后缘 (`rear`)。其余场景使用生成器提供的正常起点。

继续训练和 Stage 3 使用当前三头配置生成的 checkpoint。旧版 checkpoint 可以通过 `play.py --checkpoint` 回放 Actor。

## 论文参数与移植参数

| 模块 | 当前配置 |
|---|---|
| Actor | 五帧 84 维本体历史。当前帧拼接 32 维深度特征、HIM 的 3 维速度估计与 16 维隐变量，形成 135 维输入。四个门控专家输出 25 个关节目标。 |
| 深度 | pelvis 相机向下倾斜 60°，24×24，深度裁剪为 0.3–4 m。采用 mjlab 针孔射线，垂直视场设为 60°。 |
| Critic | 四个门控专家。输入当前本体状态、真值速度、八类物理地形标签和扫描。 |
| HIM | 速度回归与下一帧的交换原型预测，独立优化器。历史网络宽度为 512、256、128，采用 32 个原型。 |
| AMP | 每帧 357 维，窗口 4 帧。231→64→32→16 扫描编码，拼接后主干为 568→512→256，按自定义地形使用独立输出头。50,000 条回放窗口容量按已激活的头分配。 |
| PPO | 4,096 环境，24 步 rollout，5 个 epoch，4 个 mini-batch。学习率 0.001、自适应 KL 0.01、熵系数 0.01、裁剪 0.2、折扣 0.99、GAE 0.95。 |
| 判别器 | Adam 学习率 5e-6，编码器/主干与各输出头的权重衰减分别为 0.001 和 0.1。实现论文的均值对抗损失、六次梯度范数正则和风格奖励。 |
| 任务奖励 | 速度跟踪、动作变化、关节力矩、关节限位和躯干碰触。50 Hz 控制，任务权重按论文整体乘 0.5，时间步缩放由 mjlab 负责。 |
| 支撑奖励 | Stage 3 使用两个实心高箱的实际顶面尺寸和档位高度。按手膝支撑面积、边缘余量及可达性筛候选，加入接触前引导、实际接触、保持和释放等项。96,000 控制步内支撑系数从 0.1 升至 1，AMP 系数从 1 降至 0.5。 |

原文训练场景为开口框架，本文配置使用实心箱、台阶、沟壑等自定义地形。Actor/Critic 隐层宽度设为 256、128、64，门控隐藏层为 128。相机内参、普通任务奖励的具体权重和实心箱的支撑阶段边界是本移植的配置。训练中的 `traversal_success` 要求走出指定障碍区域、在行进通道内完成越障并保持一秒稳定。正式论文对比仍需固定场景、随机种子和预算后单独统计成功率。

TensorBoard 的 `combined_reward` 包含 AMP，`amp_window_ready` 是本轮实际获得 AMP 奖励的样本比例。Runner 自带的 `Mean reward` 统计环境奖励。
