# Mimic 与 AMP 数据格式

## Mimic 输入

`--motion-file` 接受 E1 25 自由度轨迹 NPZ 和引用该 NPZ 的场景 JSON。NPZ 使用默认
平地。场景 JSON 的 `motion_file` 相对 JSON 目录解析。场景描述、障碍物和轨迹对齐
分别放在 `scene` 与 `motion_to_scene`。示例见 [examples/mimic](../examples/mimic/)。

| Mimic 输入字段 | 形状 | 含义 |
|---|---|---|
| `fps` | `(1,)` | 控制采样频率，Hz |
| `joint_pos` / `joint_vel` | `(N, 25)` | E1 XML 顺序的关节角与速度，rad / rad/s |
| `body_pos_w` | `(N, B, 3)` | 身体原点世界位置，m |
| `body_quat_w` | `(N, B, 4)` | 身体世界姿态，wxyz |
| `body_lin_vel_w` / `body_ang_vel_w` | `(N, B, 3)` | 身体原点世界线速度与角速度，m/s / rad/s |
| `joint_names` / `body_names` | `(25,)` / `(B,)` | 转换器写出的关节和身体名称 |

`B` 由 E1 模型决定，排除 world，第一项为 pelvis。`motion_to_scene` 一致变换世界
位置、姿态和速度，保留关节角及关节速度。

## AMP 判别器数据

`scripts/export_reference.py` 默认每次 rollout 输出一个 NPZ。文件包含八个具名
数组和拼接后的 `float32` 矩阵 `amp_obs`。`feature_names` 与 `feature_offsets`
记录列顺序和列边界。

| 字段 | 每帧维度 | 定义 |
|---|---:|---|
| `dof_pos` | 25 | E1 关节角 |
| `body_pos_local` | 13 × 3 | 跟踪身体原点相对 pelvis 的位置，仅消除 yaw |
| `root_lin_vel` | 3 | pelvis 原点线速度，转到 yaw 对齐坐标 |
| `root_ang_vel` | 3 | pelvis 角速度，转到 yaw 对齐坐标 |
| `dof_vel` | 25 | 关节角速度 |
| `contact_state` | 6 | 左右足、手、膝的接触标志，且最强环境接触力模长至少 5 N |
| `dof_torque` | 25 | 前一控制区间内执行器关节力矩的物理子步均值 |
| `height_scan` | 231 | 当前 pelvis 高度减去射线命中面高度 |

每帧 357 维。初始化帧缺少实际施加的力矩，因此从原始第 1 帧开始记录，完整执行
`N` 帧参考时输出 `N-1` 行。`reference_frame` 和 `time` 记录对应帧和时间。
`termination` 记录结束原因，`expected_frames` 是完整执行预期的输出行数。
提前结束时保存实际执行的行。

扫描使用 `scene_json` 所描述的静态碰撞几何。网格随 pelvis 平移并按 yaw 旋转，X
范围 `[-0.5, 1.5]` 米、Y 范围 `[-0.5, 0.5]` 米、间距 0.1 米，Y 索引变化更快。
射线从 pelvis 上方 2 米向下发射，最大范围 5 米。无命中编码为 5 米，同时通过
`height_hit` 标记。在线策略样本需要沿用相同列顺序、坐标、采样间隔、接触定义和
射线设置。训练时按单条轨迹的连续帧切窗。

## 回放导出轨迹

同一个 AMP NPZ 还包含 `root_pos_w`（`N×3`，米）和 `root_quat_w`（`N×4`，
`wxyz`）。这两项记录 pelvis 在场景坐标系中的实际位置和姿态，与 `dof_pos`
及 `amp_obs` 逐行对齐。回放脚本依据 `scene_json` 恢复障碍物，并用根部位姿和
关节角重建每帧姿态。训练端继续读取 `amp_obs`，列顺序和维度见上表。
