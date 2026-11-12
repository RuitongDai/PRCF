# Mimic 数据与时间约定

## 输入与输出

输入为当前 E1 25 自由度模型对应的 mjlab NPZ，或引用该 NPZ 的 `clip.json`。
JSON 版本为 1，示例见 `examples/mimic/`。字段未知、机器人不符、非米制/
Z-up/wxyz、静态场景类型不支持等情况直接报错，避免静默忽略配置。

| NPZ 字段 | 形状 | 含义 |
|---|---|---|
| `fps` | `(1,)` | 控制采样频率 Hz |
| `joint_pos` | `(N,25)` | 关节角 rad，E1 XML 顺序 |
| `joint_vel` | `(N,25)` | 关节角速度 rad/s |
| `body_pos_w` | `(N,B,3)` | 全部机器人身体原点位置，m |
| `body_quat_w` | `(N,B,4)` | 身体原点姿态，wxyz 单位四元数 |
| `body_lin_vel_w` | `(N,B,3)` | 身体原点线速度，世界坐标，m/s |
| `body_ang_vel_w` | `(N,B,3)` | 世界坐标角速度，rad/s |
| `joint_names` | `(25,)` | 推荐提供的字符串名称，导出必带 |
| `body_names` | `(B,)` | 推荐提供的字符串名称，导出必带 |

`B` 由当前模型决定，排除 world，包括固定头部等所有机器人身体；不包含
障碍物。第一项为 pelvis。不是只导出 tracking 选中的 14 个关键身体。
姿态/速度采用 link 原点定义，不能混成质心速度。旧输入没有名称时，通过
抽样 FK 检查提高兼容性检查力度，但仍建议保留源模型与名称映射。

`motion_to_scene` 对全部身体的位置、姿态和世界系速度应用一致刚体变换。
关节角不变。导出参考已位于场景坐标，不再重复应用源变换；源变换保留在
`metadata` 中作为来源记录。世界坐标记录去掉并行环境原点。

## 扩展记录

有 N 个状态样本，间隔为 `1/fps`，对应 N-1 个控制区间。`extras.npz` 包含：

- `time`、`reference_frame`：状态采样时间和参考帧。
- `action`、`joint_position_target`：作用于前一区间的动作/目标；首帧的
  `action_valid=False`，不可用作已执行动作。
- `qfrc_actuator_snapshot`：状态刷新后的关节驱动力快照，可能已重新计算。
- `qfrc_actuator_substep`：`(N-1,D,25)`，实际物理步计算的执行器关节驱动力；
  D 为 decimation，当前为 4。与 `physics_substep_time` 一一对应。不能用
  快照替代历史施加的力矩。此字段不包含所有被动/约束广义力。
- `contact_parts`：左右足、左右手、左右膝的固定通道顺序。
- `contact_found`、`strongest_contact_force_w`、`contact_position_w`、
  `contact_normal_w`、`contact_point_velocity_w`：控制频率下的环境接触。
- 对应接触字段的 `*_substep`：`(N-1,D,...)`，物理子步记录。
- `min_environment_distance`、`min_self_distance` 及其子步版本：最近接触
  距离；没有匹配接触时记为零，用于检查负距离（穿透）。

接触只统计指定身体与 world 静态场景之间的接触，自碰撞单独检查。每部位
选择最强的一处接触，力在世界系，**不是整个足底或手掌的合力**。接触点是
模拟器返回的几何接触位置，不是关节中心，也不保证等于掌心 site。
子步的力和接触对应求解该物理步的状态；状态样本在步后 forward/sense 刷新
后采集，时间区间和状态轴分别保存。

训练奖励、参考姿态、运动学碰撞结果均不直接充当“有效支撑”真值。当前
验收中的支撑使用真实接触和力阈值，具体阈值及区间写入质量报告。后续
接触预测训练可从这些实际接触记录生成未来接触标签。
