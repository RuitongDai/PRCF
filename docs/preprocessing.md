# E1 Mimic 数据预处理

任务包括基线 `PRCF-Mimic-E1-25DoF` 和参考残差
`PRCF-Mimic-E1-25DoF-Reference`。两者分别注册在 `tasks/mimic/config/e1`
和 `tasks/mimic/config/e1_reference`，使用官方 mjlab 的 PPO runner。
参考残差任务把参考关节角用作位置目标基准，并增加双脚世界位置奖励。
加载既有 checkpoint 时使用训练时的任务 ID。

## 输入动作与场景

原数据放在 `src/prcf/asset/motion/e1/`。重定向 CSV 通过
`scripts/csv_to_npz_e1.py` 生成 mjlab NPZ。输入记录包含根部位姿和按 E1 XML
顺序排列的 25 个关节角。使用真实输入帧率与正确的四元数顺序。

```bash
uv run python scripts/csv_to_npz_e1.py \
  --input-file src/prcf/asset/motion/e1/csv/114_09.csv \
  --output-file /tmp/prcf_stairs.npz \
  --input-fps 50 --output-fps 50 --quaternion-order xyzw
```

平地可以传 `017.npz`，台阶、上箱、下箱分别传场景 JSON。JSON 中的
`motion_file` 相对于自身目录解析。固定箱体用完整尺寸和中心位置描述。
双向台阶配置由多个箱体组成。用网页回放查看参考姿态、调整障碍物及保存 JSON。

```bash
uv run python scripts/replay_motion.py --motion-file src/prcf/asset/motion/e1/npz
```

网页回放地址由终端打印。可通过 `--viewer native` 使用 MuJoCo 原生窗口，
也可以用 `--headless` 在 CPU 上逐帧走过指定的单个参考文件。

## 训练与策略回放

先用 `train.py --dry-run` 检查场景是否编译，再正式训练。训练输出目录为
`logs/<任务 ID>/<运行 ID>/`。源轨迹与静态场景在脚本中对齐，训练使用对齐后的
参考副本。所有并行 env 使用同一条动作和相同的场景，各自采样参考开始帧。

```bash
PRCF_TASK=PRCF-Mimic-E1-25DoF-Reference
PRCF_MOTION=src/prcf/asset/motion/e1/npz/stairs.json

uv run python scripts/train.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" --num-envs 4096 --device cuda:0

PRCF_CKPT=logs/PRCF-Mimic-E1-25DoF-Reference/stairs/model_6000.pt

uv run python scripts/play.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" --checkpoint "$PRCF_CKPT" --viewer viser
```

以上 checkpoint 路径是台阶训练目录的示例。替换成对应实验的 checkpoint。
`play.py` 默认也支持 `--viewer native`，使用临时文件准备参考。

## 导出

```bash
uv run python scripts/export_reference.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" --checkpoint "$PRCF_CKPT" \
  --output-file logs/exports/stairs.npz --device cuda:0
```

每次 rollout 生成一个 NPZ，包含八类特征及拼接的 `amp_obs`。脚本从参考第 0 帧
开始执行，记录每个实际完成的控制周期。碰到终止条件时保留已执行的帧，结束原因
写在 `termination`。默认一条轨迹，另可加 `--attempts` 导出多个独立文件。
已有的输出路径需要换名。采样、坐标、力矩和高度扫描规则见 [数据格式](motion_format.md)。

[训练诊断](training_diagnosis.md) 记录此前 Mimic 配置调整的原因。
