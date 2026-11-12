# PRCF

**Perception-Reliability- and Contact-Feasibility-Aware Whole-Body Control for Humanoid Locomotion in Complex Terrain**

面向复杂地形的人形机器人感知可靠性与接触可行性驱动的全身运动控制研究。

当前已完成 E1 25 自由度机器人的数据处理流程：重定向 CSV 转换、动作与场景配对、Mimic 跟踪训练、策略回放，以及物理参考轨迹的验收与导出。后续 Parkour / AMP 任务使用这些物理参考数据。

```text
重定向 CSV → mjlab NPZ → 配套场景 JSON → Mimic 训练 → 评估 → 导出物理参考
```

以下命令均在项目根目录执行。多行命令中的反斜杠 `\` 后不要添加空格。

## 1. 环境配置

项目使用 **uv + Python 3.13 + 官方 mjlab 1.6.0**。虚拟环境位于 `.venv/`，提示名称为 `PRCF`，依赖版本由 `uv.lock` 锁定。

```bash
cd /home/dai/PRCF

# 已有虚拟环境时跳过创建。
if [ ! -d .venv ]; then
  uv venv --python 3.13 --prompt PRCF
fi
uv sync --locked
```

可以直接使用 `uv run`，也可以先激活环境：

```bash
source .venv/bin/activate
```

检查安装路径、GPU 和已注册任务：

```bash
uv run python -c "import mjlab; print(mjlab.__file__)"
uv run python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
uv run python scripts/list_envs.py
```

mjlab 从虚拟环境的 `site-packages` 导入；`third_party/mjlab/` 用于查阅源码，不会自动替代已安装的版本。GPU 训练使用 `--device cuda:0`。

## 2. 数据准备与场景配置

当前动作资源放在：

```text
src/prcf/asset/motion/e1/
├── raw/                 # 原始动作，例如 BVH
├── csv/                 # 已重定向到 E1 的 CSV
└── npz/
    ├── 017.npz          # 平地动作
    ├── stairs.npz
    ├── stairs.json      # 台阶几何与轨迹关联
    ├── up_box.npz
    ├── up_box.json
    ├── down_box.npz
    └── down_box.json
```

### CSV 转 NPZ

已有 NPZ 可以跳过此步骤。输入 CSV 无表头，共 32 列：根部 XYZ、根部四元数、25 个 E1 关节角。关节顺序须与当前 E1 模型一致。

以下是首次转换台阶动作的示例：

```bash
uv run python scripts/csv_to_npz_e1.py \
  --input-file src/prcf/asset/motion/e1/csv/114_09.csv \
  --output-file src/prcf/asset/motion/e1/npz/stairs.npz \
  --input-fps 50 \
  --output-fps 50 \
  --quaternion-order xyzw
```

`--input-fps` 和四元数顺序必须与 CSV 导出设置一致。转换器会计算身体位置、姿态和速度，并拒绝覆盖已有输出。转换结果是 Mimic 的输入参考。

### 选择动作与障碍物

| 动作 | 训练、play、评估和导出时使用的输入 |
|---|---|
| 平地 | `src/prcf/asset/motion/e1/npz/017.npz` |
| 台阶 | `src/prcf/asset/motion/e1/npz/stairs.json` |
| 上箱 | `src/prcf/asset/motion/e1/npz/up_box.json` |
| 下箱 | `src/prcf/asset/motion/e1/npz/down_box.json` |

JSON 通过 `motion_file` 关联 NPZ，保存障碍物尺寸、位置、摩擦、轨迹坐标对齐和验收阈值。箱体 `size_xyz` 使用完整长宽高，`position` 使用中心位置，单位均为米。

**障碍动作应传 JSON。** 训练和策略 play 直接读取 NPZ 时使用默认平地，不会自动加载旁边的 JSON。

## 3. 回放输入轨迹与编辑障碍物

不需要训练 checkpoint，默认使用网页编辑器：

```bash
uv run python scripts/replay_motion.py \
  --motion-file src/prcf/asset/motion/e1/npz/stairs.json
```

打开终端显示的地址，通常为 `http://127.0.0.1:8080`。支持暂停、播放、逐帧查看、调速、切换动作、拖动障碍物，以及修改尺寸和位置。

点击“保存场景”更新配套 JSON，旧配置自动备份到同目录 `.scene_backups/`。切换动作保留未保存草稿。网页模式会浏览同目录动作，并优先采用已有场景 JSON。

浏览整个动作目录：

```bash
uv run python scripts/replay_motion.py --motion-file src/prcf/asset/motion/e1/npz
```

原生窗口或无界面检查：

```bash
uv run python scripts/replay_motion.py --motion-file src/prcf/asset/motion/e1/npz/stairs.json --viewer native
uv run python scripts/replay_motion.py --motion-file src/prcf/asset/motion/e1/npz/stairs.json --headless
```

端口占用时，网页编辑器可加 `--port 8081`。输入回放是运动学检查；实际物理执行由后面的策略 play 和评估完成。

## 4. Mimic 训练

| 任务 ID | 用途 |
|---|---|
| `PRCF-Mimic-E1-25DoF-Reference` | 当前使用：参考关节姿态加动作残差，并加入双脚世界位置跟踪奖励 |
| `PRCF-Mimic-E1-25DoF` | 保留的基线：相对固定站姿输出关节目标 |

两者都使用普通 PPO 和 `MjlabOnPolicyRunner`。**训练、play、评估和导出须使用与 checkpoint 一致的任务 ID。**

下面统一使用参考残差任务，以台阶动作为例；在同一终端设置后续命令使用的变量：

```bash
PRCF_TASK="PRCF-Mimic-E1-25DoF-Reference"
PRCF_MOTION="src/prcf/asset/motion/e1/npz/stairs.json"
```

切换平地、上箱或下箱动作时，修改 `PRCF_MOTION` 即可。当前每次运行跟踪一条动作，所有并行 env 共同训练一个策略。

### 校验与预检

```bash
uv run python scripts/validate_motion.py --motion-file "$PRCF_MOTION"
uv run python scripts/train.py "$PRCF_TASK" --motion-file "$PRCF_MOTION" --num-envs 1 --dry-run
```

预检会校验动作并编译场景，不启动 PPO，也不需要 CUDA。

### 开始训练

```bash
uv run python scripts/train.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --num-envs 4096 \
  --max-iterations 30000 \
  --device cuda:0
```

训练结果写入 `logs/<任务ID>/<时间戳>/`，终端会显示实际运行目录。可以用 `--log-root` 修改根目录，用 `--seed` 指定随机种子。显存不足时降低 `--num-envs`。

查看训练曲线：

```bash
uv run tensorboard --logdir logs --port 6006
```

### 选择 checkpoint 与继续训练

将下面的 `RUN_ID` 替换为实际运行目录，checkpoint 文件名替换为要使用的版本。后续 play、评估和导出共用此变量：

```bash
PRCF_CHECKPOINT="logs/$PRCF_TASK/RUN_ID/model_1500.pt"
```

继续同一任务与动作的训练：

```bash
uv run python scripts/train.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --num-envs 4096 \
  --max-iterations 3000 \
  --device cuda:0
```

这里的 `3000` 表示额外训练轮数。续训恢复模型、优化器与迭代编号，并创建新的运行目录记录本次实验。

## 5. Play：查看训练后的策略

原生 MuJoCo 窗口：

```bash
uv run python scripts/play.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --device cuda:0 \
  --viewer native
```

网页策略 viewer：

```bash
uv run python scripts/play.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --device cuda:0 \
  --viewer viser
```

`play.py` 执行训练后的策略，从参考起点开始。它使用临时目录准备输入，退出后清理，**不会新增训练日志目录**。障碍物编辑使用第 3 节的 `replay_motion.py`；保存后的场景供下次启动训练或 play 时加载。

## 6. 完整评估与物理轨迹导出

### 仅评估

```bash
uv run python scripts/evaluate.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --device cuda:0 \
  --attempts 1
```

保存 `evaluation.json`，报告完整性、跟踪误差、关节限位、穿透、滑移及已声明的接触要求，不保存 rollout 数组。

### 导出轨迹

```bash
uv run python scripts/export_reference.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --device cuda:0 \
  --attempts 1
```

评估和导出各自创建新的 `logs/<任务ID>/<时间戳>/`，与训练 run 分开。导出保留连续物理执行记录，失败时不自动重置后拼接轨迹。

```text
logs/<任务ID>/<导出运行目录>/
├── config/
├── evaluation.json
├── references/attempt_0000/     # 通过验收的尝试
│   ├── motion.npz              # 实际执行的关节、全身位置/姿态及速度
│   ├── extras.npz              # 动作、实际力矩、手/膝/足接触等附加信息
│   ├── clip.json               # 场景、来源、坐标约定和验收配置
│   └── quality.json            # 该尝试的验收结果
└── rejected/attempt_0000/       # 未通过验收的尝试，使用 rollout.npz
```

每次尝试进入 `references/` 或 `rejected/` 中对应的一处。全部尝试失败时命令返回状态码 `2`，诊断文件仍会保留；默认多次尝试是确定性名义条件下的重复运行。

后续参考池使用通过验收的 `references/attempt_XXXX/` 整个目录，以保留轨迹与场景的对应关系。导出后可检查：

```bash
uv run python scripts/validate_motion.py --motion-file logs/$PRCF_TASK/EXPORT_RUN_ID/references/attempt_0000/clip.json
```

将 `EXPORT_RUN_ID` 替换为实际导出目录。

## 7. 日志与配置文件

```text
logs/<任务ID>/<训练运行目录>/
├── config/
│   ├── env.json                # 生效的环境配置快照
│   ├── rl.json                 # 网络与 PPO 参数
│   ├── source.json             # 动作、场景、校验结果、哈希和依赖版本
│   └── reference.npz           # 应用坐标对齐后的输入参考副本
├── events.out.tfevents.*       # TensorBoard 曲线
├── model_0.pt
├── model_500.pt
└── ...
```

`config/reference.npz` 是训练输入；物理执行后通过验收的结果位于导出 run 的 `references/`。checkpoint 默认每 500 轮保存一次，正常结束还会保存最后一轮。

`logs/` 已加入 `.gitignore`。旧 `outputs → logs` 链接仅用于兼容旧路径，实际结果保存在 `logs/`。

## 8. 代码与详细文档

| 位置 | 内容 |
|---|---|
| `src/prcf/asset/robot/e1/` | E1 MJCF 与执行器配置 |
| `src/prcf/tasks/mimic/config/e1/` | 基线环境、PPO 配置与 MDP |
| `src/prcf/tasks/mimic/config/e1_reference/` | 参考残差动作与脚部跟踪奖励 |
| `src/prcf/tasks/mimic/workflow.py` | 数据准备、runner 创建、评估与导出流程 |
| `src/prcf/tasks/mimic/replay.py` | 网页动作与场景编辑器 |
| `src/prcf/motion/` | 通用数据格式、场景构建、校验、记录与导出 |
| `scripts/` | 训练、回放、评估和数据处理入口 |

- [数据处理流程与实现说明](docs/preprocessing.md)
- [NPZ 与附加数据字段](docs/motion_format.md)
- [项目目录与实验组织](docs/project_structure.md)

代码检查与测试入口：

```bash
uv run python -m unittest discover -s tests -v
```
