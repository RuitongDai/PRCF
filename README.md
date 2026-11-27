# PRCF

**Perception-Reliability- and Contact-Feasibility-Aware Whole-Body Control for Humanoid Locomotion in Complex Terrain**

面向复杂地形的人形机器人感知可靠性与接触可行性驱动的全身运动控制研究。

当前已完成 E1 25 自由度机器人的数据处理流程：重定向 CSV 转换、动作与场景配对、Mimic 跟踪训练、策略回放，以及 AMP 参考特征导出。

APEX 上箱与下箱技能已迁移到 E1 25DoF 和 mjlab。见 [APEX 训练与回放说明](src/prcf/tasks/parkour/config/apex/README.md)。

```text
重定向 CSV → mjlab NPZ → 配套场景 JSON → Mimic 训练 → 策略执行 → 导出 AMP NPZ
```

以下命令均在项目根目录执行。多行命令中的反斜杠 `\` 后不要添加空格。

## 1. 环境配置

项目使用 **uv、Python 3.13 和 PyPI 上的 mjlab 1.6.0**，依赖版本记录在 `uv.lock`。本机和服务器分别创建自己的 `.venv/`，传输项目时可以跳过这个目录。

### 安装 uv

首次配置时，如果终端找不到 `uv`，在 Linux 上执行以下命令。已有 `uv` 可以直接创建环境。[uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 创建或重建环境

先进入项目根目录。本机路径为 `/home/dai/PRCF`，服务器路径示例为 `/root/gpufree-data/PRCF`。

```bash
if [ ! -d .venv ]; then
  uv venv --python 3.13 --prompt PRCF
fi
uv sync --locked
```

缺少 Python 3.13 时，uv 会自动下载。创建后，环境目录为 `.venv/`，激活时终端显示 `(PRCF)`。以后可以直接用 `uv run` 运行脚本，也可以手动激活。

```bash
source .venv/bin/activate
```

### 验证安装

检查 Python 版本、mjlab 导入路径和已注册任务。

```bash
uv run --locked python --version
uv run --locked python -c "import mjlab; print(mjlab.__file__)"
uv run --locked python scripts/list_envs.py
```

mjlab 的导入路径应位于 `.venv/` 的 `site-packages` 中。`third_party/mjlab/` 是可选的源码查阅目录。

使用 GPU 训练前，再确认服务器能识别 NVIDIA 显卡，并检查 PyTorch 输出为 `CUDA: True`。训练命令的设备参数可用 `--device cuda:0`。

```bash
nvidia-smi
uv run --locked python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

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

| 动作 | 训练、play 和导出时使用的输入 |
|---|---|
| 平地 | `src/prcf/asset/motion/e1/npz/017.npz` |
| 台阶 | `src/prcf/asset/motion/e1/npz/stairs.json` |
| 上箱 | `src/prcf/asset/motion/e1/npz/up_box.json` |
| 下箱 | `src/prcf/asset/motion/e1/npz/down_box.json` |

JSON 通过 `motion_file` 关联 NPZ，保存障碍物尺寸、位置、摩擦和轨迹坐标对齐。箱体 `size_xyz` 使用完整长宽高，`position` 使用中心位置，单位均为米。

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

两者都使用普通 PPO 和 `MjlabOnPolicyRunner`。**训练、play 和导出须使用与 checkpoint 一致的任务 ID。**

下面统一使用参考残差任务，以台阶动作为例；在同一终端设置后续命令使用的变量：

```bash
PRCF_TASK="PRCF-Mimic-E1-25DoF-Reference"
PRCF_MOTION="src/prcf/asset/motion/e1/npz/stairs.json"
```

切换平地、上箱或下箱动作时，修改 `PRCF_MOTION` 即可。当前每次运行跟踪一条动作，所有并行 env 共同训练一个策略。

### 场景预检

```bash
uv run python scripts/train.py "$PRCF_TASK" --motion-file "$PRCF_MOTION" --num-envs 1 --dry-run
```

预检会加载动作并在 CPU 上编译场景。

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

将下面的 `RUN_ID` 替换为实际运行目录，checkpoint 文件名替换为要使用的版本。后续 play 和导出共用此变量：

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

## 6. 导出 AMP 轨迹

```bash
uv run python scripts/export_reference.py "$PRCF_TASK" \
  --motion-file "$PRCF_MOTION" \
  --checkpoint "$PRCF_CHECKPOINT" \
  --device cuda:0 \
  --output-file logs/exports/stairs.npz
```

每次执行从参考第 0 帧开始，采集策略实际运动并写出一个 NPZ。默认路径为
`logs/exports/<任务ID>/<时间戳>/amp_obs.npz`。使用 `--output-file` 可以直接指定
文件名。如果目标已存在，换一个文件名。

NPZ 包含八个具名字段和拼接好的 `amp_obs` 矩阵。放进判别器前，可按同一轨迹的连续帧
切成窗口。文件同时记录 `termination` 和 `expected_frames`，便于区分完整与提前结束的
动作。导出直接保留已执行的帧。定义见 [AMP 数据格式](docs/motion_format.md#amp-判别器数据)。

同一个 NPZ 还保存了回放所需的 pelvis 世界位姿和场景。可直接打开文件或浏览目录。

```bash
uv run python scripts/replay_motion.py --motion-file logs/exports/amp/stairs.npz
uv run python scripts/replay_motion.py --motion-file logs/exports/amp
```

训练加载 `np.load(path, allow_pickle=False)['amp_obs']`，每帧 357 维。

## 7. 日志与配置文件

```text
logs/<任务ID>/<训练运行目录>/
├── config/
│   ├── env.json                # 生效的环境配置快照
│   ├── rl.json                 # 网络与 PPO 参数
│   ├── source.json             # 动作、场景、帧数、种子与 checkpoint 来源
│   └── reference.npz           # 应用坐标对齐后的输入参考副本
├── events.out.tfevents.*       # TensorBoard 曲线
├── model_0.pt
├── model_500.pt
└── ...
```

`config/reference.npz` 保存对齐后的训练输入。导出目录里的 `amp_obs.npz` 保存实际执行的 AMP 特征。checkpoint 默认每 500 轮保存一次，正常结束还会保存最后一轮。

`logs/` 已加入 `.gitignore`。旧 `outputs → logs` 链接仅用于兼容旧路径，实际结果保存在 `logs/`。

## 8. 代码与详细文档

| 位置 | 内容 |
|---|---|
| `src/prcf/asset/robot/e1/` | E1 MJCF 与执行器配置 |
| `src/prcf/tasks/mimic/config/e1/` | 基线环境、PPO 配置与 MDP |
| `src/prcf/tasks/mimic/config/e1_reference/` | 参考残差动作与脚部跟踪奖励 |
| `src/prcf/asset/scene.py` | 共用的场景构建、动作读取和对齐 |
| `scripts/train.py` | 训练、环境准备与 runner 创建 |
| `scripts/play.py` | 策略回放 |
| `scripts/replay_motion.py` | 参考动作回放、网页场景编辑 |
| `scripts/export_reference.py` | 策略执行、轨迹采集、AMP 特征生成和保存 |
| `scripts/csv_to_npz_e1.py` | E1 CSV 转换 |

- [数据处理流程与实现说明](docs/preprocessing.md)
- [NPZ 与附加数据字段](docs/motion_format.md)
- [项目目录与实验组织](docs/project_structure.md)

代码检查与测试入口：

```bash
uv run python -m unittest discover -s tests -v
```
