# E1 Mimic：物理参考准备

当前实现任务为 `PRCF-Mimic-E1-25DoF`，配置在
`src/prcf/tasks/mimic/config/e1/`，共用运行流程在 `tasks/mimic/workflow.py`。
参数解析直接在 `scripts/train.py`、`play.py`、`evaluate.py`、
`export_reference.py` 等入口中。`--motion-file` 接受 NPZ 或场景 JSON；
旧参数 `--clip` 保留为别名。基于官方 tracking 与普通 PPO，单 GPU 训练，
本地 TensorBoard 日志，不上传 W&B。不会修改官方 mjlab 或其他本地工程。

## 1. 环境与数据

在项目根目录执行：

```bash
uv sync --locked
uv run python scripts/list_envs.py
```

当前资源位于 `src/prcf/asset/motion/e1/`，其中 `csv/` 保存重定向结果，
`npz/` 保存转换结果及配套场景 JSON；`raw/` 保存原始 BVH。

数据采用 mjlab 的 E1 NPZ，要求 25 个关节和当前 E1 模型的完整身体顺序。
不能直接使用 G1 NPZ 或旧 E1 21 自由度 NPZ。输入校验包含字段、维度、有限值、
单位四元数、关节限位、名称匹配和采样帧上的前向运动学一致性。
详细约定见 [motion_format.md](motion_format.md)。

平地动作可直接将 NPZ 路径传给 `--motion-file`。攀箱、台阶等动作使用一个配套
`clip.json`，例如：

```text
datasets/e1/up_box/retargeted/clip_001/
├── motion.npz
└── clip.json
```

参考 `examples/mimic/clip_box.example.json` 或 `clip_stairs.example.json`，
复制到动作目录后修改参数；示例本身不包含运动数据。路径相对 JSON 所在目录。
箱体尺寸是完整长宽高，位置是中心；台阶位置是底部前缘、宽度方向居中。
台阶允许绕 Z 轴旋转。`ground_height: null` 表示不创建无限地面，可以用多个
箱体构造带沟壑的支撑面。当前只支持固定 box/stairs，任意 OBJ 碰撞和移动
物体尚未实现，输入不支持的类型会直接报错。

## 2. 先校验，再做原生场景预检

以下 `path/to/clip.json` 和 checkpoint 路径需要替换成真实路径。

```bash
uv run python scripts/validate_motion.py --motion-file path/to/clip.json
uv run python scripts/train.py --motion-file path/to/clip.json --num-envs 1 --dry-run
uv run python scripts/export_reference.py --motion-file path/to/clip.json --dry-run
```

预检不需要 GPU：加载数据、应用场景对齐、编译 E1/障碍/传感器，并保存生效
配置。预检通过只表示格式与模型组装正确，不代表动力学跟踪成功。
参考 FPS 决定控制频率，每控制周期使用四个物理子步；不会静默重采样输入。

## 3. 训练

```bash
uv run python scripts/train.py PRCF-Mimic-E1-25DoF \
  --motion-file path/to/clip.json --num-envs 1024 --max-iterations 30000 --device cuda:0
```

可先用较小环境数和迭代数检查运行。首次编译 Warp 内核可能耗时。
训练从参考的不同时间段采样初始化。当前关闭观测噪声、关节初始扰动、
外力推扰和参数随机化，优先准备名义物理参考。Mimic 专用配置将软限位系数
设为 1.0，让初始化和参考使用相同的硬限位范围；输入仍须通过硬限位校验，
没有扩大机器人的物理关节范围。固定障碍任务使用 0.35 m 的三维 anchor
位置误差终止，避免 Z-only 检查遗漏水平落后。
不加入 AMP，不进行专家蒸馏。不同运动当前逐条训练，所有并行世界使用同一
动作/场景配对，尚未实现多动作在线切换。一次运行只训练一个 PPO 策略，
不是为每个 env 单独训练策略；各 env 可以从不同参考帧开始，共同采样更新它。
本地自适应采样将 50% 概率均匀分配到所有时间区间，另外 50% 根据失败
难度分配，避免大批量并行下加性均匀先验被失败计数淹没。10 个区间时
每个区间至少 5%，单个区间最多 55%；TensorBoard 的 sampling_top1_prob
反映最终实际采样分布。

场景 JSON 的 `scene` 由 `SceneBuilder` 写入编译前的 MuJoCo 场景，随后
MuJoCo Warp 使用 `nworld=num_envs` 创建独立状态。静态几何模板在各 world
复用，机器人状态和接触求解分别计算；不存在机器人争用同一个物理箱子的问题。
当前关闭 TerrainGenerator，各 world 的 `env_origins` 为零，动作和障碍物
使用一致坐标，关闭根部位置/朝向随机偏移，也关闭关节初始化扰动。
因此无需按 env 索引手动平移箱子。若以后实现多动作混训，需要同时分配
`motion_id` 和对应场景，当前不要直接把不同 NPZ 混入这条启动流程。

结果保存至 `logs/<task_id>/<时间戳>/`：

- `config/reference.npz`：坐标变换后的实际输入，原文件不覆盖。
- `config/source.json`：源文件哈希、场景、机器人资源/配置哈希、代码与依赖版本。
- `config/env.json`、`config/rl.json`：生效配置；函数使用导入路径表示。
- `model_*.pt` 和 TensorBoard 日志。

继续同一实验可传 `--checkpoint path/to/model_1000.pt`，保持动作、配置一致。
新增训练迭代数由 `--max-iterations` 指定；初始化 checkpoint 另存来源记录。

## 4. 查看策略

```bash
uv run python scripts/play.py --motion-file path/to/clip.json \
  --checkpoint logs/.../model_1000.pt --device cuda:0 --viewer native
```

无桌面可选择 `--viewer viser`。查看模式使用确定性策略，从起点开始。
参考末帧不回绕；查看模式通过明确的完成终止重新开始下一次动作。

仅检查参考动作和场景、调整障碍物时，使用网页编辑器，不需要 checkpoint：

```bash
uv run python scripts/replay_motion.py \
  --motion-file src/prcf/asset/motion/e1/npz/up_box.json
```

默认 `--viewer viser`，打开终端显示的地址，通常为 `http://127.0.0.1:8080`。
初始为暂停，可使用播放/暂停、上一帧/下一帧、帧滑条、速度和循环播放。
动作下拉框和上一个/下一个按钮会浏览输入文件所在目录；也可以直接传目录：

```bash
uv run python scripts/replay_motion.py --motion-file src/prcf/asset/motion/e1/npz
```

网页模式优先加载同目录 JSON 对应的轨迹，避免同一 NPZ 出现两次，也避免
漏掉已有场景。原生/无界面模式仍严格使用传入文件，不自动关联旁边的 JSON。

**障碍物编辑：** 选择障碍物后，调整位置 XYZ、完整长宽高 XYZ、旋转角度，
或直接拖动三维坐标轴平移。箱体调整尺寸默认保持最低点高度，方便贴地；
`stairs` 类型可修改级高、级深、宽度和级数，旋转仍只允许绕 Z 轴。
当前 `stairs.json` 是三个嵌套箱体，各层分别选择编辑。支持添加/删除箱体、
启用/关闭地面和设置地面高度。修改即时预览，切换动作保留各自草稿。

**保存：** 点击“保存场景”才写入当前动作的 JSON。原 JSON 自动备份到
同目录 `.scene_backups/`，通过临时文件原子替换；外部修改冲突会阻止覆盖。
从裸 NPZ 开始编辑会新建同名 JSON。原 NPZ、坐标对齐和验收阈值不改动。
旧场景的穿透检查报告在保存几何修改时移除，避免与新几何混用。
保存后下次训练传入该 JSON 生效；正在运行的训练已载入自己的配置与数据，
不会跟随编辑器热更新。

“检查当前帧穿透”会将草稿通过训练同一个 `SceneBuilder` 编译到原生 MuJoCo，
显示当前参考姿态的最大环境穿透，帮助定位几何冲突。它不进行物理步进，
不验证承重、摩擦或整段动作可执行性。

可使用 `--port 8081` 避开占用的端口。远程机器需显式设置 `--host 0.0.0.0`，
并使用机器地址连接。回放的数据处理和 MuJoCo 计算在 CPU 上进行，不启动 PPO。

原生窗口与无界面检查继续可用：

```bash
uv run python scripts/replay_motion.py --motion-file path/to/clip.json --viewer native
uv run python scripts/replay_motion.py --motion-file path/to/clip.json --headless
```

这些模式均为运动学回放，不验证物理执行，不依赖 CUDA。

## 5. 独立评估与导出

```bash
uv run python scripts/evaluate.py --motion-file path/to/clip.json \
  --checkpoint logs/.../model_1000.pt --device cuda:0 --attempts 1

uv run python scripts/export_reference.py --motion-file path/to/clip.json \
  --checkpoint logs/.../model_1000.pt --device cuda:0 --attempts 1
```

评估入口只保存质量报告。导出入口还保存实际连续执行记录：

```text
logs/<task_id>/<时间戳>/
├── config/
├── evaluation.json
├── references/attempt_0000/        # 只有通过全部已配置验收项才进入此处
│   ├── motion.npz
│   ├── extras.npz
│   ├── clip.json
│   └── quality.json
└── rejected/attempt_0001/          # 失败时保留 rollout.npz 和同类诊断文件
```

末帧时命令停止推进，不重置机器人。导出模式关闭环境自动重置，避免终止帧
混入下一次初始化。完整记录要求参考帧为 `0, 1, ..., T-1`，并检查实际运动
误差、关节限位、穿透、持续承重接触滑移及声明的接触要求。验收阈值在
`clip.json/quality` 中设置，当前默认值是起始工程值，需要按动作调整并报告。
滑移评价使用控制频率下持续承重的接触，不把单次冲击直接记作滑移。

攀箱动作建议显式加入手/膝支撑区间，例如（必须替换为真实帧区间）：

```json
"contact_requirements": [
  {"part": "left_hand", "start_frame": 40, "end_frame": 70, "min_fraction": 0.7}
]
```

这表示该区间内，至少 70% 的记录帧应存在超过力阈值的对应环境接触。
没有标注时，报告明确记录未验证这些指定支撑区间，不会凭动作名称编造标签。
通过验收表示通过了所列检查，不意味着轨迹“完美”或实机执行已经验证。

重复导出默认是确定性、名义条件下的重复运行，不把重复次数当作独立扰动
鲁棒性证据。若全部失败，命令返回状态码 2，报告和失败记录仍会保存。

## 6. CSV 转换

`scripts/csv_to_npz_e1.py` 已适配为 E1 原生 CPU 转换，替换原来的 G1 示例。
输入为无表头 CSV：`root_xyz + root_quaternion + 25 个 E1 关节角`，共 32 列。
关节顺序见 `E1_JOINT_NAMES`。四元数输入默认 xyzw，可以显式指定 wxyz。

```bash
uv run python scripts/csv_to_npz_e1.py \
  --input-file path/to/e1.csv --output-file path/to/motion.npz \
  --input-fps 30 --output-fps 50 --quaternion-order xyzw
```

转换器计算 E1 的全部身体位置/姿态及速度，不启动训练、不上传数据，拒绝
覆盖已有输出。转换结果仍是运动学参考，随后必须经过实际跟踪和验收。

## 7. 验证范围

CPU 测试可执行：

```bash
uv run python -m unittest discover -s tests -v
```

28 项测试覆盖错误机器人数据、身体顺序、坐标变换、场景几何、末帧停止、失败
筛选、子步穿透和导出回读，以及编辑保存、备份、外部修改冲突、场景重载和播放/切换逻辑。
网页编辑器使用真实 Viser GUI/场景 API 做离线集成测试，网络传输替换为测试桩；
当前工具会话禁止监听端口，尚未完成真实浏览器点击与渲染验证。
另外用合成短动作完成了 CPU 两环境单次 PPO 更新、
checkpoint 加载和 12 帧实际仿真轨迹导出/回读检查。2026-11-11 已加载用户提供的 4 条 NPZ 并完成格式与 FK 检查，为台阶和
上下箱补充估计场景，并验证 CPU 两个独立 world 的场景加载、状态隔离和
物理步进。当前会话没有可用 NVIDIA 设备；这些真实动作尚未完成 PPO 训练
与物理参考验收，不能把场景预检或运动学回放当作跟踪成功。

## 8. 当前台阶和箱体的估计配置

场景文件与 NPZ 位于同一目录，`motion_file` 使用相对文件名。它们来自重定向
轨迹的低速足底高度、手部碰撞胶囊和空间范围估计，没有原始物体尺寸记录。
箱体宽度与未被动作覆盖的远端边界是假设的净空，不是数据集真值。

| 输入 | 场景 | 轨迹对齐 Z 平移 |
|---|---|---|
| `017.npz` | 默认平地，不需要 JSON | 0 m |
| `stairs.json` | 三层箱体组成双向台阶，顶面高度 0.120 / 0.270 / 0.375 m | -0.015 m |
| `up_box.json` | 箱体高 0.905 m，X 范围 1.530–2.730 m | +0.008 m |
| `down_box.json` | 箱体高 0.905 m，X 范围 -0.900–0.325 m | +0.008 m |

上箱和下箱源动作使用不同坐标原点，因此箱子坐标不同；Mimic 分别训练，
不要求两段数据先对齐到同一个全局场景。`stairs.npz` 包含走过台阶后转身
返回的动作，使用嵌套箱体形成两侧可通行的台阶，而不是默认单向上升楼梯。

`motion_to_scene` 只在加载时对全部参考身体做一次一致刚体变换，原始 NPZ
没有修改。场景参数已经处于对齐后的坐标系，不要再重复平移障碍物。
JSON 中的 `metadata.scene_status` 标记为估计，`kinematic_scene_check` 保存
逐帧原生 MuJoCo 碰撞检查结果与最差帧。

当前强制回放参考姿态时，最大环境穿透约为：上箱 28.5 mm、下箱 31.6 mm、
台阶 35.9 mm，部分来自脚与地面接触的重定向误差。这些是运动学参考与
场景的残差，不是物理跟踪误差；没有为它们放宽导出验收阈值。先回放核对，
再做实际 PPO 跟踪。若稳定跟踪仍失败，需要调整场景或进行接触感知重定向，
不能据此宣布参考已经物理可行。

```bash
# 平地：直接传 NPZ
uv run python scripts/train.py \
  --motion-file src/prcf/asset/motion/e1/npz/017.npz --num-envs 1024

# 障碍动作：传 JSON，场景才会进入训练；直接传 NPZ 仍是默认平地
uv run python scripts/replay_motion.py \
  --motion-file src/prcf/asset/motion/e1/npz/up_box.json

uv run python scripts/train.py \
  --motion-file src/prcf/asset/motion/e1/npz/up_box.json --num-envs 2 --dry-run

uv run python scripts/train.py \
  --motion-file src/prcf/asset/motion/e1/npz/up_box.json --num-envs 1024 --device cuda:0
```

上下箱、台阶分别运行对应 JSON，得到各自 checkpoint 和通过验收的参考轨迹；
后续 AMP 任务再将这些物理参考组织到统一参考池中。

## 9. 上箱训练的诊断与修正

2026-11-11 对 `20261111_173125_843595` 训练日志、4500 轮 checkpoint 做了
CPU 复现，详见 [training_diagnosis.md](training_diagnosis.md)。配置已修正采样覆盖、
初始化限位冲突和水平跟踪失败检查，28 项测试及 CPU 一次 PPO 更新通过。
这些修正需要重新启动训练才生效。建议新建实验验证，先检查基础平地跟踪，
再处理攀箱参考在第 280–327 帧附近的自碰撞与接触一致性。

## 10. 日志目录与台阶跟踪实验

训练、evaluate 和 export 的默认根目录统一为 `logs/`，可通过 `--log-root`
修改；`--output-root` 保留为兼容别名。已有 `outputs/` 内容已迁移到 `logs/`。
`outputs → logs` 仅作为旧路径兼容链接，保护旧训练进程和配置中的绝对路径；
真实数据位于 `logs/`，两条路径均由 Git 忽略。

`play.py` 不再调用 `new_run_dir()`。它在系统临时目录准备对齐后的输入，
viewer 关闭或 dry-run 结束即清理，不产生新的训练 run，不写入原训练配置。

台阶 checkpoint `20261112_115305_409529/model_20000.pt` 的 CPU 确定性
回放能踏上第一层，但后续脚步明显落后，于第 256 帧附近因末端高度误差终止。
此时采样最大概率约 40%，没有复现旧上箱实验的 92% 集中现象。
原全身相对位置奖励会对多个身体取平均，并随机器人 anchor 水平重定位，
不足以单独确保脚落在固定台阶上的绝对位置。固定站姿为中心的动作表达也
要求策略从零产生完整步态。这些是此次实验调整的动机，不是已经证明的唯一根因。

新增独立对照任务 `PRCF-Mimic-E1-25DoF-Reference`：

- `config/e1_reference/` 继承 E1 基线，仍使用相同普通 PPO 和 runner。
- 位置目标改为 `q_target = q_reference + action * scale`，策略学习参考姿态的修正。
- 增加双脚世界坐标位置跟踪奖励，std=0.15 m、权重 1.0。
- 其他主要 PPO 参数保持一致，便于对照。旧任务和 checkpoint 的动作含义保留。

```bash
uv run python scripts/train.py PRCF-Mimic-E1-25DoF-Reference --motion-file src/prcf/asset/motion/e1/npz/stairs.json --num-envs 4096 --device cuda:0
```

新任务应从新策略训练。checkpoint 附近存在来源配置时，加载器会拒绝任务 ID
不匹配的 checkpoint，避免把旧任务的动作解释为参考残差。

查看新任务必须使用相同任务 ID：

```bash
uv run python scripts/play.py PRCF-Mimic-E1-25DoF-Reference --motion-file src/prcf/asset/motion/e1/npz/stairs.json --checkpoint logs/PRCF-Mimic-E1-25DoF-Reference/实际运行目录/model_1000.pt --device cuda:0
```

28 项测试通过，覆盖 play 临时文件清理、残差关节索引映射、世界坐标脚部误差、
任务不匹配拒绝等。CPU 四环境一次 PPO 更新通过，第 0、150、400 帧的零残差
目标与参考关节位置完全一致。未完成新任务的 GPU 收敛验证，不能宣称台阶跟踪
已经解决。台阶仍为估计几何，参考存在局部穿插，后续仍需独立连续评估。
