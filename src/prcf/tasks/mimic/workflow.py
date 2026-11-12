"""Mimic data preparation, runner construction and physical-reference evaluation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from prcf.motion.schema import load_clip
from prcf.motion.scene import SceneBuilder
from prcf.motion.validation import e1_model, load_motion, sha256, transform_motion, validate_motion


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f'{value.__module__}:{value.__qualname__}'
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + '\n')


def new_run_dir(root, task_id):
    directory = Path(root) / task_id / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


def prepare(task_id, source, mode, run_dir, *, num_envs=1, seed=42):
    import prcf.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from prcf.tasks.mimic.config.e1.mdp.commands import FiniteMotionCommandCfg
    from prcf.tasks.mimic.config.e1.mdp.recorders import TrajectoryRecorder, capture_substep
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    clip = load_clip(source)
    if num_envs < 1:
        raise ValueError('num_envs must be positive')
    if mode == 'export' and num_envs != 1:
        raise ValueError('Export uses one continuous rollout per attempt')
    model, joints, bodies = e1_model()
    data = load_motion(clip.motion_file)
    report = validate_motion(data, model=model, expected_fps=clip.metadata.get('declared_fps'))
    data = transform_motion(data, clip.motion_to_scene)
    validate_motion(data, model=model)
    for requirement in clip.quality.contact_requirements:
        if requirement.end_frame >= report['frames']:
            raise ValueError('Contact requirement ends beyond the motion')
    cfg = load_env_cfg(task_id, play=mode != 'train')
    if not isinstance(cfg.commands.get('motion'), FiniteMotionCommandCfg):
        raise ValueError('This launcher requires a registered PRCF finite-motion tracking task')
    cfg.scene.spec_fn = SceneBuilder(clip.scene)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # The NPZ rate is authoritative; use four physics substeps per control frame.
    cfg.decimation = 4
    cfg.sim.mujoco.timestep = 1.0 / (report['fps'] * cfg.decimation)
    cfg.episode_length_s = (report['frames'] + 2) / report['fps']
    config_dir = Path(run_dir) / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved = config_dir / 'reference.npz'
    np.savez_compressed(resolved, **data)
    cfg.commands['motion'].motion_file = str(resolved.resolve())
    if mode == 'export':
        cfg.auto_reset = False
        cfg.recorders = {'trajectory': RecorderTermCfg(func=TrajectoryRecorder)}
        cfg.metrics['record_physics_substeps'] = MetricsTermCfg(func=capture_substep, per_substep=True)
    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.seed = seed
    source_tree = Path(__file__).resolve().parents[2]  # src/prcf, including all tasks
    code_hash = hashlib.sha256()
    for file in sorted(source_tree.rglob('*.py')):
        code_hash.update(str(file.relative_to(source_tree)).encode())
        code_hash.update(file.read_bytes())
    from prcf.asset.robot.e1.e1_constants import E1_XML
    asset_hash = hashlib.sha256()
    for file in [E1_XML, *sorted((E1_XML.parent / 'meshes').glob('*'))]:
        if file.is_file():
            asset_hash.update(file.name.encode())
            asset_hash.update(file.read_bytes())
    metadata = {
        'task_id': task_id, 'seed': seed, 'scene': clip.scene, 'source_clip_id': clip.clip_id,
        'source_motion': str(clip.motion_file), 'source_motion_sha256': sha256(clip.motion_file),
        'source_manifest': str(clip.manifest_file) if clip.manifest_file else None,
        'motion_to_scene': clip.motion_to_scene, 'robot_asset_sha256': asset_hash.hexdigest(),
        'robot_config_sha256': sha256(E1_XML.parent.parent / 'e1_constants.py'),
        'code_sha256': code_hash.hexdigest(), 'python': platform.python_version(),
        'control_dt': 1 / report['fps'], 'physics_dt': cfg.sim.mujoco.timestep,
    }
    import importlib.metadata as distributions
    metadata['versions'] = {name: distributions.version(name) for name in ('mjlab', 'mujoco', 'mujoco-warp', 'torch', 'rsl-rl-lib')}
    lockfile = source_tree.parents[1] / 'uv.lock'
    if lockfile.exists():
        metadata['uv_lock_sha256'] = sha256(lockfile)
    write_json(config_dir / 'source.json', {'clip': asdict(clip), 'validation': report, 'provenance': metadata})
    write_json(config_dir / 'env.json', cfg)
    write_json(config_dir / 'rl.json', agent_cfg)
    return cfg, agent_cfg, clip, data, model, metadata


def check_device(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this session. Check nvidia-smi; use --dry-run for native scene/data checks.')


def build_runner(task_id, cfg, agent_cfg, device, *, checkpoint=None, log_dir=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    if checkpoint is not None:
        source = Path(checkpoint).resolve().parent / 'config' / 'source.json'
        if source.exists():
            saved_task = json.loads(source.read_text()).get('provenance', {}).get('task_id')
            if saved_task and saved_task != task_id:
                raise ValueError(
                    f'Checkpoint belongs to {saved_task}, requested {task_id}. '
                    'Use its matching task ID; action meanings may differ.')
    check_device(device)
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        train_cfg = asdict(agent_cfg)
        if checkpoint is not None and log_dir is None:
            saved = Path(checkpoint).resolve().parent / 'config' / 'rl.json'
            if saved.exists():
                train_cfg = json.loads(saved.read_text())
        runner = load_runner_cls(task_id)(wrapped, train_cfg, log_dir=log_dir, device=device)
        if checkpoint is not None:
            kwargs = {} if log_dir is not None else {'load_cfg': {'actor': True}}
            runner.load(str(Path(checkpoint).resolve()), map_location=device, **kwargs)
        return wrapped, runner
    except BaseException:
        env.close()
        raise


def dry_run(cfg):
    from mjlab.scene import Scene
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    return {'native_scene_compiles': True, 'nq': model.nq, 'nv': model.nv,
            'nu': model.nu, 'sensors': model.nsensor, 'num_envs': cfg.scene.num_envs,
            'control_dt': cfg.sim.mujoco.timestep * cfg.decimation}


def run_evaluation(task_id, cfg, agent_cfg, clip, reference, model, metadata, run_dir, checkpoint, device, attempts, *, save_rollouts=True):
    from prcf.motion.export import save_attempt
    from prcf.motion.quality import assess_trajectory
    if attempts < 1:
        raise ValueError('attempts must be positive')
    metadata = {**metadata, 'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': sha256(checkpoint)}
    env, runner = build_runner(task_id, cfg, agent_cfg, device, checkpoint=checkpoint)
    results = []
    try:
        policy = runner.get_inference_policy(device=device)
        with torch.inference_mode():
            for attempt in range(attempts):
                obs, _ = env.reset()
                command = env.unwrapped.command_manager.get_term('motion')
                recorder = env.unwrapped.recorder_manager.get_term('trajectory')
                failure = None
                for _ in range(len(reference['joint_pos']) - 1):
                    obs, _, _, _ = env.step(policy(obs))
                    if bool(env.unwrapped.reset_terminated[0]):
                        failure = 'environment_termination'
                        # reset_terminated excludes the explicit motion timeout.
                        break
                    if bool(env.unwrapped.reset_time_outs[0]) and not bool(command.finished[0]):
                        failure = 'premature_timeout'
                        break
                    if bool(command.finished[0]):
                        break
                states, intervals = recorder.buffer.arrays()
                report = assess_trajectory(states, intervals, reference, clip.quality,
                                           failure_reason=failure, joint_limits=model.jnt_range[1:])
                category = 'references' if report['success'] else 'rejected'
                if save_rollouts:
                    save_attempt(Path(run_dir) / category / f'attempt_{attempt:04d}', states, intervals,
                                 fps=float(reference['fps'].item()), joint_names=env.unwrapped.scene['robot'].joint_names,
                                 body_names=env.unwrapped.scene['robot'].body_names, report=report,
                                 metadata={**metadata, 'attempt': attempt, 'sampling': 'deterministic_nominal'})
                results.append(report)
                print(f"Attempt {attempt}: {'PASS' if report['success'] else 'REJECT'} {report['reasons']}")
                if hasattr(policy, 'reset'):
                    policy.reset(torch.ones(1, device=device, dtype=torch.bool))
    finally:
        env.close()
    summary = {'attempts': attempts, 'successes': sum(r['success'] for r in results),
               'success_rate': sum(r['success'] for r in results) / attempts,
               'sampling': 'repeated deterministic nominal trials; not a robustness estimate', 'results': results,
               'provenance': metadata}
    write_json(Path(run_dir) / 'evaluation.json', summary)
    return summary
