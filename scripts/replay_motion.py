"""CPU motion replay and Viser scene editing; saves scene JSON, never motion arrays."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import socket
import tempfile
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from prcf.asset.robot.e1.e1_constants import e1_model
from prcf.asset.scene import SceneBuilder, align_motion, load_clip, scene_boxes


def load_replay_motion(source):
    """Use one pose layout for reference NPZs and exported policy trajectories."""
    clip = load_clip(source)
    with np.load(clip.motion_file, allow_pickle=False) as archive:
        if 'amp_obs' in archive:
            if 'root_pos_w' not in archive or 'root_quat_w' not in archive:
                raise ValueError('此 AMP 文件缺少根部世界位姿，请用当前 export_reference.py 重新导出。')
            if clip.manifest_file is None:
                clip.scene = json.loads(archive['scene_json'].item())
            root_quat = archive['root_quat_w']
            forward = Rotation.from_quat(root_quat, scalar_first=True).apply([1., 0., 0.])
            yaw = np.arctan2(forward[:, 1], forward[:, 0])
            heading = Rotation.from_euler('z', yaw[:, None])
            motion = {
                'fps': archive['fps'],
                'joint_pos': archive['dof_pos'],
                'joint_vel': archive['dof_vel'],
                'body_pos_w': archive['root_pos_w'][:, None, :],
                'body_quat_w': root_quat[:, None, :],
                'body_lin_vel_w': heading.apply(archive['root_lin_vel'])[:, None, :],
                'body_ang_vel_w': heading.apply(archive['root_ang_vel'])[:, None, :],
            }
        else:
            motion = dict(archive)
    return clip, align_motion(motion, clip.motion_to_scene)


def set_native_frame(model, state, motion, frame):
    lower = int(frame)
    fraction = frame - lower

    def sample(values):
        start = values[lower]
        if fraction == 0:
            return start
        return start + fraction * (values[lower + 1] - start)

    quat = motion['body_quat_w'][lower, 0]
    if fraction:
        next_quat = motion['body_quat_w'][lower + 1, 0]
        if np.dot(quat, next_quat) < 0:
            next_quat = -next_quat
        quat = quat + fraction * (next_quat - quat)
        quat /= np.linalg.norm(quat)
    state.qpos[:3] = sample(motion['body_pos_w'][:, 0])
    state.qpos[3:7] = quat
    state.qpos[7:] = sample(motion['joint_pos'])
    state.qvel[:3] = sample(motion['body_lin_vel_w'][:, 0])
    rotation = Rotation.from_quat(state.qpos[3:7], scalar_first=True)
    state.qvel[3:6] = rotation.inv().apply(sample(motion['body_ang_vel_w'][:, 0]))
    state.qvel[6:] = sample(motion['joint_vel'])
    mujoco.mj_forward(model, state)


def discover_motions(source: str | Path) -> tuple[list[Path], Path]:
    """Browse siblings, preferring manifests over the NPZ they already describe."""
    source = Path(source).expanduser().resolve()
    directory = source if source.is_dir() else source.parent
    manifests, referenced = [], {}
    for path in sorted(directory.glob('*.json')):
        try:
            clip = load_clip(path)
            if clip.motion_file.is_file():
                manifests.append(path)
                referenced.setdefault(clip.motion_file, path)
        except (ValueError, KeyError, TypeError, OSError):
            continue
    paths = sorted(manifests + [p for p in directory.glob('*.npz') if p not in referenced])
    if not paths:
        raise ValueError(f'No E1 motions found in {directory}')
    initial = paths[0] if source.is_dir() else referenced.get(source, source)
    if initial not in paths:
        paths.append(initial)
    return paths, initial


class ReplayDocument:
    """One in-memory motion/scene draft; all mutations validate before committing."""

    def __init__(self, source: Path):
        self.source = Path(source).resolve()
        self.clip, self.motion = load_replay_motion(self.source)
        self.frames = len(self.motion['joint_pos'])
        self.fps = float(self.motion['fps'].item())
        self.save_path = self.clip.manifest_file or self.clip.motion_file.with_suffix('.json')
        self._disk_snapshot = self.save_path.read_bytes() if self.save_path.exists() else None
        if self.clip.manifest_file:
            self.payload = json.loads(self._disk_snapshot)
        else:
            self.payload = {
                'schema_version': 1, 'clip_id': self.clip.clip_id, 'robot_id': 'e1_25dof',
                'motion_file': os.path.relpath(self.clip.motion_file, self.save_path.parent),
                'fps': self.fps, 'scene': deepcopy(self.clip.scene),
                'motion_to_scene': deepcopy(self.clip.motion_to_scene),
                'metadata': {},
            }
        self.scene = deepcopy(self.clip.scene)
        scene_boxes(self.scene)
        self._saved_scene = deepcopy(self.scene)
        self.dirty = False
        self.frame = 0

    def replace_scene(self, scene: dict):
        scene_boxes(scene)
        self.scene = deepcopy(scene)
        self.dirty = self.scene != self._saved_scene

    def obstacle(self, name: str) -> dict:
        return next(o for o in self.scene.get('obstacles', []) if o['id'] == name)

    def edit_obstacle(self, name: str, updates: dict, *, keep_bottom=False):
        scene = deepcopy(self.scene)
        obstacle = next(o for o in scene.get('obstacles', []) if o['id'] == name)
        if keep_bottom and obstacle['type'] == 'box' and 'size_xyz' in updates:
            # Preserve the lowest point even for a rotated box.
            rotation = Rotation.from_quat(obstacle.get('quaternion_wxyz', [1, 0, 0, 0]), scalar_first=True)
            vertical = np.abs(rotation.as_matrix()[2])
            delta = float(vertical @ (np.asarray(updates['size_xyz']) - obstacle['size_xyz']) / 2)
            position = list(obstacle.get('position', [0, 0, 0]))
            position[2] += delta
            obstacle['position'] = position
        obstacle.update(deepcopy(updates))
        self.replace_scene(scene)

    def add_box(self) -> str:
        scene = deepcopy(self.scene)
        obstacles = scene.setdefault('obstacles', [])
        names = {o['id'] for o in obstacles}
        index = 1
        while f'box_{index}' in names:
            index += 1
        name = f'box_{index}'
        position = self.motion['body_pos_w'][self.frame, 0].copy()
        position[0] += 0.6
        position[2] = (scene.get('ground_height') or 0) + 0.3
        obstacles.append({'id': name, 'type': 'box', 'size_xyz': [0.6, 0.8, 0.6],
                          'position': position.tolist(), 'quaternion_wxyz': [1, 0, 0, 0], 'fixed': True})
        self.replace_scene(scene)
        return name

    def delete_obstacle(self, name):
        scene = deepcopy(self.scene)
        scene['obstacles'] = [o for o in scene.get('obstacles', []) if o['id'] != name]
        self.replace_scene(scene)

    def restore(self):
        self.scene = deepcopy(self._saved_scene)
        self.dirty = False

    def save(self) -> Path:
        """Back up and atomically replace the manifest, detecting external edits."""
        scene_boxes(self.scene)
        current = self.save_path.read_bytes() if self.save_path.exists() else None
        if current != self._disk_snapshot:
            raise RuntimeError('JSON 已被其他程序修改；为避免覆盖，请重启编辑器重新加载。')
        if not self.dirty and current is not None:
            return self.save_path
        payload = deepcopy(self.payload)
        payload['scene'] = deepcopy(self.scene)
        metadata = payload.setdefault('metadata', {})
        # Old collision checks no longer describe the edited geometry.
        metadata.pop('kinematic_scene_check', None)
        metadata.update(scene_status='manually_edited', physics_tracking_validated=False,
                        source_motion_sha256=hashlib.sha256(self.clip.motion_file.read_bytes()).hexdigest(),
                        scene_edited_at=datetime.now(timezone.utc).isoformat())
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if current is not None:
            backups = self.save_path.parent / '.scene_backups'
            backups.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            shutil.copy2(self.save_path, backups / f'{self.save_path.stem}.{stamp}.json')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.save_path.parent, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if self.save_path.exists():
                temporary.chmod(self.save_path.stat().st_mode & 0o777)
            os.replace(temporary, self.save_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.payload = payload
        self._disk_snapshot = encoded
        self._saved_scene = deepcopy(self.scene)
        self.dirty = False
        return self.save_path

    def check_current_frame(self) -> dict:
        from mjlab.entity import Entity
        from prcf.asset.robot.e1.e1_constants import get_e1_robot_cfg
        spec = Entity(get_e1_robot_cfg()).spec
        SceneBuilder(self.scene)(spec)
        model = spec.compile()
        state = mujoco.MjData(model)
        set_native_frame(model, state, self.motion, self.frame)
        contacts = []
        for contact in state.contact:
            names = [model.geom(int(i)).name or '' for i in contact.geom]
            if any(n.startswith('prcf_') for n in names) and contact.dist < 0:
                contacts.append({'distance_m': float(contact.dist), 'geoms': names})
        contacts.sort(key=lambda c: c['distance_m'])
        return {'frame': self.frame, 'max_penetration_m': -contacts[0]['distance_m'] if contacts else 0,
                'contacts': contacts[:5], 'scope': 'current-frame kinematics only'}


class WebReplayEditor:
    """Serialize client events on the replay thread; no GPU or training runner."""

    def __init__(self, source, *, host='127.0.0.1', port=8080, server=None):
        import viser
        from mjviser import ViserMujocoScene
        if server is None:
            # Viser retries failed binds internally; fail promptly for denied or
            # occupied ports instead of waiting inside its startup semaphore.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
        self.paths, initial = discover_motions(source)
        self.model, _, _ = e1_model()
        self.state = mujoco.MjData(self.model)
        self.documents = {initial: ReplayDocument(initial)}
        self.path = initial
        self.selected = None
        self.playing = False
        self.cursor = 0.0
        self.controls_elapsed = 0.0
        self.queue = Queue()
        self.box_handles = {}
        self.message = '调整后点击“保存场景”。切换动作会保留未保存草稿。'
        self.server = server or viser.ViserServer(host=host, port=port, label='PRCF 动作与场景编辑器')
        self.server.scene.set_up_direction('+z')
        self.server.gui.configure_theme(dark_mode=False)
        self.robot_view = ViserMujocoScene(self.server, self.model, num_envs=1)
        self.ground = self.server.scene.add_grid(
            '/prcf_scene/ground', infinite_grid=True, cell_size=0.1, section_size=1.0,
            plane_opacity=0.65, plane_color=(225, 230, 235), fade_distance=30.0)
        self.gizmo = self.server.scene.add_transform_controls(
            '/editor/selected', scale=0.35, disable_rotations=True, depth_test=False, visible=False)
        self.gizmo.on_update(lambda e: self._client_event(e, 'position', tuple(e.target.position)))
        self._create_gui()
        self.server.on_client_connect(lambda client: self._focus_camera(client))
        self._load(initial)

    @property
    def document(self):
        return self.documents[self.path]

    def _client_event(self, event, action, value=None):
        # Viser also invokes input callbacks on server-side .value assignments.
        if event.client_id is not None:
            self.queue.put((action, value))

    def _button(self, label, action, value=None):
        button = self.server.gui.add_button(label)
        button.on_click(lambda e: self._client_event(e, action, value))
        return button

    def _input(self, handle, action):
        handle.on_update(lambda e: self._client_event(e, action, e.target.value))
        return handle

    def _create_gui(self):
        gui = self.server.gui
        gui.add_markdown('### PRCF 动作与场景编辑器\n运动学预览 · 尺寸单位米 · 保存后供下次训练加载')
        with gui.add_folder('动作与播放'):
            self.motion_select = self._input(gui.add_dropdown(
                '动作', options=[p.name for p in self.paths], initial_value=self.path.name), 'load')
            self._button('上一个动作', 'next', -1)
            self._button('下一个动作', 'next', 1)
            self.play_button = self._button('播放', 'play')
            self._button('上一帧', 'step', -1)
            self._button('下一帧', 'step', 1)
            self.frame_slider = self._input(gui.add_slider(
                '帧', min=0, max=self.document.frames - 1, step=1, initial_value=0), 'seek')
            self.speed = self._input(gui.add_slider(
                '播放速度', min=0.1, max=2.0, step=0.1, initial_value=1.0), 'speed')
            self.loop = gui.add_checkbox('循环播放', initial_value=True)
            self._button('重置视角', 'camera')
        with gui.add_folder('障碍物'):
            self.object_select = self._input(gui.add_dropdown(
                '选择障碍物', options=['无'], initial_value='无'), 'select')
            self._button('添加箱体', 'add')
            self.delete_button = self._button('删除所选障碍物', 'delete')
            self.kind = gui.add_text('类型', initial_value='', disabled=True)
            self.position = self._input(gui.add_vector3('位置 XYZ', initial_value=(0., 0., 0.), step=0.01), 'position')
            self.euler = self._input(gui.add_vector3('旋转 XYZ（度）', initial_value=(0., 0., 0.), step=1.), 'euler')
            self.size = self._input(gui.add_vector3(
                '完整长宽高 XYZ', initial_value=(1., 1., 1.), min=(0.001, 0.001, 0.001), step=0.01), 'size')
            self.keep_bottom = gui.add_checkbox('修改尺寸时保持箱底高度', initial_value=True)
            self.step_height = self._input(gui.add_number('每级高度', initial_value=0.15, min=0.001, step=0.01), 'step_height')
            self.step_depth = self._input(gui.add_number('每级深度', initial_value=0.3, min=0.001, step=0.01), 'step_depth')
            self.width = self._input(gui.add_number('楼梯宽度', initial_value=1.2, min=0.001, step=0.01), 'width')
            self.num_steps = self._input(gui.add_number('台阶数量', initial_value=3, min=1, max=100, step=1), 'num_steps')
        with gui.add_folder('地面与保存'):
            self.ground_enabled = self._input(gui.add_checkbox('启用地面', initial_value=True), 'ground_enabled')
            self.ground_height = self._input(gui.add_number('地面高度', initial_value=0., step=0.01), 'ground_height')
            self._button('检查当前帧穿透', 'check')
            self._button('保存场景', 'save')
            self._button('恢复已保存场景', 'restore')
            gui.add_markdown('保存场景时会备份原 JSON。')
        self.status = gui.add_markdown('')

    def _load(self, path):
        # Construct first so a malformed file cannot replace the active draft.
        if path not in self.documents:
            self.documents[path] = ReplayDocument(path)
        self.path = path
        self.document.frame = 0
        self.cursor = 0.
        self.playing = False
        self.selected = next((o['id'] for o in self.document.scene.get('obstacles', [])), None)
        self.frame_slider.value = 0
        self.frame_slider.max = self.document.frames - 1
        self.motion_select.value = path.name
        self._sync_scene()
        self._sync_inputs()
        self._render_frame()
        self._focus_camera()

    def _sync_scene(self):
        boxes = scene_boxes(self.document.scene)
        names = {name for name, *_ in boxes}
        for name in set(self.box_handles) - names:
            self.box_handles.pop(name).remove()
        for name, size, position, quat, _ in boxes:
            selected = self.selected and (name == self.selected or name.startswith(self.selected + '_step_'))
            color = (62, 175, 190) if selected else (205, 177, 112)
            if name not in self.box_handles:
                self.box_handles[name] = self.server.scene.add_box(
                    '/prcf_scene/' + name, dimensions=size, position=position, wxyz=quat, color=color)
            else:
                handle = self.box_handles[name]
                handle.dimensions = tuple(size)
                handle.position = position
                handle.wxyz = quat
                handle.color = color
        ground = self.document.scene.get('ground_height', 0.)
        self.ground.visible = ground is not None
        self.ground.position = (0., 0., float(ground or 0))
        self.gizmo.visible = self.selected is not None
        if self.selected:
            self.gizmo.position = tuple(self.document.obstacle(self.selected).get('position', [0, 0, 0]))

    def _sync_inputs(self):
        names = [o['id'] for o in self.document.scene.get('obstacles', [])]
        self.object_select.options = names or ['无']
        self.object_select.value = self.selected or '无'
        has_object = self.selected is not None
        for handle in (self.position, self.euler, self.size, self.delete_button):
            handle.disabled = not has_object
        box = has_object and self.document.obstacle(self.selected)['type'] == 'box'
        self.size.visible = self.keep_bottom.visible = bool(box)
        for handle in (self.step_height, self.step_depth, self.width, self.num_steps):
            handle.visible = has_object and not box
        if has_object:
            obstacle = self.document.obstacle(self.selected)
            self.kind.value = obstacle['type']
            self.position.value = tuple(obstacle.get('position', [0., 0., 0.]))
            self.euler.value = tuple(Rotation.from_quat(
                obstacle.get('quaternion_wxyz', [1, 0, 0, 0]), scalar_first=True).as_euler('xyz', degrees=True))
            if box:
                self.size.value = tuple(obstacle['size_xyz'])
            else:
                for key in ('step_height', 'step_depth', 'width', 'num_steps'):
                    getattr(self, key).value = obstacle[key]
        else:
            self.kind.value = ''
        ground = self.document.scene.get('ground_height', 0.)
        self.ground_enabled.value = ground is not None
        self.ground_height.disabled = ground is None
        self.ground_height.value = float(ground or 0)
        self.play_button.label = '暂停' if self.playing else '播放'
        self._status()

    def _status(self):
        doc = self.document
        dirty = [p.name for p, draft in self.documents.items() if draft.dirty]
        self.status.content = (
            f'**{self.path.name}** · 帧 {doc.frame}/{doc.frames - 1} · '
            f'{doc.frame / doc.fps:.2f} 秒 · 源轨迹 {doc.fps:g} Hz\n\n'
            f'保存路径：`{doc.save_path}`\n\n'
            f'未保存：{", ".join(dirty) if dirty else "无"}\n\n{self.message}')

    def _render_frame(self, *, update_controls=True):
        doc = self.document
        # The final pose holds until the loop restarts at the first pose.
        set_native_frame(self.model, self.state, doc.motion, min(self.cursor, doc.frames - 1))
        self.robot_view.update_from_mjdata(self.state)
        if update_controls:
            self.controls_elapsed = 0.
            self.frame_slider.value = doc.frame
            self._status()

    def _focus_camera(self, client=None):
        points = self.document.motion['body_pos_w'][:, 0]
        bounds = [points.min(axis=0), points.max(axis=0)]
        for _, size, pos, _, _ in scene_boxes(self.document.scene):
            bounds.extend([pos - size / 2, pos + size / 2])
        bounds = np.array(bounds)
        center = (bounds.min(0) + bounds.max(0)) / 2
        center[2] = max(0.5, center[2])
        distance = max(3.5, float(np.max(np.ptp(bounds, axis=0))) * 1.3)
        for current in ([client] if client else self.server.get_clients().values()):
            current.camera.up_direction = (0, 0, 1)
            current.camera.look_at = tuple(center)
            current.camera.position = tuple(center + np.array([-0.8, -1., 0.65]) * distance)

    def dispatch(self, action, value=None):
        """Apply one serialized GUI action. Public for local integration checks."""
        doc = self.document
        self.message = ''
        if action == 'load':
            self._load(next(p for p in self.paths if p.name == value))
            return
        if action == 'next':
            self._load(self.paths[(self.paths.index(self.path) + value) % len(self.paths)])
            return
        if action == 'play':
            if self.playing:
                self.cursor = float(doc.frame)
            elif doc.frame == doc.frames - 1:
                self.cursor = 0.
                doc.frame = 0
            self.playing = not self.playing
        elif action in ('step', 'seek'):
            self.playing = False
            doc.frame = int(np.clip(doc.frame + value if action == 'step' else value, 0, doc.frames - 1))
            self.cursor = float(doc.frame)
        elif action == 'camera':
            self._focus_camera()
        elif action == 'select':
            self.selected = None if value == '无' else value
        elif action == 'add':
            self.selected = doc.add_box()
        elif action == 'delete':
            if self.selected:
                doc.delete_obstacle(self.selected)
                self.selected = next((o['id'] for o in doc.scene.get('obstacles', [])), None)
        elif action in ('position', 'size', 'euler', 'step_height', 'step_depth', 'width', 'num_steps'):
            if self.selected:
                if action == 'euler':
                    updates = {'quaternion_wxyz': Rotation.from_euler('xyz', value, degrees=True).as_quat(scalar_first=True).tolist()}
                else:
                    updates = { {'size': 'size_xyz'}.get(action, action): list(value) if action in ('position', 'size') else value }
                doc.edit_obstacle(self.selected, updates, keep_bottom=self.keep_bottom.value)
        elif action in ('ground_height', 'ground_enabled'):
            scene = deepcopy(doc.scene)
            scene['ground_height'] = float(value) if action == 'ground_height' else (float(self.ground_height.value) if value else None)
            doc.replace_scene(scene)
        elif action == 'save':
            path = doc.save()
            self.message = f'已保存 `{path.name}`。后续训练加载此 JSON 生效；正在运行的训练不变。'
        elif action == 'restore':
            doc.restore()
            self.selected = next((o['id'] for o in doc.scene.get('obstacles', [])), None)
            self.message = '已恢复到本次最后保存的场景。'
        elif action == 'check':
            check = doc.check_current_frame()
            self.message = f'当前帧最大环境穿透：**{check["max_penetration_m"] * 1000:.1f} mm**（仅运动学检查）。'
        self._sync_scene()
        self._sync_inputs()
        self._render_frame()

    def tick(self, elapsed):
        if not self.playing:
            return
        doc = self.document
        previous_cursor = self.cursor
        self.cursor += elapsed * doc.fps * self.speed.value
        if self.cursor >= doc.frames:
            if self.loop.value:
                self.cursor %= doc.frames
            else:
                self.cursor = doc.frames - 1
                self.playing = False
                self.play_button.label = '播放'
        doc.frame = int(self.cursor)
        self.controls_elapsed += elapsed
        update_controls = not self.playing or self.controls_elapsed >= 0.1
        if self.cursor != previous_cursor or update_controls:
            self._render_frame(update_controls=update_controls)

    def run(self):
        print(f'PRCF web replay: http://{self.server.get_host()}:{self.server.get_port()}', flush=True)
        print('Ctrl+C 停止。场景仅在点击“保存场景”时写入 JSON。', flush=True)
        previous = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                elapsed = min(now - previous, 0.25)
                previous = now
                while True:
                    try:
                        action, value = self.queue.get_nowait()
                    except Empty:
                        break
                    try:
                        self.dispatch(action, value)
                    except (ValueError, KeyError, RuntimeError, OSError, StopIteration) as error:
                        self.message = f'操作失败：{error}'
                        self._sync_inputs()
                        self.motion_select.value = self.path.name
                self.tick(elapsed)
                time.sleep(max(0., 1 / 60 - (time.monotonic() - now)))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 reference or exported AMP NPZ, scene JSON, or a directory in web mode')
    parser.add_argument('--viewer', choices=('viser', 'native'), default='viser',
                        help='web scene editor (default), or native replay')
    parser.add_argument('--host', default='127.0.0.1', help='web viewer bind address')
    parser.add_argument('--port', type=int, default=8080, help='web viewer port')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.viewer == 'viser' and not args.headless:
        try:
            WebReplayEditor(args.motion_file, host=args.host, port=args.port).run()
        except (OSError, ValueError) as error:
            parser.exit(2, f'无法启动网页回放：{error}\n可使用 --port 指定其他端口，或 --viewer native。\n')
        return 0
    if Path(args.motion_file).is_dir():
        parser.error('native/headless replay requires a single NPZ or JSON file')
    try:
        clip, motion = load_replay_motion(args.motion_file)
    except (OSError, ValueError) as error:
        parser.exit(2, f'无法加载回放：{error}\n')
    frames = len(motion['joint_pos'])
    fps = float(motion['fps'].item())
    from mjlab.scene import Scene
    from prcf.tasks.mimic.config.e1.env_cfgs import e1_mimic_env_cfg
    cfg = e1_mimic_env_cfg('play')
    cfg.scene.spec_fn = SceneBuilder(clip.scene)
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    state = mujoco.MjData(model)
    if args.headless:
        for frame in range(frames):
            set_native_frame(model, state, motion, frame)
        print(f"Replayed {frames} frames using native forward kinematics; no physics rollout performed.")
    else:
        from mujoco import viewer as native_viewer
        set_native_frame(model, state, motion, 0)
        with native_viewer.launch_passive(model, state) as viewer:
            for frame in range(frames):
                if not viewer.is_running():
                    break
                start = time.monotonic()
                set_native_frame(model, state, motion, frame)
                viewer.sync()
                time.sleep(max(0, 1 / fps - (time.monotonic() - start)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
