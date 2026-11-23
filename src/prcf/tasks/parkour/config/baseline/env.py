"""Environment with a true 10 Hz camera and a 50 Hz control policy."""

from __future__ import annotations

from functools import partial
import sys
import types

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv

from .sensors import PeriodicCameraSensor


class BaselineEnv(ManagerBasedRlEnv):
  """Run camera rendering every fifth policy step while sensing rays at 50 Hz."""

  def __init__(self, cfg, device, render_mode=None, *, cuda_debug=False, **kwargs) -> None:
    super().__init__(cfg, device, render_mode, **kwargs)
    self._cuda_debug = cuda_debug

    simulation = self.sim
    if cuda_debug:
      for name in ("step", "reset", "forward"):
        operation = getattr(simulation, name)
        setattr(simulation, name, partial(self._cuda_stage, f"simulation.{name}", operation))
    original_create_graph = simulation.create_graph

    # The Python render-cadence decision cannot be captured in a CUDA graph.
    # Keep the normal physics/forward/reset graphs and execute sensing directly.
    def create_graph_without_sense_graph(simulation_self) -> None:
      original_create_graph()
      simulation_self.sense_graph = None

    simulation.create_graph = types.MethodType(
      create_graph_without_sense_graph, simulation
    )
    simulation.sense_graph = None
    self._install_periodic_camera_sense()

  def _cuda_stage(self, name, operation, *args, **kwargs):
    if not self._cuda_debug:
      return operation(*args, **kwargs)
    point = f"before {name}"
    try:
      torch.cuda.synchronize(self.device)
      point = f"during {name}"
      result = operation(*args, **kwargs)
      torch.cuda.synchronize(self.device)
      return result
    except RuntimeError as error:
      message = f"Baseline CUDA failure {point}, control step {self.common_step_counter}: {error}"
      print(message, file=sys.stderr, flush=True)
      raise RuntimeError(message) from error

  def _install_periodic_camera_sense(self) -> None:
    simulation = self.sim
    context = self.scene.sensor_context
    if context is None:
      raise RuntimeError("Periodic camera task requires a sensor context.")

    cameras = [
      sensor
      for sensor in context.camera_sensors
      if isinstance(sensor, PeriodicCameraSensor)
    ]
    if len(cameras) != 1:
      raise RuntimeError(
        "Expected exactly one PeriodicCameraSensor, found "
        f"{len(cameras)}."
      )
    camera = cameras[0]

    def sense_with_periodic_camera(simulation_self) -> None:
      ctx = simulation_self._sensor_context
      if ctx is None:
        return

      self._cuda_stage("sensor preparation", ctx.prepare)
      render_now = camera.should_render()
      with wp.ScopedDevice(simulation_self.wp_device):
        rc = ctx.render_context
        self._cuda_stage(
          "BVH refit", mjwarp.refit_bvh,
          simulation_self.wp_model,
          simulation_self.wp_data,
          rc,
        )
        if render_now:
          self._cuda_stage(
            "depth rendering", mjwarp.render,
            simulation_self.wp_model,
            simulation_self.wp_data,
            rc,
          )
          self._cuda_stage("camera buffer unpack", ctx.unpack_rgb)
        for ray_sensor in ctx.raycast_sensors:
          self._cuda_stage(f"raycast {ray_sensor.cfg.name}", ray_sensor.raycast_kernel, rc=rc)
      self._cuda_stage("sensor readback", ctx.finalize)
      camera.finish_sense(render_now)

    simulation.sense = types.MethodType(
      sense_with_periodic_camera, simulation
    )


__all__ = ["BaselineEnv"]
