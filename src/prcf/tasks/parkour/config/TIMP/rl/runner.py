"""Keep the inherited training loop and align Stage 3's support schedule on resume."""
from mjlab.rl import MjlabOnPolicyRunner


class TimpRunner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        raw = self.env.unwrapped
        raw.timp_stage_start = raw.common_step_counter - self.alg.stage_step
        return infos
