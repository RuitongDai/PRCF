"""Register vision-based parkour tasks."""
from importlib.util import find_spec

from .config import apex as apex
from .config import baseline as baseline
from .config import box_obstacles as box_obstacles

if find_spec(f"{__name__}.config.TIMP") is not None:
    from .config import TIMP as TIMP
