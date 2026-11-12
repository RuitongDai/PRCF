"""Separate state samples (N) from the physics intervals between them (N-1)."""

import numpy as np


class TrajectoryBuffer:
    def __init__(self):
        self.states = []
        self.intervals = []

    def add_state(self, state):
        self.states.append({key: np.asarray(value).copy() for key, value in state.items()})

    def add_interval(self, substeps):
        if not substeps:
            raise RuntimeError("No physics-substep data were recorded")
        self.intervals.append({key: np.stack([step[key] for step in substeps]) for key in substeps[0]})

    def arrays(self):
        if not self.states:
            raise RuntimeError("Empty trajectory")
        states = {key: np.stack([state[key] for state in self.states]) for key in self.states[0]}
        intervals = {
            key: np.stack([interval[key] for interval in self.intervals]) for key in self.intervals[0]
        } if self.intervals else {}
        if len(self.intervals) != len(self.states) - 1:
            raise RuntimeError("Expected exactly N-1 physics intervals for N states")
        return states, intervals
