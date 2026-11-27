"""APEX contact and reward regressions."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from prcf.tasks.parkour.config.apex.sensors import GroundContact


class GroundContactTests(unittest.TestCase):
    def test_ground_reaction_stops_air_timer(self):
        sensor = GroundContact.__new__(GroundContact)
        state = SimpleNamespace(
            current_air_time=torch.full((1, 3), 0.1),
            last_air_time=torch.zeros(1, 3),
            current_contact_time=torch.zeros(1, 3),
            last_contact_time=torch.zeros(1, 3),
        )
        sensor._air_time_state = state
        # Foot on the ground, limb on a vertical face, and upper contact from above.
        contacts = SimpleNamespace(force=torch.tensor([[[0., 0., -143.], [50., 0., 0.], [0., 0., 20.]]]))
        with patch.object(sensor, "_extract_sensor_data", return_value=contacts):
            sensor._update_air_time_tracking(0.005)
        torch.testing.assert_close(state.current_air_time, torch.tensor([[0.0, 0.105, 0.105]]))
        torch.testing.assert_close(state.current_contact_time, torch.tensor([[0.005, 0.0, 0.0]]))
        torch.testing.assert_close(state.last_air_time, torch.tensor([[0.105, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
