from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from backend.arm import H2ArmController


class _Publisher:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.writes = 0

    def Write(self, _cmd):
        if self.fail:
            raise RuntimeError("publish failed")
        self.writes += 1


def _controller(publisher: _Publisher) -> H2ArmController:
    ctl = H2ArmController.__new__(H2ArmController)
    ctl.n = 2
    ctl._lock = threading.Lock()
    ctl._low_cmd = SimpleNamespace(
        motor_cmd=[SimpleNamespace(tau=0.0, q=0.0, dq=0.0, kp=0.0, kd=0.0)
                   for _ in range(32)],
        crc=0,
    )
    ctl._jog_indices = [22, 23]
    ctl._other_indices = [15, 16]
    ctl._other_tau = np.array([1.0, 2.0])
    ctl._other_hold_q = np.array([0.3, 0.4])
    ctl.kp_vec = np.array([80.0, 50.0])
    ctl.kd_vec = np.array([1.5, 2.0])
    ctl.hand_move_kd = 2.0
    ctl._crc = SimpleNamespace(Crc=lambda _cmd: 123)
    ctl._publisher = publisher
    ctl._last_sent_q = np.zeros(2)
    ctl._last_sent_tau_ff = np.zeros(2)
    ctl._last_sent_at = None
    ctl._last_sent_sequence = 0
    return ctl


class ArmCommandSnapshotTests(unittest.TestCase):
    def test_snapshot_updates_only_after_successful_publish(self):
        ctl = _controller(_Publisher())
        q = np.array([0.12, -0.34])
        tau = np.array([3.5, -1.2])

        ctl._write_command(q, False, 1.0, tau)
        snapshot = ctl.command_snapshot()

        np.testing.assert_allclose(snapshot["q_rad"], q)
        np.testing.assert_allclose(snapshot["tau_ff_nm"], tau)
        self.assertEqual(snapshot["sequence"], 1)
        snapshot["q_rad"][0] = 99.0
        self.assertAlmostEqual(ctl.command_snapshot()["q_rad"][0], q[0])

    def test_failed_publish_does_not_advance_snapshot(self):
        ctl = _controller(_Publisher(fail=True))

        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            ctl._write_command(np.array([0.1, 0.2]), False, 1.0, np.zeros(2))
        with self.assertRaisesRegex(RuntimeError, "has not published"):
            ctl.command_snapshot()


if __name__ == "__main__":
    unittest.main()
