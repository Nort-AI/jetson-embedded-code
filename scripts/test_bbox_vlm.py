"""
scripts/test_bbox_vlm.py — Smoke-test for bbox_renderer and vlm_session.

Run from project root:
    python scripts/test_bbox_vlm.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import unittest
from unittest.mock import patch, MagicMock

# ── Test bbox_renderer ─────────────────────────────────────────────────────────
class TestBboxRenderer(unittest.TestCase):

    def setUp(self):
        from core import bbox_renderer
        self.renderer = bbox_renderer
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.bbox  = (100, 100, 300, 500)

    def _frame_was_modified(self, frame):
        return frame.sum() > 0

    def test_draw_hud_box_no_vlm(self):
        f = self.frame.copy()
        self.renderer.draw_hud_box(f, self.bbox, global_id=42,
                                   local_track_id=1, camera_id="cam1")
        self.assertTrue(self._frame_was_modified(f), "Frame should be modified")

    def test_draw_hud_box_vlm_pending(self):
        f = self.frame.copy()
        self.renderer.draw_hud_box(f, self.bbox, global_id=42,
                                   local_track_id=1, camera_id="cam1",
                                   vlm_result={"status": "pending", "text": ""})
        self.assertTrue(self._frame_was_modified(f))

    def test_draw_hud_box_vlm_done(self):
        f = self.frame.copy()
        self.renderer.draw_hud_box(f, self.bbox, global_id=42,
                                   local_track_id=1, camera_id="cam1",
                                   vlm_result={"status": "done",
                                               "text": "Person wearing blue jacket carrying a bag."})
        self.assertTrue(self._frame_was_modified(f))

    def test_draw_hud_box_vlm_error(self):
        f = self.frame.copy()
        self.renderer.draw_hud_box(f, self.bbox, global_id=99,
                                   local_track_id=3, camera_id="cam2",
                                   vlm_result={"status": "error", "text": "timeout"})
        self.assertTrue(self._frame_was_modified(f))

    def test_draw_selection_hud(self):
        f = self.frame.copy()
        self.renderer.draw_selection_hud(f, self.bbox, frame_counter=42)
        self.assertTrue(self._frame_was_modified(f))

    def test_clear_track_state(self):
        f = self.frame.copy()
        # Render twice to build scan state, then clear
        self.renderer.draw_hud_box(f, self.bbox, global_id=7, local_track_id=1)
        self.renderer.draw_hud_box(f, self.bbox, global_id=7, local_track_id=1)
        self.renderer.clear_track_state(7)
        # Should not raise after clear
        self.renderer.draw_hud_box(f, self.bbox, global_id=7, local_track_id=1)

    def test_out_of_bounds_bbox(self):
        f = self.frame.copy()
        # Should not crash even if bbox is partially outside frame
        self.renderer.draw_hud_box(f, (-50, -50, 50, 50), global_id=1,
                                   local_track_id=1, camera_id="cam1")
        self.renderer.draw_hud_box(f, (1200, 600, 1400, 800), global_id=2,
                                   local_track_id=2, camera_id="cam1")


# ── Test vlm_session ───────────────────────────────────────────────────────────
class TestVLMSession(unittest.TestCase):

    def _make_session(self, cooldown=5.0):
        from core import vlm_session as mod
        # Patch vlm_analyst functions used by VLMSession
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", return_value=True), \
             patch.object(mod.vlm_analyst, "get_result",
                          return_value={"status": "not_found", "text": "", "ts": 0.0, "mode": ""}):
            session = mod.VLMSession("test_cam", cooldown_s=cooldown)
        return session

    def test_request_first_call_succeeds(self):
        from core import vlm_session as mod
        session = self._make_session()
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", return_value=True):
            result = session.request("track_1")
        self.assertTrue(result)

    def test_request_throttled_during_cooldown(self):
        from core import vlm_session as mod
        session = self._make_session(cooldown=60.0)
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", return_value=True):
            first  = session.request("track_2")
            second = session.request("track_2")   # immediate retry
        self.assertTrue(first)
        self.assertFalse(second, "Should be throttled by cooldown")

    def test_request_allowed_after_cooldown(self):
        from core import vlm_session as mod
        session = self._make_session(cooldown=0.01)
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", return_value=True):
            session.request("track_3")
            time.sleep(0.05)   # > cooldown
            result = session.request("track_3")
        self.assertTrue(result, "Should succeed after cooldown expires")

    def test_tick_empty_list(self):
        from core import vlm_session as mod
        session = self._make_session()
        # Should not raise
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True):
            session.tick([])

    def test_tick_fires_auto_scan(self):
        from core import vlm_session as mod
        session = self._make_session(cooldown=0.0)
        submitted = []
        def fake_submit(track_id, **kwargs):
            submitted.append(track_id)
            return True

        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", side_effect=fake_submit):
            session._last_auto_scan_ts = 0   # force scan
            session.tick(["global_7", "global_8"])

        self.assertTrue(len(submitted) >= 1, "Auto-scan should have submitted a job")

    def test_clear_track_removes_cooldown(self):
        from core import vlm_session as mod
        session = self._make_session(cooldown=60.0)
        with patch.object(mod.vlm_analyst, "is_enabled", return_value=True), \
             patch.object(mod.vlm_analyst, "has_crop", return_value=True), \
             patch.object(mod.vlm_analyst, "submit_analysis", return_value=True):
            session.request("track_x")
            session.clear_track("track_x")
            result = session.request("track_x")  # should work again
        self.assertTrue(result, "After clear_track cooldown should reset")

    def test_result_callback_fires_once(self):
        from core import vlm_session as mod
        session = self._make_session()
        fired = []
        session.set_result_callback(lambda tid, r: fired.append(tid))
        done_result = {"status": "done", "text": "blue jacket", "ts": 99.0, "mode": "describe"}

        with patch.object(mod.vlm_analyst, "get_result", return_value=done_result):
            session.get("track_y")
            session.get("track_y")   # second call — should NOT fire again

        self.assertEqual(len(fired), 1, "Callback should fire exactly once")

    def test_get_session_registry(self):
        from core import vlm_session as mod
        s1 = mod.get_session("cam_alpha")
        s2 = mod.get_session("cam_alpha")
        self.assertIs(s1, s2, "Same camera_id should return the same session object")


if __name__ == "__main__":
    unittest.main(verbosity=2)
