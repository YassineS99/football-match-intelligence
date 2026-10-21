"""Regression tests for the read-only Stage 3 pitch explorer."""

import unittest

import numpy as np
import pandas as pd

import pitch_explorer as pitch


class PitchExplorerTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([
            # Run A, one connected segment, mapped track 24 at frames 100-101.
            dict(run_id="run-a", frame=100, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=12, pitch_y_m=30, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=101, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=13, pitch_y_m=31, predicted_role="team_a_outfield"),
            # Gap at 102: two disconnected mapped pieces MUST NOT join.
            dict(run_id="run-a", frame=103, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=14, pitch_y_m=32, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=104, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=15, pitch_y_m=33, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=100, track_id="31", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="1",
                 pitch_x_m=7, pitch_y_m=18, predicted_role="team_b_outfield"),
            dict(run_id="run-a", frame=105, track_id="24", class_id=0,
                 camera_segment_id="segment_001", pitch_mapping_valid="True",
                 pitch_x_m=30, pitch_y_m=28, predicted_role="team_a_outfield"),
            # Invalid/out-of-region points are not drawn even when finite.
            dict(run_id="run-a", frame=106, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="False",
                 pitch_x_m=50, pitch_y_m=60, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=107, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=110, pitch_y_m=60, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=108, track_id="24", class_id=0,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=np.nan, pitch_y_m=22, predicted_role="team_a_outfield"),
            dict(run_id="run-a", frame=109, track_id="99", class_id=32,
                 camera_segment_id="segment_000", pitch_mapping_valid="True",
                 pitch_x_m=10, pitch_y_m=20, predicted_role="unknown"),
        ])

    def filtered(self, **kwargs):
        return pitch.valid_run_positions(self.rows(), run_id="run-a", segment_id="segment_000",
                                         length=105, width=68, **kwargs)

    def test_accepted_samples_only_and_no_cross_segment_merging(self):
        result = self.filtered()
        self.assertEqual(len(result), 5)
        self.assertEqual(set(result["track_id"]), {"24", "31"})
        self.assertEqual(set(result["frame"]), {100, 101, 103, 104})
        self.assertTrue(result["pitch_x_m"].between(0, 105).all())

    def test_frame_role_and_track_filters(self):
        result = self.filtered(roles=["team_a_outfield"], track_ids=["24"],
                               start_frame=101, end_frame=103)
        self.assertEqual(result["frame"].tolist(), [101, 103])
        self.assertEqual(len(self.filtered(roles=[])), 0)
        self.assertEqual(len(self.filtered(track_ids=[])), 0)

    def test_run_id_mismatch_and_same_track_id_other_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "tracking run"):
            pitch.valid_run_positions(self.rows(), run_id="run-b", segment_id="segment_000",
                                      length=105, width=68)
        other = self.rows().iloc[:1].copy()
        other["run_id"] = "run-b"
        with self.assertRaisesRegex(ValueError, "tracking run"):
            pitch.valid_run_positions(pd.concat([self.rows(), other]), run_id="run-a",
                                      segment_id="segment_000", length=105, width=68)

    def test_duplicate_observation_is_not_silently_drawn_twice(self):
        combined = pd.concat([self.rows(), self.rows().iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pitch.valid_run_positions(combined, run_id="run-a", segment_id="segment_000",
                                      length=105, width=68)

    def test_missing_mapping_returns_empty_with_required_columns(self):
        result = pitch.valid_run_positions(None, run_id="run-a", segment_id="segment_000",
                                           length=105, width=68)
        self.assertTrue(result.empty)
        self.assertIn("predicted_role", result)

    def test_safe_pitch_dimensions_finite_and_assumed(self):
        self.assertEqual(pitch.safe_pitch_geometry({"pitch_geometry": {"length_m": 105, "width_m": 68}}),
                         (105.0, 68.0))
        for invalid in (float('inf'), float('nan'), 0, -100, 1000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pitch.safe_pitch_geometry({"pitch_geometry": {"length_m": invalid, "width_m": 68}})

    def test_hull_requires_verified_calibration_and_same_segment(self):
        good = {"valid_for_mapping": True, "camera_segment_id": "segment_000",
                "pitch_calibration_hull": [[2, 20], [15, 20], [5, 36]]}
        self.assertEqual(pitch.supported_hull(good, camera_segment_id="segment_000",
                                             length=105, width=68).shape, (3, 2))
        self.assertIsNone(pitch.supported_hull(good, camera_segment_id="segment_001",
                                                length=105, width=68))
        good["valid_for_mapping"] = False
        self.assertIsNone(pitch.supported_hull(good, camera_segment_id="segment_000",
                                                length=105, width=68))

    def test_hull_does_not_infer_polygon_from_player_positions(self):
        self.assertIsNone(pitch.supported_hull(None, camera_segment_id="segment_000",
                                                length=105, width=68))
        candidate = {"valid_for_mapping": True, "camera_segment_id": "segment_000",
                     "pitch_calibration_hull": [[1, 2], [1, 2], [1, 2]]}
        self.assertIsNone(pitch.supported_hull(candidate, camera_segment_id="segment_000",
                                                length=105, width=68))
        candidate["pitch_calibration_hull"] = [[1, 2], [1, 5], [150, 7]]
        self.assertIsNone(pitch.supported_hull(candidate, camera_segment_id="segment_000",
                                                length=105, width=68))

    def test_trajectory_segments_split_at_unmapped_frames(self):
        groups = pitch.continuous_trajectories(self.filtered(track_ids=["24"]))
        self.assertEqual([group["rows"]["frame"].tolist() for group in groups],
                         [[100, 101], [103, 104]])

    def test_continuous_trajectories_never_join_camera_segments(self):
        a = self.filtered(track_ids=["24"])
        b = pitch.valid_run_positions(self.rows(), run_id="run-a",
                                      segment_id="segment_001", length=105, width=68)
        groups = pitch.continuous_trajectories(pd.concat([a, b]))
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[-1]["segment_id"], "segment_001")

    def test_frame_plot_contains_only_selected_frame(self):
        fig = pitch.make_pitch_figure(self.filtered(), length=105, width=68,
                                      segment_id="segment_000", mode="frame", frame_number=101)
        all_x = [x for trace in fig.data if trace.name and "team" in trace.name for x in trace.x]
        self.assertEqual(all_x, [13.0])
        self.assertEqual(fig.layout.yaxis.range[0], 70)

    def test_trajectory_plot_draws_only_connected_pairs(self):
        fig = pitch.make_pitch_figure(self.filtered(track_ids=["24"]), length=105, width=68,
                                      segment_id="segment_000", mode="trajectories")
        groups = [trace for trace in fig.data if trace.name and trace.name.startswith("Track 24")]
        self.assertEqual(len(groups), 2)
        self.assertEqual([list(trace.x) for trace in groups], [[12.0, 13.0], [14.0, 15.0]])

    def test_sample_density_uses_only_observed_points_no_interpolation(self):
        rows = self.filtered()
        fig = pitch.make_pitch_figure(rows, length=105, width=68,
                                      segment_id="segment_000", mode="samples")
        samples = [trace for trace in fig.data if trace.name and "outfield" in trace.name]
        self.assertEqual(sum(len(trace.x) for trace in samples), len(rows))
        self.assertFalse(any(getattr(trace, "type", "") == "heatmap" for trace in fig.data))

    def test_pitch_canvas_is_dark_and_field_shading_stays_within_boundaries(self):
        fig = pitch.make_pitch_figure(self.filtered(), length=105, width=68,
                                      segment_id="segment_000", mode="frame", frame_number=101)
        self.assertNotEqual(fig.layout.plot_bgcolor, "#164e2e")
        backgrounds = [shape for shape in fig.layout.shapes
                       if getattr(shape, "fillcolor", None)]
        self.assertGreaterEqual(len(backgrounds), 2)
        for shape in backgrounds:
            self.assertGreaterEqual(shape.x0, 0)
            self.assertLessEqual(shape.x1, 105)
            self.assertGreaterEqual(shape.y0, 0)
            self.assertLessEqual(shape.y1, 68)

    def test_trajectory_legend_is_role_based_not_repeated_for_each_fragment(self):
        fig = pitch.make_pitch_figure(self.filtered(track_ids=["24"]), length=105,
                                      width=68, segment_id="segment_000", mode="trajectories")
        segments = [trace for trace in fig.data if str(trace.name).startswith("Track 24")]
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(trace.showlegend is False for trace in segments))
        legend_roles = [trace for trace in fig.data if trace.showlegend is True]
        self.assertEqual(len(legend_roles), 1)
        self.assertEqual(legend_roles[0].name, "team a outfield")
        self.assertEqual(len(legend_roles[0].x), 0)

    def test_unrecognized_mode_rejected(self):
        with self.assertRaises(ValueError):
            pitch.make_pitch_figure(self.filtered(), length=105, width=68,
                                    segment_id="segment_000", mode="invented")


if __name__ == "__main__":
    unittest.main()
