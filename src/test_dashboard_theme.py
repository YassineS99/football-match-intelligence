"""Regression checks for read-only Night Match presentation helpers."""

import unittest

import dashboard_theme as ui


class NightMatchThemeTests(unittest.TestCase):
    def test_every_workspace_has_navigation_and_header(self):
        self.assertEqual(len(ui.NAV_LABELS), 9)
        for page in ui.NAV_LABELS:
            with self.subTest(page=page):
                hero = ui.page_hero(page, run_id="run-1", frames=368,
                                    duration=14.72, directory="tracking_0_367_retry1")
                self.assertIn("fmi-hero", hero)
                self.assertIn("368", hero)
                self.assertIn("14.72s", hero)
                self.assertIn("tracking_0_367_retry1", hero)

    def test_metadata_is_html_escaped(self):
        hero = ui.page_hero("Overview", run_id='<script>alert(1)</script>',
                            frames=100, duration=4.0, directory='"<img src=x>')
        self.assertNotIn("<script>", hero)
        self.assertNotIn("<img", hero)
        self.assertIn("&lt;", hero)
        self.assertIn("&quot;", hero)

    def test_mapping_acceptance_is_not_pitch_area(self):
        coverage = ui.mapping_coverage_panel(854, 5156)
        self.assertIn("16.6%", coverage)
        self.assertIn("854", coverage)
        self.assertIn("5,156", coverage)
        self.assertIn("not the percentage of pitch area", coverage)

    def test_missing_mapping_is_unavailable_not_zero_coverage(self):
        coverage = ui.mapping_coverage_panel(0, 0)
        self.assertIn("Unavailable", coverage)
        self.assertIn("width:0.00%", coverage)

    def test_theme_has_no_external_fonts_or_urls(self):
        self.assertIn("#080f1b", ui.CSS.lower())
        self.assertNotIn("@import", ui.CSS)
        self.assertNotIn("https://", ui.CSS)


if __name__ == "__main__":
    unittest.main()
