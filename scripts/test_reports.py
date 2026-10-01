import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from render_training_report import render
from evaluate_requests import evaluate


class ReportTests(unittest.TestCase):
    def test_report_escapes_metadata_and_has_two_charts(self):
        manifest = json.loads((Path(__file__).resolve().parents[1]/'testdata/artifact/tiny-q/float32/manifest.json').read_text())
        manifest['limitations'].append('<script>alert(1)</script>')
        result = render(manifest)
        self.assertEqual(result.count('<svg'), 2)
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('dev_cal_nll', result)

    def test_evaluation_counts_errors_without_fabricated_predictions(self):
        cases = [{'request': {'id':'one'}, 'expected': {'support':'contradicts'}}]
        with patch('evaluate_requests.subprocess.run', return_value=SimpleNamespace(stdout='{"id":"one","error":{"code":"input_truncated"}}\n')):
            report = evaluate('binary', [], cases)
        self.assertEqual(report['errors'], 1)
        self.assertEqual(report['correct'], 0)
        self.assertEqual(report['false_supports'], 0)
        self.assertIsNone(report['median_ms'])

    def test_evaluation_refuses_response_identity_and_count_drift(self):
        cases = [{'request': {'id':'one'}, 'expected': {'support':'supports'}}]
        for output in ['', '{"id":"wrong","error":{}}\n']:
            with patch('evaluate_requests.subprocess.run', return_value=SimpleNamespace(stdout=output)):
                with self.assertRaises(ValueError):
                    evaluate('binary', [], cases)
