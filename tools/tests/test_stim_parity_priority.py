import json
import tempfile
import unittest
from pathlib import Path

from tools.stim_parity_priority import render_markdown, summarize_missing


class StimParityPriorityTests(unittest.TestCase):
    def test_summarize_missing_groups_by_family(self) -> None:
        report = {
            'covered_rows': 10,
            'missing_rows': 4,
            'missing': [
                {'family': 'stim.Circuit', 'taxonomy_class': 'D0', 'api_name': 'stim.Circuit.diagram'},
                {'family': 'stim.Circuit', 'taxonomy_class': 'P0', 'api_name': 'stim.Circuit.compile_sampler'},
                {'family': 'stim.Flow', 'taxonomy_class': 'D0', 'api_name': 'stim.Flow.output_copy'},
                {'family': 'stim.TableauSimulator', 'taxonomy_class': 'P0', 'api_name': 'stim.TableauSimulator.h'},
            ],
        }
        summary = summarize_missing(report)
        self.assertEqual(summary['covered_rows'], 10)
        self.assertEqual(summary['missing_rows'], 4)
        self.assertEqual(summary['families'][0]['family'], 'stim.Circuit')
        self.assertEqual(summary['families'][0]['missing_count'], 2)
        self.assertEqual(summary['families'][0]['taxonomy_counts'], {'D0': 1, 'P0': 1})

    def test_render_markdown_includes_examples(self) -> None:
        summary = {
            'covered_rows': 10,
            'missing_rows': 2,
            'families': [
                {
                    'family': 'stim.Flow',
                    'missing_count': 2,
                    'taxonomy_counts': {'D0': 2},
                    'examples': ['stim.Flow.output_copy', 'stim.Flow.input_copy'],
                }
            ],
        }
        markdown = render_markdown(summary)
        self.assertIn('stim.Flow', markdown)
        self.assertIn('stim.Flow.output_copy', markdown)
        self.assertIn('D0=2', markdown)


if __name__ == '__main__':
    unittest.main()
