#!/usr/bin/env python3
"""Provenance checks use temporary synthetic artifacts, never model benchmarks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import export_benchmark_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'benchmarks/results/record.json'
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text('{"runs": [], "prompt": "synthetic", "logical_cpus": 24}\n')
        self.artifact = {
            'path': self.raw.relative_to(self.root).as_posix(),
            'sha256': hashlib.sha256(self.raw.read_bytes()).hexdigest(),
            'kind': 'generation', 'testbed': 'unknown',
            'status': 'historical_unverified', 'aliases': ['archive/record.json'],
        }
        self.manifest = {'schema_version': 1, 'artifacts': [self.artifact]}

    def errors(self):
        return evidence.validate_manifest(self.manifest, self.root)

    def test_unknown_hardware_stays_unknown_despite_24_threads(self):
        self.assertEqual(self.errors(), [])
        report = evidence.render_report(self.manifest, self.root)
        self.assertIn('| unknown |', report)
        self.assertNotIn('0.0', report)
        self.assertIn('chưa ghi nhận', report)

    def test_wrong_digest_and_modified_payload_fail(self):
        self.raw.write_text('{"runs": [], "prompt": "changed"}')
        self.assertTrue(any('sha256 mismatch' in error for error in self.errors()))

    def test_duplicate_content_and_paths_fail(self):
        self.manifest['artifacts'].append(copy.deepcopy(self.artifact))
        errors = self.errors()
        self.assertTrue(any('duplicate artifact content' in error for error in errors))
        self.assertTrue(any('duplicate canonical path' in error for error in errors))

    def test_invalid_schema_fields_and_machine_rejected(self):
        self.artifact['testbed'] = 'i9'
        self.assertTrue(any('unsupported testbed' in error for error in self.errors()))
        del self.artifact['testbed']
        self.assertTrue(any('requires exactly' in error for error in self.errors()))
        self.manifest['schema_version'] = True
        self.assertEqual(self.errors(), ['schema_version must be 1'])

    def test_alias_cannot_shadow_canonical_or_escape_root(self):
        self.artifact['aliases'] = [self.artifact['path']]
        self.assertTrue(any('collides' in error for error in self.errors()))
        self.artifact['path'] = '../outside.json'
        self.assertTrue(any('unsafe relative path' in error for error in self.errors()))

    def test_deleted_historical_alias_allowed(self):
        self.assertFalse((self.root / self.artifact['aliases'][0]).exists())
        self.assertEqual(self.errors(), [])

    def test_each_raw_kind_classifies_without_metrics_defaults(self):
        shapes = {
            'quality': {'pass_rate': 1, 'results': []},
            'http': {'checks': [], 'runs': []},
            'microbenchmark': {'iterations_per_sample': 1},
            'suite': {'suite': {}, 'runs': []},
            'generation': {'max_tokens': 2, 'runs': []},
            'unknown': {'environment': {'logical_cpus': 12}},
        }
        for kind, shape in shapes.items():
            with self.subTest(kind=kind):
                self.assertEqual(evidence.classify(shape), kind)
        self.artifact['kind'] = 'suite'
        self.assertTrue(any('kind must be' in error for error in self.errors()))

    def test_render_is_deterministic(self):
        other = dict(self.artifact, path='benchmarks/results/aaa.json', aliases=[])
        self.manifest['artifacts'].append(other)
        first = evidence.render_report(self.manifest, self.root)
        self.manifest['artifacts'].reverse()
        self.assertEqual(first, evidence.render_report(self.manifest, self.root))

    def test_cli_check_freshness_coverage_and_no_writes(self):
        manifest_path = self.root / 'benchmarks/manifest.json'
        manifest_path.write_text(json.dumps(self.manifest))
        report_path = self.root / 'docs/BENCHMARKS.md'
        args = ['--root', str(self.root), '--manifest', str(manifest_path), '--output', str(report_path)]
        raw_before = self.raw.read_bytes()
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(evidence.main(args + ['--check']), 1)
            self.assertFalse(report_path.exists())
            self.assertEqual(evidence.main(args), 0)
            self.assertEqual(evidence.main(args + ['--check']), 0)
            report_path.write_text('stale')
            self.assertEqual(evidence.main(args + ['--check']), 1)
            self.assertEqual(report_path.read_text(), 'stale')
            (self.raw.parent / 'unlisted.json').write_text('{}')
            self.assertEqual(evidence.main(args), 1)
            self.assertEqual(report_path.read_text(), 'stale')
        self.assertEqual(self.raw.read_bytes(), raw_before)


if __name__ == '__main__':
    unittest.main()
