"""CPU-only tests for command planning and checkpoint-file assembly."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'experiments').mkdir()
        shutil.copy2(ROOT / 'experiments/train.py', self.root / 'experiments/train.py')
        (self.root / 'worker.py').write_text('print("worker ran")\n')
        self.stage = {'id': 'example', 'script': 'worker.py', 'gpus': 1,
                      'requires': [], 'args': [], 'environment': {},
                      'products': [], 'source_recipe': 'worker.py'}
        (self.root / 'experiments/training.json').write_text(json.dumps([self.stage]))

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, *args):
        return subprocess.run([sys.executable, str(self.root / 'experiments/train.py'),
                               'example', *args], capture_output=True, text=True)

    def test_plan_has_no_run_directory_side_effect(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'runs').exists())

    def test_failure_logged_and_rerun_does_not_overwrite(self):
        (self.root / 'worker.py').write_text('print("retained failure")\nraise SystemExit(7)\n')
        first = self.launch('--run')
        self.assertEqual(first.returncode, 7, first.stderr)
        folder = self.root / 'runs/example'
        self.assertEqual(json.loads((folder / 'run.json').read_text())['exit_code'], 7)
        original = (folder / 'train.log').read_bytes()
        self.assertIn(b'retained failure', original)
        second = self.launch('--run')
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((folder / 'train.log').read_bytes(), original)

    def test_transfer_copies_correct_sources_and_rejects_overwrite(self):
        parent = self.root / 'parent'
        target = self.root / 'target'
        parent.mkdir(); target.mkdir()
        first = ['adapter_config.json', 'adapter_model.safetensors', 'special_tokens.pt']
        second = ['input_projector.pt', 'dual_head.pt', 'autoencoder2d.pt']
        for name in first:
            (parent / name).write_text('parent-' + name)
        for name in second:
            (target / name).write_text('target-' + name)
        out = self.root / 'mixed'
        command = [sys.executable, str(ROOT / 'experiments/prepare_init.py'),
                   'gs-l2-mixed', '--parent', str(parent), '--target', str(target), '--out', str(out)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in first:
            self.assertEqual((out / name).read_bytes(), (parent / name).read_bytes())
        for name in second:
            self.assertEqual((out / name).read_bytes(), (target / name).read_bytes())
        self.assertEqual(len(json.loads((out / 'components.json').read_text())), 6)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_transfer_input_leaves_no_output(self):
        out = self.root / 'missing'
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/prepare_init.py'),
                                 'elliptic-2d', '--parent', str(self.root / 'absent'),
                                 '--ae', str(self.root / 'absent.pt'), '--out', str(out)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
