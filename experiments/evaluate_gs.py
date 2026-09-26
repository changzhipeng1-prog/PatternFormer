"""Evaluate a final Gray-Scott checkpoint into a new output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup', choices=['L1', 'L2'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--limit', type=int, default=1, help='Test parameters; 0 evaluates the full test list')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.limit < 0:
        p.error('--limit must be nonnegative')
    base = ROOT / 'GS' / a.setup
    suffix = 'big' if a.setup == 'L1' else 'L2'
    checkpoint = (a.checkpoint or base / 'ckpt/epoch_60').resolve()
    inputs = [checkpoint / n for n in ['adapter_config.json', 'adapter_model.safetensors',
              'autoencoder2d.pt', 'input_projector.pt', 'dual_head.pt', 'special_tokens.pt']]
    inputs += [base / 'data' / (n + '_' + suffix + '.pt') for n in ['gs_lookup', 'test_p_idx', 'norm_stats']]
    inputs += [base / 'code/op_n128.mat']
    missing = [str(x) for x in inputs if not x.is_file()]
    if missing:
        p.error('Missing required inputs: ' + ', '.join(missing))
    output = a.output.expanduser().resolve()
    if output.exists():
        p.error('Choose a new output directory.')
    command = [sys.executable, str(base / 'code/eval_ord.py'), '--ckpt', str(checkpoint),
               '--K', '24', '--n', str(a.limit or -1), '--out', str(output / 'results.json'),
               '--maxiter', '8000' if a.setup == 'L1' else '30000',
               '--early_iter', '3000' if a.setup == 'L1' else '0']
    if a.setup == 'L2':
        command += ['--L2']
    output.mkdir(parents=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(b)
        return h.hexdigest()
    record = {'setup': a.setup, 'limit': a.limit, 'command': command,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'inputs': [{'path': str(x), 'sha256': digest(x)} for x in inputs]}
    try:
        with (output / 'environment.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        with (output / 'evaluation.log').open('w') as f:
            result = subprocess.run(command, cwd=output, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        record['exit_code'] = result.returncode
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(output / 'results.json')


if __name__ == '__main__':
    main()
