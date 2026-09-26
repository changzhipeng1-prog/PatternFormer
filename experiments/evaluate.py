"""Run elliptic checkpoint inference and refinement into a new output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    '1D_p': ['p_solutions_lookup.pt', 'test_p_idx.pt'],
    'a2a4': ['bvp_region_test.pt'],
    '2D': ['p_solutions_lookup_2d_filtered.pt', 'test_p_idx.pt'],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem', required=True, choices=DATA)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=2,
                   help='Number of test entries (default: 2); use 0 for the full test set.')
    p.add_argument('--cpus', type=int, default=4)
    a = p.parse_args()
    if a.limit < 0 or a.cpus < 1:
        p.error('Use a nonnegative limit and a positive CPU count.')
    base = ROOT / a.problem
    checkpoint = (a.checkpoint or base / 'best_ckpt/best_model').expanduser().resolve()
    output = a.output.expanduser().resolve()
    ae = 'autoencoder2d.pt' if a.problem == '2D' else 'unet.pt'
    components = ['adapter_config.json', 'adapter_model.safetensors', ae,
                  'input_projector.pt', 'output_projector.pt', 'dual_head.pt', 'special_tokens.pt']
    inputs = [base / 'data' / name for name in DATA[a.problem]]
    inputs += [checkpoint / name for name in components]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        p.error('Missing input files: ' + ', '.join(missing))
    if output.exists():
        p.error('Select a new output directory; existing runs are not overwritten.')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update({'CKPT_DIR': str(checkpoint), 'GEN_OUT': str(output / 'generated_solutions.pt'),
                'STATS_OUT': str(output / 'stats.pt'), 'STATS_CSV': str(output / 'stats.csv'),
                'STAT_NCPU': str(a.cpus), 'OMP_NUM_THREADS': str(a.cpus),
                'MKL_NUM_THREADS': str(a.cpus), 'PYTHONUNBUFFERED': '1'})
    commands = [
        [sys.executable, str(base / 'test/generate_test.py'), '--limit', str(a.limit)],
        [sys.executable, str(base / 'test/compute_stats.py')],
    ]
    sources = sorted(set((base / 'code').rglob('*.py')) | set((base / 'test').glob('*.py')))
    sources += [Path(__file__).resolve()]
    record = {'problem': a.problem, 'limit': a.limit, 'commands': commands,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'inputs': [{'path': str(x), 'bytes': x.stat().st_size, 'sha256': digest(x)} for x in inputs],
              'sources': [{'path': str(x.relative_to(ROOT)), 'sha256': digest(x)} for x in sources],
              'steps': []}
    def save():
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        with (output / 'environment.txt').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        for label, command in zip(['generation', 'statistics'], commands):
            print(label, '->', output / (label + '.log'), flush=True)
            with (output / (label + '.log')).open('w') as log:
                proc = subprocess.run(command, cwd=output, env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
            record['steps'].append({'name': label, 'exit_code': proc.returncode})
            save()
            if proc.returncode:
                raise SystemExit(proc.returncode)
        record['outputs'] = [{'path': x.name, 'bytes': x.stat().st_size, 'sha256': digest(x)}
                             for x in sorted(output.iterdir()) if x.is_file() and x.name != 'run.json']
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
