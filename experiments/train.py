"""Display or execute a training stage using the repository's training implementations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--run', action='store_true', help='Start training in the current allocation')
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--ae', type=Path, help='Input autoencoder weight file')
    parser.add_argument('--parent', type=Path, help='Input model or assembled initialization directory')
    args = parser.parse_args()
    stages = json.loads((ROOT / 'experiments/training.json').read_text())
    if args.list or not args.stage:
        for stage in stages:
            print(f"{stage['id']:28} GPUs={stage['gpus']}  {stage['script']}")
        return
    stage = next((s for s in stages if s['id'] == args.stage), None)
    if stage is None:
        parser.error('Unknown stage; use --list.')
    output = args.run_root.expanduser().resolve() / stage['id']
    context = {'repo': str(ROOT), 'output': str(output)}
    for name in stage['requires']:
        value = getattr(args, name)
        if value is None:
            parser.error(f'{stage["id"]} requires --{name}.')
        context[name] = str(value.expanduser().resolve())
    command = [sys.executable]
    if stage['gpus'] > 1:
        command += ['-m', 'torch.distributed.run', '--standalone',
                    f"--nproc_per_node={stage['gpus']}"]
    command += [str(ROOT / stage['script'])]
    command += [a.format(**context) for a in stage['args']]
    plan = {'stage': stage['id'], 'command': command, 'cwd': str(output),
            'gpus': stage['gpus'], 'environment': stage['environment'],
            'products': [str(output / p) for p in stage['products']],
            'source_recipe': stage['source_recipe']}
    print('Command:', shlex.join(command), flush=True)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    for name in stage['requires']:
        p = Path(context[name])
        if not (p.is_file() if name == 'ae' else p.is_dir()):
            parser.error(f'Input --{name} does not exist: {p}')
    if output.exists():
        parser.error(f'Output already exists; choose a new --run-root: {output}')
    output.mkdir(parents=True)
    source = ROOT / stage['script']
    plan['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    plan['started_utc'] = datetime.now(timezone.utc).isoformat()
    plan['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    plan['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    record = output / 'run.json'
    record.write_text(json.dumps(plan, indent=2) + '\n')
    env = os.environ.copy()
    env.update({'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'OMP_NUM_THREADS': '4', **stage['environment']})
    print('Training output:', output / 'train.log', flush=True)
    try:
        with (output / 'environment.txt').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        with (output / 'train.log').open('w') as log:
            result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        plan['exit_code'] = result.returncode
    except BaseException as error:
        plan['error'] = repr(error)
        raise
    finally:
        plan['finished_utc'] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(plan, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
