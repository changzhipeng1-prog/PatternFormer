"""Find and run paper figure scripts without importing training code."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', nargs='?', help='For example: main-fig-02')
    parser.add_argument('--list', action='store_true', help='List paper figure entries')
    parser.add_argument('--run', action='store_true', help='Execute the displayed figure steps')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'experiments/figures.json').read_text())
    if args.list or not args.figure:
        for entry in entries:
            print(f"{entry['id']:14} {entry['paper']:26} {entry['title']}")
        return
    entry = next((e for e in entries if e['id'] == args.figure), None)
    if entry is None:
        parser.error('Unknown figure; use --list for available entries.')
    print(f"{entry['paper']}: {entry['title']}", flush=True)
    for source in entry['experiment_sources']:
        print('Experiment source:', source, flush=True)
    for path in entry['primary_inputs']:
        print('Input:', path, flush=True)
    commands = []
    for step in entry['figure_steps']:
        command = [sys.executable, str(ROOT / step['script']), *step['args']]
        cwd = ROOT / step['cwd']
        print(f'Working directory: {cwd}\nCommand: {shlex.join(command)}', flush=True)
        commands.append((command, cwd))
    for path in entry['outputs']:
        print('Output:', path, flush=True)
    if not args.run:
        return
    if not commands:
        parser.error('This entry lists experiment sources. See docs/EXPERIMENTS.md for its workflow.')
    missing = [p for p in entry['primary_inputs'] if not (ROOT / p).is_file()]
    if missing:
        parser.error('Place these inputs under the repository root first: ' + ', '.join(missing))
    for command, cwd in commands:
        result = subprocess.run(command, cwd=cwd)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
