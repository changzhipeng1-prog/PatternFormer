"""Render a paper figure in a separate directory from evaluation outputs and saved inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'main-fig-02': ['1D_p'], 'supp-fig-01': ['a2a4'],
            'supp-fig-02': ['2D'], 'supp-fig-03': ['2D'], 'supp-fig-04': ['2D'],
            'main-fig-05': ['1D_p', 'a2a4', '2D']}
PREPARE = {
    '1D_p': ['test/select_examples.py', 'test/plot_examples.py', 'test/plot_box.py',
             'data_gen/make_bifurcation_1Dp.py', 'test/timing/plot_efficiency.py'],
    'a2a4': ['test/plot_examples.py', 'test/plot_box.py', 'test/timing/plot_efficiency_a2a4.py'],
    '2D': ['test/plot_examples.py', 'test/plot_box.py', 'test/timing/plot_efficiency_2d.py'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('figure')
    p.add_argument('--evaluation', action='append', default=[], metavar='PROBLEM=DIRECTORY')
    p.add_argument('--gs-evaluation', action='append', default=[], metavar='SETUP=JSON')
    p.add_argument('--saved-inputs', action='store_true', help='Use the downloaded historical figure inputs')
    p.add_argument('--output', type=Path, required=True, help='New directory outside the repository')
    a = p.parse_args()
    entries = {x['id']: x for x in json.loads((ROOT / 'experiments/figures.json').read_text())}
    if a.figure not in entries or not entries[a.figure]['figure_steps']:
        p.error('Select an executable figure from experiments/run.py --list.')
    output = a.output.expanduser().resolve()
    if output.exists() or output.is_relative_to(ROOT):
        p.error('Choose a new output directory outside this repository.')
    evaluations = {}
    for value in a.evaluation:
        problem, sep, directory = value.partition('=')
        if not sep or problem not in PREPARE or problem in evaluations:
            p.error('--evaluation must be a unique 1D_p=DIR, a2a4=DIR or 2D=DIR.')
        evaluations[problem] = Path(directory).expanduser().resolve()
    gs_evaluations = {}
    for value in a.gs_evaluation:
        setup, sep, path = value.partition('=')
        if not sep or setup not in ['L1', 'L2'] or setup in gs_evaluations:
            p.error('--gs-evaluation must be L1=JSON or L2=JSON.')
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            p.error('Missing GS evaluation JSON: ' + str(path))
        gs_evaluations[setup] = path
    if gs_evaluations and (a.figure not in ['main-fig-03', 'supp-fig-07'] or evaluations):
        p.error('--gs-evaluation is for the GS composite figures only.')
    if gs_evaluations and ('L2' if a.figure == 'main-fig-03' else 'L1') not in gs_evaluations:
        p.error('Pass the evaluation for this figure diffusion regime.')
    if a.saved_inputs and (evaluations or gs_evaluations):
        p.error('Use either explicit evaluation outputs or --saved-inputs.')
    needed = REQUIRED.get(a.figure, [])
    if not a.saved_inputs and set(evaluations) != set(needed):
        p.error('This figure requires --evaluation for: ' + ', '.join(needed) + '; or use --saved-inputs.')
    if not needed and not a.saved_inputs and not gs_evaluations:
        p.error('Use --saved-inputs for this composite of separate experiments; see REVIEWER_WORKFLOW.md.')
    for problem, directory in evaluations.items():
        for name in ['generated_solutions.pt', 'stats.csv', 'stats.pt']:
            if not (directory / name).is_file():
                p.error('Missing evaluation output: ' + str(directory / name))
    # Data lookups are read-only links; figure inputs are copied because some plot
    # scripts update tables. No original input or evaluation output is overwritten.
    data_dirs = {ROOT / name / 'data' for name in ['1D_p', 'a2a4', '2D', 'GS/L1', 'GS/L2']}
    weight_names = {'adapter_model.safetensors', 'unet.pt', 'autoencoder2d.pt',
                    'input_projector.pt', 'output_projector.pt', 'special_tokens.pt', 'dual_head.pt'}
    def ignore(directory, names):
        return [n for n in names if n in {'.git', '.artifact-cache', 'runs', '__pycache__', '.venv'}
                or n in weight_names or Path(directory) / n in data_dirs]
    work = output / 'workspace'
    shutil.copytree(ROOT, work, ignore=ignore)
    for data in data_dirs:
        if data.is_dir():
            target = work / data.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(data, target_is_directory=True)
    for problem, directory in evaluations.items():
        for name in ['generated_solutions.pt', 'stats.csv', 'stats.pt']:
            shutil.copy2(directory / name, work / problem / 'test' / name)
    (output / 'inputs.json').write_text(json.dumps({'figure': a.figure,
        'evaluation_outputs': {k: str(v) for k, v in evaluations.items()},
        'gs_evaluation_outputs': {k: str(v) for k, v in gs_evaluations.items()},
        'saved_figure_inputs': str(ROOT), 'workspace': str(work)}, indent=2) + '\n')
    commands = []
    if a.figure == 'main-fig-05':
        commands.append(['1D_p/test/select_examples.py'])
        commands.extend([[name + '/test/_who_npz.py'] for name in PREPARE])
    elif needed:
        for problem in needed:
            scripts = PREPARE[problem]
            if a.figure in ['supp-fig-03', 'supp-fig-04']:
                scripts = []  # The catalogue directly invokes this plotter.
            commands.extend([[problem + '/' + script] for script in scripts])
    env = {**os.environ, **{f'PF_GS_{k}_EVAL': str(v) for k, v in gs_evaluations.items()}}
    with (output / 'plot.log').open('w') as log:
        for command in commands:
            subprocess.run([sys.executable, str(work / command[0]), *command[1:]],
                           cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
        subprocess.run([sys.executable, str(work / 'experiments/run.py'), a.figure, '--run'],
                       cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    print('Figure files:')
    for name in entries[a.figure]['outputs']:
        path = work / name
        if not path.is_file():
            raise FileNotFoundError(path)
        print(path)


if __name__ == '__main__':
    main()
