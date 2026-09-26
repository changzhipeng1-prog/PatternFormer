"""Select the five paper example parameters from newly generated test solutions."""
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
PARAMETERS = [1.622, 6.799, 15.997, 16.378, 17.809]


def main():
    records = torch.load(HERE / 'generated_solutions.pt', map_location='cpu', weights_only=False)
    selected = []
    for value in PARAMETERS:
        matches = [r for r in records if abs(float(r['p']) - value) < 1e-5]
        if len(matches) != 1:
            raise ValueError(f'Expected one test record at p={value}; found {len(matches)}. '
                             'Run the full test evaluation before plotting the paper examples.')
        selected.append(matches[0])
    torch.save(selected, HERE / 'examples_solutions.pt')
    print('Selected paper parameters:', PARAMETERS)


if __name__ == '__main__':
    main()
