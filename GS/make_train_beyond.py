"""Render Supplementary Figure S8 from training-parameter evaluation fields."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_plot_style import apply_style
from make_fig1 import panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'L2/try/results/train_beyond.pt')
    p.add_argument('--indices', type=int, nargs='+', default=[638, 482])
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'fig_train_beyond')
    a = p.parse_args()
    records = torch.load(a.input, map_location='cpu', weights_only=False)
    for ti in a.indices:
        if ti not in records or len(records[ti]['sols']) == 0:
            p.error(f'Input needs nonempty saved fields for training index {ti}.')
    apply_style()
    fig = plt.figure(figsize=(15, 5.5 * len(a.indices)))
    outer = fig.add_gridspec(len(a.indices), 1, hspace=0.16)
    for row, ti in enumerate(a.indices):
        record = records[ti]
        fields = record['sols'][:, 0].numpy()
        matched = record['is_gt'].numpy().astype(bool)
        means = fields.reshape(len(fields), -1).mean(1)
        order = sorted(range(len(fields)), key=lambda i: (not matched[i], means[i]))
        panel(fig, outer[row], fields[order], matched[order], 8, 'L2', record['param'])
        print(f'ti={ti}: total={len(fields)}, matched={int(matched.sum())}, new={int((~matched).sum())}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(a.output) + '.pdf', bbox_inches='tight')
    fig.savefig(str(a.output) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
