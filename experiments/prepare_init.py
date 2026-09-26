"""Assemble explicit component transfers into a new checkpoint directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['elliptic-2d', 'gs-l2-mixed'])
    p.add_argument('--parent', type=Path, required=True,
                   help='a2a4 checkpoint or L1 ordered checkpoint')
    p.add_argument('--ae', type=Path, help='2D autoencoder for elliptic-2d')
    p.add_argument('--target', type=Path, help='L2 STOP checkpoint for gs-l2-mixed')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    common = ['adapter_config.json', 'adapter_model.safetensors', 'special_tokens.pt']
    sources = {name: a.parent / name for name in common}
    if a.kind == 'elliptic-2d':
        if a.ae is None:
            p.error('elliptic-2d requires --ae')
        sources.update({name: a.parent / name for name in ['output_projector.pt', 'dual_head.pt']})
        sources['autoencoder2d.pt'] = a.ae
        # A fresh input projector handles the parameter-dimension transition.
    else:
        if a.target is None:
            p.error('gs-l2-mixed requires --target')
        sources.update({name: a.target / name for name in
                        ['input_projector.pt', 'dual_head.pt', 'autoencoder2d.pt']})
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        p.error('Missing component files: ' + ', '.join(missing))
    if a.out.exists():
        p.error('Output already exists; select a new directory.')
    a.out.mkdir(parents=True)
    records = []
    for name, source in sources.items():
        destination = a.out / name
        shutil.copy2(source, destination)
        records.append({'component': name, 'source': str(source.resolve()),
                        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    (a.out / 'components.json').write_text(json.dumps(records, indent=2) + '\n')
    print(a.out.resolve())


if __name__ == '__main__':
    main()
