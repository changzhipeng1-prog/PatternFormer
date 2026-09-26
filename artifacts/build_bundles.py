"""Build final evaluation checkpoint bundles with member and download checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

BUNDLES = {'elliptic-1d': '1D_p', 'elliptic-two-parameter': 'a2a4',
           'elliptic-2d': '2D', 'gray-scott-l1': 'GS/L1', 'gray-scott-l2': 'GS/L2'}


def digest_stream(stream):
    h = hashlib.sha256()
    for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        h.update(data)
    return h.hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return digest_stream(stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Root containing 1D_p, a2a4, 2D and GS')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory.')
    a.output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'bundles': []}
    for key, problem in BUNDLES.items():
        checkpoint = 'ckpt/epoch_60' if problem.startswith('GS') else 'best_ckpt/best_model'
        files = sorted(x for part in [checkpoint] for x in (a.source / problem / part).rglob('*')
                       if x.is_file() and x.suffix in {'.pt', '.safetensors', '.json'})
        if not files or any(not x.is_file() for x in files):
            raise FileNotFoundError(problem)
        members = [{'path': x.relative_to(a.source).as_posix(), 'bytes': x.stat().st_size,
                    'sha256': digest(x)} for x in files]
        archive = a.output / f'patternformer-{key}.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
            for x, member in zip(files, members):
                tar.add(x, arcname=member['path'], recursive=False)
        # Verify the bytes inside the archive, not just the source filenames.
        expected = {x['path']: x for x in members}
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                with tar.extractfile(member) as stream:
                    assert digest_stream(stream) == expected[member.name]['sha256'], member.name
        parts = []
        chunk_size = 1024 ** 3  # Keep each GitHub release asset below its per-file limit.
        if archive.stat().st_size > chunk_size:
            with archive.open('rb') as stream:
                index = 0
                while True:
                    data = stream.read(chunk_size)
                    if not data:
                        break
                    path = archive.with_name(archive.name + f'.part{index:02}')
                    path.write_bytes(data)
                    parts.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
                    index += 1
        else:
            parts.append({'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest(archive)})
        manifest['bundles'].append({'id': key, 'archive': archive.name, 'archive_sha256': digest(archive),
                                    'parts': parts, 'members': members})
        (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(key, len(members), 'files;', archive.stat().st_size, 'archive bytes', flush=True)


if __name__ == '__main__':
    main()
