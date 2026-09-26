"""Download a named data/model bundle, verify checksums, and install its files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return Path(*p.parts)


def install(archive, bundle, destination):
    destination = destination.resolve()
    members = {x['path']: x for x in bundle['members']}
    if len(members) != len(bundle['members']):
        raise ValueError('Duplicate manifest member')
    # Inspect all targets before extracting anything. Existing matching files are reusable.
    for name, item in members.items():
        target = destination / relative_path(name)
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Destination symlink escapes the output directory: ' + name)
        if target.exists() and (not target.is_file() or digest(target) != item['sha256']):
            raise FileExistsError('Existing file differs; select another destination: ' + str(target))
    with tempfile.TemporaryDirectory(prefix='.pf-extract-', dir=destination) as staging:
        stage = Path(staging)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for entry in tar:
                if not entry.isfile() or entry.name not in members or entry.name in seen:
                    raise ValueError('Unexpected archive member: ' + entry.name)
                item = members[entry.name]
                if entry.size != item['bytes']:
                    raise ValueError('Wrong member size: ' + entry.name)
                target = stage / relative_path(entry.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                if digest(target) != item['sha256']:
                    raise ValueError('Wrong member checksum: ' + entry.name)
                seen.add(entry.name)
        if seen != set(members):
            raise ValueError('Archive is missing manifest members')
        for name in members:
            target = destination / relative_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(stage / relative_path(name), target)  # Atomic and never overwrites a file.
            except FileExistsError:
                if not target.is_file() or digest(target) != members[name]['sha256']:
                    raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    p.add_argument('--bundle', required=True)
    p.add_argument('--dest', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    bundle = next((x for x in manifest['bundles'] if x['id'] == a.bundle), None)
    if bundle is None:
        p.error('Unknown bundle: ' + a.bundle)
    base_url = bundle.get('base_url', manifest.get('base_url', ''))
    if not base_url.startswith('https://'):
        p.error('The manifest must contain an HTTPS release base_url.')
    destination = a.dest.expanduser().resolve()
    cache = destination / '.artifact-cache'
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for item in bundle['parts']:
        name = item['name']
        if relative_path(name).name != name:
            p.error('Asset name must be a basename.')
        path = cache / name
        if not path.is_file() or digest(path) != item['sha256']:
            temporary = path.with_name(path.name + '.download')
            url = base_url.rstrip('/') + '/' + name
            print('Downloading:', url, flush=True)
            try:
                with urllib.request.urlopen(url) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
                    raise ValueError('Asset checksum or size mismatch: ' + name)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        parts.append(path)
    archive_name = bundle['archive']
    if relative_path(archive_name).name != archive_name:
        p.error('Archive name must be a basename.')
    archive = cache / archive_name
    if len(parts) != 1 or parts[0] != archive:
        with archive.open('wb') as dst:
            for part in parts:
                with part.open('rb') as src:
                    shutil.copyfileobj(src, dst)
    if digest(archive) != bundle['archive_sha256']:
        raise ValueError('Combined archive checksum mismatch')
    install(archive, bundle, destination)
    print('Installed:', bundle['id'], 'into', destination)


if __name__ == '__main__':
    main()
