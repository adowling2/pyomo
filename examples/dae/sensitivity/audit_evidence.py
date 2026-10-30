# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Audit checkpoint source claims against retained Git snapshots.

This inspects historical source hashes, not numerical accuracy. A completed
audit may report unrecovered pre-commit source snapshots; inspect the flags.
External wheel/library and original supplied-probe hashes are outside this
repository-file audit. Use --through to reproduce a fixed checkpoint scope.
"""

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path
import subprocess

BASELINE = 'a6c7d20b27b379e9e54aa25986fcd50202fbf10c'
FOLDER = Path('examples/dae/sensitivity')


def audit(root, revision):
    """Match each recorded repository source hash to a local checkpoint revision."""

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root)

    through = git('rev-parse', revision).decode().strip()
    commits = git('rev-list', f'{BASELINE}..{through}').decode().splitlines() + [
        BASELINE
    ]

    @cache
    def versions(path):
        result = {}
        for commit in commits:
            run = subprocess.run(
                ['git', 'show', f'{commit}:{path}'], cwd=root, capture_output=True
            )
            if run.returncode == 0:
                result.setdefault(hashlib.sha256(run.stdout).hexdigest(), commit)
        return result

    def repository_path(name):
        return str(
            Path(name)
            if name.startswith(('pyomo/', 'doc/', 'examples/'))
            else FOLDER / name
        )

    records = []
    paths = (
        git('ls-tree', '-r', '--name-only', through, '--', str(FOLDER / 'results'))
        .decode()
        .splitlines()
    )
    for name in paths:
        if not name.endswith('.json'):
            continue
        report = json.loads(git('show', f'{through}:{name}'))
        claims = {}
        for key in ('repository_files', 'source_files', 'files', 'source_sha256'):
            if isinstance(report.get(key), dict):
                for path, value in report[key].items():
                    digest = value.get('sha256') if isinstance(value, dict) else value
                    if isinstance(digest, str) and len(digest) == 64:
                        claims[repository_path(path)] = digest
        if isinstance(report.get('source_sha256'), str):
            script = {
                'dae_guard': 'dae_guard_reproducer',
                'dae_native': 'dae_qualification',
                'dae_time': 'dae_time_reproducer',
                'dae_zero_rate': 'dae_zero_rate_reproducer',
            }
            stem = Path(name).stem.rsplit('-', 1)[0]
            if stem in script:
                claims[str(FOLDER / (script[stem] + '.py'))] = report['source_sha256']
        for path, digest in claims.items():
            records.append(
                {
                    'record': name,
                    'path': path,
                    'sha256': digest,
                    'matching_commit': versions(path).get(digest),
                }
            )
    unmatched = [r for r in records if r['matching_commit'] is None]
    return {
        'purpose': 'Repository source-hash recovery, not a numerical capability test',
        'baseline_commit': BASELINE,
        'through_commit': through,
        'claims_count': len(records),
        'matched_count': len(records) - len(unmatched),
        'all_hashes_recovered': not unmatched,
        'claims': records,
        'unrecovered_claims': unmatched,
        'scope_exclusions': [
            'External wheels, libraries and original supplied-probe hashes',
            'Uncommitted final packaging files beyond the requested revision',
        ],
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    args.output.write_text(json.dumps(audit(root, args.through), indent=2) + '\n')


if __name__ == '__main__':
    main()
