# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Build the Simulator sensitivity guide/API and execute its examples.

The build uses a temporary source copy so project-generated files do not
modify the checkout. Only the DAE guide, compile API and referenced citation
are selected; this is not a full manual build or an external link check.
Use --skip-build for numerical environments without Sphinx installed.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

from pyomo.common.fileutils import import_file


def check_examples(root):
    """Execute all five map guide blocks and four analytic examples."""
    source = (root / 'doc/OnlineDocs/explanation/modeling/dae.rst').read_text()
    section = source.split('Differentiable ODE Simulation\n', 1)[1].split(
        'Specifying Time-Varying Inputs\n', 1
    )[0]
    blocks = re.findall(r'\.\. code-block:: python\n\n((?:   [^\n]*\n|\n)+)', section)
    if len(blocks) != 5:
        raise AssertionError('Expected five guide blocks, found %s' % len(blocks))
    namespace = {}
    for index, block in enumerate(blocks):
        exec(textwrap.dedent(block), namespace)
        if index == 1:
            expected = 1.2 * 1.3 * math.exp(-0.7 * 1.2)
            assert math.isclose(
                float(namespace['gradient']['adj_p']), expected, rel_tol=3e-7
            )
            assert math.isclose(
                float(namespace['derivative']),
                math.exp(-0.7 * 1.2) * (2 * (-0.7) + 1.2 * (-0.7) ** 2),
                rel_tol=3e-7,
            )
        if index == 3:
            assert math.isclose(
                float(namespace['total_direct']['fwd_J']),
                float(namespace['total_adjoint']['adj_theta']),
                rel_tol=3e-7,
            )
        if index == 4:
            x = 1.3 * math.exp(-0.7 * 1.4 * 1.2)
            expected = [-(1.4**2) * 1.2 * x, x * (1 - 0.7 * 1.4 * 1.2)]
            for i, derivative in enumerate(expected):
                assert math.isclose(
                    float(namespace['dae_gradient']['adj_p'][i]),
                    derivative,
                    rel_tol=3e-7,
                    abs_tol=3e-7,
                )
            for i, t in enumerate((0, 0.2, 0.7, 1.2)):
                assert math.isclose(
                    float(namespace['dae_direction']['fwd_zf'][i]),
                    -(1.4**2) * t * 1.3 * math.exp(-0.7 * 1.4 * t),
                    rel_tol=3e-7,
                    abs_tol=3e-7,
                )
    for name in (
        'simulator_sensitivity_example.py',
        'simulator_control_sensitivity_example.py',
        'simulator_dae_sensitivity_example.py',
        'simulator_dae_control_sensitivity_example.py',
    ):
        import_file(str(root / 'examples/dae' / name)).run_example()
    print('Five guide snippets and four analytic examples passed.')


def build_guide(root, target):
    """Build selected unmodified guide/API sources with warnings treated as errors."""
    for name in ('doc/OnlineDocs', 'doc/logos', 'examples'):
        shutil.copytree(root / name, target / name, dirs_exist_ok=True)
    source = target / 'doc/OnlineDocs'
    conf = source / 'conf.py'
    text = conf.read_text().replace(
        'html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]',
        'html_theme_path = []  # Installed theme is discoverable in modern Sphinx.',
    )
    text += '''
# Focused build: no network inventory fetches or unrelated manual pages.
intersphinx_mapping = {}
from pathlib import Path
_selected = {'index.rst', 'explanation/modeling/dae.rst', 'reference/bibliography.rst'}
exclude_patterns = [str(p.relative_to(Path(__file__).parent)) for p in Path(__file__).parent.rglob('*.rst') if str(p.relative_to(Path(__file__).parent)) not in _selected]
'''
    conf.write_text(text)
    citation = (source / 'reference/bibliography.rst').read_text()
    citation = (
        '.. [PyomoDAE-paper]'
        + citation.split('.. [PyomoDAE-paper]', 1)[1].split('\n.. [', 1)[0]
    )
    (source / 'reference/bibliography.rst').write_text(
        'References\n==========\n\n' + citation
    )
    (source / 'index.rst').write_text('''ODE documentation check
=======================

.. toctree::

   explanation/modeling/dae
   reference/bibliography

Compiled map API
----------------

.. autoclass:: pyomo.dae.Simulator
   :members: compile
''')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root)
    subprocess.run(
        [
            sys.executable,
            '-m',
            'sphinx',
            '-E',
            '-b',
            'html',
            '-W',
            '--keep-going',
            str(source),
            str(target / 'html'),
        ],
        env=environment,
        check=True,
    )
    for name in (
        'simulator_sensitivity_example.py',
        'simulator_control_sensitivity_example.py',
        'simulator_dae_sensitivity_example.py',
        'simulator_dae_control_sensitivity_example.py',
    ):
        matches = list((target / 'html/_downloads').rglob(name))
        expected = hashlib.sha256(
            (root / 'examples/dae' / name).read_bytes()
        ).hexdigest()
        assert (
            len(matches) == 1
            and hashlib.sha256(matches[0].read_bytes()).hexdigest() == expected
        )
    print('Guide/API HTML and four download hashes passed:', target / 'html')


def main():
    """Run the focused documentation gate without changing repository sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--build-directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    check_examples(root)
    if not args.skip_build:
        if args.build_directory:
            build_guide(root, args.build_directory)
        else:
            with tempfile.TemporaryDirectory(prefix='pyomo-ode-docs-') as directory:
                build_guide(root, Path(directory))


if __name__ == '__main__':
    main()
