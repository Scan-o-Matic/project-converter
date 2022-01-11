#!/usr/bin/env python


#
# DEPENDENCIES
#

import os
import sys
from subprocess import Popen, PIPE, call
import json


#
# PREPARING INSTALLATION
#

package_dependencies = [
    'argparse', 'matplotlib', 'multiprocessing', 'odfpy',
    'numpy', 'sh', 'nmap', 'configparse', 'skimage',
    'uuid', 'PIL', 'scipy', 'setproctitle', 'psutil', 'flask', 'requests', 'pandas']

scripts = [
    os.path.join("scripts", p) for p in [
        "scan-o-matic",
        "scan-o-matic_server",
    ]
]

packages = [
    "scanomatic",
    "scanomatic.generics",
    "scanomatic.models",
    "scanomatic.models.factories",
    "scanomatic.io",
    "scanomatic.qc",
    "scanomatic.server",
    "scanomatic.image_analysis",
    "scanomatic.data_processing",
    "scanomatic.data_processing.phases",
    "scanomatic.util",
    "scanomatic.ui_server"
]

#
# Parsing and removing argument for accepting all questions as default
#

silent_install = any(arg.lower() == '--default' for arg in sys.argv)
if silent_install:
    sys.argv = [arg for arg in sys.argv if arg.lower() != '--default']

#
# Parsing and removing arguments for branch information
#

branch = None
branch_info = tuple(i for i, arg in enumerate(sys.argv) if arg.lower() == '--branch')

if branch_info:
    branch_info = branch_info[0]
    branch = sys.argv[branch_info + 1] if len(sys.argv) > branch_info + 1 else None
    sys.argv = sys.argv[:branch_info] + sys.argv[branch_info + 2:]


#
# Parsing and removing version upgrade in argument
#

version_update = {i: v for i, v in enumerate(sys.argv) if v.lower().startswith("--version")}
if version_update:
    id_argument = version_update.keys()[0]
    sys.argv = sys.argv[:id_argument] + sys.argv[id_argument + 1:]
    version_update = version_update[id_argument].lower().split("-")[-2:]
    version_update[0] = True
    version_update[1] = version_update[1] if version_update[1] in ('minor', 'major') else False


#
# Python-setup
#


if len(sys.argv) > 1:


    #
    # INSTALLING SCAN-O-MATIC
    #

    from distutils.core import setup
    from scanomatic.__init__ import get_version
    print("Setting up Scan-o-Matic on the system")

    setup(
        name="Scan-o-Matic",
        version=get_version(),
        description="High Throughput Solid Media Image Phenotyping Platform",
        long_description="""Scan-o-Matic is a high precision phenotyping platform
        that uses scanners to obtain images of yeast colonies growing on solid
        substrate.

        The package contains a user interface as well as an extensive package
        for yeast colony analysis from scanned images.
        """,
        author="Martin Zackrisson",
        author_email="martin.zackrisson@gu.se",
        url="www.gitorious.org/scannomatic",
        packages=packages,

        package_data={
            "scanomatic": [
                'ui_server_data/*.html',
                'ui_server_data/js/*.js',
                'ui_server_data/js/external/*.js',
                'ui_server_data/style/*.css',
                'ui_server_data/fonts/*',
                'ui_server_data/templates/*',
                'images/*',
            ]
        },

        scripts=scripts,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: X11 Application :: GTK',
            'Environment :: Console',
            'Intended Autdience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Bio-Informatics'
        ],
        requires=package_dependencies
    )

    from setup_tools import install_data_files

    install_data_files(silent=True)


    from scanomatic.io.paths import Paths

    try:
        with open(Paths().source_location_file, mode='w') as fh:
            directory = os.path.dirname(os.path.join(os.path.abspath(os.path.expanduser(os.path.curdir)), sys.argv[0]))
            json.dump({'location': directory, 'branch': branch}, fh)

    except IOError:
        print("Could not write info for future upgrades. You should stick to manual upgrades")
