from pathlib import Path
import re
from setuptools import setup, find_packages


def stream_requirements(fd):
    """For a given requirements file descriptor, generate lines of
    distribution requirements, ignoring comments and chained requirement
    files.
    """
    for line in fd:
        cleaned = re.sub(r'#.*$', '', line).strip()
        if cleaned and not cleaned.startswith('-r'):
            yield cleaned


# ---------------------------------------------------------------------------- #
#                                   Requirements                               #
# ---------------------------------------------------------------------------- #

ROOT_PATH = Path(__file__).parent
README_PATH = ROOT_PATH / 'README.md'
REQUIREMENTS_PATH = ROOT_PATH / 'requirements' / 'main.txt'
# REQUIREMENTS_TEST_PATH = ROOT_PATH / 'requirements' / 'test.txt'

with REQUIREMENTS_PATH.open() as requirements_file:
    REQUIREMENTS = list(stream_requirements(requirements_file))

# with REQUIREMENTS_TEST_PATH.open() as test_requirements_file:
#     REQUIREMENTS_TEST = REQUIREMENTS[:]
#     REQUIREMENTS_TEST.extend(stream_requirements(test_requirements_file))


# ---------------------------------------------------------------------------- #
#                                   SETUP                                      #
# ---------------------------------------------------------------------------- #
setup(
    name='timeshap',
    version='0.0.0',
    description="KernelSHAP adaptation for recurrent models.",
    keywords=['explainability', 'TimeShap'],

    author="Feedzai",
    url="https://github.com/feedzai/timeshap",

    package_dir={'': 'src'},
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_data={
        '': ['*.yaml, *.yml'],
    },
    include_package_data=True,

    python_requires='>=3.6.*',

    install_requires=REQUIREMENTS,

    zip_safe=False,

    #test_suite='tests',
    #tests_require=REQUIREMENTS_TEST,
)
