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
print(ROOT_PATH)
README_PATH = ROOT_PATH / 'README.md'
print(README_PATH)
REQUIREMENTS_PATH = ROOT_PATH / 'requirements' / 'main.txt'
print(REQUIREMENTS_PATH)
# REQUIREMENTS_TEST_PATH = ROOT_PATH / 'requirements' / 'test.txt'

with REQUIREMENTS_PATH.open() as requirements_file:
    requirements = list(stream_requirements(requirements_file))

# with REQUIREMENTS_TEST_PATH.open() as test_requirements_file:
#     requirements_test = requirements[:]
#     requirements_test.extend(stream_requirements(test_requirements_file))


# ---------------------------------------------------------------------------- #
#                                   Version                                    #
# ---------------------------------------------------------------------------- #
SRC_PATH = ROOT_PATH / 'src' / 'timeshap'
VERSION_PATH = SRC_PATH / 'version.py'

with VERSION_PATH.open('rb') as version_file:
    exec(version_file.read())


# ---------------------------------------------------------------------------- #
#                                   SETUP                                      #
# ---------------------------------------------------------------------------- #
setup(
    name='timeshap',
    version=__version__,
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

    install_requires=requirements,

    zip_safe=False,

    #test_suite='tests',
    #tests_require=REQUIREMENTS_TEST,
)
