"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name='lion-sp',
    version='0.1.1',
    description='LInear Optimization Networks - shortest path algorithms',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Nina Wiedemann',
    author_email=('nwiedemann@uos.de'),
    license="GPLv3",
    url='https://github.com/NinaWie/lion',
    install_requires=['numpy', 'numba', 'scipy', 'coverage'],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.'),
    python_requires='>=3.6',
    scripts=scripts
)
