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
    version='2.0.0',
    description='Linear optimization networks - shortest path algorithms',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Nina Wiedemann',
    author_email=('nwiedemann@uos.de'),
    url='https://github.com/NinaWie/lion',
    license='MIT',
    install_requires=['numpy', 'numba', 'scipy'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.'),
    python_requires='>=3.6',
    scripts=scripts
)
