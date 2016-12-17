#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('docs/history.rst') as history_file:
    history = history_file.read()

requirements = [
    'gym',
    'tflearn',
    'numpy',
    'Pillow'
]

test_requirements = [
    'bumpversion',
    'wheel',
    'flake8',
    'tox',
    'coverage',
    'Sphinx',
    'pytest',
    'path.py'
]

setup(
    name='rlflow',
    version='0.1.0',
    description="A framework for learning about and experimenting with reinforcement learning algorithms",
    long_description=readme + '\n\n' + history,
    author="Trevor Barron",
    author_email='barron.trevor@gmail.com',
    url='https://github.com/tpbarron/rlflow',
    packages=[
        'rlflow',
    ],
    package_dir={'rlflow':
                 'rlflow'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='mdp machine learning reinforcement neural network',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
