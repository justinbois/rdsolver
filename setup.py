#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'scipy',
    'numba',
    'bokeh',
    'tqdm'
]

setup_requirements = [
    'pytest-runner',
    # TODO(justinbois): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='rdsolver',
    version='0.1.6',
    description="Solver for 2D reaction-diffusion systems.",
#    long_description=readme,
    author="Justin Bois",
    author_email='bois@caltech.edu',
    url='https://github.com/justinbois/rdsolver',
    packages=find_packages(include=['rdsolver']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='rdsolver',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
