#!/usr/bin/env python
# coding: utf-8

import setuptools

setuptools.setup(
    # includes all other files
    include_package_data=True,
    # package name
    name = "ptbo",
    # project dir
    packages = setuptools.find_packages(),
    # description
    description = "PyTorch-based Perturbed Oracle Tool",
    # version
    version = "0.1.0",
    # Github repo
    url = "https://github.com/youssouf1994/PyPTBO",
    # author name
    author = "Youssouf Emine",
    # mail address
    author_email = "youssouf.emine@polymtl.ca",
    # dependencies
    install_requires = [
        "python>=3.10.0",
        "torch>=1.13.1"
    ],
    # classifiers
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ]
)