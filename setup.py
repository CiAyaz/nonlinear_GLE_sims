#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

if __name__ == "__main__":
    # cihan nonlinear GLE's
    setup(name='NL_GLE_sims',
          packages=find_packages(),
          version="0.1",
          license='MIT',
          description=('Generate nonlinear GLE trajectory.'),
          author="Cihan Ayaz",
          zip_safe=False,
          requires=['numpy (>=1.10.4)', 'numba (>=0.37.0)'],
          install_requires=['numpy>=1.10.4', 'numba>=0.37.0'])

