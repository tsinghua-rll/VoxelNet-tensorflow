#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : setup.py
# Purpose :
# Creation Date : 11-12-2017
# Last Modified : 2017年12月11日 星期一 22时11分16秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'box overlaps',
    ext_modules = cythonize('box_overlaps.pyx')
)
