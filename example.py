#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:17:42 2019

@author: stefan
"""

# usage example

import numpy as np

from patch_handling import get_patches, get_volume


A = np.random.random_sample((100, 100, 100))
divs = (2,4,2)
offset = (3,4,0)

print('Splitting A into', np.prod(divs), 'patches')

A_p = get_patches(A, divs, offset)

print(A_p.shape)

A_ = get_volume(A_p, divs, offset)

print(np.all(A_==A))

