#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:41:18 2019

@author: stefan
"""
import numpy as np

from patch_handling import get_patches, get_volume


def test(A, divs, offset):
    A_p = get_patches(A, divs, offset)
    A_ = get_volume(A_p, divs, offset)

    if A_.shape == A.shape:
        if np.all(A_ == A):
            print('Test passed.')
        else:
            print('Test failed.',A.shape, divs, offset) 
    else:
        print('Test failed.', A.shape, A_.shape, divs, offset) 


# SPECIAL CASES
# do nothing.
V = np.random.random_sample((100, 100, 100))
test(V,(1,1,1),(0,0,0))

V = np.random.random_sample((1, 1, 1))
test(V,(1,1,1),(0,0,0))

V = np.random.random_sample((1, 3))
test(V,(1),(0))

V = np.random.random_sample((1))
test(V,(1),(0))

V = np.random.random_sample((1))
test(V,(1),(5))

# TEST 1D
V = np.random.random_sample((100))
test(V, 2, 6)
test(V, 100, 0)
test(V, 50, 10)
test(V, 2, 9)

# TEST 2D
V = np.random.random_sample((100, 100))
test(V,(2,2),(3,3))
test(V,(2,4),(9,0))

V = np.random.random_sample((40, 60))
test(V,(2,2),(3,3))
test(V,(2,4),(9,0))

# TEST 2D RGB
V = np.random.random_sample((100, 100, 3))
test(V,(2,2),(3,3))
test(V,(2,4),(9,0))


# TEST 3D
V = np.random.random_sample((100, 100, 100))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,10),(9,0,7))

V = np.random.random_sample((40, 60, 30))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,5),(9,0,1))
test(V,(2,2,1),(3,3,3))

# TEST 3D with fake singleton dimension
V = np.random.random_sample((100, 100, 100, 1))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,5),(9,0,7))

V = np.random.random_sample((40, 60, 30, 1))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,5),(9,0,1))
test(V,(2,2,1),(3,3,3))

# TEST 3D RGB
V = np.random.random_sample((100, 100, 100, 3))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,2),(9,0,7))

V = np.random.random_sample((40, 60, 30, 3))
test(V,(2,2,2),(3,3,3))
test(V,(2,4,5),(9,0,1))
test(V,(2,2,1),(3,3,3))

# TEST 4D
V = np.random.random_sample((100, 100, 100, 100))
test(V,(2,2,2,2),(3,3,3,3))
test(V,(2,4,5,2),(9,3,7,3))

V = np.random.random_sample((40, 60, 30, 100))
test(V,(2,2,2,2),(3,3,3,3))
test(V,(2,4,5,2),(9,0,1,5))

print('10 random tests')
# totally random
for _ in np.arange(10):
    Ndim = np.random.randint(2,7)
    MultiChannel = np.random.randint(2)
    divs = np.random.randint(1,4,size=Ndim-MultiChannel)
    multipliers =  np.random.randint(1,4,size=Ndim-MultiChannel)
    offset = tuple(np.random.randint(1,10,size=Ndim-MultiChannel))
    dimensions = divs*multipliers
    print('Ndim',Ndim, 'MultiChannel?', bool(MultiChannel))
    print('divs:', divs, 'offset:', offset, 'dimensions:', dimensions)
    V = np.random.random_sample(tuple(dimensions))
    test(V, divs, offset)

