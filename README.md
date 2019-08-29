# nd-array-patches

Splits an numpy array of arbitrary dimension >= 1 into small patches and reconstructs the original array from the patches.
Works for plain n-d arrays or for multi-channel n-d arrays, eg RGB images or RGB volumes.
Useful e.g. for convolutional neural networks.




```python
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
```
