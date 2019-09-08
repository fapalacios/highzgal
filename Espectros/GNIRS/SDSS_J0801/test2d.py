#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:59:46 2019

@author: felicia
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

hdu_list = fits.open('j2_2d.fits')

image_data = hdu_list[1].data
header = hdu_list[1].header
l_0 = header['CRVAL1']
step = header['CDELT1']


x =  np.arange(0, len(image_data[0]))*step + l_0
x_pos = np.arange(0, len(x), 5)
x_label = x[: : 10]

plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.ylim(400, 450)
#plt.xticks(x_pos, x_label)
#plt.xlim(16772 - 50, 16772 + 50)
