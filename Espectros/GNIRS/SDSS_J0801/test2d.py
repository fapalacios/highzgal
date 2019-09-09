#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:59:46 2019

@author: felicia
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

hdu_list = fits.open('j2_2d.fits')

image_data = hdu_list[1].data
header = hdu_list[1].header
l_0 = header['CRVAL2']
step = header['CDELT2']

lim = pxoiii1 = (12610 - l_0)/(step)


rotated_img = ndimage.rotate(image_data, -90)

x =  np.arange(0, len(rotated_img[0]))*step + l_0
x_pos = np.arange(0, len(x), 20)
x_label = x[: : 20]
x_label = np.around(x_label)

oiii1img = plt.imshow(rotated_img, vmin=-0.172, vmax=0.119)
plt.xticks(x_pos, x_label)
plt.xlim(lim - 50, lim)
plt.ylim(20, 115)
oiii1img.axes.get_yaxis().set_visible(False)