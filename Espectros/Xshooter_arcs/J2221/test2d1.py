#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:59:46 2019

@author: felicia
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

hdu_list = fits.open('J2221_SCI_SLIT_FLUX_MERGE2D_NIR.fits')

image_data = hdu_list[0].data
header = hdu_list[0].header
l_0 = header['CRVAL1']
step = header['CDELT1']
lim = pxoiii1 = (16772 - l_0*10)/(step*10)

x =  np.arange(0, len(image_data[0]))*step*10 + l_0*10
x_pos = np.arange(0, len(x), 20)
x_label = x[: : 20]

imoiii = plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.title('OIII')
plt.xticks(x_pos, x_label)
plt.xlim(lim - 50, lim + 50)
plt.ylim(55, 20)
imoiii.axes.get_yaxis().set_visible(False)