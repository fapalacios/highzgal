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

plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.colorbar()
plt.xlim(11385 - 50, 11385 + 50)