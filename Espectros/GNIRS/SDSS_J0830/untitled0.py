#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:31:14 2019

@author: felicia
"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

hdu = fits.open('spec_j2_calib.fits')

header = hdu[0].header
data = hdu[0].data
cube = data/1e-19

n = len(data)
l0 = header['CRVAL1']
step = header['CDELT1']

lam = np.arange(n)*step + l0


lam_oiii1 = 5006.843                                       
lam_oiii2 = 4958.911
lam_hbeta = 4861.333
lam_halpha = 6562.819
lam_nii1 = 6583.460
lam_nii2 = 6548.050
lam_oii1 = 3726.032
lam_oii2 = 3728.815

cen_oiii1 = 11669
cen_oiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_hbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_halpha = lam_halpha + (lam_halpha/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)

min_oiii = cen_oiii2 - 50
max_oiii = cen_oiii1 + 50

imin_oiii = (np.abs(lam - min_oiii)).argmin()
imax_oiii = (np.abs(lam - max_oiii)).argmin()

lam1 = lam[imin_oiii : imax_oiii]
cube1 = cube[imin_oiii : imax_oiii]

plt.plot(lam1, cube1)
