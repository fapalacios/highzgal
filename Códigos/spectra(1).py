import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt


lam_oiii1 = 5006.843                                          ; lambda de laboratorio - OIII
lam_oiii2 = 4958.911
lam_hbeta = 4861.333
lam_halpha = 6562.819
lam_nii1 = 6583.460
lam_nii2 = 6548.050
lam_oii1 = 3726.032
lam_oii2 = 3728.815


file = 'local_starforming_1.fits'

hdu = fits.open(file)
head = hdu[0].header
data = hdu[0].data

cube = data

n = len(data)
l0 = head['CRVAL1']
step = head['CDELT1']
unit = head['CUNIT1']

if unit == 'Angstrom':
   lam = np.arange(n)*step + l0
else:
    if unit == 'nm':
        lam = np.arange(n)*step*10. + l0*10.
        




