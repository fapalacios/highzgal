import astropy.io.fits as fits
import matplotlib.pyplot as plt

hdu = fits.open('j2_2d.fits')

image_data = hdu[1].data

plt.imshow(image_data, vmin=-2.472, vmax=2.519)
