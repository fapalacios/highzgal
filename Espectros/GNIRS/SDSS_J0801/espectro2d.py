import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
import math
import copy

#comprimentos de onda de laboratório
lam_oiii1 = 5006.843                                       
lam_oiii2 = 4958.911
lam_hbeta = 4861.333
lam_halpha = 6562.819
lam_nii1 = 6583.460
lam_nii2 = 6548.050
lam_oii1 = 3726.032
lam_oii2 = 3728.815

#abre os arquivos .fits
file_h = 'spec_h_calib.fits'
file_j1 = 'spec_j1_calib.fits'
file_j2 = 'spec_j2_calib.fits'

#centro das linhas observadas
cen_oiii1 = 12610
cen_oiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_hbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_halpha = lam_halpha + (lam_halpha/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)

#extrai o cabeçalho, os dados e os erros
hdu_j2 = fits.open(file_j2)
head_j2 = hdu_j2[0].header
data_j2 = hdu_j2[0].data
cube_j2 = data_j2/1e-19
cube_j2[0] = 0
cube_j2[-1] = 0

hdu_h = fits.open(file_h)
head_h = hdu_h[0].header
data_h = hdu_h[0].data
cube_h = data_h/1e-19
cube_h[0] = 0
cube_h[-1] = 0

n_j2 = len(data_j2)
l0_j2 = head_j2['CRVAL1']
step_j2 = head_j2['CDELT1']

lam_j2 = np.arange(n_j2)*step_j2 + l0_j2


#cria janela do OIII
min_oiii = cen_oiii2 - 50
max_oiii = cen_oiii1 + 50

imin_oiii = (np.abs(lam_j2 - min_oiii)).argmin()
imax_oiii = (np.abs(lam_j2 - max_oiii)).argmin()

lam1 = lam_j2[imin_oiii : imax_oiii]
cube1 = cube_j2[imin_oiii : imax_oiii]
#sig1 = sig_j1[imin_oiii : imax_oiii]

#cria janela do H beta
min_hbeta = cen_hbeta - 50
max_hbeta = cen_hbeta + 50

imin_hbeta = (np.abs(lam_j2 - min_hbeta)).argmin()
imax_hbeta = (np.abs(lam_j2 - max_hbeta)).argmin()

lam2 = lam_j2[imin_hbeta : imax_hbeta]
cube2 = cube_j2[imin_hbeta : imax_hbeta]
#sig2 = sig_j1[imin_hbeta : imax_hbeta]

n_h = len(cube_h)
l0_h = head_h['CRVAL1']
step_h = head_h['CDELT1']
lam_h = np.arange(n_h)*step_h + l0_h

#cria janela do NII
min_nii = cen_nii2 - 50
max_nii = cen_nii1 + 50

imin_nii = (np.abs(lam_h - min_nii)).argmin()
imax_nii = (np.abs(lam_h - max_nii)).argmin()

lam3 = lam_h[imin_nii : imax_nii]
cube3 = cube_h[imin_nii : imax_nii]

#cria janela do OII
min_oii = cen_oii1 - 50
max_oii = cen_oii2 + 50

imin_oii = (np.abs(lam_j2 - min_oii)).argmin()
imax_oii = (np.abs(lam_j2 - max_oii)).argmin()

lam4 = lam_j2[imin_oii : imax_oii]
cube4 = cube_j2[imin_oii : imax_oii]



#junta as janelas
lam_fin1 = np.append(lam1,lam2)
cube_fin1 = np.append(cube1,cube2)
lam_fin2 = np.append(lam_fin1, lam3)
cube_fin2 = np.append(cube_fin1, cube3)
lam_fin = np.append(lam_fin2, lam4)
cube_fin = np.append(cube_fin2, cube4)


#declara as função gaussianas e os contínuos que serão ajustados aos dados
def gauss(x, a, b, c, d, e, f, g, h, i, j, k, l):
    step_min_oiii = np.heaviside(lam_fin - min_oiii, 0)
    step_max_oiii = np. heaviside(lam_fin - max_oiii, 0)
    step_min_hbeta = np.heaviside(lam_fin - min_hbeta, 0)
    step_max_hbeta = np.heaviside(lam_fin - max_hbeta, 0)
    step_min_halpha = np.heaviside(lam_fin - min_nii, 0)
    step_max_halpha = np.heaviside(lam_fin - max_nii, 0)
    step_min_oii = np.heaviside(lam_fin - min_oii, 0)
    step_max_oii = np.heaviside(lam_fin - max_oii, 0)
    
    r1 = a*(step_min_oiii - step_max_oiii)
    r2 = b*(step_min_halpha - step_max_halpha)
    r3 = c*(step_min_oii - step_max_oii)
    r4 = d*(step_min_hbeta - step_max_hbeta)
    
    w_oiii1 = (x - e)/f
    alpha_oiii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oiii1**2.)/2.)*g
    
    cenoiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(e - lam_oiii1)
    sigoiii2 = (lam_oiii2/lam_oiii1)*f
    ampoiii2 = g/3.
    w_oiii2 = (x - cenoiii2)/sigoiii2
    alpha_oiii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oiii2**2.)/2.)*ampoiii2
    
    cenhbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(e - lam_oiii1)
    sighbeta = (lam_hbeta/lam_oiii1)*f
    w_hbeta = (x - cenhbeta)/sighbeta
    alpha_hbeta = (1./math.sqrt(2.*math.pi))*np.exp((-w_hbeta**2.)/2.)*h
    
    cenhalpha = lam_halpha + (lam_halpha/lam_oiii1)*(e - lam_oiii1)
    sighalpha = (lam_halpha/lam_oiii1)*f
    w_halpha = (x - cenhalpha)/sighalpha
    alpha_halpha = (1./math.sqrt(2.*math.pi))*np.exp((-w_halpha**2.)/2.)*i
    
    cennii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(e - lam_oiii1)
    signii1 = (lam_nii1/lam_oiii1)*f
    w_nii1 = (x - cennii1)/signii1
    alpha_nii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_nii1**2.)/2.)*j
    
    cennii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(e - lam_oiii1)
    signii2 = (lam_nii2/lam_oiii1)*f
    ampnii2 = j/3.
    w_nii2 = (x - cennii2)/signii2
    alpha_nii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_nii2**2.)/2.)*ampnii2
    
    cenoii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(e - lam_oiii1)
    sigoii1 = (lam_oii1/lam_oiii1)*f
    w_oii1 = (x - cenoii1)/sigoii1
    alpha_oii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oii1**2.)/2.)*k
    
    cenoii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(e - lam_oiii1)
    sigoii2 = (lam_oii2/lam_oiii1)*f
    w_oii2 = (x - cenoii2)/sigoii2
    alpha_oii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oii2**2.)/2.)*l
    
    return (r1 + r2 + r3 + r4+ (alpha_oiii1/f) + (alpha_oiii2/sigoiii2) + (alpha_hbeta/sighbeta)
+ (alpha_halpha/sighalpha) + (alpha_nii1/signii1) + (alpha_nii2/signii2) + (alpha_oii1/sigoii1) 
+ (alpha_oii2/sigoii2))

#chute inicial dos parâmetros 
guess = np.array([10, 10, 10, 10, cen_oiii1, 200*cen_oiii1/(2.99798*1e5), 1000, 1000, 1000, 1000, 1000, 1000])

#ajuste das funções aos dados    
p, pv = curve_fit(gauss, lam_fin, cube_fin, guess, bounds=([0]*4 + [12603] + [0]*7,
                                                           [np.inf]*4 + [12612] + [np.inf]*7))

#parâmetros do ajuste
a1 = p[0]
a2 = p[1]
a3 = p[2]
a4 = p[3]
center_oiii1 = p[4]
sig_oiii1 = copy.deepcopy(p[5])
amp_oiii1 = p[6]
amp_hbeta = p[7]
amp_halpha = p[8]
amp_nii1 = p[9]
amp_oii1 = p[10]
amp_oii2 = p[11]

#outros parâmetros
center_oiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_hbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_halpha = lam_halpha + (lam_halpha/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_nii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_nii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_oii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_oii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(center_oiii1 - lam_oiii1)

sigma_oiii2 = (lam_oiii2/lam_oiii1)*sig_oiii1
sigma_hbeta = (lam_hbeta/lam_oiii1)*sig_oiii1
sigma_halpha = (lam_halpha/lam_oiii1)*sig_oiii1
sigma_nii1 = (lam_nii1/lam_oiii1)*sig_oiii1
sigma_nii2 = (lam_nii2/lam_oiii1)*sig_oiii1
sigma_oii1 = (lam_oii1/lam_oiii1)*sig_oiii1
sigma_oii2 = (lam_oii2/lam_oiii1)*sig_oiii1

amp_oiii2 = amp_oiii1/3.
amp_nii2 = amp_nii1/3.


p5 = p[5]*(299792.58/p[4])

print (p)

# Imagens 2D

hdu_list_j2 = fits.open('j2_2d.fits')
hdu_list_h = fits.open('h_2d.fits')

image_data_j2 = hdu_list_j2[1].data
image_data_h = hdu_list_h[1].data

rotated_img_j2 = ndimage.rotate(image_data_j2, -90)
rotated_img_h = ndimage.rotate(image_data_h, -90)

pxoiii1 = (center_oiii1 - l0_j2)/(step_j2)
pxhbeta = (center_hbeta - l0_j2)/(step_j2)
pxnii1 = (center_nii1 - l0_h)/(step_h)
pxhalpha = (center_halpha - l0_h)/(step_h)
pxoii1 = (center_oii1 - l0_j2)/(step_j2)
pxoii2 = (center_oii2 - l0_j2)/(step_j2)

x_pos_j2 = np.arange(0, n_j2, 20)
x_label_j2 = lam_j2[: : 20]
x_label_j2 = np.round(x_label_j2)

x_pos_h = np.arange(0, n_h, 20)
x_label_h = lam_h[: : 20]
x_label_h = np.round(x_label_h)

#plot dos espectros e ajustes
fig = plt.figure(figsize=(10,12))
fig.subplots_adjust(hspace=0.3)
plt.subplot(3,2,1)
x1 = plt.imshow(rotated_img_j2, vmin=-1.118, vmax=0.548)
plt.xticks(x_pos_j2, x_label_j2)
plt.xlim(pxoiii1 - 50, pxoiii1 + 50)
plt.ylim(20, 100)
plt.title('[OIII]5007\u212b')
x1.axes.get_yaxis().set_visible(False)
#fig.savefig('plot_oiii.png')

#plt.figure(figsize=(2,4))
plt.subplot(3,2,2)
x2 = plt.imshow(rotated_img_j2, vmin=-1.118, vmax=1.548)
plt.xticks(x_pos_j2, x_label_j2)
plt.xlim(pxhbeta - 50, pxhbeta + 50)
plt.ylim(20, 100)
plt.title('H\u03b2')
x2.axes.get_yaxis().set_visible(False)
#fig2.savefig('plot_hbeta.png')

plt.subplot(3,2,3)
x3 = plt.imshow(rotated_img_h, vmin=-1.118, vmax=1.548)
plt.xticks(x_pos_h, x_label_h)
plt.xlim(pxhalpha - 50, pxhalpha + 50)
plt.ylim(20, 100)
plt.title('H\u03b1')
x3.axes.get_yaxis().set_visible(False)
#fig3.savefig('plot_nii.png')

#plt.figure(figsize=(2,4))
plt.subplot(3,2,4)
plt.plot(lam3, cube3)
x4 = plt.imshow(rotated_img_h, vmin=-1.118, vmax=1.548)
plt.xticks(x_pos_h, x_label_h)
plt.xlim(pxnii1 - 50, pxnii1 + 50)
plt.ylim(20, 100)
plt.title('[NII]')
x4.axes.get_yaxis().set_visible(False)

#plt.figure(figsize=(2,4))
plt.subplot(3,2,5)
x5 = plt.imshow(rotated_img_j2, vmin=-1.118, vmax=1.54)
plt.xticks(x_pos_j2, x_label_j2)
plt.xlim(pxoii1 - 40, pxoii2 + 45)
plt.ylim(10, 100)
plt.title('OII Window')
x5.axes.get_yaxis().set_visible(False)

plt.show()


