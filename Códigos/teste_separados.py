# coding=utf-8
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

#comprimentos de onda de laboratório
lam_oiii1 = 5006.843                                       
lam_oiii2 = 4958.911
lam_hbeta = 4861.333
lam_halpha = 6562.819
lam_nii1 = 6583.460
lam_nii2 = 6548.050
lam_oii1 = 3726.032
lam_oii2 = 3728.815


file_ir = 'local_starforming_1.fits'
file_vis = 'local_starforming_1.fits'

hdu = fits.open(file_ir)
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
        lam = np.arange(n)*step*10 + l0*10

#center_oiii1 vai ser um dos parâmetros do ajuste (substituir depois)

center_oiii1 = 5100
center_oiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_hbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_halpha = lam_halpha + (lam_halpha/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_nii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_nii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_oii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(center_oiii1 - lam_oiii1)
center_oii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(center_oiii1 - lam_oiii1)

#janelas
#janela do oiii
min_oiii = center_hbeta - 100
max_oiii = center_oiii1 + 100

#janela do h alpha
min_halpha = center_nii2 - 100
max_halpha = center_nii1 + 100

#janela do oii
min_oii = center_oii1 - 100
max_oii = center_oii2 + 100

#cortar em torno das janelas
#oiii

amin_oiii = round(min_oiii)
amax_oiii = round(max_oiii)

imin_oiii = (np.abs(lam - amin_oiii)).argmin()
imax_oiii = (np.abs(lam - amax_oiii)).argmin()

lam1 = lam[imin_oiii : imax_oiii]
cube1 = cube[imin_oiii : imax_oiii]

#h alpha
amin_halpha = round(min_halpha)
amax_halpha = round(max_halpha)

imin_halpha = (np.abs(lam - amin_halpha)).argmin()
imax_halpha = (np.abs(lam - amax_halpha)).argmin()

lam2 = lam[imin_halpha : imax_halpha]
cube2 = cube[imin_halpha : imax_halpha]

#oii
if min_oii < 10000:
    hdu_vis = fits.open(file_vis)
    head_vis = hdu_vis[0].header
    data_vis = hdu_vis[0].data
    
    cube_vis = data_vis
    
    n_vis = len(cube_vis)
    l0_vis = head_vis['CRVAL1']
    step_vis = head_vis['CDELT1']
    unit_vis = head_vis['CUNIT1']
    if unit_vis == 'Angstrom':
        lam_vis = np.arange(n_vis)*step_vis + l0_vis
    else:
        if unit_vis == 'np':
            lam_vis = np.arange(n_vis)*step_vis*10 + l0_vis*10
    
    amin_oii = round(min_oii, -1)
    amax_oii = round(max_oii, -1)

    imin_oii = (np.abs(lam - amin_oii)).argmin()
    imax_oii = (np.abs(lam - amax_oii)).argmin()

    lam3 = lam_vis[imin_oii : imax_oii]
    cube3 = cube_vis[imin_oii : imax_oii]
else:
    amin_oii = round(min_oii, -1)
    amax_oii = round(max_oii, -1)

    imin_oii = (np.abs(lam - amin_oii)).argmin()
    imax_oii = (np.abs(lam - amax_oii)).argmin()

    lam3 = lam[imin_oii : imax_oii]
    cube3 = cube[imin_oii : imax_oii]


lam_fin1 = np.append(lam1,lam2)
cube_fin1 = np.append(cube1,cube2)
lam_fin = np.append(lam3, lam_fin1)
cube_fin = np.append(cube3, lam_fin1)
        
def oiii (x, a, b, c, d, f, g, i, j):
    w_oiii1 = (x - a)/b
    alpha_oiii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oiii1**2.)/2.)
    sigoiii2 = (lam_oiii2/lam_oiii1)*b
    w_oiii2 = (x - d)/sigoiii2
    alpha_oiii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oiii2**2.)/2.)
    sighbeta = (lam_hbeta/lam_oiii1)*b
    w_hbeta = (x - g)/sighbeta
    alpha_hbeta = (1./math.sqrt(2.*math.pi))*np.exp((-w_hbeta**2.)/2.)
    r1 = j 
    return r1 + c*alpha_oiii1/b + f*alpha_oiii2/sigoiii2 + i*alpha_hbeta/sighbeta

guess_oiii = np.array([center_oiii1, 1, 1, center_oiii2, 1, center_hbeta, 1, 10])

p, pv = curve_fit(oiii, lam1, cube1, guess_oiii)

cen_oiii1 = p[0]
sigma_oiii1 = p[1]
amp_oiii1 = p[2]
cen_oiii2 = p[3]
amp_oiii2 = p[4]
cen_hbeta = p[5]
amp_hbeta = p[6]
a1 = p[7]
sigma_oiii2 = (lam_oiii2/lam_oiii1)*sigma_oiii1
sigma_hbeta = (lam_hbeta/lam_oiii1)*sigma_oiii1

def halpha (x, a, c, d, f, g, i, j):
    sighalpha = (lam_halpha/lam_oiii1)*sigma_oiii1
    w_halpha = (x - a)/sighalpha
    alpha_halpha = (1./math.sqrt(2.*math.pi))*np.exp((-w_halpha**2.)/2.)
    signii1 = (lam_nii1/lam_oiii1)*sigma_oiii1
    w_nii1 = (x - d)/signii1
    alpha_nii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_nii1**2.)/2.)
    signii2 = (lam_nii2/lam_oiii1)*sigma_oiii1
    w_nii2 = (x - g)/signii2
    alpha_nii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_nii2**2.)/2.)
    r2 = j  
    return r2 + c*alpha_halpha/sighalpha + f*alpha_nii1/signii1 + i*alpha_nii2/signii2

guess_halpha = np.array([center_halpha, 100, center_nii1, 100, center_nii2, 100, 10])

p1, pv1 = curve_fit(halpha, lam2, cube2, guess_halpha)

cen_halpha = p1[0]
amp_halpha = p1[1]
cen_nii1 = p1[2]
amp_nii1 = p1[3]
cen_nii2 = p1[4]
amp_nii2 = p1[5]
a2 = p1[6]
sigma_halpha = (lam_halpha/lam_oiii1)*sigma_oiii1
sigma_nii1 = (lam_nii1/lam_oiii1)*sigma_oiii1
sigma_nii2 = (lam_nii2/lam_oiii1)*sigma_oiii1


def oii (x, a, c, d, f, g):
    sigoii1 = (lam_oii1/lam_oiii1)*sigma_oiii1
    w_oii1 = (x - a)/sigoii1
    alpha_oii1 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oii1**2.)/2.)
    sigoii2 = (lam_oii1/lam_oiii1)*sigma_oiii1
    w_oii2 = (x - d)/sigoii2
    alpha_oii2 = (1./math.sqrt(2.*math.pi))*np.exp((-w_oii2**2.)/2.)
    r3 = g
    return r3 + c*alpha_oii1/sigoii1 + f*alpha_oii2/sigoii2

guess_oii = np.array([center_oii1, 100, center_oii2, 100, 10])

p2, pv2 = curve_fit(oii, lam3, cube3, guess_oii)

cen_oii1 = p2[0]
amp_oii1 = p2[1]
cen_oii2 = p2[2]
amp_oii2 = p2[3]
a3 = p2[4]
sigma_oii1 = (lam_oii1/lam_oiii1)*sigma_oiii1
sigma_oii2 = (lam_oii2/lam_oiii1)*sigma_oiii1

plt.subplot(3,1,1)
plt.plot(lam1,cube1)
x = lam1
y = (a1 + amp_oiii1*(1./math.sqrt(2.*math.pi))*np.exp((-((x - cen_oiii1)/sigma_oiii1)**2.)/2.)/sigma_oiii1
     + amp_oiii2*(1./math.sqrt(2.*math.pi))*np.exp((-((x - cen_oiii2)/sigma_oiii2)**2.)/2.)/sigma_oiii2
     + amp_hbeta*(1./math.sqrt(2.*math.pi))*np.exp((-((x - cen_hbeta)/sigma_hbeta)**2.)/2.)/sigma_hbeta)
plt.plot(x,y, color = 'red')

plt.subplot(3,1,2)
plt.plot(lam2, cube2)
x1 = lam2
y1 = (a2 + amp_halpha*(1./math.sqrt(2.*math.pi))*np.exp((-((x1 - cen_halpha)/sigma_halpha)**2.)/2.)/sigma_halpha
     + amp_nii1*(1./math.sqrt(2.*math.pi))*np.exp((-((x1 - cen_nii1)/sigma_nii1)**2.)/2.)/sigma_nii1
     + amp_nii2*(1./math.sqrt(2.*math.pi))*np.exp((-((x1 - cen_nii2)/sigma_nii2)**2.)/2.)/sigma_nii2)
plt.plot(x1,y1, color = 'red')

plt.subplot(3,1,3)
plt.plot(lam3,cube3)
x2 = lam3
y2 = (a3 + amp_oii1*(1./math.sqrt(2.*math.pi))*np.exp((-((x2 - cen_oii1)/sigma_oii1)**2.)/2.)/sigma_oii1
     + amp_oii2*(1./math.sqrt(2.*math.pi))*np.exp((-((x2 - cen_oii2)/sigma_oii2)**2.)/2.)/sigma_oii2)
plt.plot(x2,y2, color = 'red')

print(p)
print (p1)
print (p2)




