import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
file_ir = 'J023251_1_SCI_SLIT_FLUX_MERGE1D_NIR.fits'
file_vis = 'J023251_1_SCI_SLIT_FLUX_MERGE1D_VIS.fits'

#extrai o cabeçalho, os dados e os erros
hdu = fits.open(file_ir)
head = hdu[0].header
data = hdu[0].data
sig = hdu[1].data/1e-19

cube = data/1e-19

n = len(data)
l0 = head['CRVAL1']
step = head['CDELT1']
unit = head['CUNIT1']

#cria um vetor com comprimentos de onda
if unit == 'Angstrom':
   lam = np.arange(n)*step + l0
else:
    if unit == 'nm':
        lam = np.arange(n)*step*10 + l0*10

#centro das linhas observadas
cen_oiii1 = 16741.7
cen_oiii2 = lam_oiii2 + (lam_oiii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_hbeta = lam_hbeta + (lam_hbeta/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_halpha = lam_halpha + (lam_halpha/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii1 = lam_nii1 + (lam_nii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_nii2 = lam_nii2 + (lam_nii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii1 = lam_oii1 + (lam_oii1/lam_oiii1)*(cen_oiii1 - lam_oiii1)
cen_oii2 = lam_oii2 + (lam_oii2/lam_oiii1)*(cen_oiii1 - lam_oiii1)

#cria janela do OIII
min_oiii = cen_oiii2 - 50
max_oiii = cen_oiii1 + 50

imin_oiii = (np.abs(lam - min_oiii)).argmin()
imax_oiii = (np.abs(lam - max_oiii)).argmin()

lam1 = lam[imin_oiii : imax_oiii]
cube1 = cube[imin_oiii : imax_oiii]
sig1 = sig[imin_oiii : imax_oiii]

#cria janela do H beta
min_hbeta = cen_hbeta - 50
max_hbeta = cen_hbeta + 50

imin_hbeta = (np.abs(lam - min_hbeta)).argmin()
imax_hbeta = (np.abs(lam - max_hbeta)).argmin()

lam2 = lam[imin_hbeta : imax_hbeta]
cube2 = cube[imin_hbeta : imax_hbeta]
sig2 = sig[imin_hbeta : imax_hbeta]

#cria janela do NII
min_nii = cen_nii2 - 50
max_nii = cen_nii1 + 50

imin_nii = (np.abs(lam - min_nii)).argmin()
imax_nii = (np.abs(lam - max_nii)).argmin()

lam3 = lam[imin_nii : imax_nii]
cube3 = cube[imin_nii : imax_nii]
sig3 = sig[imin_nii : imax_nii]

#cria janela do OII
min_oii = cen_oii1 - 50
max_oii = cen_oii2 + 50

#abre o arquivo do espectro visual se as linhas de OII cairem fora do infravermelho
if min_oii < 10000:
    hdu_vis = fits.open(file_vis)
    head_vis = hdu_vis[0].header
    data_vis = hdu_vis[0].data
    sig_vis = hdu_vis[1].data/1e-19
    
    cube_vis = data_vis/1e-19
    
    n_vis = len(cube_vis)
    l0_vis = head_vis['CRVAL1']
    step_vis = head_vis['CDELT1']
    unit_vis = head_vis['CUNIT1']
    
    if unit_vis == 'Angstrom':
       lam_vis = np.arange(n_vis)*step_vis + l0_vis
    else:
        if unit_vis == 'nm':
            lam_vis = np.arange(n_vis)*step_vis*10 + l0_vis*10    

    imin_oii = (np.abs(lam_vis - min_oii)).argmin()
    imax_oii = (np.abs(lam_vis - max_oii)).argmin()

    lam4 = lam_vis[imin_oii : imax_oii]
    cube4 = cube_vis[imin_oii : imax_oii]
    sig4 = sig_vis[imin_oii : imax_oii]

else:

    imin_oii = (np.abs(lam - min_oii)).argmin()
    imax_oii = (np.abs(lam - max_oii)).argmin()

    lam4 = lam[imin_oii : imax_oii]
    cube4 = cube[imin_oii : imax_oii]
    sig4 = sig[imin_oii : imax_oii]

#junta as janelas
lam_fin1 = np.append(lam1,lam2)
cube_fin1 = np.append(cube1,cube2)
sig_fin1 = np.append(sig1,sig2)
lam_fin2 = np.append(lam_fin1, lam3)
cube_fin2 = np.append(cube_fin1, cube3)
sig_fin2 = np.append(sig_fin1, sig3)
lam_fin = np.append(lam_fin2, lam4)
cube_fin = np.append(cube_fin2, cube4)
sig = np.append(sig_fin2, sig4)

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
guess = np.array([10, 10, 10, 10, cen_oiii1, 60*cen_oiii1/(2.99798*1e5), 1000, 1000, 1000, 1000, 1000, 1000])

#ajuste das funções aos dados    
p, pv = curve_fit(gauss, lam_fin, cube_fin, guess, sig)

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

#Dados 2D
hdu_list = fits.open('J023251_1_SCI_SLIT_FLUX_MERGE2D_NIR.fits')

image_data = hdu_list[0].data

pxoiii1 = (center_oiii1 - l0*10)/(step*10)
pxhbeta = (center_hbeta - l0*10)/(step*10)
pxnii1 = (center_nii1 - l0*10)/(step*10)
pxhalpha = (center_halpha - l0*10)/(step*10)
pxoii1 = (center_oii1 - l0*10)/(step*10)
pxoii2 = (center_oii2 - l0*10)/(step*10)

#plot dos espectros e ajustes
fig = plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.xlim(pxoiii1 - 50, pxoiii1 + 50)
plt.title('[OIII]5007\u212b')
#fig.savefig('plot_oiii.png')

#plt.figure(figsize=(2,4))
plt.subplot(3,2,2)
plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.xlim(pxhbeta - 50, pxhbeta + 50)
plt.title('H\u03b2')
#fig2.savefig('plot_hbeta.png')


#plt.figure(figsize=(2,4))
plt.subplot(3,2,3)
plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.xlim(pxhalpha - 50, pxhalpha + 50)
plt.title('H\u03b1')
#fig3.savefig('plot_nii.png')

#plt.figure(figsize=(2,4))
plt.subplot(3,2,4)
plt.plot(lam3, cube3)
plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.xlim(pxnii1 - 50, pxnii1 + 50)
plt.title('[NII]')

#plt.figure(figsize=(2,4))
plt.subplot(3,2,5)
plt.imshow(image_data, vmin=-3.118e-18, vmax=1.548e-18)
plt.xlim(pxoii1 - 40, pxoii2 + 45)
plt.title('OII Window')

plt.show()

#razões de fluxo
#ratio1 = math.log10(amp_oiii1/amp_hbeta)
#ratio2 = math.log10(amp_nii1/amp_halpha)
#ratio1 = str(ratio1)
#ratio2 = str(ratio2)
#
#file_path = '/graduacao/fpalacios/UFRGS/IC/'
#line_ratio = open(file_path + 'razao_de_linha.txt', 'a',)
#line_ratio.write(ratio1 + ratio2 + '\n')
#line_ratio.close()