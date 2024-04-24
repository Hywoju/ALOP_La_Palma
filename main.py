import matplotlib.pyplot as plt
import numpy as np
import T03_ObsM37_tools as WT
from astropy.io import fits
import matplotlib.colors as colors

path = "C:\Leiden Universiteit\ALOP\La_Palma"


'Calculates a S/N ration form the old pictures'
# data = fits.open(path+'\Drew_data\\r376531-4_test2.fits')[0]
# image = data.data
# plt.imshow(image, cmap='Greys')#,norm=colors.PowerNorm(gamma=0.2))
# plt.show()
#
# coords = WT.regextract(path + '\Drew_data\SN_regions_r376531-4.reg')
# print(coords)
#
# fluxes = np.zeros(3)
#
# index = 0
# for section in coords:
#     flux, eflux, sky, skyerr = WT.aper(image, xc=[section[0]], yc=[section[1]], apr=[section[2]], phpadu=1.,
#                             skyrad=[-1], flux=True, setskyval=0., silent=True)
#     fluxes[index] = flux
#     index += 1
# print(fluxes)
# print(fluxes[0]/fluxes[1])


data_old = fits.open(path+'\Drew_data\\r376531-4_test2.fits')[0]
data_new = fits.open(path+'\Drew_data\\r376531-4_test2.fits')[0]

def normalize(array):
    return array/np.sum(array)

def diff_image(array_1,array_2):
    return array_1 - array_2
