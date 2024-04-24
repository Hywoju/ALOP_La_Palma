import matplotlib.pyplot as plt
import numpy as np
import T03_ObsM37_tools as WT
from astropy.io import fits
import matplotlib.colors as colors
from regions import Regions

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

"""Reads in regions form DS9, normalized data and subtracts two data sets"""

def normalize(array):
    return array / np.sum(array)


def diff_image(array_1, array_2):
    return array_1 - array_2


data_old = fits.open(path + '\Drew_data\\r376531-4_test2.fits')[0]
data_new = fits.open(path + '\Drew_data\\r376531-4_test2.fits')[0]
reg_str_old = '\Drew_data\\r376531-4_test2_2box.reg'
reg_str_new = '\Drew_data\\r376531-4_test2_2box.reg'

regs_old = Regions.read(path + reg_str_old, format='ds9')
regs_new = Regions.read(path + reg_str_new, format='ds9')

slices = np.zeros(1,dtype=[('old','O'),('new','O')])

for i,j,k in zip([data_old, data_new],[reg_str_old,reg_str_new],['old','new']):
    regs = Regions.read(path + j, format='ds9')


    for i,reg in enumerate(regs):
        slice = data_old.data[int(reg.center.x - (0.5 * reg.width)):int(reg.center.x + (0.5 * reg.width)),
                      int(reg.center.y - (0.5 * reg.height)):int(reg.center.y + (0.5 * reg.height))]


