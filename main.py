import matplotlib.pyplot as plt
import numpy as np
import T03_ObsM37_tools as WT
from astropy.io import fits
import matplotlib.colors as colors
from regions import Regions
import matplotlib.colors as colors
import fits_compress

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


# data_old = fits.open(path + '\Drew_data\sections\\region_1_2003\\r376535-4.fits')[1]
# data_old = fits_compress.decompress_fits([data_old])[0]
data_new = fits.open(path + '\Drew_data\sections\\region_1_2024\\coadd.fits')[0]
reg_str_IM_old = '\Regions\Section1_2003_1box_IM.reg'
reg_str_IM_new = '\Regions\Section1_2024_1box_IM.reg'

print(data_old.data.shape)

regs_old = Regions.read(path + reg_str_IM_old, format='ds9')
regs_new = Regions.read(path + reg_str_IM_new, format='ds9')

slices = np.zeros(len(regs_old), dtype=[('old', 'O'), ('new', 'O')])

for dat, regs, k in zip([data_old, data_new], [regs_old, regs_new], ['old', 'new']):

    for i, reg in enumerate(regs):
        slice = dat.data[int(reg.center.x - (0.5 * reg.width)):int(reg.center.x + (0.5 * reg.width)),
                int(reg.center.y - (0.5 * reg.height)):int(reg.center.y + (0.5 * reg.height))]
        slices[k][i] = normalize(slice)

print(slices['old'][0].shape)
plt.imshow(np.rot90(slices['old'][0]), cmap='hot', origin='lower', norm=colors.LogNorm(1e-1,1e-5))
plt.colorbar(orientation='horizontal')
plt.show()

for i in range(len(slices)):
    diff = diff_image(slices['old'][i]-slices['new'][i])
    hdu = fits.PrimaryHDU(data=diff)
    hdu.writeto(path + f'\Diff_images\\section1_diffim_{i}.fits')



