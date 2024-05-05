
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('train_data_05.fits')
num = 5001 # the 5001st spectra in this fits file

flux = hdulist[0].data[num-1]
objid = hdulist[1].data['objid'][num-1]
label = hdulist[1].data['label'][num-1]
wavelength = np.linspace(3900,9000,3000)

c = {0:'GALAXY',1:'QSO',2:'STAR'}
plt.plot(wavelength,flux)
plt.title(f'class:{c[label]}')
plt.xlabel('wavelength ({})'.format(f'$\AA$'))
plt.ylabel('flux')
plt.show()
