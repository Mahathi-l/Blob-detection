import numpy as np
import scipy.stats as st
from pylab import *

# Gaussian Kernel generation 
def gkern(sigma):
    kernlen = 2*int(3 * sigma + 0.5) + 1
    x = np.linspace(-sigma, sigma, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    #print(np.amax(kern2d), np.amin(kern2d))
    plt.imshow(kern2d, cmap="gray")
    plt.show()
    return kern2d/kern2d.sum()

#LOG Kernel
def LoG(sigma):
    #window size 
    n = np.ceil(sigma*6)
    y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    filter_max = np.amax(final_filter)
    filter_min = np.amin(final_filter)
    #print(filter_max, filter_min)
    #final_filter = np.array(255*(final_filter - filter_min)/(filter_max - filter_min))
    #final_filter = final_filter.astype(int)
    #filter_Img = Image.fromarray(final_filter)
    plt.imshow(final_filter, cmap="gray")
    plt.show()
    return final_filter

class fourier():
    # Fourier 2D fft using 1D fft
    def dft(self, img):
        return np.fft.fft(np.fft.fft(img, axis=0), axis=1)

    def idft(self, img):
        conjugate = np.conj(img)
        reconstruct = self.dft(conjugate)/(img.shape[0]*img.shape[1])
                
        return reconstruct