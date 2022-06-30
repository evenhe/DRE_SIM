# -*- coding: utf-8 -*-
import scipy as scipy
from scipy import special as spc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_AiryDisk(sizeImage, NA, wavelength, sizePixel, MA, CF,factor1,factor2):
    """
    this function generate the PSF(point spread function) and OTF(optical transmit function)
    of a objective len with NA(numberical aperture) in a lambda wavelenght light condition
    
    Input:
      NA -- numberical aperture of objetive len
      MA -- the magnification times of objetive len
      lambda -- the wavelenght of light, unit nm(nanometer)
      sizePixel -- the space cycle of the PSF0, unit um(micrometer)
      sizeImage -- the size of the PSF0 array
    Output:
      PSF0 -- the PSF array
      OTF0 -- the OTF array
      centrical point of array is location at (sizeImage/2+1, sizeImage/2+1)
      
    demo:
        sizeImage = 512;
        NA = 0.8;
        wavelength = 580;
        sizePixel = 6.5;
        MA = 60;
        CF = 1;
    """
    NA=NA*factor2
    wl = wavelength*1e-9 # unit, m
    k = 2*np.pi/wl
    sizePixel = sizePixel*1e-6 # unit, m
    w = sizeImage
    wo = int(w/2)
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,w-1,w)
    aX, aY = np.meshgrid(x,y)
    
    # Generation of the PSF with Besselj
    aR = np.sqrt((aX-wo)**2+(aY-wo)**2)*sizePixel
    aZ = (k*NA/MA)*aR
    temp_bj = spc.jv(1, aZ)
    PSF0 = (2*temp_bj/(aZ+np.spacing(1)))**2
    PSF0[wo, wo] = 1
    PSF0 = PSF0/np.sum(PSF0)
    PSF = np.fft.fftshift(PSF0)
    
    # generation OTF
    
    OTF = np.abs( np.fft.fft2( PSF ) )
    
    OTF0 = np.fft.fftshift(OTF)
    
    # curvature function
    curMap = CF**(aR/wo)
    OTF0 = curMap*OTF0
    OTF = np.fft.ifftshift(OTF0)
    
    # get PSF0
    PSF = np.abs( np.fft.ifft2( OTF ) )
    PSF0 = np.fft.ifftshift(PSF)
    OTF0=OTF0**factor1
    return PSF0, OTF0
    

def get_fsPower(video_in):
    
    nc, nt, nh, nw = video_in.shape
    fsPower_out = np.zeros(video_in.shape)
    for ic in range(nc):
        for it in range(nt):
            temp = np.fft.fft2(video_in[ic,it])
            fsPower_out[ic,it] = np.abs( np.fft.fftshift(temp) )
    
    return fsPower_out
def get_3d(otf,size,max_):

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = np.arange(0, size, 1)
  Y = np.arange(0, size, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, otf, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  ax.set_zlim(0, max_) 
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()