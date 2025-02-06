from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb
from astropy.visualization import MinMaxInterval
from astropy.visualization import LogStretch
from scipy.stats import norm
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

#####entry and output foder
#outputfolder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/DOMINGUEZ_galaxies/DOMINGUEZ2018_images_rgb_norm/'
#filenames = glob.glob('/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/DOMINGUEZ_galaxies/DOMINGUES2018_images/imagensdescomp/*.fits')
filenames = glob.glob('/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/G10_galaxies/G10_images/*.fits')
outputfolder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/G10_galaxies/testing_images/'
print(len(filenames))

####its a funtions to obtain just the filename instead of the full path
def getName(string):
    name = string.split('/')[7] ###how many bars has in filenames
    return name

###get sky function
#####this function get the sky measure
def getsky(oneim, tag, plotit=False):
    """
    measure the sky and sky sigma in an image. This is done by fitting a
    guassian assymetrically to the pixel distribution: we iteratively
    ignore pixels higher than the sky value, to get an estimate mostly
    from pixels that are lower.
    """
    oneim = oneim.flatten()
    qui=(abs(oneim) > 1e-6) & (oneim < 6e4)
    if (qui.size > oneim.size/2.):  # make sure the image has enough pixel with values!
        s = np.sort(oneim[qui])  # get initial estimates from the full pixel distribtion with the top and bottom 10% of value clipped 
        n = s.size
        sky1 = np.median(s[int(np.floor(0.1*n)):int(np.floor(0.9*n))])
        sky2 = np.std(s[int(np.floor(0.1*n)):int(np.floor(0.9*n))])
        binsize = np.floor(sky2/20.)+1
        hmax = sky1+sky2*5

        if plotit:
            print('Iteration 1')
            n, bins, patches = plt.hist(oneim[qui], bins=np.arange(0,hmax,binsize), density=True)  # Make a histogram of pixel values
            _ = plt.title(tag)
#             hx = np.arange(oneim[qui].size)*binsize+sky2/200  # with histogram ranges set y above estimates.
#             plot,hx,honeim,charsize=2,title=tag             
        (mu, sigma) = norm.fit(np.sort(oneim[qui]))  # fit with gaussian
#         g1=gaussfit(hx,honeim,nterm=3,fg)
        if plotit:
            y = norm.pdf(bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            _ = plt.show()
#             oplot,hx,g1,col = 255
#         qui = hx < mu + sigma
            print(f'mu, sigma: {mu}, {sigma}')
        qui = oneim < mu + sigma
        if (qui.size > 10):  # guards against weird failures with bright objects
            (mu, sigma) = norm.fit(np.sort(oneim[qui]))  # refit with gaussian for all pixel values < sky+skysigma
#             g2 = gaussfit(hx(qui),honeim(qui),nterm=3,fg)  # refit with gaussian for all pixel values < sky+skysigma
            if plotit:
                print('Iteration 2')
                n, bins, patches = plt.hist(oneim[qui], bins=np.arange(0,hmax,binsize), density=True)  # Make a histogram of pixel values
                y = norm.pdf(bins, mu, sigma)
                l = plt.plot(bins, y, 'r--', linewidth=2)
                _ = plt.show()
#                 oplot,hx(qui),g2,col='00ffff'x
#                 qui = hx < mu + sigma/2.
                print(f'mu, sigma: {mu}, {sigma}')
            qui = oneim < mu + sigma/2.
            if (qui.size > 10):  # guards against wierd failures with bright objects
                (mu, sigma) = norm.fit(np.sort(oneim[qui]))  #  # refit with gaussian for all pixel values < sky+skysigma/2
#                     g3 = gaussfit(hx(qui),honeim(qui),nterm=3,fg)  # refit with gaussian for all pixel values < sky+skysigma/2
                if plotit:
                    print('Iteration 3')
                    n, bins, patches = plt.hist(oneim[qui], bins=np.arange(0,hmax,binsize), density=True)  # Make a histogram of pixel values
                    y = norm.pdf(bins, mu, sigma)
                    l = plt.plot(bins, y, 'r--', linewidth=2)
                    _ = plt.show()
#                         oplot,hx(qui),g3,col='00ff00'x
                    print(f'mu, sigma: {mu}, {sigma}')
    else:
        (mu, sigma) = (0, 1)
#         fg=[1,0,1]
        if plotit:
            print('Iteration 0')
            _ = plt.hist(oneim)
            _ = plt.title(tag)
#             plot,[0],title=tag
    
    return mu, sigma
    # fg (or mu,sigma) is the output the fitted gaussian paramters that describe the sky.


###make RGB function
def makergb1(imarray, gcut=9., rcut=8., zcut=10., lcut=-3., logcut=1., logstretch=1., plotit=False):
    """
    
    """
    g = imarray[0]
    mu_g, sigma_g = getsky(g, 'g-band', plotit) # get sky for g-band
    r = imarray[1]
    mu_r, sigma_r = getsky(r, 'r-band', plotit) # get sky for r-band
    i = imarray[2]
    mu_i, sigma_i = getsky(i, 'i-band', plotit) # get sky for i-band
    z = imarray[3]
    mu_z, sigma_z = getsky(z, 'z-band', plotit) # get sky for z-band

    # create a compound i+z image, if data allows
    iz = i-i
    izw = i-i
    
    qui = abs(i) > 1e-6
    if (sum(qui.flatten()) != 0): iz[qui] = iz[qui] + i[qui] - mu_i
    if (sum(qui.flatten()) != 0): izw[qui] = izw[qui] + 1

    qui = abs(z) > 1e-6
    if (sum(qui.flatten()) != 0): iz[qui] = iz[qui] + z[qui] - mu_z
    if (sum(qui.flatten()) != 0): izw[qui] = izw[qui] + 1

    iz = iz/izw + max([mu_i, mu_z])
    qui = izw < 1e-6
    if (sum(qui.flatten()) != 0): iz[qui] = 0.000000000

    mu_iz, sigma_iz = getsky(iz, 'i+z-band', plotit)  # get sky for i+z-band

    # ok now we have all the filter images pulled and sky values estimated. Now make the color jpeg.

    # tunable paramters to scale the
    # bands. Controls + andlimits on the
    #                         RGB channels
    #                         (like z1,z2
    #                         in ds9)
    gcut=gcut
    rcut=rcut
    zcut=zcut
    lcut=lcut

    # count the number of pixels in each image
    ng = sum(abs(g.flatten()) > 1e-6)
    nr = sum(abs(r.flatten()) > 1e-6)
    ni = sum(abs(i.flatten()) > 1e-6)
    nz = sum(abs(z.flatten()) > 1e-6)

    doit = 0  # flag to show if at least 3 filters have data

    # if all 4 bands have data, use g, r, iz
    if (ng > 10) & (nr > 10) & (ni > 10) & (nz > 10):
        # then B=g, G=r, R=i+z
        doit=1
        (mu_gs, sigma_gs) = (mu_g, sigma_g)
        (mu_rs, sigma_rs) = (mu_r, sigma_r)
        (mu_zs, sigma_zs) = (mu_iz, sigma_iz)
        bim = g - mu_gs
        gim = r - mu_rs
        rim = iz - mu_zs

    # otherwise assign RGB channels for other filter combinations if 
    # 3 of 4 filters exist. If not, doit flag doesn't get tripped
    # and no jpeg will get made.
    elif (ng <= 10) & (nr > 10) & (ni > 10) & (nz > 10):
        doit=1
        (mu_gs, sigma_gs) = (mu_r, sigma_r)
        (mu_rs, sigma_rs) = (mu_i, sigma_i)
        (mu_zs, sigma_zs) = (mu_z, sigma_z)
        bim = r - mu_gs
        gim = i - mu_rs
        rim = z - mu_zs
    elif (ng > 10) & (nr <= 10) & (ni > 10) & (nz > 10):
        doit=1
        (mu_gs, sigma_gs) = (mu_g, sigma_g)
        (mu_rs, sigma_rs) = (mu_i, sigma_i)
        (mu_zs, sigma_zs) = (mu_z, sigma_z)
        bim = g - mu_gs
        gim = i - mu_rs
        rim = z - mu_zs
    elif (ng > 10) & (nr > 10) & (ni <= 10) & (nz > 10):
        doit=1
        (mu_gs, sigma_gs) = (mu_g, sigma_g)
        (mu_rs, sigma_rs) = (mu_r, sigma_r)
        (mu_zs, sigma_zs) = (mu_z, sigma_z)
        bim = g - mu_gs
        gim = r - mu_rs
        rim = z - mu_zs
    elif (ng > 10) & (nr > 10) & (ni > 10) & (nz <= 10):
        doit=1
        (mu_gs, sigma_gs) = (mu_g, sigma_g)
        (mu_rs, sigma_rs) = (mu_r, sigma_r)
        (mu_zs, sigma_zs) = (mu_i, sigma_i)
        bim = g - mu_gs
        gim = r - mu_rs
        rim = i - mu_zs
    else:
        bim = g - g
        gim = r - r
        rim = i - i

    # if at least 3 filter images exist, we can make a color jpeg
    if doit:

        # set pixels lower than z1 to z1: use
        # same lower cut in multiples of sky
        # sigma for all three channels   
        qui = bim < lcut * sigma_gs
        if (sum(qui.flatten()) != 0): bim[qui] = lcut * sigma_gs
        qui = gim < lcut * sigma_rs
        if (sum(qui.flatten()) != 0): gim[qui] = lcut * sigma_rs
        qui = rim < lcut * sigma_zs
        if (sum(qui.flatten()) != 0): rim[qui] = lcut * sigma_zs

        # set pixels larger than z2 to z2: use
        # different upper cuts for each
        # channel, in multiples of sky sigma
        qui = bim > gcut * sigma_gs
        if (sum(qui.flatten()) != 0): bim[qui] = gcut * sigma_gs
        qui = gim > rcut * sigma_rs
        if (sum(qui.flatten()) != 0): gim[qui] = rcut * sigma_rs
        qui = rim > zcut * sigma_zs
        if (sum(qui.flatten()) != 0): rim[qui] = zcut * sigma_zs

        # set all images to run between 0 and 1
        bim = bim - sigma_gs * lcut
        gim = gim - sigma_rs * lcut
        rim = rim - sigma_zs * lcut
        bim /= ((gcut - lcut)*sigma_gs)
        gim /= ((rcut - lcut)*sigma_rs)
        rim /= ((zcut - lcut)*sigma_zs)

        # take base10 log; logcut parameter
        # selects how sky value is mapped into
        # log space. This choice makes sky
        # fairly bright which is good for
        # spooting finat low surface brightness stuff
        # transformation is log(logstretch*x + logcut)/log(logstretch + logcut)
        logcut = logcut
        logstretch = logstretch
        bim = np.log10(logstretch*bim+logcut)/np.log10(logstretch+logcut)
        gim = np.log10(logstretch*gim+logcut)/np.log10(logstretch+logcut)
        rim = np.log10(logstretch*rim+logcut)/np.log10(logstretch+logcut)

        # scale from 0-255 (8 bits)
        bim = np.floor(bim/max(bim.flatten()) * 255).astype(int)
        gim = np.floor(gim/max(gim.flatten()) * 255).astype(int)
        rim = np.floor(rim/max(rim.flatten()) * 255).astype(int)

#     else:
#         print('Not enough complete bands for RGB')
    
    return np.stack([rim, gim, bim], axis=2)


#####it is a function to handle with the filenames to be the ID number

#####creating rgb images
# Define a function to create RGB images from FITS files
def create_rgb_images(fname):
    try:
        with fits.open(fname) as hdul:
            #hdul.info()
            image_data_g = hdul[0].data
            image_data_r = hdul[1].data
            image_data_i = hdul[2].data
            image_data_z = hdul[3].data
            IMG_HEIGHT = 256
            IMG_WIDTH = 256
            imarray = np.zeros((4, IMG_HEIGHT, IMG_WIDTH))
            imarray[0, :, :] = image_data_g
            imarray[1, :, :] = image_data_r
            imarray[2, :, :] = image_data_i
            imarray[3, :, :] = image_data_z
            
            #imarray_normalized = imarray / 255.0
            #print(imarray_normalized)
            
            rgb = makergb1(imarray, lcut=0., gcut=100., rcut=100., zcut=100., logstretch=50) ##original uses imarray
            image_data = rgb.astype('uint8')
            #print(image_data)
            
            
            plt.imsave(outputfolder + getName(fname) + '.png', image_data, dpi=200, format='png')
    except:
        print("Missing band in " + getName(fname) + " file")


Parallel(n_jobs=20)(delayed(create_rgb_images)(fname) for fname in tqdm(filenames))