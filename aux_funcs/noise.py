import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def addGaussianNoise(img: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """
    Adds Gaussian noise to an image.
    Used to simulate read noise.
    Input format: CxHxW
    """
    img_in = np.copy(img)
    img_in = img_in + np.random.normal(mean, variance, img.shape)
    img_in = np.clip(img_in,0,1)
    return img_in

def addSaltAndPepperNoise(img: np.ndarray, saltCount: int, pepperCount: int) -> np.ndarray:
    """
    Adds salt and pepper noise to an image.
    Not used in the paper but cool for testing.
    Input format: CxHxW
    """
    l, h, w = img.shape
    for i in range(saltCount):
        y_coord=random.randint(0, w - 1)
        x_coord=random.randint(0, h - 1)
        img[:,x_coord,y_coord] = 1
    for i in range(pepperCount):
        y_coord=random.randint(0, w - 1)
        x_coord=random.randint(0, h - 1)
        img[:,x_coord,y_coord] = 0
    return img

def addPoissonNoise(img: np.ndarray, exp: float) -> np.ndarray:
    """
    Adds Poisson noise to an image.
    Used to simulate shot noise.
    Input format: CxHxW
    """
    photons_per_second = 9.6*np.power(10.0,7.0) #https://warrenmars.com/photography/technical/resolution/photons.htm
    num_channel = 31
    photon_count_per_channel = photons_per_second / num_channel * exp
    img_in = np.copy(img).astype(float)
    img_in = img_in * photon_count_per_channel
    img_in = np.random.poisson(img_in)
    img_in = img_in / photon_count_per_channel
    return img_in

def modifyIllumination(hpim, lams, illumination_spectra_address, visualize=False):
    """
    Scales channels in hpim based on values in illumination_spectra.
    """
    light_vec = np.zeros((len(lams),1))
    fh = open(illumination_spectra_address)
    lines = fh.readlines()
    fh.close()
    data_amount = len(lines) - 1
    data_arr = np.zeros((data_amount, 2))
    for i, line in enumerate(lines):
        if i == 0: continue
        nums = line.split()
        data_arr[i-1,0] = float(nums[0])
        data_arr[i-1,1] = float(nums[1])
    lam_min = np.min(lams)
    lam_max = np.max(lams)
    nlam = len(lams)
    lam_bin_size = (lam_max - lam_min) / (nlam - 1)
    lam_bin_min =  lam_min - lam_bin_size / 2
    lam_bin_max =  lam_max + lam_bin_size / 2
    bins = np.linspace(lam_bin_min, lam_bin_max, nlam + 1, endpoint=True)
    data_arr[:,0] = np.digitize(data_arr[:,0],bins,right=False)
    for i in range(nlam + 1):
        if i == 0: continue
        for j in range(data_arr.shape[0]):
            if data_arr[j,0] == i:
                light_vec[i-1] += data_arr[j,1]
    light_vec = light_vec / np.max(light_vec)
    light_vec = light_vec[:, :, np.newaxis]
    hpim_new = light_vec * hpim
    light_vec = np.reshape(light_vec, (np.size(light_vec), 1))
    if visualize:
        plt.plot(lams, light_vec)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title("Illumination Spectra")
        plt.show()
    return hpim_new, light_vec