import numpy as np
import aux_funcs.initialization as init
import aux_funcs.metrics as qm
import aux_funcs.spectral_dimensionality as dim
import aux_funcs.colorization as clr


def gaussian(dist, sigma):
    return np.exp(-np.square(dist)/(2*np.square(sigma)))

def propagate_spectral_clues_Cao2015(spectral_clues: np.ndarray, rgb_img: np.ndarray, sigma_spatial:float = 21, sigma_g:float = 0.03, filter_size:int = 31) -> np.ndarray:
    """
    Implements Equation 1 from "High Resolution Multispectral Video Capture with a Hybrid Camera System" by Xun Cao (2015).
    Implementation is semi-vectorized, it can be sped up more with more vectorization.
    spectral_clues: sparse spectral cluess, np.ndarray of shape (n_channels, height, width)
    rgb_img: guide rgb image, np.ndarray of shape (height, width, 3)
    sigma_spatial: float, hyperparameter, controlls the effectiveness of spatial distance in color propagation.
    sigma_g: float, hyperparameter, controlls the effectiveness of rgb distance in color propagation.
    filter_size: int, hyperparameter, controlls the size of the filter used for color propagation.
    note: increasing the filter size beyond 91 marginally improves the results, but increases the runtime significantly.
    """
    result_img = np.copy(spectral_clues)
    rgb_img = np.moveaxis(rgb_img, -1, 0)
    v = int(np.floor(filter_size/2))
    for i in range(spectral_clues.shape[1]):
        for j in range(spectral_clues.shape[2]):
            if np.sum(spectral_clues[:,i,j]) != 0:
                continue
            window_min = int(max(0, i-v)), int(max(0, j-v))
            window_max = int(min(spectral_clues.shape[1], i+v)), int(min(spectral_clues.shape[2], j+v))
            window_center = i - window_min[0], j - window_min[1]
            window = spectral_clues[:, window_min[0]:window_max[0], window_min[1]:window_max[1]]
            window_rgb = rgb_img[:,window_min[0]:window_max[0], window_min[1]:window_max[1]]
            axis_x, axis_y = np.meshgrid(np.arange(-window_center[1], window.shape[2]-window_center[1]), np.arange(-window_center[0], window.shape[1]-window_center[0]))
            mask = np.sum(window, axis=0) != 0
            spatial_dist = np.sqrt(np.square(axis_x[mask]) + np.square(axis_y[mask]))
            central_rgb = np.reshape(window_rgb[:,window_center[0], window_center[1]], (3,1))
            spectral_dist = np.sqrt(np.sum(np.square(window_rgb[:,mask]-central_rgb),axis=0))
            spatial_dist = gaussian(spatial_dist, sigma_spatial)
            spectral_dist = gaussian(spectral_dist, sigma_g)
            weight = spatial_dist * spectral_dist
            result_img[:,i,j] = np.sum(weight * window[:,mask], axis=1) / (np.sum(weight) + 1e-10)
    return result_img

def Cao2015Wrapper(hpim_n:np.ndarray, rgb_img:np.ndarray, sampling_ratio:float):
    '''
    Wrapper function to generate results of Cao2015.
    '''
    spectral_clues = hpim_n * init.get_uniform_sampling_pattern(hpim_n.shape[1], hpim_n.shape[2], sampling_ratio)
    result = propagate_spectral_clues_Cao2015(spectral_clues, rgb_img)
    return result

def hp_grid_search_Cao2015(hpim, rgb_img):
    '''
    Performs a grid search over the hyperparameters of the Cao2015's equation 1.
    Best hyperparameters were: filter size: 151, sigma_spatial: 16.6, sigma_g: 0.015
    I gave up some accuracy for speed, and used filter size: 71, sigma_spatial: 16.6, sigma_g: 0.015.
    '''
    sampling_ratio = 0.01
    filter_sizes = [151.0]
    sigma_spatials = [16.6]
    sigma_gs = [0.01]
    spectral_clues = init.get_uniform_sampling_pattern(hpim.shape[1], hpim.shape[1], sampling_ratio)
    best_score = 0
    for filter_size in filter_sizes:
        for sigma_spatial in sigma_spatials:
            for sigma_g in sigma_gs:
                result = propagate_spectral_clues_Cao2015(spectral_clues, rgb_img, sigma_spatial, sigma_g, filter_size)
                score = qm.PSNR(result, hpim)
                print(f"Filter size: {filter_size}, sigma_spatial: {sigma_spatial}, sigma_g: {sigma_g}, score: {score}")
                if score > best_score:
                    best_score = score
                    best_filter_size = filter_size
                    best_sigma_spatial = sigma_spatial
                    best_sigma_g = sigma_g
    print(f"Best filter size: {best_filter_size}, best sigma_spatial: {best_sigma_spatial}, best sigma_g: {best_sigma_g}, best score: {best_score}")
    return best_filter_size, best_sigma_spatial, best_sigma_g


def hyperColorizationWrapper(gray:np.ndarray, hpim:np.ndarray, lams:np.ndarray, sd: dim.SpectralDim, sampling_ratio: float, applySmartFilter = True):
    '''
    Wrapper for my results.
    '''
    spectral_clues = hpim * init.get_uniform_sampling_pattern(hpim.shape[2], hpim.shape[1], sampling_ratio)
    ideal_dim = int(np.clip(sd.intrinsicDimension(spectral_clues, 'two_features'), 0, 31)) 
    spectral_clues = sd.project(spectral_clues, ideal_dim)
    GClr = clr.GlobalColorizer(gray, spectral_clues, lams, '1931')
    if applySmartFilter:
        GClr.smartFilter()
    return GClr.hyperColorize(sd)