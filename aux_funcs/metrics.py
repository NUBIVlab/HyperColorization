import numpy as np
import numpy as np
from skimage.metrics import structural_similarity
import scipy

def getQualityMetrics() -> tuple[dict, dict, dict]:
    """
    Returns a dictionary of quality metrics and a dictionary of boolean values indicating whether the metric is a high-is-better metric or not.
    """
    qm_dict_full = {'MSE' : MSE, 'GFC' : GFC, 'SSV' : SSV, 'PSNR' : PSNR, 'RMS' : RMS, 'SID' : SID, 'SSIM' : SSIM, 'EMD': EMD, 'MLPD': MLPD}
    hib_dict = {'MSE' : False, 'GFC' : True, 'SSV' : False, 'PSNR': True, 'RMS' : False, 'SID' : False, 'SSIM' : True, 'EMD' : False, 'MLPD': False}
    qm_dict = {'PSNR' : PSNR, 'GFC' : GFC, 'SSV' : SSV, 'EMD': EMD}
    return qm_dict, hib_dict, qm_dict_full
    
def getQualityMetrics2() -> tuple[dict, dict, dict]:
    """
    Returns a dictionary of quality metrics and a dictionary of boolean values indicating whether the metric is a high-is-better metric or not.
    Includes SSIM. 
    """
    qm_dict_full = {'MSE' : MSE, 'GFC' : GFC, 'SSV' : SSV, 'PSNR' : PSNR, 'RMS' : RMS, 'SID' : SID, 'SSIM' : SSIM, 'EMD': EMD, 'MLPD': MLPD}
    hib_dict = {'MSE' : False, 'GFC' : True, 'SSV' : False, 'PSNR': True, 'RMS' : False, 'SID' : False, 'SSIM' : True, 'EMD' : False, 'MLPD': False}
    qm_dict = {'PSNR' : PSNR, 'GFC' : GFC, 'SSV' : SSV, 'EMD': EMD, 'SSIM': SSIM}
    return qm_dict, hib_dict, qm_dict_full

def MLPD(img1: np.ndarray, img2: np.ndarray, p: float = 1.0, return_map: bool = False):
    """
    Mean L-p distance between two images.
    return_map: if True, returns the local values of L-p distance.
    """
    assert img1.shape == img2.shape
    l,h,w = img1.shape
    diff = np.abs(img1 - img2)
    local_lp = np.power(np.sum(np.power(diff, p), axis = 0), 1/p)
    lp = np.mean(local_lp)
    if return_map:
        return lp, local_lp
    else:
        return lp
    

def EMD(img1: np.ndarray, img2: np.ndarray, return_map: bool = False):
    """
    Mean Earth Movers distance, aka Wasserstein distance, between two hyperspectral images.
    return_map: if True, returns the local values of Wasserstein distance.
    """
    assert img1.shape == img2.shape
    l,h,w = img1.shape
    local_emd = np.zeros((h,w))
    for i in range(img1.shape[1]):
        for j in range(img1.shape[2]):
            local_emd[i,j] = scipy.stats.wasserstein_distance(img1[:,i,j], img2[:,i,j])
    emd = np.mean(local_emd)
    if return_map:
        return emd, local_emd
    else:
        return emd


def PSNR(img1: np.ndarray, img2: np.ndarray, return_map: bool = False):
    """
    Peak signal-to-noise ratio between two images.
    return_map: if True, returns the local values of PSNR.
    """
    assert img1.shape == img2.shape
    if return_map:
        rms, local_rms  = RMS(img1, img2, return_map = True)
        local_psnr = 20 * np.log10(1/(local_rms+1e-10))
        psnr = 20 * np.log10(1/(rms+1e-10))
        return psnr, local_psnr
    else:
        rms = RMS(img1, img2, return_map = False)
        psnr = 20 * np.log10(1/(rms+1e-10))
        return psnr
    
def SSIM(img1: np.ndarray, img2: np.ndarray):
    """
    Structural similarity index between two images.
    There is no local value for SSIM.
    Input format: CxHxW
    """
    if len(img1.shape) == 2:
        return structural_similarity(img1, img2, data_range = 1.0)
    else:
        return structural_similarity(img1, img2, data_range = 1.0, channel_axis = 0)
    
def MSE(orig: np.ndarray, test: np.ndarray, return_map: bool = False):
    """
    Mean squared error between two images.
    return_map: if True, returns the local values of MSE.
    Input format: CxHxW
    """
    assert orig.shape == test.shape
    diff = orig - test
    if return_map:
        return np.sum(np.square(diff)) / diff.size, np.sum(np.square(diff), axis=0)
    else:
        return np.sum(np.square(diff)) / diff.size

def GFC(data1: np.ndarray, data2: np.ndarray, return_map: bool = False):
    """
    Goodnes factor coefficient between two images.
    return_map: if True, returns the local values of GFC.
    Input format: CxHxW
    """
    assert data1.shape == data2.shape
    l, h, w = data1.shape
    norm1 = np.linalg.norm(data1, axis = 0, keepdims = True)
    norm2 = np.linalg.norm(data2, axis = 0, keepdims = True)
    data1 = data1 / (norm1 + 1e-10)
    data2 = data2 / (norm2 + 1e-10)
    correlation = data1 * data2
    if return_map:
        return np.sum(correlation) / (h * w), np.sum(correlation, axis = 0)
    else:
        return np.sum(correlation) / (h * w) 

def SSV(orig: np.ndarray, test: np.ndarray, return_map: bool = False):
    """
    Spectral similarity value between two images.
    return_map: if True, returns the local values of SSV.
    Input format: CxHxW
    """
    assert orig.shape == test.shape
    l, h, w = orig.shape
    diff = orig - test
    m_squared = np.sum(np.square(diff), axis = 0) / l
    orig_mean = np.mean(orig, axis = 0)
    orig_std = np.std(orig, axis = 0)
    test_mean = np.mean(test, axis = 0)
    test_std = np.std(test, axis = 0)
    std = orig_std * test_std
    correlation = np.mean((orig - orig_mean) * (test - test_mean), axis = 0)
    s_squared = 1 - np.square(correlation / (std + 1e-10))
    ssv_local = np.sqrt(m_squared + s_squared)
    ssv = np.mean(ssv_local)
    if return_map:
        return ssv, ssv_local
    else:
        return ssv

def RMS(orig: np.ndarray, test: np.ndarray, return_map: bool = False):
    """
    Root mean squared error between two images.
    return_map: if True, returns the local values of RMS.
    Input format: CxHxW
    """
    assert orig.shape == test.shape
    l, h, w = orig.shape
    diff = orig - test
    rms_local = np.sqrt(np.sum(np.square(diff), axis = 0) / l)
    rms = np.sum(rms_local) / (h * w)
    if return_map:
        return rms, rms_local
    else:
        return rms

def SID(orig: np.ndarray, test: np.ndarray, return_map: bool = False):
    """
    Spectral information divergence between two images.
    return_map: if True, returns the local values of SID.
    Input format: CxHxW
    """
    assert orig.shape == test.shape
    l, h, w = orig.shape
    orig_gray = np.sum(orig, axis = 0)
    test_gray = np.sum(test, axis = 0)
    orig_scaled = orig / (orig_gray + 1e-10) 
    test_scaled = test / (test_gray + 1e-10)
    diff = test_scaled - orig_scaled
    log_diff = np.log((test_scaled+1e-10)) - np.log((orig_scaled+1e-10))
    sid_local = np.sum(diff * log_diff, axis = 0)
    sid = np.mean(sid_local)
    if return_map:
        return sid, sid_local
    else:
        return sid



    