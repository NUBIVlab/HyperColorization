import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import aux_funcs.colorization as clr
import aux_funcs.noise as noise
import aux_funcs.spectral_dimensionality as spec
from aux_funcs.visualization import draw_hpim

def get_uniform_sampling_pattern(h: int,w: int ,sampling_ratio: float) -> np.ndarray:
    """
    Returns a sampling pattern with uniform sampling distance.
    h: height of the image
    w: width of the image
    sampling_ratio: ratio of sampled pixels to total pixels
    """
    sample_dist = np.ceil(np.sqrt(1/sampling_ratio)).astype(np.int32)
    col = np.arange(0,h)
    row = np.arange(0,w)
    x,y = np.meshgrid(col,row)
    mask = np.logical_and(x % sample_dist == sample_dist//2,  y % sample_dist == sample_dist//2)
    return mask

def get_random_sampling_pattern(h: int,w: int ,sampling_count: float = None, sampling_ratio: float=None) -> np.ndarray:
    """
    Returns a randomly selected lcoations on the image.
    h: height of the image
    w: width of the image
    sampling_count: number of sampled pixels
    sampling_ratio: ratio of sampled pixels to total pixels, if sampling_count is not provided
    """
    if sampling_count == None:
        sampling_count = int(h * w * sampling_ratio)
    x = np.arange(0,w)
    y = np.arange(0,h)
    x, y = np.meshgrid(x, y)
    x, y = np.reshape(x, -1), np.reshape(y, -1)
    coords = np.stack((x,y), axis=1)
    np.random.shuffle(coords)
    coords = coords[:sampling_count, :]
    x ,y = coords[:,0], coords[:,1]
    mask = np.zeros((h,w))
    mask[y,x] = 1
    return mask

def get_guided_sampling_pattern(gray_img, sampling_ratio, bin_count:int = 16, optional_weight = 0.7) -> np.ndarray:
    """
    Intelligent sampling of HSI based on the gray image.
    gray_img: gray image
    sampling_ratio: ratio of sampled pixels to total pixels
    bin_count: number of bins for quantization of gray image
    optional_weight: weight based on corner detection
    """
    default_dist = np.ceil(np.sqrt(1/sampling_ratio))
    gray_img = np.squeeze(np.copy(gray_img))
    h,w = gray_img.shape
    gray_img_binned = np.squeeze(gray_img / np.max(gray_img))
    gray_img_binned = np.floor(gray_img_binned * 64) / 64
    priority = np.zeros(h)
    for i in range(h):
        priority[i] = float(np.size(np.unique(gray_img_binned[i,:]))) 
    priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    priority = 0.1 + 0.9 * priority
    priority = priority / np.mean(priority)
    if optional_weight != 0:
        priority2 = np.zeros(h)
        corners = cv2.goodFeaturesToTrack((255*gray_img_binned).astype(np.uint8),int(h*w*0.1),0.01,int(h/25))
        corners = np.squeeze(np.int0(corners))
        output = np.zeros_like(np.squeeze(gray_img))
        output[corners[:,1],corners[:,0]] = 1
        output = cv2.blur(output,(int(h/10), int(w/10)))
        priority2 = np.sum(output, axis=1)
        priority2 = (priority2 - np.min(priority2)) / (np.max(priority2) - np.min(priority2))
        priority2 = 0.1 + 0.9 * priority2
        priority2 = priority2 / np.mean(priority2)
        priority = priority * (1-optional_weight) + priority2 * optional_weight
    count = np.floor(h / default_dist) + 1
    rows = np.zeros(count.astype(np.int32))
    cols = np.zeros((count.astype(np.int32), int(w / default_dist) + 1))
    pool = 0
    j = 0
    for i in range(0, h):
        pool += priority[i]
        if pool > default_dist:
            pool -= default_dist
            rows[j] = i
            j += 1
    if(rows[-1]) == 0:
        rtm1, rtm2 = np.copy(rows[:-1]), np.copy(rows[:-1])
        rtm1 = np.insert(rtm1, 0, 0)
        rtm2 = np.append(rtm2, h)
        grad = rtm2 - rtm1
        h=np.argmax(grad)
        rows[-1] = np.floor((rtm1[h] + rtm2[h]) / 2).astype(np.int32)
    rows = np.sort(rows)
    window_size = 21
    hws = 10
    windowed_view = np.lib.stride_tricks.sliding_window_view(gray_img_binned, (window_size, window_size))
    priority = np.zeros((windowed_view.shape[0], windowed_view.shape[1]))
    for i in range(windowed_view.shape[0]):
        for j in range(windowed_view.shape[1]):
            priority[i, j] = float(np.size(np.unique(windowed_view[i,j,:,:])))
    #priority = np.var(windowed_view, axis=(2,3))
    priority = np.pad(priority, ((hws,hws),(hws,hws)), 'constant', constant_values = 0)
    priority = priority[rows.astype(np.int32)]
    priority = (priority - np.min(priority, axis = 1, keepdims= True)) / (np.max(priority, axis = 1, keepdims= True) - np.min(priority, axis = 1, keepdims= True) + 1e-6)
    priority = 0.1 + 0.9 * priority
    priority = priority / np.mean(priority, axis = 1, keepdims=True)
    for i, row in enumerate(rows.astype(np.int32)):
        pool = 0
        k = 0
        for j in range(0,w):
            pool += priority[i, j]
            if pool > default_dist:
                pool -= default_dist
                cols[i,k] = j
                k += 1
        if(cols[i,-1]) == 0:
            rtm1, rtm2 = np.copy(cols[i,:-1]), np.copy(cols[i,:-1])
            rtm1 = np.insert(rtm1, 0, 0)
            rtm2 = np.append(rtm2, h)
            grad = rtm2 - rtm1
            h=np.argmax(grad)
            cols[i,-1] = np.floor((rtm1[h] + rtm2[h]) / 2).astype(np.int32)
    mask = np.zeros_like(gray_img)
    rows = np.reshape(rows, (rows.shape[0], 1))
    rows = np.tile(rows, (1, count.astype(np.int32)))
    rows, cols = np.reshape(rows, -1), np.reshape(cols, -1)
    mask[rows.astype(np.int32), cols.astype(np.int32)] = 1
    return mask

def get_uniform_pushbroom(h:int, w:int, sampling_ratio:float) -> np.ndarray:
    """
    Get a pushbroom mask with uniform sampling ratio
    h: height of the image
    w: width of the image
    sampling_ratio: sampling ratio of the pushbroom mask
    """
    col = np.arange(0,h)
    mask = np.zeros((h,w))
    mask[col%(np.ceil(1/sampling_ratio))==5,:] = 1
    return mask

def get_random_pushbroom(h:int, w:int, sampling_ratio:float) -> np.ndarray:
    """
    Get randomly selected rows from the HSI.
    h: height of the image
    w: width of the image
    sampling_ratio: sampling ratio of the pushbroom mask
    """
    col = np.random.choice(h, size = np.ceil(h*sampling_ratio).astype(np.int32), replace = False)
    mask = np.zeros((h,w))
    mask[col.astype(np.int32),:] = 1
    return mask

def get_guided_pushbroom(gray_img:np.ndarray, sampling_ratio: float, bin_count:int = 16, optional_weight:float = 0.7) -> np.ndarray:
    """
    Intelligent sampling of rows on HSI based on the gray image.
    gray_img: gray image
    sampling_ratio: sampling ratio of the pushbroom mask
    bin_count: number of bins for quantization of gray image
    optional_weight: weight based on corner detection
    """
    default_dist = np.ceil(1/sampling_ratio)
    gray_img = np.squeeze(np.copy(gray_img))
    h,w = gray_img.shape
    gray_img_binned = np.squeeze(gray_img / np.max(gray_img))
    gray_img_binned = np.floor(gray_img_binned * 64) / 64
    priority = np.zeros(h)
    for i in range(h):
        priority[i] = float(np.size(np.unique(gray_img_binned[i,:]))) 
    priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    priority = 0.1 + 0.9 * priority
    priority = priority / np.mean(priority)
    if optional_weight != 0:
        priority2 = np.zeros(h)
        corners = cv2.goodFeaturesToTrack((255*gray_img_binned).astype(np.uint8),int(h*w*0.1),0.01,int(h/25))
        corners = np.squeeze(np.int0(corners))
        output = np.zeros_like(np.squeeze(gray_img))
        output[corners[:,1],corners[:,0]] = 1
        output = cv2.blur(output,(int(h/10), int(w/10)))
        priority2 = np.sum(output, axis=1)
        priority2 = (priority2 - np.min(priority2)) / (np.max(priority2) - np.min(priority2))
        priority2 = 0.1 + 0.9 * priority2
        priority2 = priority2 / np.mean(priority2)
        priority = priority * (1-optional_weight) + priority2 * optional_weight
    count = np.floor(h / default_dist) + 1
    rows = np.zeros(count.astype(np.int32))
    pool = 0
    j = 0
    for i in range(0, h):
        pool += priority[i]
        if pool > default_dist:
            pool -= default_dist
            rows[j] = i
            j += 1
    if(rows[-1]) == 0:
        rtm1, rtm2 = np.copy(rows[:-1]), np.copy(rows[:-1])
        rtm1 = np.insert(rtm1, 0, 0)
        rtm2 = np.append(rtm2, h)
        grad = rtm2 - rtm1
        h=np.argmax(grad)
        rows[-1] = np.floor((rtm1[h] + rtm2[h]) / 2).astype(np.int32)
    mask = np.zeros_like(gray_img)
    mask[rows.astype(np.int32), :] = 1
    return mask

def testPatternsPushbroom(colorizer:clr.GlobalColorizer, hpim:np.ndarray, gray_img:np.ndarray, lams:np.ndarray, sampling_ratio:float, metrics:dict, metric:str, bin_count:int = 16, optional_weight:float = 0, draw_method = '1931', save=None) -> None:
    """
    Used for comparing different pushbroom sampling patterns.
    colorizer: colorizer object
    hpim: hyperspectral image
    gray_img: gray image
    lams: wavelength of the hyperspectral image
    sampling_ratio: sampling ratio of the pushbroom mask
    metrics: dictionary of metrics
    metric: metric to be used for comparison
    bin_count: number of bins for quantization of gray image
    optional_weight: weight based on corner detection
    """
    l,h,w = hpim.shape
    if save is not None:
        save1 = save + 'uni_pb.png'
        save2 = save + 'rnd_pb.png'
        save3 = save + 'gdd_pb.png'
    else:
        save1 = None
        save2 = None
        save3 = None
    #uniform init
    mask = get_uniform_pushbroom(h,w,sampling_ratio)
    uni_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save1, brightness_boost=1.5)
    uniform_score = metrics[metric](hpim, colorizer.get_result())
    #random init
    mask = get_random_pushbroom(h,w,sampling_ratio)
    rnd_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save2, brightness_boost=1.5)
    random_score = metrics[metric](hpim, colorizer.get_result())
    #guided init
    mask = get_guided_pushbroom(np.squeeze(gray_img/np.max(gray_img)), sampling_ratio, bin_count=bin_count, optional_weight=optional_weight)
    guided_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save3, brightness_boost=1.5)
    guided_score = metrics[metric](hpim, colorizer.get_result())
    print(metric+':\t\t\t\t'+'Sample Count'+':\nuni: ' + str(np.round(uniform_score, 5))+'\t\t\t'+
          str(uni_sample_cnt.astype(np.int32))+ '\nrnd: '+ str(np.round(random_score,5))+'\t\t\t'+
          str(rnd_sample_cnt.astype(np.int32))+ '\ngdd: '+ str(np.round(guided_score,5))+'\t\t\t'+
          str(guided_sample_cnt.astype(np.int32)))
    

def testPatternsWhiskbroom(colorizer:clr.GlobalColorizer, hpim:np.ndarray, gray_img:np.ndarray, lams:np.ndarray, sampling_ratio:float, metrics:dict, metric:str, bin_count:int = 64, optional_weight:float = 0.5, draw_method = '1931', save=None) -> None:
    """
    Used for comparing different whiskbroom sampling patterns.
    colorizer: colorizer object
    hpim: hyperspectral image
    gray_img: gray image
    lams: wavelength of the hyperspectral image
    sampling_ratio: sampling ratio of the whiskbroom mask
    metrics: dictionary of metrics
    metric: metric to be used for comparison
    bin_count: number of bins for quantization of gray image
    optional_weight: weight based on corner detection
    """
    l,h,w = hpim.shape
    if save is not None:
        save1 = save + 'uni_wb.png'
        save2 = save + 'rnd_wb.png'
        save3 = save + 'gdd_wb.png'
    else:
        save1 = None
        save2 = None
        save3 = None
    #uniform init
    mask = get_uniform_sampling_pattern(h,w,sampling_ratio)
    uni_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save1, convolve=np.ones((2,2)))
    uniform_score = metrics[metric](hpim, colorizer.get_result())
    #random init
    mask = get_random_sampling_pattern(h,w,uni_sample_cnt)
    rnd_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save2, convolve=np.ones((2,2)))
    random_score = metrics[metric](hpim, colorizer.get_result())
    #guided init
    mask = get_guided_sampling_pattern(np.squeeze(gray_img), sampling_ratio, bin_count=bin_count, optional_weight = optional_weight)
    guided_sample_cnt = np.sum(mask)
    colorizer.__init__(gray_img, mask*hpim, lams=lams, draw_method=draw_method)
    colorizer.hyperColorize(draw_result=True)
    colorizer.plot_spectral_clue_result(save_figure=save3, convolve=np.ones((2,2)))
    guided_score = metrics[metric](hpim, colorizer.get_result())
    print(metric+':\t\t\t\t'+'Sample Count'+':\nuni: ' + str(np.round(uniform_score, 5))+'\t\t\t'+
          str(uni_sample_cnt.astype(np.int32))+ '\nrnd: '+ str(np.round(random_score,5))+'\t\t\t'+
          str(rnd_sample_cnt.astype(np.int32))+ '\ngdd: '+ str(np.round(guided_score,5))+'\t\t\t'+
          str(guided_sample_cnt.astype(np.int32)))
    
def ShotNoiseClueCountTrade(hpim:np.ndarray,iter_c:int, basis:spec.SpectralDim, colorizer:clr.GlobalColorizer, metrics:dict, rec_dim:list = None) -> tuple[dict, np.ndarray]:
    """
    Used for analysis on trade off between shot noise and clue count.
    hpim: hyperspectral image
    iter_c: number of tests
    basis: spectral basis.
    colorizer: colorizer object
    metrics: dictionary of metrics
    rec_dim: reconstruction dimension (optional but recommended)
    """
    gray_img = np.sum(hpim, axis=0, keepdims=True)
    l,h,w = hpim.shape
    iteration = np.arange(iter_c)
    exposure_time = np.power(2, iteration) / (8192 * 96)
    min_samples = h*w*0.000244140625
    total_time = min_samples * exposure_time[-1]
    num_samples = total_time / exposure_time
    sampling_ratio = num_samples / (h*w)
    output = {'sf_disabled':dict(), 'sf_enabled':dict(), 'sf_disabled_quarter':dict(), 'sf_enabled_quarter':dict()}
    output['sampling_ratio'] = np.asarray(sampling_ratio)
    output['exposure_time'] = np.asarray(exposure_time)
    raw_data = np.zeros((2,h*w,iter_c))
    for metric in metrics.keys():
        output['sf_disabled'][metric] = np.zeros(iter_c)
        output['sf_enabled'][metric] = np.zeros(iter_c)
        output['sf_disabled_quarter'][metric] = np.zeros((iter_c,4))
        output['sf_enabled_quarter'][metric] = np.zeros((iter_c,4))
    print('Total time spent scanning: ', total_time)
    print('Sampling ratio: ', sampling_ratio)
    print('Exposure time: ', exposure_time)
    if rec_dim is None:
        dimensionality = np.zeros(iter_c).astype(np.int32)
    else:
        dimensionality = np.copy(rec_dim)
    for i in tqdm(range(len(sampling_ratio))):
        hp_n = noise.addPoissonNoise(hpim, exposure_time[i])
        mask = get_uniform_sampling_pattern(h,w, sampling_ratio[i])
        if rec_dim is None:
            dimensionality[i] = basis.intrinsicDimension(clr.smartFilter(gray_img, hp_n*mask), 'two_features')
            ldim_hpim_noised = basis.project(hp_n, dimensionality[i])
        else:
            ldim_hpim_noised = basis.project(hp_n, dimensionality[i])
        ldim_spectral_clues = ldim_hpim_noised * mask
        colorizer.__init__(gray_img, ldim_spectral_clues)
        result1 = colorizer.hyperColorize(sd = basis)
        colorizer.smartFilter()
        result2 = colorizer.hyperColorize(sd = basis)
        for metric in metrics.keys():
            output['sf_disabled'][metric][i], map_disabled= metrics[metric](hpim, result1, return_map = True)
            output['sf_enabled'][metric][i], map_enabled = metrics[metric](hpim, result2, return_map = True)
            raw_data[0,:,i] = np.reshape(map_disabled, -1)
            raw_data[1,:,i] = np.reshape(map_enabled, -1)
    output['recon_dims'] = dimensionality
    return output, raw_data


def constantTimeViolinFigure(output:dict, raw_data:np.ndarray, metric: str = 'EMD', y_range:tuple=(0,3.7)) -> None:
    """
    Uses the output of ShotNoiseClueCountTrade to generate a violin plot.
    output: output of ShotNoiseClueCountTrade
    raw_data: raw data from ShotNoiseClueCountTrade
    metric: metric to be used
    """
    exposure_time = output['exposure_time']
    sampling_ratio = output['sampling_ratio']
    rec_dims = output['recon_dims']
    raw_data = raw_data * 100
    result_sf_enabled = 100 * output['sf_enabled'][metric]
    result_sf_disabled = 100 * output['sf_disabled'][metric]
    data_locations = np.log10(exposure_time)
    data_locations2 = np.log10(sampling_ratio)
    p10, quartile1, quartile3, p90 = np.percentile(raw_data, [10, 25, 75, 90], axis=1)
    num_elements = raw_data.shape[1]
    num_remove = int(num_elements * (2 / 100))
    sorted_arr = np.sort(raw_data, axis=1)
    trimmed_arr = sorted_arr[:, 0:num_elements - num_remove, :]
    #Figure 1
    figure = plt.figure(figsize = (11, 5))
    ax1 = figure.add_subplot(1,1,1)
    ax1.set_title('''Optimizing Sampling Ratio with a Fixed Time Budget''', pad=25)
    ax1.set_ylim(y_range)
    ax1.set_xlabel('Exposure Time Per Pixel (ms)')
    ax1.set_ylabel(metric +'  ' +r'$\times10^{-2}$')
    tick_locations = np.arange(-6, -1.99, 1.0)
    ax1.set_xticks(tick_locations, np.power(10.0, tick_locations+3))
    #ax1.vlines(data_locations-0.01, quartile1[0,:], quartile3[0,:], color='navy', linestyle='-', lw=2.5, alpha=0.6, label = 'SFD Quartile Range')
    #ax1.vlines(data_locations-0.01, p10[0,:], p90[0,:], color='navy', linestyle='-', lw=1, alpha=0.6, label='SFD 10-90 Percentile Range')
    parts = ax1.violinplot(trimmed_arr[0,:,:],data_locations, showmeans = False, showextrema = False, showmedians = False, widths = 0.25)
    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_alpha(0.2)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
    ax1.plot(data_locations, result_sf_disabled, alpha = 0.7, color = 'blue', marker='o', mfc='white', label = 'Mean Error - Raw Measurements')
    #ax1.vlines(data_locations+0.01, quartile1[1,:], quartile3[1,:], color='maroon', linestyle='-', lw=2.5, alpha=0.6, label = 'SFE Quartile Range')
    #ax1.vlines(data_locations+0.01, p10[1,:], p90[1,:], color='maroon', linestyle='-', lw=1, alpha=0.6, label = 'SFE 10-90 Percentile Range')
    parts = ax1.violinplot(trimmed_arr[1,:,:],data_locations, showmeans = False, showextrema = False, showmedians = False, widths = 0.25)
    for pc in parts['bodies']:
        pc.set_facecolor('coral')
        pc.set_alpha(0.2)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    ax1.plot(data_locations, result_sf_enabled, alpha = 0.7, color = 'red', marker='o', label = 'Mean Error - Filtered Measurements', mfc='white')
    ax2 = ax1.twiny()
    ax2.plot(data_locations2, result_sf_disabled, marker = '.', alpha=0.0)
    ax2.set_xlabel('Sampling Ratio (%)')
    ax2.invert_xaxis()
    tick_locations = np.arange(-1,-4.05, -1.0)
    ax2.set_xticks(tick_locations, np.power(10.0, tick_locations+2))
    ax1.fill_between(tick_locations[-1:-2],0,0, facecolor='blue', alpha=0.2, label = 'Raw Measurements')
    ax1.fill_between(tick_locations[-1:-2],0,0, facecolor='coral', alpha=0.2, label = 'Filtered Measurements')
    ax1.legend(loc = 'upper right', fontsize=12)