import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_hpim(hpim: np.ndarray, lams:np.ndarray = None, draw:bool = False, method:str = '1931', normalize = True) -> np.ndarray:
    """
    Draw hyperspectral image in RGB format.
    hpim: hyperspectral image
    lams: wavelength of each band
    draw: whether to draw the image or just return the rgb array.
    method: '1931' or 'Gaussians'
    1931: use the CIE 1931 standard observer to convert the hyperspectral image to RGB image.
    Gaussians: use Gaussian functions to convert the hyperspectral image to RGB image.
    return rgb_img in HWC format. 
    """
    l,h,w = np.shape(hpim)
    if lams is None:
        lams = np.linspace(400, 700, num = l)
    if method == '1931':
        data = np.reshape(hpim, (l, -1), order ='C')
        delta_lam = lams[1] - lams[0]
        start_lam, end_lam = lams[0] - delta_lam / 2, lams[-1] + delta_lam / 2
        CIEXYZ_1931_table = np.load('data/CIEXYZ_1931_table.npy')
        scaling = 1 / np.sum(CIEXYZ_1931_table, axis=0)[2]
        CIEXYZ_1931_table[:, 1:4] = CIEXYZ_1931_table[:, 1:4] * scaling
        start_index, end_index = (start_lam - CIEXYZ_1931_table[0,0]).astype(np.int32), (end_lam - CIEXYZ_1931_table[0,0]).astype(np.int32)
        X = np.sum(CIEXYZ_1931_table[start_index:end_index, 1].reshape(-1, delta_lam.astype(np.int32)), axis=1)
        Y = np.sum(CIEXYZ_1931_table[start_index:end_index, 2].reshape(-1, delta_lam.astype(np.int32)), axis=1)
        Z = np.sum(CIEXYZ_1931_table[start_index:end_index, 3].reshape(-1, delta_lam.astype(np.int32)), axis=1)
        hs2xyz = np.stack([X, Y, Z], axis=1).T
        xyz_data = hs2xyz @ data
        #xyz2rgb = np.asarray([2.3706743, -0.9000405, -0.4706338, -0.5138850, 1.4253036, 0.0885814, 0.0052982, -0.0146949, 1.0093968]).reshape(3,3)
        xyz2rgb = np.asarray([2.0413690, -0.5649464, -0.3446944, -0.9692660, 1.8760108, 0.0415560, 0.0134474, -0.1183897,1.0154096]).reshape(3,3)
        rgb_data = xyz2rgb @ xyz_data
        if normalize:
            rgb_data = (rgb_data - np.min(rgb_data, axis = 1, keepdims=True)) / (np.max(rgb_data, axis = 1, keepdims=True) - np.min(rgb_data, axis = 1, keepdims=True) + 1e-16)
        rgb_img = np.reshape(rgb_data, (3, h, w), order='C')
        rgb_img = np.moveaxis(rgb_img, 0, -1)
        rgb_img = np.clip(rgb_img, 0, np.inf)
        if draw:
            plt.imshow(rgb_img)
            plt.axis('off')
        return rgb_img
    elif method == 'Gaussians':
        R = .55*np.exp(-(lams-600)**2/2500)
        G = np.exp(-(lams-525)**2/4000)
        B = .85*np.exp(-(lams-450)**2/5000)
        wb = [0.972952272645604, 2.0642231002049565, 1.6066716258050628]
        R = R/wb[0]
        G = G/wb[1]
        B = B/wb[2]
        Rim = np.moveaxis(np.tile(R,[h,w,1]),2,0)
        Gim = np.moveaxis(np.tile(G,[h,w,1]),2,0)
        Bim = np.moveaxis(np.tile(B,[h,w,1]),2,0)
        rgb_img = np.stack([np.sum(Rim*hpim,axis=0),np.sum(Gim*hpim,axis=0),np.sum(Bim*hpim,axis=0)],axis=-1)
        rgb_img = np.clip((1/3) * rgb_img, 0,1)
        if draw:
            plt.imshow(rgb_img)
            plt.axis('off')
        return rgb_img

def save_hpim(path:str, hpim: np.ndarray, lams: np.ndarray, method:str = '1931', normalize:bool = False) ->  None:
    '''
    Calls draw_hpim, but instead of showing the image, it saves it.
    path: path to save the image.
    hpim: hyperspectral image
    lams: wavelength of each band
    draw: whether to draw the image or just return the rgb array.
    method: '1931' or 'Gaussians'
    1931: use the CIE 1931 standard observer to convert the hyperspectral image to RGB image.
    Gaussians: use Gaussian functions to convert the hyperspectral image to RGB image.
    '''
    rgb_img = draw_hpim(hpim, lams, draw=False, method=method, normalize=normalize)
    rgb_img = np.clip(rgb_img, 0, 1)
    plt.imsave(path, rgb_img)
    return

def slider_visualization_colorized(hpim:np.ndarray, lams:np.ndarray) -> None:
    """
    Visualize hyperspectral image with a slider to change the wavelength.
    Band are colorized according to their wavelength.
    hpim: hyperspectral image
    lams: wavelength of each band
    """
    def on_slide(value):
        img = hpim[value,:,:]
        img = np.tile(np.expand_dims(img, 2), (1,1,3))
        img[:,:,0] = img[:,:,0] * B[value]
        img[:,:,1] = img[:,:,1] * G[value]
        img[:,:,2] = img[:,:,2] * R[value]
        cv2.imshow(windowName, img)
        return img
    R = .55*np.exp(-(lams-600)**2/2500)
    G = np.exp(-(lams-525)**2/4000)
    B = .85*np.exp(-(lams-450)**2/5000)
    wb = 1.04*np.array([2.787597850164129, 2.6481827452593056, 1.0046574012759086])
    R = R/wb[0]
    G = G/wb[1]
    B = B/wb[2]
    windowName = 'Data'    
    cv2.imshow(windowName, on_slide(0))
    cv2.createTrackbar('slider', windowName, 0, hpim.shape[0]-1, on_slide)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def slider_visualization_gray(hpim:np.ndarray) -> None:
    """
    Visualize hyperspectral image with a slider to change the wavelength.
    Band are grayscale.
    hpim: hyperspectral image
    """
    def on_slide(value):
        img = hpim[value,:,:]
        cv2.imshow(windowName, img)
        return img
    windowName = 'Data'    
    cv2.imshow(windowName, on_slide(0))
    cv2.createTrackbar('slider', windowName, 0, hpim.shape[0]-1, on_slide)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicker_visualization(lams, hpim1, hpim2 = None, hpim3 = None, hpim4 = None, hpim5 = None, name1 = '1', name2 = '2', name3 = '3', name4 = '4', name5 = '5', draw: int = 1, draw_method = 'Gaussians') -> None:
    """
    Visualize hyperspectral image click on a location  to plot the corresponding spectral response.
    You can choose up to 5 different results to plot and also assign their names for the legend.
    draw_method: 'Gaussians' or '1931'
    draw: which hpim should be shown in the figure. 1, 2, 3, 4 or 5.
    """
    def mouseHyperspectral(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < w and y < h:
                fig = plt.figure(figsize = (10,10), facecolor='w', dpi=DPI)
                data1 = hpim1[:,y,x]
                if hpim2 is not None:
                    data2 = hpim2[:,y,x]
                if hpim3 is not None:
                    data3 = hpim3[:,y,x]
                if hpim4 is not None:
                    data4 = hpim4[:,y,x]
                if hpim5 is not None:
                    data5 = hpim5[:,y,x]
                plt.clf()
                plt.plot(lams, data1, label = name1, linewidth = '3', color = 'black')
                if hpim2 is not None:
                    plt.plot(lams, data2, label = name2, linewidth = '2', color = 'blue')
                if hpim3 is not None:
                    plt.plot(lams, data3, label = name3, linewidth = '2', color = 'red')
                if hpim4 is not None:
                    plt.plot(lams, data4, label = name4, linewidth = '2', color = 'green')
                if hpim5 is not None:
                    plt.plot(lams, data5, label = name5, linewidth = '2', color = 'purple')
                plt.legend()
                plt.title('x:' + str(x) + ' y:' + str(y))
                fig.savefig('ui/test.png', dpi=DPI)
                plot_rgb = np.reshape(plt.imread('ui/test.png'),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
                if h > 800:
                    plot_rgb = cv2.resize(plot_rgb, (hpim1.shape[1], hpim1.shape[2]))
                else:
                    plot_rgb = cv2.resize(plot_rgb, (800, 800))
                plot_rgb = plot_rgb[:,:,0:3]
                plot_rgb = plot_rgb[...,::-1]
                draw_rgb = np.concatenate((result_bgr, plot_rgb), axis = 1)
                plt.close(fig)
                cv2.imshow('image', draw_rgb)
    DPI = 256
    img_list = [hpim1, hpim2, hpim3, hpim4, hpim5]
    result_rgb = draw_hpim(img_list[draw - 1], draw=False, method=draw_method, lams=lams)
    result_bgr = result_rgb[...,::-1]
    h,w,l = result_bgr.shape
    if h < 800:
        box = np.ones((800-h,w,l))
        result_bgr = np.concatenate((result_bgr, box), axis = 0)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mouseHyperspectral)
    cv2.imshow('image',result_bgr)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
