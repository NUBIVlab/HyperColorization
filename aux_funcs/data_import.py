import numpy as np
import scipy
import cv2
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import OpenEXR as exr
import Imath


class importer:
#Functions that import different datasets
    def RGYPepperC():
        """
        hyperspectral data downloaded from:
        https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes
        returns format: CxHxW
        Not used in the paper.
        """
        f = scipy.io.loadmat('data/Pepper/RGYPepperC_2500.mat') 
        lams = np.squeeze(f['Wavelengths'])
        hpim0 = np.array(f['S']) 
        hpim = np.moveaxis(hpim0,1,2) # rotate all images
        return (lams, hpim)

    def BearAndFruit(rsz:float = 408/2010)-> tuple[np.ndarray, np.ndarray]:
        """
        hyperspectral data downloaded from:
        https://color2.psych.upenn.edu/hyperspectral/bearfruitgray/bearfruitgray.html
        returns format: CxHxW
        Cited as [10] in the paper.
        """
        f = scipy.io.loadmat('data/BearFruitGrayB/data.mat')
        lams = np.squeeze(f['wavelengths'])
        lams = lams.astype("double")
        hpim0 = np.array(f['img_stack'])
        hpim0 = hpim0[:, 5:-5, 5:-5]
        l,h,w = np.shape(hpim0)
        hpim = np.zeros((l,np.rint(rsz*h).astype(np.int32), np.rint(rsz*w).astype(np.int32)))
        for i in range(len(lams)):
            hpim[i,:,:] = cv2.resize(np.squeeze(hpim0[i,:,:]), (int(np.rint(rsz*h)), int(np.rint(rsz*w))))
        return(lams, hpim)
    
    def BearAndFruit_low_res()-> tuple[np.ndarray, np.ndarray]:
        """
        This function loads the low resolution version of the BearAndFruit dataset on google colab.
        See BearAndFruit() for more details.
        """
        f = scipy.io.loadmat('data/BearFruitGrayB/data_low_res.mat')
        lams = np.squeeze(f['lams'])
        lams = lams.astype("double")
        hpim = np.array(f['hpim'])
        return(lams, hpim)
    
    def load_Harvard_db(filepath:str = 'data/HarvardDataset/', correct_camera_sensitivity:bool = True) -> np.ndarray:
        """
        hyperspectral data downloaded from:
        http://vision.seas.harvard.edu/hyperspec/d2x5g3/
        Cited as [8] in the paper.
        This function unrolls the images to a Matrix so that each row occupies a pixel and each column a wavelength.
        returns format: NxC
        Used for learning the singular vectors.
        """
        cam_sens = np.genfromtxt(filepath+'calib.txt')
        spectral_data = np.zeros((31,1))
        for i,file in enumerate(tqdm(glob.glob(filepath+'*.mat'))):
            data = scipy.io.loadmat(file)
            image = data['ref'].astype(np.float32)
            mask = data['lbl'].astype(np.float32)
            image = cv2.resize(image, (130, 174))
            mask = cv2.resize(mask, (130, 174))
            image = np.moveaxis(image, 2, 0)
            l,w,h = image.shape
            if correct_camera_sensitivity:
                image = image / np.reshape(cam_sens, (cam_sens.shape[0], 1, 1))
            image = image[:,mask > 0]
            spectral_data = np.append(spectral_data, image, axis = 1)
        return spectral_data
    
    def load_Harvard_img(file_to_open:int = 0, rsz:float = 408/1040, correct_camera_sensitivity:bool = True)-> tuple[np.ndarray, np.ndarray]:
        """
        hyperspectral data downloaded from:
        http://vision.seas.harvard.edu/hyperspec/d2x5g3/
        returns format: CxHxW
        Cited as [8] in the paper.
        """
        filepath = 'data/HarvardDataset/'
        cam_sens = np.genfromtxt(filepath+'calib.txt')
        lams = np.arange(420,721,10)
        file = glob.glob(filepath+'*.mat')[file_to_open]
        data = scipy.io.loadmat(file)
        image = data['ref'].astype(np.float32)
        image = np.transpose(image, (2,0,1))
        image = image[:, :, (1392-1040):1392]
        image = image / np.reshape(np.max(image, axis=(1,2)), (image.shape[0],1,1))
        l,h,w = image.shape
        hpim = np.zeros((l,int(rsz*h), int(rsz*w)))
        for i in range(len(lams)):
            hpim[i,:,:] = cv2.resize(np.squeeze(image[i,:,:]), None, fx=rsz, fy=rsz)
        return lams, hpim
    
    def load_KAIST_img(file_to_open:int = 0, rsz:float = 408/2704) -> tuple[np.ndarray, np.ndarray]:
        """
        hyperspectral data downloaded from:
        http://vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset.html
        returns format: CxHxW
        Cited as [14] in the paper.
        """
        lams = np.arange(420,721,10)
        filepath = 'data/KAISTDataset/'
        file = glob.glob(filepath+'*.exr')[file_to_open]
        exrfile = exr.InputFile(file)
        header = exrfile.header()
        dw = header['dataWindow']
        h,w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
        wavelength_keys = {}
        l = 0
        for key in header['channels']:
            if key.startswith('w'):
                wavelength_keys[key] = header['channels'][key]
                l += 1
        im_dim = min(h,w)
        hpim = np.zeros((l,int(im_dim*rsz),int(im_dim*rsz)))
        for i,c in enumerate(wavelength_keys):
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.fromstring(C, dtype=np.float32)
            C = np.reshape(C, (h,w))
            C = C[:im_dim, :im_dim]
            C = cv2.resize(C, (int(im_dim*rsz),int(im_dim*rsz)))
            hpim[i,:,:] = C
        return lams, hpim

    def load_CAVE_db(filepath:str) -> np.ndarray:
        """
        hyperspectral data downloaded from:
        http://vision.seas.harvard.edu/hyperspec/d2x5g3/
        Cited as [11] in the paper.
        This function unrolls the images to a Matrix so that each row occupies a pixel and each column a wavelength.
        returns format: NxC
        Used for learning the singular vectors.
        """
        dataset = np.zeros((31, 1))
        for i,folder in enumerate(tqdm(glob.glob(filepath+'*'))):
            img = np.zeros((31,512,512))
            for j,file in enumerate(glob.glob(folder+'/*.png')):
                if i == 31:
                    print(file)
                img[j,:,:] = plt.imread(file)
            img = np.reshape(img, (31, -1))
            img = img[:, np.sum(img, axis=0) > 0.8]
            dataset = np.append(dataset, img, axis=1)
        return dataset
    
    
    def load_CAVE_img(file_to_open: int, rsz:float = 408/512) -> tuple[np.ndarray, np.ndarray]:
        """
        hyperspectral data downloaded from:
        https://www.cs.columbia.edu/CAVE/databases/multispectral/
        returns format: CxHxW
        Cited as [11] in the paper.
        """
        l,h,w = 31, np.floor(512*rsz).astype(np.int32), np.floor(512*rsz).astype(np.int32)
        filepath = 'data/CAVEDataset/'
        folder = glob.glob(filepath+'*')[file_to_open]
        dataname = folder.split('\\')[1]
        img = np.zeros((l,h,w))
        lams = np.arange(400, 705, 10)
        for j,file in enumerate(glob.glob(folder+'/*.png')):
             img[j,:,:] = cv2.resize(np.squeeze(plt.imread(file)), None, fx=rsz, fy=rsz)
        return lams, img
    
    def load_ICVL_img(rsz = 408/1306, spectral_downsample = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        hyperspectral data downloaded from:
        https://icvl.cs.bgu.ac.il/hyperspectral/
        returns format: CxHxW
        Cited as [13] in the paper.
        """
        converter = lambda s: float(s.decode('utf8').strip(','))
        data_path = 'data/ICVLDataset/'
        file = 'bulb_0822-0909'
        lams = np.loadtxt(data_path+file+'.hdr', dtype=np.float32, skiprows=34, converters = converter, max_rows=519)
        img = np.fromfile(data_path+file+'.raw', dtype=np.uint16, sep="")
        img = np.reshape(img, (1306, 519, 1392))
        img = np.flip(np.transpose(img, (1,2,0)), axis = 1)
        a = (1392 - 1306) //2
        img = img[:, a:(a+1306),:]
        hpim = np.zeros((519, int(img.shape[1]*rsz), int(img.shape[2]*rsz)))
        for i in range(img.shape[0]):
            hpim[i,:,:] = cv2.resize(img[i,:,:], (int(img.shape[1]*rsz), int(img.shape[2]*rsz)))
        hpim = hpim.astype(np.float32) / np.power(2.0, 15)
        if spectral_downsample != 1.0:
            c,h,w = hpim.shape
            ds_ratio = int(1/spectral_downsample)
            new_c = np.floor(c/ds_ratio).astype(np.int32)
            old_c = new_c * ds_ratio
            hpim = hpim[:old_c,:,:]
            lams = lams[:old_c]
            hpim = np.moveaxis(hpim, 0, -1)
            hpim = np.reshape(hpim, (h,w,-1,ds_ratio))
            hpim = np.sum(hpim, axis=-1)
            hpim = np.moveaxis(hpim, -1, 0)
            lams = np.reshape(lams, (-1,ds_ratio))
            lams = np.mean(lams, axis=-1)
        return lams, hpim

