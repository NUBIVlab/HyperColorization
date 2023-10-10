import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import aux_funcs.noise as noise
from kneed import KneeLocator
from tqdm import tqdm
from aux_funcs.metrics import MSE, GFC, SSV, PSNR, RMS, SID, SSIM
import aux_funcs.visualization as  vis

class SpectralDim:
    """ Class that contaons the functions and data for lower dimensional representation of hyperspectral images."""
    def __init__(self) -> None:
        self.basis = None
        self.singular_values = None

    def learnSpectralBasisDB(self, data: np.ndarray, saveLocation: str = None) -> dict:
        """
        Learns the spectral basis of a hyperspectral image dataset using the data matrix.
        data: Unrolled hyperspectral image of shape NxC, where N is the number of pixels and C is the number of spectral bands.
        saveLocation: If not None, saves the basis and singular values to the specified location.
        """
        sigma = (data@data.T) / data.shape[1]
        self.basis,self.singular_values,_ = np.linalg.svd(sigma)
        tbr = {'basis': self.basis, 'singular_values': self.singular_values}
        if saveLocation is not None:
            np.save(saveLocation, tbr)
        return tbr
    
    def learnSpectralBasis(self, hpim: np.ndarray, saveLocation: str = None) -> dict:
        """
        Learns the spectral basis of a hyperspectral image dataset which is still not unrolled.
        hpim: hyperspectral image of shape CxHxW.
        saveLocation: If not None, saves the basis and singular values to the specified location.
        """
        l,h,w = hpim.shape
        data = np.reshape(hpim, (l, -1), order ='F')
        sigma = (data@data.T) / data.shape[1]
        self.basis,self.singular_values,_ = np.linalg.svd(sigma)
        tbr = {'basis': self.basis, 'singular_values': self.singular_values}
        if saveLocation is not None:
            np.save(saveLocation, tbr)
        return tbr

    def loadSpectralBasis(self, dataLocation: str) -> dict:
        """
        Loads the spectral basis and singular values from a file.
        """
        data = np.load(dataLocation, allow_pickle=True)
        data = data.item()
        self.basis = data['basis']
        self.singular_values = data['singular_values']
        return data
    
    def intrinsicDimension(self, hpim_n:np.ndarray, method:str = 'two_features') -> int:
        """
        Estimates the intrinsic dimension of a hyperspectral image from the spectral clues.
        hpim_n: hyperspectral image of shape CxHxW, probably sparse.
        method: 'knee_locator' or 'two_features'
        """
        l,h,w = hpim_n.shape
        if method == 'knee_locator':
            u = SpectralDim()
            u.learnSpectralBasis(hpim_n)
            kl = KneeLocator(range(1,l+1), np.log(u.singular_values), curve="convex", direction="decreasing")
            return kl.knee
        if method == 'two_features':
            projection_power = np.var(self.project(hpim_n, 31), axis=(1,2))
            projection_power = np.log2(projection_power)
            projection_power = -np.sort(-projection_power)
            feature1 = np.abs(np.min(projection_power))
            feature2 = self.intrinsicDimension(hpim_n, 'knee_locator')
            X = np.asarray([float(feature1), float(feature2), float(np.power(feature1,2)), float(np.power(feature2,2)), 1.0])
            weights = np.asarray([0.6354399, 2.65882429, 0.0247422, -0.10020802, -14.625279710070192])
            pred = np.clip(np.round(np.sum(X*weights)), 1, 31)
            return int(pred)

    def project(self, hpim: np.ndarray, dim: int) -> np.ndarray:
        """
        Projects a hyperspectral image to a lower dimensional space.
        hpim: hyperspectral image of shape CxHxW.
        dim: dimension of the projection.
        returns: projected hyperspectral image of shape PxHxW.
        """
        l,h,w = hpim.shape
        data = np.reshape(hpim, (l, -1), order ='C')
        z = data.T @ self.basis[:, 0:dim]
        hpim_projected = np.reshape(z.T, (dim,h,w))
        return hpim_projected
    
    def recover(self, hpim: np.ndarray) -> np.ndarray:
        """
        Recovers a hyperspectral image from its lower dimensional representation.
        hpim: hyperspectral image of shape PxHxW.
        """
        dim,h,w = hpim.shape
        l, u = self.basis.shape
        data = np.reshape(hpim, (dim, -1), order ='C')
        x_rec = (data.T @ self.basis[:, 0:dim].T).T
        hpim_recovered = np.reshape(x_rec, (l,h,w))
        return hpim_recovered
    
    def projectRecover(self, hpim: np.ndarray , dim: int) -> np.ndarray:
        """
        Projects a hyperspectral image to a lower dimensional space and recovers it.
        hpim: hyperspectral image of shape CxHxW.
        dim: dimension of the projection.
        """
        l,h,w = hpim.shape
        data = np.reshape(hpim, (l, -1), order ='C')
        z = data.T @ self.basis[:, 0:dim]
        x_rec = (z @ self.basis[:, 0:dim].T).T
        hpim_recovered = np.reshape(x_rec, (l,h,w))
        return hpim_recovered
    
def get_uniform_sampling_pattern(h: int,w: int ,sampling_ratio: float) -> np.ndarray:
    """
    Redeclared here to fix a bug. I will remove this once I fix the cross referencing.
    I recommed using: aux_funcs.initializations.get_uniform_sampling_pattern(...)
    """
    sample_dist = math.ceil(math.sqrt(1/sampling_ratio))
    col = np.arange(0,h)
    row = np.arange(0,w)
    x,y = np.meshgrid(col,row)
    mask = np.logical_and(x % sample_dist == sample_dist//2,  y % sample_dist == sample_dist//2)
    return mask


def numberOfBaseAnalysis(hpim:np.ndarray, spec:SpectralDim, maxNumBase:int) -> plt.figure:
    """ 
    Draws the recovered hyperspectral images for different number of basis and plots the MSE.
    hpim: hyperspectral image of shape CxHxW.
    spec: SpectralDim object.
    maxNumBase: maximum number of basis to use.
    """
    mse_list = list()
    subplotdim = int(np.ceil(np.sqrt(maxNumBase + 1)))
    fig = plt.figure(figsize=(subplotdim*4, subplotdim*4))
    for k in range(1, maxNumBase + 1):
        hpim_recovered = spec.projectRecover(hpim, k)
        mse = MSE(hpim, hpim_recovered)
        mse_list.append(mse)
        rgb_img = vis.draw_hpim(hpim_recovered, draw=False)
        fig.add_subplot(subplotdim, subplotdim, k+1).set_title(k)
        plt.imshow(rgb_img)
        plt.axis('off')
    fig.add_subplot(subplotdim, subplotdim, 1).set_title('Reprojection Error (MSE) vs # of basis')
    plt.plot(range(1, maxNumBase + 1), mse_list)
    plt.ylabel('Reprojection Error (MSE)')
    plt.xlabel('Number of basis')
    plt.show()
    return fig
    
def drawSpectralBasis(spec:SpectralDim, lams:np.ndarray, invert: list() = [1, 1, 1, 1, 0, 0]) -> None:
    """
    Draws the top 6 singular vectors and the corresponding colors.
    spec: SpectralDim object.
    lams: wavelengths.
    """
    fig = plt.figure(figsize=(12,8))
    for i in range(6):
        ax = plt.subplot2grid((3, 16), (i//2, i%2*8), colspan=5, rowspan=1)
        ax.set_title(str(i+1), fontsize = 14)
        ax.set_xlabel(r'$\lambda (nm)$', fontsize = 14)
        ax.set_ylim([-1, 1])
        if invert[i]:
             ax.plot(lams, -spec.basis[:,i], label = 'Base ' + str(i+1), color = 'black')
        else:
            ax.plot(lams, spec.basis[:,i], label = 'Base ' + str(i+1), color = 'black')
        ax.plot(np.arange(np.min(lams),np.max(lams)+1,10),np.zeros_like(lams), linestyle=':', color='black')
        ax.tick_params(labelsize=12)
        base = np.zeros((3,20,3))
        l, h = -15, 15
        #print('BASE '   + str(i+1))
        for j in np.linspace(l,h,50):
            color = np.clip(np.tile(np.reshape(spec.basis[:,i], (31, 1, 1)) * j, (1, 3, 20)), 0, np.inf)
            a = vis.draw_hpim(color, draw = False, lams = lams, method = '1931', normalize=False)
            a = np.clip(a, 0, 1)
            base = np.concatenate((base, a), axis=0)
        base = base[3:, :, :]
        ax = plt.subplot2grid((3, 16), (i//2, i%2*8+5), colspan=2, rowspan=1)
        if invert[i]:
            ax.imshow(np.flip(base, axis=0))
        else:
            ax.imshow(base)
        ax.set_ylabel('Projected Color of Base ' + str(i+1), fontsize = 12)
        plt.yticks([0, 74, 149], [str(l), '0', str(h)], fontsize = 12)
        ax.yaxis.tick_right()
        ax.tick_params(left = False , labelleft = False, bottom = False, labelbottom = False)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        '''
        if i == 5:
                l, h = -20, 20
        for j in np.linspace(l,h,50):
            color_ld = np.zeros((31,3,20))
            color_ld[i, :, :] = j
            color = spec.recover(color_ld)
            a = vis.draw_hpim(color, draw = False, lams = lams, method = 'Gaussians', normalize=False)
            a = np.clip(a, 0, np.inf)
            base = np.concatenate((base, a), axis=0)
        base = base[3:, :, :]
        ax = plt.subplot2grid((3, 16), (i//2, i%2*8+5), colspan=2, rowspan=1)
        if invert[i]:
            ax.imshow(np.flip(base, axis=0))
        else:
            ax.imshow(base)
        ax.set_ylabel('Projected Color of Base ' + str(i+1))
        plt.yticks([0, 74, 149], [str(l), '0', str(h)])
        ax.yaxis.tick_right()
        ax.tick_params(left = False , labelleft = False, bottom = False, labelbottom = False)
        plt.subplots_adjust(wspace=0.0, hspace=0.0  )
        '''
    fig.tight_layout(w_pad = 0.0)
    plt.show()

def analysisForPaper(hpim: np.ndarray, iter_c: int, spec:SpectralDim, colorizer, metrics: dict, enable_sf = False) -> dict:
    """
    Ideal dimensionality analysis.
    hpim: hyperspectral image of shape CxHxW.
    iter_c: number of iterations for the poisson noise.
    spec: SpectralDim object.
    colorizer: colorizer object.
    metrics: dictionary of metrics to use.
    enable_sf: True to enable the smart filter. (Eq. 4 in paper.)
    """
    output = dict()
    l,h,w = hpim.shape
    num_basis = spec.basis.shape[0]
    gray_img = np.sum(hpim, axis=0, keepdims=True)
    mask = get_uniform_sampling_pattern(h,w,0.03)
    iteration = np.power(2, np.arange(iter_c)) / (1024*96)
    print(iteration)
    output['exposure_time'] = iteration
    output['intrinsic_dim'] = np.zeros(len(iteration))
    for metric in metrics.keys():
        output[metric] = np.zeros((len(iteration),spec.basis.shape[0]))
    for i,iter in enumerate(tqdm(iteration)):
        hp_n = noise.addPoissonNoise(hpim, iter)
        #hp_n = noise.addGaussianNoise(hp_n, 0, 0.01*np.var(hpim))
        output['intrinsic_dim'][i] = spec.intrinsicDimension(hp_n * mask, 'two_features')
        for k,j in enumerate(range(1, num_basis+1)):
            hpim_ldim = spec.project(hp_n, j)
            visual_clues = mask * hpim_ldim
            colorizer.__init__(gray_img, visual_clues)
            if enable_sf:
                colorizer.smartFilter()
            result = colorizer.hyperColorize(sd=spec)
            for metric in metrics.keys():
                output[metric][i,k] = metrics[metric](hpim, result)
    return output

def dimensionalityFigures(analysis_output: dict, metric: str) -> None:
    '''
    analysis_output: output of analysisForPaper.
    metric: metric to use.
    higher_is_better: True if higher value of metric is better.
    gamma: Makes color scales easier to understand.
    '''
    figure = plt.figure(figsize = (10,5))
    lines_selected = [0,1,2,3,4,5,12]
    #FIGURE 1
    plt.title('Reconstruction Error Under Different Poisson Noise Levels')
    qv = analysis_output[metric]
    exposure_time = analysis_output['exposure_time']
    intrinsic_dim = analysis_output['intrinsic_dim']
    noise_count, dim_count = qv.shape
    colormap = plt.get_cmap('coolwarm')
    for i,j in enumerate(lines_selected):
        line, = plt.plot(np.arange(1,dim_count+1),qv[i,:], color=colormap((len(lines_selected)-j)/len(lines_selected)), linewidth=2)
        line.set_label(str(f'{exposure_time[j]:.2e}'))
    plt.xlabel('Number of Basis')
    plt.xscale('log', base=2)
    plt.ylabel(metric)
    plt.legend(title = "Exposure Time per Measurement", prop={'size': 10}, ncol=2)
    #plt.grid(visible=True, which='both')
    #FIGURE 2
    figure = plt.figure(figsize = (12,6))
    quality_metrics = ['EMD', 'SSV', 'PSNR', 'GFC']
    gamma = [0.001, 2.0, 1.0, 75.0]
    higher_is_better = [False, False, True, True]
    plt.suptitle('''Comparison Between The Actual Best Dimensionality and Our Estimation''')
    for i,metric in enumerate(quality_metrics):
        if metric == 'EMD':
            ax = plt.subplot2grid((3, 8), (0, 0), colspan=6, rowspan=3)
        if metric == 'SSV':
            ax = plt.subplot2grid((3, 8), (0, 6), colspan=2, rowspan=1)
        if metric == 'PSNR':
            ax = plt.subplot2grid((3, 8), (1, 6), colspan=2, rowspan=1)
        if metric == 'GFC':
            ax = plt.subplot2grid((3, 8), (2, 6), colspan=2, rowspan=1)
        qv = analysis_output[metric]
        X = np.power(2, np.log2(exposure_time) - 0.5)
        xlim_max = np.power(2,np.log2(exposure_time[-1]) + 0.5)
        xlim_min = np.power(2,np.log2(exposure_time[0]) - 0.5)
        X = np.append(X,xlim_max)
        Y = np.arange(0.5,32, 1)
        pmap = qv.T
        if higher_is_better[i]:
            best_location = np.argmax(qv, axis=1)
            best_case = np.max(qv, axis=1)
            colormap2 = plt.get_cmap('RdYlGn')
        else:
            best_location = np.argmin(qv, axis=1)
            best_case = np.min(qv, axis=1)
            colormap2 = plt.get_cmap('RdYlGn').reversed()
        if metric == 'EMD':
            plt.pcolor(X,Y, np.power(pmap,gamma[i]) , cmap = colormap2, shading = 'flat', edgecolors = 'black')
        else:
            plt.pcolor(X,Y, np.power(pmap,gamma[i]) , cmap = colormap2, shading = 'flat')
        cb = plt.colorbar()
        cb.set_label(metric)
        oldlabels = cb.ax.get_yticklabels()
        if metric == 'EMD':
            newlabels = map(lambda x: str(f'{np.power(float(x.get_text()), 1/gamma[i]):.3f}'), oldlabels)
        elif metric == 'PSNR':
            newlabels = map(lambda x: str(int(np.power(float(x.get_text()), 1/gamma[i]))), oldlabels)
        else:
            newlabels = map(lambda x: str(f'{np.power(float(x.get_text()), 1/gamma[i]):.2f}'), oldlabels)
        oldlabel_vals = map(lambda x: float(x.get_text()), oldlabels)
        cb.ax.set_yticks(list(oldlabel_vals)[1:-1])
        cb.ax.set_yticklabels(list(newlabels)[1:-1])
        plt.xscale('log', base=2)
        plt.xlim((xlim_min,xlim_max))
        x_tick_locations = np.power(10, np.arange(-5, -1.9, 1.0))
        plt.xticks(x_tick_locations, [r'$10^{a}$'.replace('a', str(int(np.log10(i)+3))) for i in x_tick_locations])   
        if metric == 'EMD':
            plt.plot(exposure_time, intrinsic_dim, marker='o', label = 'Predicted Dimensionality')
            plt.plot(exposure_time, best_location + 1, marker='o', label = 'Lowest Error Dimensionality', color = 'purple')
            plt.yticks(np.arange(1,33,2))
            plt.xlabel('Exposure Time (ms)')
            plt.ylabel('Number of Basis')
            plt.legend()
        else:
            plt.plot(exposure_time, intrinsic_dim, marker='o', label = 'Ideal Dimensionality Estimation', markersize = 3)
            plt.plot(exposure_time, best_location + 1, marker='o', label = 'Actual Best Dimensionality', color = 'purple', markersize = 3)
            plt.tick_params(right = False, labelbottom = False, bottom = False)
            plt.yticks([1,16,31])
            if metric == 'GFC':
                plt.xlabel('Exposure Time')
            
    plt.tight_layout()
    plt.savefig('results/graphs/dimensionality.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
