import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from aux_funcs.spectral_dimensionality import SpectralDim
import aux_funcs.visualization as  vis
import cv2


class GlobalColorizer:
    """ This class stores and implements the colorization algorithm. (Eq. 1-4) """
    def __init__(self, gray_img: np.ndarray, visual_clues: np.ndarray, lams: np.ndarray = None, draw_method:str = '1931') -> None:
        """
        gray_img: The grayscale image to be colorized. Should be in form 1xHxW.
        visual_clues: The spectral clues to be used for colorization. Should be in form CxHxW.
        lams: The wavelengths of the spectral clues. should have size C.
        draw_method: The method to be used for drawing the spectral clues and the result.
        """
        if lams is None:
            lams = np.arange(400, 701, 10)
        self.lams = lams
        self.draw_method = draw_method
        (self.channel_count, self.image_rows, self.image_cols) = visual_clues.shape
        self.gray_img = gray_img
        self.gray_img_f = self.gray_img / self.channel_count
        self.image_clues =  visual_clues
        self.map_colored = np.abs(np.sum(self.image_clues, axis=0)) > 0.0001
        try:
            self.image_clues_rgb = vis.draw_hpim(self.image_clues, draw=False, lams = self.lams, method = self.draw_method)
        except:
            clue_locations = np.abs(np.sum(self.image_clues, axis=0)) > 1e-8
            self.image_clues_rgb = np.tile(np.expand_dims(clue_locations, axis=0), (3,1,1))
        self.result = None
        self.B = None
        self.W = None
        self.calculate_weights = True
            
    def position_to_id(self, x:np.ndarray, y:np.ndarray, m:np.ndarray) -> np.ndarray:
        """
        Given a location in the image (x and), returns the corresponding index in the unrolled image.
        x: row index
        y: column index
        m: number of columns
        """
        return x * m + y
    
    def find_neighbour(self, x:int, y:int, n:int, m:int, d:int = 2) -> list[list[int, int]]:
        """
        Given a location in the image, returns the indices of the neighbouring pixels.
        x: row index
        y: column index
        n: number of rows
        m: number of columns
        d: the size of the neighbourhood
        """
        neighbour = []
        for i in range(max(0, x - d), min(n, x + d + 1)):
            for j in range(max(0, y - d), min(m, y + d + 1)):
                if (i != x) or (j != y):
                    neighbour.append([i, j])
        return neighbour
    
    def plot_inputs(self, figure_size: tuple[int, int] = (12, 6), brightness_boost:float=1.0, convolve:np.ndarray=None) -> None:
        """ Plots the grayscale image and the spectral clues. """
        figure = plt.figure(figsize=figure_size)
        figure.add_subplot(1, 2, 1).set_title('Black & White')
        plt.imshow(np.squeeze(self.gray_img_f), cmap='gray')
        plt.axis('off')
        figure.add_subplot(1, 2, 2).set_title('Color Hints')
        tbd = self.image_clues_rgb * brightness_boost
        if convolve is not None:
            for i in range(self.image_clues_rgb.shape[2]):
                tbd[:,:,i] = sp.signal.convolve2d(tbd[:,:,i], convolve, 'same')
        plt.imshow(tbd)
        plt.axis('off')
        plt.show()

    def plot_results(self) -> None:
        # Plots the result.
        result_rgb = vis.draw_hpim(self.result, draw=False, lams = self.lams, method = self.draw_method)
        plt.imshow(result_rgb)
        plt.title('Result of Colorization')
        plt.axis('off')
        plt.show()

    def plot_spectral_clue_result(self, figure_size:tuple[int,int]=(12, 6), show_colors:bool=True, brightness_boost:float=1.0, convolve:np.ndarray=None,save_figure:str=None) -> None:
        """
        Plots the spectral clues and the result.
        figure_size: The size of the figure.
        show_colors: Whether to show the spectral clues in color or in white.
        save_figure: The path to save the figure. If None, the figure will onnly be shown.
        """
        result = vis.draw_hpim(self.result, draw = False, lams = self.lams, method=self.draw_method)
        if show_colors:
            spectral_clue = vis.draw_hpim(self.image_clues, lams = self.lams, method=self.draw_method)
        else:
            spectral_clue = (np.sum(self.image_clues, axis=0) != 0).astype(np.float32)
        if convolve is not None:
            if len(spectral_clue.shape) == 2:
                spectral_clue = sp.signal.convolve2d(spectral_clue, convolve, 'same')
            else:
                for i in range(spectral_clue.shape[2]):
                    spectral_clue[:,:,i] = sp.signal.convolve2d(spectral_clue[:,:,i], convolve, 'same')
        spectral_clue = np.clip(spectral_clue * brightness_boost, 0, 1)
        figure = plt.figure(figsize=figure_size)
        figure.add_subplot(1,2,1).set_title('Clues')
        if show_colors:
            plt.imshow(spectral_clue)
        else:
            plt.imshow(spectral_clue, cmap='gray')
        plt.axis('off')
        figure.add_subplot(1,2,2).set_title('Result')
        plt.imshow(result)
        plt.axis('off')
        if save_figure is not None:
            plt.savefig(save_figure, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
        

    def get_result(self,) -> np.ndarray:
        """ Returns the result. """
        return self.result
    
    def smartFilter(self, draw_canny = False):
        """ 
        Applies the smart filter described by Eq. 4 in the paper.
        draw_canny: Visualization of Zeta in Eq. 4. 
        """
        edges = cv2.Canny((255*cv2.GaussianBlur(np.squeeze(self.gray_img), (31, 31),11,11)).astype(np.uint8), 600, 600)
        blurred_edges = cv2.blur(edges, (21,21))
        map = np.clip(edges/(np.max(edges)*2) + blurred_edges/(0.8*np.max(blurred_edges)), 0, 1)
        if draw_canny:
            plt.imshow(map, cmap = 'gray')
        l,h,w = self.image_clues.shape
        for i in range(h):
            for j in range(w):
                if sum(self.image_clues[:,i,j]) == 0:
                    continue
                else:
                    spectral_clue_locations = np.sum(self.image_clues[:,i-10:i+11, j-10:j+11], axis = 0) != 0
                    alfa = map[i,j] * self.image_clues[:,i,j]
                    beta = (1-map[i,j]) * np.sum(self.image_clues[:,i-10:i+11, j-10:j+11], axis = (1,2)) / (np.sum(spectral_clue_locations) + 10e-6)
                    self.image_clues[:,i,j] = alfa + beta
        rows, cols = self.map_colored.nonzero()
        ids = self.position_to_id(rows, cols, self.image_cols)
        self.B = np.zeros((self.channel_count, self.image_rows * self.image_cols))
        self.B[:, ids] = self.image_clues[:,rows, cols]
    
    def hyperColorize(self, sd:SpectralDim = None, draw_result: bool = False, save_result:str = None):
        """
        Applies the HyperColorization algorithm.
        sd: The spectral basis to use. Required for propagating colors in a lower dimension.
        draw_result: Whether to draw the result.
        save_result: The path to save the result. If None, the result will not be saved.
        """
        self.colorize()
        self.shade(spec=sd)
        if draw_result:
            self.plot_spectral_clue_result(show_colors=(sd is None))
        if save_result is not None:
            self.plot_spectral_clue_result(show_colors=(sd is None), save_figure=save_result)
        return self.get_result()
    
    def shade(self, spec:SpectralDim = None):
        """
        Rebalances luminance information in the HSI based on the gray image. (Eq. 3 in the paper)
        spec: The spectral basis to use. Required if propagating colors in a lower dimension.
        """
        if spec is None:
            sum_img = np.sum(self.result, axis=0, keepdims=True)
            scale_img = np.divide(self.gray_img, sum_img + 1e-10)
            self.result = self.result * scale_img
        else:     
            (_, h, w) = self.result.shape
            l = spec.basis.shape[0]
            new_result = np.zeros((l, h, w))
            current_img = spec.recover(self.result)
            sum_img = np.sum(current_img, axis=0, keepdims=True)
            scale_img = np.divide(self.gray_img, sum_img + 1e-10)
            new_result = np.clip(current_img * scale_img, 0, 1)
            self.result = new_result

    # def colorize(self) -> np.ndarray:
    #     """
    #     Applies the colorization algorithm given by Eq. 1 and 2 in the paper.
    #     Nested for loops implement eqauation 2, it calculates the weight term using the gray image.
    #     Minimization of equation 1 is achieved by solving a linear system of equations.
    #     """
    #     size = self.image_rows * self.image_cols
    #     if self.calculate_weights:
    #         self.W = sp.sparse.lil_matrix((size, size), dtype = float)
    #         self.B = np.zeros((self.channel_count, size))
    #         for i in range(self.image_rows):
    #             for j in range(self.image_cols):
    #                 if self.map_colored[i,j]:
    #                     id = self.position_to_id(i, j, self.image_cols)
    #                     self.W[id, id] = 1
    #                     self.B[:,id] = self.image_clues[:,i, j]
    #                 Y = self.gray_img[0, i, j]
    #                 id = self.position_to_id(i, j, self.image_cols)
    #                 neighbour = np.asarray(self.find_neighbour(i, j, self.image_rows, self.image_cols, 1), dtype= np.int32).T
    #                 Ys = self.gray_img[0, neighbour[0,:], neighbour[1,:]]
    #                 ids = self.position_to_id(neighbour[0,:], neighbour[1,:], self.image_cols)
    #                 sigma = np.std(np.append(Ys, Y))
    #                 #sigma = np.std(Ys)
    #                 if sigma > 1e-3:
    #                     weights = np.exp(-1 * (Ys - Y) * (Ys - Y) / (2 * sigma * sigma))
    #                 else:
    #                     weights = np.ones_like(Ys)
    #                 sum = np.sum(weights)
    #                 weights = weights / sum
    #                 self.W[id, ids] = -1 * weights
    #                 self.W[id, id] += 1 
    #         self.calculate_weights = False
    #     result = np.zeros(shape = (self.channel_count, self.image_rows, self.image_cols))
    #     self.W = self.W.tocsc()
    #     unrolled = sp.sparse.linalg.spsolve(self.W, self.B.T)
    #     self.W = self.W.tolil()
    #     result = np.reshape(unrolled.T, (self.channel_count, self.image_rows, self.image_cols))
    #     self.result = result
    #     return result
    
    def colorize(self) -> np.ndarray:
        """
        Applies the colorization algorithm given by Eq. 1 and 2 in the paper.
        This is the vectorized implementation by Yi-Chun Hung in Northwestern University.
        The colorize function commented out is non-vetorized and slower, however, the code
        structure is easier to follow. 
        Minimization of equation 1 is achieved by solving a linear system of equations.
        """
        window_size = 3 #which is the size-1 neighbor
        size = self.image_rows * self.image_cols
        if self.calculate_weights:
            self.W = sp.sparse.eye(size, dtype=float).tolil()
            self.B = np.zeros((self.channel_count, size))
            self.map_colored = np.abs(np.sum(self.image_clues, axis=0)) > 0.0001
            colored_idx, colored_idy = np.where(self.map_colored==True)
            colored_id = self.position_to_id(colored_idx, colored_idy, self.image_cols)
            self.W[colored_id, colored_id] = self.W[colored_id, colored_id]+np.ones(len(colored_id))
            self.B[:, colored_id] = self.image_clues[:,colored_idx, colored_idy]
            
            neighbor_counter = np.ones((window_size, window_size))

            all_one = np.ones((self.image_rows, self.image_cols))
            neighbor_count_map = sp.signal.convolve2d(all_one, neighbor_counter, 'same')
            Y = self.gray_img[0,...]
            rolling_mean = sp.signal.convolve2d(Y, neighbor_counter, 'same')
            rolling_mean = rolling_mean / neighbor_count_map
            rolling_sqr_mean = sp.signal.convolve2d(Y**2, neighbor_counter, 'same')
            rolling_sqr_mean = rolling_sqr_mean / neighbor_count_map
            sigma = np.sqrt(rolling_sqr_mean - rolling_mean**2)
            
            filter_bank = np.eye(window_size*window_size)
            filter_num = window_size**2
            filter_bank[:,(filter_num-1)//2] = -1
            filter_bank = filter_bank.reshape(filter_num, window_size, window_size)
            
            filter_bank = np.delete(filter_bank, (filter_num-1)//2, 0)
            weights = []
            for f in filter_bank:
                residual = sp.signal.convolve2d(Y, f[::-1,::-1], "same", fillvalue=10**100)
                prob_resi = np.exp(-residual**2/(2*sigma**2))
                prob_resi[sigma<1e-3] = 1
                weights.append(prob_resi)
            weights = np.array(weights)
            sum = np.sum(weights, axis=0)
            weights = weights / sum
            for f, w in zip(filter_bank, weights):
                offset_x, offset_y = np.where(f==1)
                offset_x -= (window_size - 1)//2
                offset_y -= (window_size - 1)//2
                ids_x, ids_y = np.where(w!=0)
                id = self.position_to_id(ids_x, ids_y, self.image_cols)
                # calculate the ids
                ids_x = ids_x + offset_x
                ids_y = ids_y + offset_y
                ids = self.position_to_id(ids_x, ids_y, self.image_cols)
                self.W[id, ids] = -1* w[w!=0]
            
        self.W = self.W.tocsc()
        unrolled = sp.sparse.linalg.spsolve(self.W, self.B.T)
        self.W = self.W.tolil()
        result = np.reshape(unrolled.T, (self.channel_count, self.image_rows, self.image_cols))
        self.result = result
        return result
