<h1> HyperColorization: Propagating spatially sparse noisy spectral clues over hyperspectral images </h1><br>
<b>Abstract</b>: Hyperspectral cameras face challenging spatial-spectral resolution trade-offs and are more affected by shot noise than RGB photos taken over the same total exposure time. Here, we present a colorization algorithm to reconstruct hyperspectral images from a grayscale guide image and spatially sparse spectral clues. We demonstrate that our algorithm generalizes to lower dimensional models for hyperspectral images, and show that colorizing in a low-rank space reduces compute time and the impact of shot noise. To enhance robustness, we incorporate guided sampling, edge-aware filtering, and dimensionality estimation techniques. Our method surpasses previous algorithms in various performance metrics, including SSIM, PSNR, GFC, and EMD, which we propose as valuable metrics for characterizing hyperspectral image quality. Collectively, these findings provide a promising avenue for overcoming the time-space-wavelength resolution trade-off by reconstructing a dense hyperspectral image from samples obtained by whisk or push broom scanners, as well as computational imaging systems. 
<p align="center">
  <img src="ui/F1.png">
</p>
<h2>Installation </h2>
! You can also use <dtrong>Google Collab</strong> to check out our results without any installation! <br>
  Demo:<a href="https://colab.research.google.com/github/NUBIVlab/HyperColorization/blob/master/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br>
  Demo for figures: <a href="https://colab.research.google.com/github/NUBIVlab/HyperColorization/blob/master/demo_for_figures.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br>
  
* Clone this repository to your local machine. Example local path: ../dev/HyperColorization/
* Install [Anaconda]( https://www.anaconda.com/):
* Run Anaconda Prompt as Administrator.
* Navigate to the directory where you cloned this repository with "cd ../dev/HyperColorization/".
* Run: "conda create --name HyperColorization --file requirements.txt".
* Run "conda activate HyperColorization".<br>
* Open demo.ipynb from your favorite IDE that supports Jupiter notebooks.<br>
* Select HyperColorization as the interpreter. You should be able to work on the project now!<br>

<h2> FAQ </h2>

* I am trying to use the KAIST hyperspectral dataset. OpenEXR won't run.
  > You need the binary files of OpenEXR for the Python library to work. If you are on Linux or Mac, follow this [guide](https://openexr.com/en/latest/install.html). On Windows, compiling the binary is more complicated. However, we can still download the precompiled binary. Open Anaconda Prompt as Administrator, and activate your environment. First run: "pip install pipwin". After a successful install, run: "pipwin install openexr".

* Why do I get blue errors close to the edges when colorizing the Bear & Fruit Image?
  > Unfortunately, it's a problem with the data. If you are too annoyed, the best fix is to remove a couple of columns from the left and a couple of rows from the bottom after importing.

* Can I run this on my pre-existing environment?
  > Probably yes, but you might have version conflicts. Other than numpy, scipy, matplotlib, scikit-image opencv, and tqdm (which are probably already installed in most image processing-related environments), you will need kneed and openexr (openexr is optional, you can comment it out, but you won't be able to open the Kaist dataset).

* How do I learn a spectral basis for my dataset?
  > If you have a big dataset, you need to have a lot of RAM or downsample your images. Unroll your dataset to have shape CxN where C are spectral vectors and N is the number of pixels. Then, call LearnSpectralBasisDB function of the SpectralDim class. You can create unrolled versions of existing datasets by calling load_CAVE_db or load_Harvard_db inside the importer class.
  
