# DRE-SIM
we raised a postprocessing method to further improve the lateral resolution of SIM with the help of an untrained neural network. Taking the diffraction limit and SIM reconstruction into consideration, the method can further extend spatial frequency components of SIM images by exploring the implicit prior with the network. 


# Install

list of libraries:
- python=3.7
- jupyter
- nb_conda
- numpy
- pyyaml
- mkl
- setuptools
- cmake
- cffi
- matplotlib
- scikit-image
- pytorch=1.8.0
- torchvision=0.9.0
- torchaudio=0.8.0
- cudatoolkit=10.2
- pillow=6.2.1
- opencv-python

You can create an conda env with all dependencies via environment file

```
conda env create -f DRE_env.yml
```

## Test image
