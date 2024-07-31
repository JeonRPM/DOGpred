<h1 align="center">
    DOGpred
    <br>
<h1>

<h4 align="center">Standalone program for the DOGpred paper</h4>

# Introduction
This repository provides the standalone program for DOGpred framework. The virtual environment, extracted features, and final models are available via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13141341.svg)](https://doi.org/10.5281/zenodo.13141341)

# Installation
## Software requirements
* Ubuntu 20.04.6 LTS (This source code has been already tested on Ubuntu 20.04.6 LTS with NVIDIA RTX A5000)
* CUDA 11.7 (with GPU suport)
* cuDNN 8.6.0.163 (with GPU support)
* Python 3



## Cloning this repository
```shell
git clone https://github.com/JeonRPM/DOGpred.git
```
```shell
cd DOGpred
```

## Creating virtual environment
* Please download the virtual environment (_**dogpred.tar.gz**_) via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13141341.svg)](https://doi.org/10.5281/zenodo.13141341)
* Please extract it into the [dogpred](https://github.com/JeonRPM/DOGpred/tree/main/dogpred) folder as below:
```
tar -xzf dogpred.tar.gz -C dogpred 
```
* Activate the virtual environment as below:
```
source dogpred/bin/activate
```

# Getting started
## Downloading all extracted features for the independent datasets
* Please download extracted features for the independent datasets via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13141341.svg)](https://doi.org/10.5281/zenodo.13141341)
* Please extract _**features.zip**_ file downloaded via Zenodo and put **CFDs** and **PLMs** folders into the [features/CFDs](https://github.com/JeonRPM/DOGpred/tree/main/features/CFDs) and [features/PLMs](https://github.com/JeonRPM/DOGpred/tree/main/features/PLMs) folders, respectively.


## Downloading all final models
* Please download all final models via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13141341.svg)](https://doi.org/10.5281/zenodo.13141341)
* Please extract _**final_models.zip**_ file downloaded via Zenodo and put all *.h5 files into the [final_models](https://github.com/JeonRPM/DOGpred/tree/main/final_models) folder.


## Running prediction
### Usage
```shell
CUDA_VISIBLE_DEVICES=<GPU_NUMBER> python predictor.py 
```
### Example for GPU
```shell
CUDA_VISIBLE_DEVICES=0 python predictor.py
```

### Example for CPU
```shell
CUDA_VISIBLE_DEVICES=-1 python predictor.py
```

## Extracting features (Optional)

**Note:** We provide snippets for extracting CFDs and PLM-based embeddings. Please modify the source code to extract all features.

### Extracting CFDS using _iFeatureOmega_<sup>[1]</sup>
```shell
python extract_cfds.py
```
```shell
python convert_cfds_to_tensors.py
```

### Extracting PLMs using _bio_embeddings_<sup>[2]</sup>
```shell
python extract_plms.py
```

# Citation
_**If you use this code or any part of it, as well as the refined datasets, please cite the following papers:**_
## Main
```
@article{pham2024leveraging,
  title={DOGpred: A Novel Deep Learning Framework for Accurate Identification of Human O-linked Threonine Glycosylation Sites},
  author={Lee, Ki Wook, and Pham, Nhat Truong and Min, Hye Jung and Park, Hyun Woo and Lee, Ji Won and Lo, Han-En and Kwon, Na Young and Seo, Jimin and Shaginyan, Illia and Cho, Heeje and Manavalan, Balachandran  and Jeon, Young-Jun},
  journal={},
  volume={},
  number={},
  pages={},
  year={2024},
  publisher={}
}
```

# References
[1] Chen, Z., Liu, X., Zhao, P., Li, C., Wang, Y., Li, F., Akutsu, T., Bain, C., Gasser, R.B., Li, J. & Song, J. (2022). iFeatureOmega: an integrative platform for engineering, visualization and analysis of features from molecular sequences, structural and ligand data sets. <i>Nucleic acids research</i>, 50(W1), W434-W447. <a href="https://doi.org/10.1093/nar/gkac351"><img src="https://zenodo.org/badge/doi/10.1093/nar/gkac351.svg" alt="DOI"></a> <br>
[2] Dallago, C., Schütze, K., Heinzinger, M., Olenyi, T., Littmann, M., Lu, A. X., Yang, K. K., Min, S., Yoon, S., Morton, J. T., & Rost, B. (2021). Learned embeddings from deep learning to visualize and predict protein sets. <i>Current Protocols</i>, 1, e113. <a href="https://doi.org/10.1002/cpz1.113"><img src="https://zenodo.org/badge/doi/10.1002/cpz1.113.svg" alt="DOI"></a>
