# BART Morphological
## Training a BART model from scratch using PyTorch
Refer to the original [BART](https://arxiv.org/abs/1910.13461) paper.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Installation and Setup
### Git Clone
Git clone the repository by running the following commands:
```bash
# change directory to Documents
cd Documents
git clone https://github.com/TahaBinhuraib/BART_morphological.git
cd BART_morphological
```
### Anaconda
You need to install [Anaconda](https://www.anaconda.com/products/distribution), then run below:
```bash
conda create --name bart python=3.9
conda activate bart
pip install -r requirements.txt
```
## Running the Python File :snake:
``` bash
python main.py --epochs 64
```
You can take a look at the flags for any other hyperparameters you would like to experiment with.



## Contributing
Please feel free to contribute. Pull requests are welcome. My goal is to manage a repositor that is comprehensive in nature, and easy to understand for beginners. 