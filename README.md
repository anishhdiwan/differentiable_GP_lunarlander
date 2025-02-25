# Lunar Lander w/ GP

Evolving a lunar lander with differentiable genetic programming as part of CS4205 Evolutionary Algorithms at TU Delft

## Installation
Since this project requires older versions of some packages, it is recommended to install them in a virtual environment. You could either create a native python venv using the `requirement.txt` file in this repo (untested) or create a conda environment with the provided `environment.yml` file (tested). To do this, follow these steps.

1. Assuming you have conda installed. Download the environment.yml file 
2. The older versions of some packages are not available on the common conda channels. Moreover, different systems/hardware usually need different torch variants. These will be installed via pip in the next step. For now, run `conda env create -f environment.yml`
3. Activate the new env using `conda activate EA_env` and install the pending packages
4. `pip install pygame=2.1.0` | `pip install pyglet==1.5.21`
5. Find out the correct torch version for your system and GPU variant (cuda version). Install it from the [Torch Website](https://pytorch.org/)
6. Install git via `conda install -c anaconda git` and clone this repo. Enable the python kernel for this environment `python -m ipykernel install --user --name=python3` and start the notebook via `jupyter notebook` 
7. You can now start playing around with our solution
