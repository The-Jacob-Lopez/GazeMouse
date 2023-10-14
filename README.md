# GazeMouse: An accessible and mouse replacement

### Prerequisites
For dependency management we use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Go ahead and install it if you don't have it already.

### Installation

## Getting Started
1. Clone the repository 
   ```sh
   git clone https://github.com/The-Jacob-Lopez/GazeMouse.git
   ```
2. Create a new conda environment and install the required dependencies
   ```sh
   conda env create -f environment.yml
   ```
   Note that if you intended on using GPU, you will also have to install the `pytorch-cuda` package. This can be done by create the environment using the appropriate environment file
   ```sh
   conda env create -f environment-cuda.yml
   ```