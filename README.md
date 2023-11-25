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
## Running the App
You can run the app by entering the repo through a shell and running the following commands
   ```sh
   conda activate GazeMouse
   python ./GazeMouse/src/app/dev_app.py
   ```

## Note on Conda Solver
Sometimes creating a large environment from an `environment.yml` file can take ludicoursly long. It is recommended to use the libmamba solver to speed up this process. Instructions on how to do so can be found (here)[https://www.anaconda.com/blog/conda-is-fast-now]. 

## Note on Installation for MacOS
There is some interesting and unexpected behavior with how pip interacts with MacOS versions which is documented (here)[https://stackoverflow.com/questions/65290242/pythons-platform-mac-ver-reports-incorrect-macos-version/65402241#65402241]. If you run into issues downloading the correct version of `opencv-contrib-python` then prepend `SYSTEM_VERSION_COMPAT=0` to the previously mentioned conda commans as such 
```sh
SYSTEM_VERSION_COMPAT=0 conda env create -f environment.yml
```