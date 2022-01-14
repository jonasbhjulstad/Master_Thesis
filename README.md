# TTK4900 Master Thesis (FIPOPT)


## Prerequisites (Ubuntu)
Install python-dev
```bash
sudo apt-get install python-dev
```

Install pybind11, CUTEst and python binders

```bash
git clone https://github.com/pybind/pybind11.git
git clone https://github.com/ralna/CUTEst.git
```
(Follow provided installation instructions)
## Docker Installation
```bash
Docker pull jonashj/ttk4900-master-thesis-fipopt:latest
```
Initialize container with shell attached:
```bash
docker image list
(find image ID) ...

docker run -ti <image> /bin/bash
```
## Path Configuration
Using a replace all-tool it is safe to rename "/home/build/FIPOPT" to chosen PROJECT_ROOT

## Usage

```bash
@PROJECT_ROOT$ mkdir Release
@PROJECT_ROOT$ cd Release
@PROJECT_ROOT$ cmake .. -DSIF_PROBLEM=<SIF-Problem> -DCMAKE_BUILD_TYPE=Release
```
Some targets may be disabled in CMakeLists.txt, which needs manual uncommenting in the executables directory's CMakeLists.txt.

Automated execution of SIF-files are available by executing PROJECT_ROOT/Plot/SIF/Run_HS.sh from inside the directory. 

## Results
Results are available under PROJECT_ROOT/Data/{QP,NC_QP, NE_QP} for QPs, and PROJECT_ROOT/Data/SIF/HS/*.SIF/ for SIF-problems. 

## Plotting
Plotting tools are available for QP/SIF under PROJECT_ROOT/Plot/. 

Objective_Plot, Multiplier_Plot and Trajectory_Plot (.py) provides convergence illustrations.






