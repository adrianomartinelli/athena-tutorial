# ATHENA tutorial
ATHENA workflows presented at WIS.

# Install Package
```{bash}
# create a new virtual environment with Python 3.8
conda create -y -n athena python=3.8
conda activate athena

# install spatialOmics data container
pip install "git+https://github.com/histocartography/spatial-omics.git@master"

# install ATHENA package
pip install "git+https://github.com/histocartography/athena.git@master"

# install tutorial specific packages
pip install colorcet
pip install jupyter lab

```
