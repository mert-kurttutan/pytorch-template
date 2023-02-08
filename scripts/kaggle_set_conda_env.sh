# Create New Conda Environment and Use Conda Channel 
conda create -n newCondaEnvironment -c cctbx202208 -y
source /opt/conda/bin/activate newCondaEnvironment && conda install -c cctbx202208 python -y

/opt/conda/envs/newCondaEnvironment/bin/python3 --version

sudo rm /opt/conda/bin/python3
sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3

sudo rm /opt/conda/bin/python3.7
sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3.7

sudo rm /opt/conda/bin/python
sudo ln -s /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python

python --version