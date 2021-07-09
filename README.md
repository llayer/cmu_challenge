# cmu_challenge

## Machine learning for muon energy reconstruction in a high-granularity calorimeter


## Step I: Install custom python packages

Login to
```
ssh <user_name>@bridges2.psc.edu
```
and execute:
```
module load AI
pip install --user uproot
pip install --user awkward
pip install --user xgboost
pip install --user sparse
pip install --user --upgrade torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Step II: Create a custom kernel for your jupyter

```
mkdir -p ~/.local/share/jupyter/kernels/
cp -r /ocean/projects/cis210053p/shared/common/python3-AI ~/.local/share/jupyter/kernels/
```

### Step III: Open the notebook in OnDemand

To open a live notebook from the OnDemand interface, follow these steps:
- log in on https://ondemand.bridges2.psc.edu/
- click on "Jupyter Notebook" in the `Interactive Apps` menu in the top bar
- You can use the default settings for a standard CPU node, for GPU, select the `GPU-shared` partition, and use 
this in the extra ARgs: `--gpus=v100-16:1`
- Once your Jupyter instance is allocated, open it, and you should be able to create a new notebook, with kernel "Python 3 - AI"
