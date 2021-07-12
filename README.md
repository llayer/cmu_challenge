# CMU Challenge: Machine learning for muon energy reconstruction in a high-granularity calorimeter

## Setup

Setup the environment on CMU bridges

### Step I: Install custom python packages

Login to
```
ssh <user_name>@bridges2.psc.edu
```
and execute:
```
module load AI
pip install --user uproot==4.0.4 awkward==1.4.0 xgboost==1.4.2 sparse==0.11.2 fastprogress==0.1.21
pip install --user --upgrade torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step II: Create a custom kernel for your jupyter

```
mkdir -p ~/.local/share/jupyter/kernels/
cp -r /ocean/projects/cis210053p/shared/common/python3-AI ~/.local/share/jupyter/kernels/
```

### Step III: Open the notebook in OnDemand

To open a live notebook from the OnDemand interface, follow these steps:
- log in on https://ondemand.bridges2.psc.edu/
- click on "Jupyter Notebook" in the `Interactive Apps` menu in the top bar
- Set the number of hours to ~24 so you can keep the session running during the night if you want to continue training
- You can use the default settings for a standard CPU node, for GPU, select the `GPU-shared` partition, and use 
this in the extra ARgs: `--gpus=v100-16:1`
- Once your Jupyter instance is allocated, open it, and you should be able to create a new notebook, with kernel "Python 3 - AI"
- If you open a notebook from a repository you eventually have to click on 'Kernel' and then 'Change Kernel' to change to "Python 3 - AI"

### Step IV: Copy the data and clone the repository

From a bridges node (login via ssh) you can copy the data via:
```
cp -r /ocean/projects/cis210053p/shared/muon_reg .
```
And clone the repository:
```
git clone https://github.com/llayer/cmu_challenge.git
```

## Imporvement ideas
 
 - Combine HL features with the CNN output
 - Change the loss function to better focus on lower energy
 - Apply a bias correction to the regressor predictions to reduce residual bias at high energy
 - Improve CNN architecture
 - Construct new HL features from raw data
 - Improve CNN training
 - Ensemble different models


## Data citation

If reused, the data should be citated as:
```
@misc{kieseler2021calorimetric,
      title={Calorimetric Measurement of Multi-TeV Muons via Deep Regression}, 
      author={Jan Kieseler and Giles C. Strong and Filippo Chiandotto and Tommaso Dorigo and Lukas Layer},
      year={2021},
      eprint={2107.02119},
      archivePrefix={arXiv},
      primaryClass={physics.ins-det}
}
```

We'll soon be releasing the fulldatasets on Zenodo, anyway, at which point they will have their own DOI and citation.
