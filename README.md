# cmu_challenge
Muon regression challenge

To set up the notebook kernel login to 

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
mkdir -p ~/.local/share/jupyter/kernels/
cp -r /ocean/projects/cis210053p/shared/common/python3-AI ~/.local/share/jupyter/kernels/
```
