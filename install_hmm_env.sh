#creating conda environment
conda create --name hmm_env_shared --file hmm_requirements.txt


conda activate hmm_env_shared

#initializing ipykernel for jupyter notebook to connect to
python -m ipykernel install --user --name=hmm_env_shared

#building hmmlearn from source

cd ./.hmmlearn
python3 setup.py
cd ../



