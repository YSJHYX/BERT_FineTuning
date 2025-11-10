#!/bin/bash
#SBATCH --job-name=MLHA6         # create a short name for your job
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdtsuperpod      # only require for multiple projects

module purge                     # clear environment modules inherited from submission
# module load cuda/11.8
module load Anaconda3/2023.09-0  # load the exact modules required

# --- Initialize Conda ---
source $(conda info --base)/etc/profile.d/conda.sh

# --- Construct and Activate Conda enviroment ---
conda create -y -n ha6env python=3.10
conda activate ha6env

# --- Intsall PyTorch（GPU version）、Transformers、Datasets、scikit-learn ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install scikit-learn
pip install numpy
pip install pandas

# --- Task Dir ---
cd ~/jobs/ha6/

# --- Run Python sbatch ---
echo "Start running HA6Code.py ..."
python HA6Code.py "IMDB Dataset.csv"

echo "Task Completed!"