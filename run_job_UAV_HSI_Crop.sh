#!/bin/bash

#SBATCH --job-name=TransUNet                 # Job name
#SBATCH --output=%j-[STDOUT].txt             # Output log file (%j will be replaced by the job ID)
#SBATCH --error=%j-[STDERR].txt              # Error log file
#SBATCH --ntasks=1                           # Number of tasks (typically 1 for a single Python script)
#SBATCH --cpus-per-task=20                   # Number of CPU cores per task
#SBATCH --gres=gpu:1                         # Request 1 GPU (adjust based on the available resources)
#SBATCH --mem=40G                            # Memory allocation
#SBATCH --time=12:00:00                      # Maximum run time (HH:MM:SS)
#SBATCH --partition=bigTiger                 # Partition to submit to (adjust based on availability)

# ########################################
# cd /home/mhssain9/TransUNet/
# conda activate trans_u_env
# git reset --hard
# git pull
# chmod +x run_job_UAV_HSI_Crop.sh
# sbatch run_job_UAV_HSI_Crop.sh
#
# cd /home/mhssain9/TransUNet/ && conda activate trans_u_env && git reset --hard && git pull && chmod +x run_job_UAV_HSI_Crop.sh && sbatch run_job_UAV_HSI_Crop.sh
# squeue -p bigTiger -u $USER -O jobid,partition,username,state,reasonlist,gres

# sacct -o JobId%20,MaxVMSize,MaxRSS,NCPUS,ReqTRES%25,ReqMem --units=G -j [jobid]

# scancel [jobid]
# scancel -n TransUNet
# sinfo -a
# sprio -j [jobid]

# scontrol show partition
# scontrol show job [jobid]
# scontrol update job [jobid] VARIABLE=value
# scontrol update job 211385 gres=gpu:N

# https://unix.stackexchange.com/a/646046/405424
# du -h -d 1  /home/mhssain9/
# du -h -d 1  /project/mhssain9/
# ########################################
DIR_NAME=/project/mhssain9
MODEL_NAME=ViT-B_16
DATASET=UAV_HSI_Crop

CHECKPOINT_DIR=${DIR_NAME}/model/vit_checkpoint/imagenet21k
# CHECKPOINT_DIR=${DIR_NAME}/model/vit_checkpoint/imagenet21k+imagenet2012
SNAPSHOT_DIR="/project/mhssain9/Experiment_02/exp_09"
# rm -rf $SNAPSHOT_DIR

# Download the pre-trained checkpoint.
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  mkdir -p ${CHECKPOINT_DIR}
fi

if [[ ! -f ${CHECKPOINT_DIR}/${MODEL_NAME}.npz ]]; then
  wget "https://storage.googleapis.com/vit_models/imagenet21k/${MODEL_NAME}.npz"
  # wget "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/${MODEL_NAME}.npz"
  
  mv ${MODEL_NAME}.npz ${CHECKPOINT_DIR}
fi

# nvcc -V
nvidia-smi

# # Activate Anaconda work environment
# source /home/${USER}/.bashrc
# conda activate trans_u_env

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  mkdir -p ${SNAPSHOT_DIR}
fi

echo "########################################################################"
printf "#\n# Test with imagenet21k+imagenet2012 loss transformer 3  \n#\n"
echo "########################################################################"
echo "To restart the same experiment delete the SNAPSHOT_DIR:" 
echo "rm -rf '$SNAPSHOT_DIR'" 
echo "########################################################################"
echo "To check the experiment results:" 
echo "tensorboard --logdir='$SNAPSHOT_DIR' --port=65535"
echo "ssh -N -L 65535:localhost:65535 mhssain9@itiger.memphis.edu"
echo "########################################################################"

# Run the classification task using the dataset and subset variables
python train.py --dataset ${DATASET}  --vit_name ${MODEL_NAME} --batch_size 16 --base_lr 0.01 --img_size 96 --snapshot_dir $SNAPSHOT_DIR --max_epochs 3000

# Evaluate the trained model
python test.py --dataset ${DATASET} --vit_name ${MODEL_NAME} --batch_size 16 --base_lr 0.01 --img_size 96 --snapshot_dir $SNAPSHOT_DIR --max_epochs 3000 --is_savenii
