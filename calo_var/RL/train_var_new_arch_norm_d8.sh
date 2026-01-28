#!/bin/bash
#SBATCH --job-name=VAR_new_arch_norm_d8
#SBATCH --output=slurm_var_new_arch_norm_d8_%j.log
#SBATCH --error=slurm_var_new_arch_norm_d8_%j.err
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH -A plgdyplomancipw2-gpu-a100
#SBATCH --signal=SIGTERM@30

# ============================================
# VAR Training: new_arch_norm, depth=8, shared AdaLN
# ============================================

module load CUDA
module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgfw01169396/.conda/envs/condaVAR2

cd /net/people/plgrid/plgfw01169396/caloVQVAR

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$((29500 + RANDOM % 1000))

# VQ Model paths
TRAINED_MODELS="/net/people/plgrid/plgfw01169396/calo-VQ-expanded/trained_models"

echo "============================================"
echo "VAR Training: new_arch_norm depth=8"
echo "============================================"

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:${MASTER_PORT} \
    --nnodes=1 \
    --nproc-per-node=4 \
    train.py \
    --exp_name "VAR_new_arch_norm_d8_saln_stable" \
    --vqmodel_config "${TRAINED_MODELS}/new_arch_norm/configs/ds3_norm_step1_multiscale_ema_new_arch_hlf_bs256.yaml" \
    --vqmodel_ckpt "${TRAINED_MODELS}/new_arch_norm/checkpoints/epoch=001423.ckpt" \
    --depth 8 \
    --pn "1_2_3_4_5_6_9_13_18_26" \
    --saln False \
    --tfast 0 \
    --fuse=True \
    --bs 256 \
    --ep 300 \
    --predict_R True \
    --R_regression False \
    --R_scale0 True
