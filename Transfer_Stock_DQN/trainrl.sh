#!/bin/bash
#SBATCH --job-name=rl_stock
#SBATCH --time=05:00:00
#SBATCH --account=free -p short
#SBATCH --gres=gpu:0
#SBATCH --output=/insomnia001/home/ik2592/slurm_outs/llada4.4_long_%j.out
#SBATCH --error=/insomnia001/home/ik2592/slurm_outs/llada4.4_long_%j.err

# Load conda
source ~/.bashrc
conda activate py39


# nvidia-smi
# # Run the generation script
# echo "genlen 256 steps 80 blocklen 16"
# python llada.py --genlen 256 --steps 80 --blocklen 16

# echo "genlen 256 steps 160 blocklen 16"
# python llada.py --genlen 256 --steps 160 --blocklen 16

# echo "genlen 256 steps 320 blocklen 16"
# python llada.py --genlen 256 --steps 320 --blocklen 16

# echo "genlen 256 steps 640 blocklen 16"
# python llada.py --genlen 256 --steps 320 --blocklen 16


# echo "STEPS 128 Conf 0.9"
# python speculative.py --steps 128 --conf 0.9
# echo "STEPS 64 Conf 0.9"
# python speculative.py --steps 64 --conf 0.9
# echo "STEPS 32 Conf 0.9"
# python speculative.py --steps 32 --conf 0.9

# echo "STEPS 128 Conf 0.7"
# python speculative.py --steps 128 --conf 0.7
# echo "STEPS 64 Conf 0.7"
# python speculative.py --steps 64 --conf 0.7
# echo "STEPS 32 Conf 0.7"
# python speculative.py --steps 32 --conf 0.7


python main.py --train_stock ./data/AXP_market_data.csv --val_stock ./data/AAPL_market_data.csv
