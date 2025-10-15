#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=scope_genalpha2 ##faith_compactusinganswer #usingsubquery ##compactedusinganswer #usingquery #appendinguserqueryevalfaith ##faithmixed##selfrag2bysentproba#2mixed #segmentationMixedCum # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds
#SBATCH --gpus=1 # nombre de gpu
#SBATCH --mem-per-gpu=80G
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=48:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate scope
#conda activate alignscore
cd /home/djeddal/Documents/Code/scope-decoding

#python src/scope/launcher.py main_model.model_path="meta-llama/Llama-3.1-8B-Instruct" noise_model.model_path="meta-llama/Llama-3.1-8B-Instruct" generation.mixture_alpha=0.3 data.dataset_path="hanane/attributionbenchmark-scopeaugmented" out_path=fulltrainalpa3
python src/scope/launcher.py main_model.model_path="meta-llama/Llama-3.1-8B-Instruct" noise_model.model_path="meta-llama/Llama-3.1-8B-Instruct" generation.mixture_alpha=0.3 data.dataset_path="hanane/attributionbenchmark-scopeaugmented" out_path=fulltrainalpa2