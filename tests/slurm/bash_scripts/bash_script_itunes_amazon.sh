#!/bin/sh
#SBATCH --job-name=detect_s1_s2_itunes_amazon
#SBATCH --array=0-479%10 #5000 Jobs, 8 max en parallèle
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=nada.kired@outlook.fr   # fin de l'exécution du job.
#SBATCH --partition=48CPUNodes
#SBATCH --output=results/detect_s1_s2_itunes_amazon.out
#SBATCH --error=results/detect_s1_s2_itunes_amazon.err

/projets/sig/nkired/data-matching/tests/slurm/scripts/scripts_itunes_amazon/script_job$SLURM_ARRAY_TASK_ID.sh