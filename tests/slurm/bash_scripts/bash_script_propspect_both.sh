#!/bin/sh
#SBATCH --job-name=detect_s1_s2_propspect_both
#SBATCH --array=0-479%10  #5000 Jobs, 8 max en parallèle
#SBATCH --mail-type=END             # Notification par email de la
#SBATCH --mail-user=nada.kired@outlook.fr   # fin de l'exécution du job.
#SBATCH --partition=48CPUNodes
#SBATCH --output=results/detect_s1_s2_propspect_both.out
#SBATCH --error=results/detect_s1_s2_propspect_both.err

/projets/sig/nkired/data-matching/tests/slurm/scripts/scripts_propspect_both/script_job$SLURM_ARRAY_TASK_ID.sh