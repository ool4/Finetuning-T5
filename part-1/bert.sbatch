#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append # Append to the output file. If you don't use this, the output will be overwritten.
#SBATCH --output=./%j_%x.out # %j is the job id, %x is the job name.
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL # Export all the environment variables to the job.
#SBATCH --time=00:10:00 
#SBATCH --gres=gpu:1 # Request 1 GPU.
#SBATCH --requeue # Requeue the job if it fails.


singularity exec --bind /scratch --nv --overlay  /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/$USER/ubuntu-20.04.3.sif /bin/bash -c " conda activate nlp_env; cd /scratch/$USER/; python main.py --train --eval --debug_train; "

