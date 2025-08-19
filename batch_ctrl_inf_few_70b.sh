#!/usr/bin/zsh

############################################################
### Slurm flags
############################################################

#SBATCH --partition=c23g            # request partition with GPU nodes
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --ntasks-per-node=1         # request desired number of processes (or MPI tasks)

#SBATCH --cpus-per-task=24          # request desired number of CPU cores or threads per process (default: 1)
                                    # Note: available main memory is also scaling with
                                    #       number of cores if not specified otherwise
                                    # Note: On CLAIX-2023 each GPU can be used with 24 cores

#SBATCH --gres=gpu:1                # specify desired number of GPUs per node
#SBATCH --time=12:00:00             # max. run time of the job
#SBATCH --mem=64G
#SBATCH --job-name=a-zero-7b-job     # set the job name
#SBATCH --output=./atbfew7bscript.log      # redirects stdout and stderr to stdout.txt
#SBATCH --error=./atbfew7bscript.err
#SBATCH --account=thes2020          # insert your project-id or delete this line


############################################################
### Parameters and Settings
############################################################

# print some information about current system
echo "Job nodes: ${SLURM_JOB_NODELIST}"
echo "Current machine: $(hostname)"
nvidia-smi

############################################################
### Execution / Commands
############################################################

# Optional: Load desired models for GPU such as CUDA
# Load required modules
module purge                              # Unload all currently loaded modules
module load intel/2024a
module load GCCcore/14.2.0                # Load the base GCC core (compiler infrastructure)
module load GCC/14.2.0                    # Load full GCC compiler (needed for C/C++/CUDA code)
module load CUDA/12.3.0                   # Load CUDA toolkit (you need this for GPU offloading)
module load SQLite/3.47.2                 # Required by some Python dependencies (e.g., llama-cpp)


# Example: 1:2 mapping between MPI processes and GPUs
#          Process intened to use both GPUs. If your code is based on CUDA,
#          you might internally need to use cudaSetDevice to target the individual GPUs.
#cd /rwthfs/rz/cluster/home/wz516401/jupyterlab/ ########
#source myvenv2/bin/activate  #venv/bin/activate
source $HOME/myvenv2/bin/activate

#srun python3 text_to_persona_generation_7B.py
#finish infzero7b
#srun python3 Rong_inferred_fewshot_7B.py
#srun python3 Rong_inferred_zeroshot_70B.py
#srun python3 Rong_inferred_fewshot_70B.py
#srun python3 Attribute-Controlled_zeroshot_7B.py
#srun python3 Attribute-Controlled_zeroshot_70B.py 
#srun python3 Attribute-Controlled_fewshot_70B.py
#srun python3 Attribute-Controlled_fewshot_7B.py
srun python3 ctrl_Rong_inferred_fewshot_70B.py