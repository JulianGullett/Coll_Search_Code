#!/bin/bash
#SBATCH --job-name=CSSL_ABM	        ## set job name
#SBATCH --nodes=40                  ## uncomment this line - does performance improve?
#SBATCH --ntasks-per-node=32        ## tasks per node
#SBATCH --cpus-per-task=1           ## cpus per task
#SBATCH --time=72:00:00		        ## request job time limit
#SBATCH --partition=netsi_standard           ## run on partition "netsi_standard"

######## change number of MPI tasks here: #################
mpiNtasks=1280
##########################

#Clean env of other conda modules or installations:
## Deactivate your existing conda environment - uncomment the below line if you have a conda environment automatically loaded through your ~/.bashrc
#conda deactivate
module purge

#Load the python environment to run MPI with python:
module load discovery anaconda3/2022.01

#Activate Conda Env
source activate NetSI23

#Reset temp_data_1
rm -r temp_data_1
mkdir temp_data_1

#MPI Run Command
mpirun -n $mpiNtasks python Attempt14_Memory.py temp_data_1