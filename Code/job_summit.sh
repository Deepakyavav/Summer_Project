#!/bin/bash


## DO NOT EDIT ANY LINE BELOW ##
## THIS IS THE STANDARD TEMPLATE FOR SUBMITTING A JOB IN DGX BOX ##
## The line above is used by computer to process it as
## bash script.

## This file serves as the template of the bash script which
## should be used to run the codes required for experiments 
## associated with the ISL Lab.


## The following are some necessary commands associated with 
## the SLURM.
#SBATCH --job-name=data ## Job name
#SBATCH --ntasks=2 ## Run on a single CPU
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=longq ## ( partition name )
#SBATCH --qos=longq ## ( QUEUE name )
#SBATCH --mem=64G ## Memory requested

## The output of the scuessfully executed code will be saved
## in the file mentioned below. %u -> user, %x -> jon-name, 
## %N -> the compute node (dgx1), %j-> job ID.
#SBATCH --output=/home1/cs23m105/Desktop/success/%u/%x-%N-%j.out ##Output file

## The errors associated will be saved in the file below
#SBATCH --error=/home1/cs23m105/Desktop/error/%u/%x-%N-%j.err ## Error file


## the following command ensures successful loading of modules
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

## DO NOT EDIT ANY LINE ABOVE ###

## Uncomment the following to import torch
eval "$(conda shell.bash hook)"
conda activate pytorch_gpu 

## put all your required python code in demo.py
## call your python script here

##python tf.py
python Creating_rtc_from_mask.py

## command to execute this script:
## sbatch job_submit.sh

## note down the jobid for the submitted job
## check your job status using the below command
## sacct -u <username> 
## once your job status is shown as completed
## then cd to the directory /scratch/<username>/ 
## the file isl-dgx1-<jobid>.out -- contains the output of your program
## the file isl-dgx1-<jobid>.err -- contains the errors encountered

##SBATCH --cpus-per-task=2
##SBATCH --mem=64G
##SBATCH --nodes=1
##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:a100:2
