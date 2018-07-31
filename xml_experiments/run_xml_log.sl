#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64gb
#SBATCH --time=20:00:00
#SBATCH -p standard
#PBS -A 243

bash run_xml_log.sh $1 $2 "${@:3}"
