#!/bin/bash
#SBATCH -t 14-0
#SBATCH --job-name rf_map4
#SBATCH --mem=64GB
#SBATCH -q primary
#SBATCH --cpus-per-task=20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o rf_map4.log
#SBATCH -e rf_map4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gi1632@wayne.edu

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda not found, installing Miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "export PATH=\${PATH}:\${HOME}/miniconda/bin" >> \${HOME}/.bashrc
    export PATH=\${PATH}:\${HOME}/miniconda/bin
    conda init bash
else
	source /wsu/home/gi/gi16/gi1632/.bashrc
	#conda init bash
fi

# Display conda version
echo "Using conda version:" 
echo | which conda

# Check if the conda environment exists
if conda env list | grep -q 'ML'
then
    echo "Conda environment ML exists"
else
    echo "Conda environment not found"
	exit
fi

# Activate the conda environment
conda activate ML

# Check if the conda environment was activated successfully
if [[ $(conda env list | grep '*' | awk '{print $1}') != "ML" ]]
then
    echo "Failed to activate conda environment ML"
    exit
else
	echo "'ML' conda environment activated successfully."
fi

python rf_map4.py