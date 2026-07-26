#!/bin/bash

for i in 0 1 2 3 4 5 6 7
do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=branch_${i}
#SBATCH --output=logs/branch_${i}_%j.out
#SBATCH --error=logs/branch_${i}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
module load python
module load anaconda
conda activate myenv

python3 -u run_branch.py ${i}

EOF
    echo "Submitted branch ${i}"
done