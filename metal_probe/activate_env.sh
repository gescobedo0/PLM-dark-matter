# Sourced by the slurm_*.sh jobs to set up the Python environment on the node.
# EDIT THIS ONCE for your cluster, then you never pass ENV_SETUP on the command line.
# Examples (uncomment/edit one):
# module load miniconda && conda activate metalprobe
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate metalprobe
# source ~/venvs/metalprobe/bin/activate
module load CUDA 2>/dev/null || true
