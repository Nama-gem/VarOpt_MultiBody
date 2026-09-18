#!/bin/bash

#SBATCH --job-name=varopt_multibody
#SBATCH --partition=jila
#SBATCH --qos=long

#SBATCH --array=0-14

#SBATCH --time=7-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

#SBATCH --output=/users/rey/raka3858/output/out-%x.%A_%a.out
#SBATCH --error=/users/rey/raka3858/output/err-%x.%A_%a.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@colorado.edu


# =============================================================================
# Paths
# =============================================================================

PROJECT_DIR="/users/rey/raka3858/VarOpt_MultiBody"

PYTHON_SCRIPT="${PROJECT_DIR}/cluster_optimization.py"

# IMPORTANT:
# This should point to the Python executable in the environment
# where numpy, scipy, numba, quspin, filelock, etc. are installed.
PYTHON="/users/rey/raka3858/.conda/envs/varopt39/bin/python"


# =============================================================================
# Create output directory
# =============================================================================

mkdir -p "/users/rey/raka3858/output"


# =============================================================================
# Check paths
# =============================================================================

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "ERROR: Project directory does not exist:"
    echo "  ${PROJECT_DIR}"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script does not exist:"
    echo "  ${PYTHON_SCRIPT}"
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python executable does not exist or is not executable:"
    echo "  ${PYTHON}"
    exit 1
fi


# =============================================================================
# Move into project directory
# =============================================================================

cd "${PROJECT_DIR}" || exit 1


# =============================================================================
# Prevent contamination from other Python installations
# =============================================================================

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

hash -r


# =============================================================================
# Job information
# =============================================================================

echo "========================================================================"
echo "SLURM job ID:        ${SLURM_JOB_ID}"
echo "SLURM array job ID:  ${SLURM_ARRAY_JOB_ID:-not-set}"
echo "SLURM task ID:       ${SLURM_ARRAY_TASK_ID:-not-set}"
echo "Node:                ${SLURMD_NODENAME:-$(hostname)}"
echo "Working directory:   $(pwd)"
echo "Python executable:   ${PYTHON}"
echo "Start time:          $(date)"
echo "========================================================================"


# =============================================================================
# Verify Python environment
# =============================================================================

"${PYTHON}" -c '
import sys
import numpy
import scipy
import numba
import quspin
import filelock

print("sys.executable:", sys.executable)
print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Numba:", numba.__version__)
print("QuSpin:", quspin.__version__)
print("filelock:", filelock.__version__)
print("Environment import test succeeded.")
'


# Stop immediately if environment test failed
if [[ $? -ne 0 ]]; then
    echo "ERROR: Python environment test failed."
    exit 1
fi


# =============================================================================
# Run optimization
# =============================================================================

echo
echo "Starting cluster optimization..."
echo

"${PYTHON}" -u "${PYTHON_SCRIPT}"


# =============================================================================
# Check exit status
# =============================================================================

EXIT_CODE=$?

echo
echo "========================================================================"

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Task completed successfully."
else
    echo "ERROR: Task exited with code ${EXIT_CODE}"
fi

echo "Finished task: ${SLURM_ARRAY_TASK_ID:-not-set}"
echo "End time:      $(date)"
echo "========================================================================"

exit ${EXIT_CODE}