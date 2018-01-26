#!/bin/bash
#MSUB -A b1011
#MSUB -q ligo
#MSUB -l walltime=1:00:00:00
#MSUB -l nodes=1:ppn=1
#MSUB -l partition=quest6
#MSUB -N retirement_${MOAB_JOBARRAYINDEX}
#MOAB -W umask=022
#MSUB -j oe
#MSUB -d /projects/b1011/mzevin/gravity_spy/GravitySpy/

num_cores=20

module load python

cd /projects/b1011/mzevin/gravity_spy/GravitySpy/API/
python retireimage.py --num-cores ${num_cores} --index ${MOAB_JOBARRAYINDEX} --file-name 'test'
