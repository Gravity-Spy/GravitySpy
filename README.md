# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

The gravityspy project contains three components. The code to produce the images of the glitches that users will evalutate. The code that will evaluate user performance and image retireability colloquially known as the Crowd Sourcing code. Fianlly, the code that determines the Machine Learning labelling of an image, speifically a set of values indicating the classifiers confidence that it belongs in a given category.

## Installing Gravity Spy on Clusters

```
git checkout O2GravitySpy
cd setup
mkdir -p /home/"username"/src/O2/
mkdir -p /home/"username"/opt/O2/
vi install.sh
./install.sh
```

add . /home/"username"/opt/O2/GravitySpy/bin/activate to you .bash_profile

## Test on Gravity Spy on Clusters
```
cd pyomega/examples/
vi wscan.param
bash wscan.param
```

Image should look like: https://ldas-jobs.ligo-la.caltech.edu/~scoughlin/GravitySpy/Test/Classified/123abc1234.png
Score should be  0.995164

Can find score by running
```
ipython
import pandas as pd
pd.read_hdf('/home/scoughlin/public_html/GravitySpy/Test/ML_GSpy_123abc1234.h5').Scratchy
```

Every glitch you make an image of outputs labelling information into the public_html path you specify.
