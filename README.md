# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

The gravityspy project contains three components. The code to produce the images of the glitches that users will evalutate. The code that will evaluate user performance and image retireability colloquially known as the Crowd Sourcing code. Fianlly, the code that determines the Machine Learning labelling of an image, speifically a set of values indicating the classifiers confidence that it belongs in a given category.

# Docs

<https://gravity-spy.github.io/>

# Installation

The easiest method to install cosmic is using pip directly:

```
conda install -c gravityspy gravityspy
```

# Installation on LDG Clusters
```
source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate /home/gravityspy/.conda/envs/gravityspy-gpu-py37/
```

[![Build Status](https://travis-ci.com/Gravity-Spy/GravitySpy.svg?branch=develop)](https://travis-ci.com/Gravity-Spy/GravitySpy)
[![Coverage Status](https://coveralls.io/repos/github/Gravity-Spy/GravitySpy/badge.svg?branch=develop)](https://coveralls.io/github/Gravity-Spy/GravitySpy?branch=develop)
