# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

The gravityspy project contains three components. The code to produce the images of the glitches that users will evalutate. The code that will evaluate user performance and image retireability colloquially known as the Crowd Sourcing code. Fianlly, the code that determines the Machine Learning labelling of an image, speifically a set of values indicating the classifiers confidence that it belongs in a given category.

## Installing Gravity Spy on Clusters

git checkout O2GravitySpy

cd setup

vi install.sh (Changing "scoughlin" to your username and making the necessary directories mkdir -p /home/scoughlin/src/O2/ and mkdir -p /home/scoughlin/opt/O2/ (yes this should be a part of the install.sh script)

./install.sh

add . /home/"username"/opt/O2/GravitySpy/bin/activate to you .bash_profile

# ER10/O2 Gravity Spy

# Livingston
* Start Time of ER10: 1161907217
* O1 Start time = 1126400000 September 15, 2015 -- End time = 1137250000 January 18, 2016
# Hanford
* Start Time of ER10: 1163203217
* O1: Start time = 1126400000 September 15, 2015 -- End time = 1137250000 January 18, 2016

[![Build Status](https://travis-ci.org/scottcoughlin2014/GravitySpy.svg?branch=master)](https://travis-ci.org/scottcoughlin2014/GravitySpy)
[![Coverage Status](https://coveralls.io/repos/github/scottcoughlin2014/GravitySpy/badge.svg?branch=master)](https://coveralls.io/github/scottcoughlin2014/GravitySpy?branch=master)
