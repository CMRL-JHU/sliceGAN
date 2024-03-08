# SliceGAN

This repository is a modification of the original SliceGAN code developed by Steve Kench and Samuel Cooper at Imperial College [1], [github link](https://github.com/stke9/SliceGAN/tree/master).

SliceGAN requires a single 2D training image of an isotropic microstructure, or three 2D images taken at perpendicular angles of an anisotropic microstructure. Images can be colour, grayscale or n-phase.

The main difference in this version is the link between Dream3D and sliceGAN. Dream3D is used to pack grains and get statistical data from the generated microstructures. 

This code was tested and used using the following environment:

- python 3.9.0
- cuda 11.1.0
- pytorch 1.9.0+cu111

The developments of these code were done by Joshua Stickel and Brayan Murgas in the Computational Mechanics Research Laboratory (CMRL) under the supervision of prof. Somnath Ghosh at Johns Hopkins University. Baltimore, Maryland, USA.

---

## Some changes from the original base code:

The original structure of the code was kepts. The main changes are:

1. Input files from Dream3D using hdmf5 files
2. Input variables are defined in the json files
3. A dream3d_support folder was added to keep an interface between sliceGAN and Dream3D
4. A postprocessing folder was added to get data from the generated images

---

## Results

TODO

---

## Preparation of the data

In the Input folder you can find some of the dream3D files used to generate a cold-sprayed Al alloy and an additively manufactured Ti64 alloy. The EBSD maps were obtained in the University of Alabama by prof Luke Brewer's group and APL-JHU by Steven Storck's group.

---

## Training

Run the following command:
```
python ./run_slicegan.py 1 --inp=inputs_3d_mask.json
```
or using the bash file slicegan.train.scr

---

## Generating

Run the following command:
```
python ./run_slicegan.py 0 --inp=inputs_3d_mask.json
```
or using the bash file slicegan.generate.scr

---

## References

1. Kench, S. & Cooper, S. J. Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion. Nat. Mach. Intell. 3, 299–305 (2021)

