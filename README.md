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

## Testing

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

---

## Create a file

Next, you’ll add a new file to this repository.

1. Click the **New file** button at the top of the **Source** page.
2. Give the file a filename of **contributors.txt**.
3. Enter your name in the empty file space.
4. Click **Commit** and then **Commit** again in the dialog.
5. Go back to the **Source** page.

Before you move on, go ahead and explore the repository. You've already seen the **Source** page, but check out the **Commits**, **Branches**, and **Settings** pages.

---

## Clone a repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

Now that you're more familiar with your Bitbucket repository, go ahead and add a new file locally. You can [push your change back to Bitbucket with SourceTree](https://confluence.atlassian.com/x/iqyBMg), or you can [add, commit,](https://confluence.atlassian.com/x/8QhODQ) and [push from the command line](https://confluence.atlassian.com/x/NQ0zDQ).
