# TransPose

Code for our SIGGRAPH 2021 [paper](https://xinyu-yi.github.io/TransPose/files/TransPose.pdf) "TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors". This repository contains the system implementation, evaluation, and some example IMU data which you can easily run with. [Project Page](https://xinyu-yi.github.io/TransPose/)

![Live Demo 1](data/figures/1.gif)![Live Demo 2](data/figures/2.gif)

*Note： I have put the live demo codes in the `dirty code/`. It contains sensor calibration details.*

## Usage

### Install dependencies

We use `python 3.7.6`. You should install the newest `pytorch chumpy vctoolkit open3d`.

### Prepare SMPL body model

1. Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. Then unzip it.
2. In `config.py`, set `paths.smpl_file` to the model path.

### Prepare pre-trained network weights

1. Download weights from [here](https://xinyu-yi.github.io/TransPose/files/weights.pt).
2. In `config.py`, set `paths.weights_file` to the weights path.

### Prepare test datasets (optional)

1. Download DIP-IMU dataset from [here](https://dip.is.tue.mpg.de/). We use the raw (unnormalized) data.
2. Download TotalCapture dataset from [here](https://cvssp.org/data/totalcapture/). The ground-truth SMPL poses used in our evaluation are provided by the DIP authors. So you may also need to contact the DIP authors for them.
3. In `config.py`, set `paths.raw_dipimu_dir` to the DIP-IMU dataset path; set `paths.raw_totalcapture_dip_dir` to the TotalCapture SMPL poses (from DIP authors) path; and set `paths.raw_totalcapture_official_dir` to the TotalCapture official `gt` path. Please refer to the comments in the codes for more details.

### Run the example

To run the whole system with the provided example IMU measurement sequence, just use:

```shell
python example.py
```

The rendering results in Open3D may be upside down. You can use your mouse to rotate the view.

### Run the evaluation

You should preprocess the datasets before evaluation:

```shell
python preprocess.py
python evaluate.py
```

Both offline and online results for DIP-IMU and TotalCapture test datasets will be printed.

## Citation

If you find the project helpful, please consider citing us:

```
@article{TransPoseSIGGRAPH2021,
    author = {Yi, Xinyu and Zhou, Yuxiao and Xu, Feng},
    title = {TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors},
    journal = {ACM Transactions on Graphics}, 
    year = {2021}, 
    month = {08},
    volume = {40},
    number = {4}, 
    articleno = {86},
    publisher = {ACM}
} 
```

