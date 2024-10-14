# TransPose

Code for our SIGGRAPH 2021 [paper](https://xinyu-yi.github.io/TransPose/files/TransPose.pdf) "TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors". This repository contains the system implementation, evaluation, and some example IMU data which you can easily run with. [Project Page](https://xinyu-yi.github.io/TransPose/)

![Live Demo 1](data/figures/1.gif)![Live Demo 2](data/figures/2.gif)

## Usage

### Install dependencies

We use `python 3.7.6`. You should install the newest `pytorch chumpy vctoolkit open3d`.

*If the newest `vctoolkit` reports errors, please use `vctoolkit==0.1.5.39`.*

*Installing `pytorch` with CUDA is recommended. The system can only run at ~40 fps on a CPU (i7-8700) and ~90 fps on a GPU (GTX 1080Ti).*

```
conda create -n "imuposer" python=3.7
conda activate imuposer
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

python -m pip install -r requirements.txt
python -m pip install -e src/
```

We use <ROOT> to refer to the root path of this repository in your file system. Prepare folders in the following format:
```
<ROOT>
    └── TransPose
        └── data
            └── dataset_raw
```

### Prepare SMPL body model

1. Register an account in https://smpl.is.tue.mpg.de/download.php. Click on ```Download version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)```. The ```SMPL_python_v.1.0.0.zip``` file will be downloaded. Put it in ```<ROOT>/TransPose/data/dataset_raw```.

```
<ROOT>/data/dataset_raw$ unzip SMPL_python_v.1.0.0.zip
rm -r __MACOSX/
rm -r SMPL_python_v.1.0.0.zip
<ROOT>/data/dataset_raw$ mv smpl/models ../../
<ROOT>/data/dataset_raw$ rm -r smpl/
```
2. In `config.py`, set `paths.smpl_file` to the model path.

### Prepare pre-trained network weights

1. Download weights from [here](https://xinyu-yi.github.io/TransPose/files/weights.pt).
```
<ROOT>/data/dataset_raw$ wget https://xinyu-yi.github.io/TransPose/files/weights.pt
<ROOT>/data/dataset_raw$ mv weights.pt ..
```
2. In `config.py`, set `paths.weights_file` to the weights path.

### Prepare test datasets (optional)

1. Register an account and download DIP-IMU dataset from [here](https://dip.is.tue.mpg.de/). Click on  ```DIP IMU AND OTHERS - DOWNLOAD SERVER 1 ```  (approx. 2.5GB). We use the raw (unnormalized) data.
```
<ROOT>/data/dataset_raw$ unzip DIPIMUandOthers.zip
<ROOT>/data/dataset_raw$ rm DIPIMUandOthers.zip
<ROOT>/data/dataset_raw/DIP_IMU_and_Others$ unzip DIP_IMU.zip
<ROOT>/data/dataset_raw/DIP_IMU_and_Others$ mv DIP_IMU ..
<ROOT>/data/dataset_raw$ rm -r DIP_IMU_and_Others
```
Follow ```Prepare AMASS and DIP_IMU``` or ```3. Download training data``` from https://github.com/bryanbocao/IMUPoser/blob/main/README.md to download dataset AMASS.

2. Download TotalCapture dataset from https://cvssp.org/data/totalcapture/data. Select ```Vicon Groundtruth - The real world position and orinetation```
The following 5 subjects' data (```Subject1 Subject2 Subject3 Subject4 Subject5```) files will be downloaded:
```
s1_vicon_pos_ori.tar.gz
s2_vicon_pos_ori.tar.gz
s3_vicon_pos_ori.tar.gz
s4_vicon_pos_ori.tar.gz
s5_vicon_pos_ori.tar.gz
```
Put them into this folder: ```<ROOT>/data/dataset_raw/TotalCapture/official```. Untar the files by
```
<ROOT>/data/dataset_raw/TotalCapture/official$ for file in *.tar.gz; do tar -xvzf "$file" -C .; done
<ROOT>/data/dataset_raw/TotalCapture/official$ rm -r s2
<ROOT>/data/dataset_raw/TotalCapture/official$ rm -r *.tar.gz
```

Where to find the ```DIP_recalculate``` data:

https://github.com/Xinyu-Yi/TransPose/blob/4963e71ae33c3ea5ac24fcc053015804e9705ad1/config.py#L21
```
    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/dataset_raw/TotalCapture/DIP_recalculate'  # contain ground-truth SMPL pose (*.pkl)
```
Load ground-truth SMPL poses and IMUs from the TotalCapture dataset.

Pointers:
```
https://github.com/eth-ait/dip18?tab=readme-ov-file
https://github.com/eth-ait/aitviewer/blob/main/examples/load_DIP_TC.py
https://github.com/eth-ait/aitviewer/blob/8fb6d4661303579ef04b3bf63ac907dbaecff2ff/examples/load_DIP_TC.py#L14
```

From https://dip.is.tue.mpg.de/download.php, select ```ORIGINAL TotalCapture DATA W/ CORRESPONDING REFERENCE SMPL Poses (wo/ normalization, approx. 250MB)```. The file named ```TotalCapture_Real_60FPS.zip``` will be downloaded.
```
<ROOT>/data/dataset_raw/TotalCapture/DIP_recalculate$ unzip TotalCapture_Real_60FPS.zip
Archive:  TotalCapture_Real_60FPS.zip
   creating: TotalCapture_Real_60FPS/
  inflating: TotalCapture_Real_60FPS/s4_acting3.pkl
  inflating: TotalCapture_Real_60FPS/s5_freestyle3.pkl
  inflating: TotalCapture_Real_60FPS/s3_freestyle3.pkl
  inflating: TotalCapture_Real_60FPS/s3_acting2.pkl
  inflating: TotalCapture_Real_60FPS/s1_rom3.pkl
  ...
<ROOT>/data/dataset_raw/TotalCapture/DIP_recalculate$ rm TotalCapture_Real_60FPS.zip
<ROOT>/data/dataset_raw/TotalCapture/DIP_recalculate$ mv TotalCapture_Real_60FPS/* .
<ROOT>/data/dataset_raw/TotalCapture/DIP_recalculate$ rm -r TotalCapture_Real_60FPS
```

The ground-truth SMPL poses used in our evaluation are provided by the DIP authors. So you may also need to contact the DIP authors for them.

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

### Run your live demo

We provide `live_demo.py` which uses NOTIOM Legacy IMU sensors. This file contains sensor calibration details which may be useful for you.

```
python live_demo.py
```

The estimated poses and translations are sent to Unity3D for visualization using a socket in real-time. You may need to write a client to receive these data to run the live demo codes (or modify the codes a bit).

### Synthesize AMASS dataset

Prepare the raw AMASS dataset and modify `config.py` accordingly. Then, uncomment the `process_amass()` in `preprocess.py` and run:

```
python preprocess.py
```

The saved files are:

- `joint.pt`, which contains a list of tensors in shape [#frames, 24, 3] for 24 absolute joint 3D positions.
- `pose.pt`, which contains a list of tensors in shape [#frames, 24, 3] for 24 relative joint rotations (in axis-angles).
- `shape.pt`, which contains a list of tensors in shape [10] for the subject shape (SMPL parameter).
- `tran.pt`, which contains a list of tensors in shape [#frames, 3] for the global (root) 3D positions.
- `vacc.pt`, which contains a list of tensors in shape [#frames, 6, 3] for 6 synthetic IMU acceleration measurements (global).
- `vrot.pt`, which contains a list of tensors in shape [#frames, 6, 3, 3] for 6 synthetic IMU orientation measurements (global).

All sequences are in 60 fps.

*Please note that these synthesized data should not be directly used in training. They need normalization/coordinate frame transformation according to the paper.*

### Visualize the result in Unity3D

1. Download the unity package from [here](https://xinyu-yi.github.io/TransPose/files/visualizer.unitypackage).
2. Load the package in Unity3D (>=2019.4.16) and open the `Example` scene.
3. Run `example_server.py`. Wait till the server starts. Then play the unity scene.

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

