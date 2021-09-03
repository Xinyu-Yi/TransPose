r"""
    Test the system with an example IMU measurement sequence.
"""


import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TransPoseNet().to(device)
acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))
x = normalize_and_concat(acc, ori).to(device)
pose, tran = net.forward_offline(x)     # offline
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])
