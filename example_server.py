r"""
    Test the system with an example IMU measurement sequence. Use unity3d to visualize.
    Run this file first, then run Unity3D demo.
"""

import socket
import torch
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate.math as M
from pygame.time import Clock

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TransPoseNet().to(device)
acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))
x = normalize_and_concat(acc, ori).to(device)
pose, tran = net.forward_offline(x)
pose = M.rotation_matrix_to_axis_angle(pose).view(-1, 72)

server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_for_unity.bind(('127.0.0.1', 8888))
server_for_unity.listen(1)
print('Server start. Waiting for unity3d to connect.')
conn, addr = server_for_unity.accept()
clock = Clock()

while True:
    for p, t in zip(pose, tran):
        clock.tick(60)
        s = ','.join(['%g' % v for v in p]) + '#' + \
            ','.join(['%g' % v for v in t]) + '$'
        conn.send(s.encode('utf8'))
