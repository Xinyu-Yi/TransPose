r"""
    Preprocess DIP-IMU and TotalCapture test dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""


import torch
import os
import pickle
from config import paths


def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)


def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)


if __name__ == '__main__':
    process_dipimu()
    process_totalcapture()
