r"""
    Evaluate the pose estimation.
"""

import torch
import tqdm
from net import TransPoseNet
from config import *
import os
import articulate as art
from utils import normalize_and_concat


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def evaluate_pose(dataset, num_past_frame=20, num_future_frame=5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = PoseEvaluator()
    net = TransPoseNet(num_past_frame, num_future_frame).to(device)
    data = torch.load(os.path.join(dataset, 'test.pt'))
    xs = [normalize_and_concat(a, r).to(device) for a, r in zip(data['acc'], data['ori'])]
    ys = [(art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t) for p, t in zip(data['pose'], data['tran'])]
    offline_errs, online_errs = [], []
    for x, y in tqdm.tqdm(list(zip(xs, ys))):
        net.reset()
        online_results = [net.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
        pose_p_online, tran_p_online = [torch.stack(_)[num_future_frame:] for _ in zip(*online_results)]
        pose_p_offline, tran_p_offline = net.forward_offline(x)
        pose_t, tran_t = y
        offline_errs.append(evaluator.eval(pose_p_offline, pose_t))
        online_errs.append(evaluator.eval(pose_p_online, pose_t))
    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))
    print('============== online ================')
    evaluator.print(torch.stack(online_errs).mean(dim=0))


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False   # if cudnn error, uncomment this line
    evaluate_pose(paths.dipimu_dir)
    evaluate_pose(paths.totalcapture_dir)
