r"""
    Angular math utils that contain calculations of angles.
"""


__all__ = ['RotationRepresentation', 'to_rotation_matrix', 'radian_to_degree', 'degree_to_radian', 'normalize_angle',
           'angle_difference', 'angle_between', 'svd_rotate', 'generate_random_rotation_matrix',
           'axis_angle_to_rotation_matrix', 'rotation_matrix_to_axis_angle', 'r6d_to_rotation_matrix',
           'rotation_matrix_to_r6d', 'quaternion_to_axis_angle', 'axis_angle_to_quaternion',
           'quaternion_to_rotation_matrix', 'rotation_matrix_to_euler_angle', 'euler_angle_to_rotation_matrix',
           'rotation_matrix_to_euler_angle_np', 'euler_angle_to_rotation_matrix_np', 'euler_convert_np']


from .general import *
import enum
import numpy as np
import torch


class RotationRepresentation(enum.Enum):
    r"""
    Rotation representations. Quaternions are in wxyz. Euler angles are in local XYZ.
    """
    AXIS_ANGLE = 0
    ROTATION_MATRIX = 1
    QUATERNION = 2
    R6D = 3
    EULER_ANGLE = 4


def to_rotation_matrix(r: torch.Tensor, rep: RotationRepresentation):
    r"""
    Convert any rotations into rotation matrices. (torch, batch)

    :param r: Rotation tensor.
    :param rep: The rotation representation used in the input.
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    if rep == RotationRepresentation.AXIS_ANGLE:
        return axis_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.QUATERNION:
        return quaternion_to_rotation_matrix(r)
    elif rep == RotationRepresentation.R6D:
        return r6d_to_rotation_matrix(r)
    elif rep == RotationRepresentation.EULER_ANGLE:
        return euler_angle_to_rotation_matrix(r)
    elif rep == RotationRepresentation.ROTATION_MATRIX:
        return r.view(-1, 3, 3)
    else:
        raise Exception('unknown rotation representation')


def radian_to_degree(q):
    r"""
    Convert radians to degrees.
    """
    return q * 180.0 / np.pi


def degree_to_radian(q):
    r"""
    Convert degrees to radians.
    """
    return q / 180.0 * np.pi


def normalize_angle(q):
    r"""
    Normalize radians into [-pi, pi). (np/torch, batch)

    :param q: A tensor (np/torch) of angles in radians.
    :return: The normalized tensor where each angle is in [-pi, pi).
    """
    mod = q % (2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def angle_difference(target, source):
    r"""
    Calculate normalized target - source. (np/torch, batch)
    """
    return normalize_angle(target - source)


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor, rep=RotationRepresentation.ROTATION_MATRIX):
    r"""
    Calculate the angle in radians between two rotations. (torch, batch)

    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot1 = to_rotation_matrix(rot1, rep)
    rot2 = to_rotation_matrix(rot2, rep)
    offsets = rot1.transpose(1, 2).bmm(rot2)
    angles = rotation_matrix_to_axis_angle(offsets).norm(dim=1)
    return angles


def svd_rotate(source_points: torch.Tensor, target_points: torch.Tensor):
    r"""
    Get the rotation that rotates source points to the corresponding target points. (torch, batch)

    :param source_points: Source points in shape [batch_size, m, n]. m is the number of the points. n is the dim.
    :param target_points: Target points in shape [batch_size, m, n]. m is the number of the points. n is the dim.
    :return: Rotation matrices in shape [batch_size, 3, 3] that rotate source points to target points.
    """
    usv = [m.svd() for m in source_points.transpose(1, 2).bmm(target_points)]
    u = torch.stack([_[0] for _ in usv])
    v = torch.stack([_[2] for _ in usv])
    vut = v.bmm(u.transpose(1, 2))
    for i in range(vut.shape[0]):
        if vut[i].det() < -0.9:
            v[i, 2].neg_()
            vut[i] = v[i].mm(u[i].t())
    return vut


def generate_random_rotation_matrix(n=1):
    r"""
    Generate random rotation matrices. (torch, batch)

    :param n: Number of rotation matrices to generate.
    :return: Random rotation matrices of shape [n, 3, 3].
    """
    q = torch.zeros(n, 4)
    while True:
        n = q.norm(dim=1)
        mask = (n == 0) | (n > 1)
        if q[mask].shape[0] == 0:
            break
        q[mask] = torch.rand_like(q[mask]) * 2 - 1
    q = q / q.norm(dim=1, keepdim=True)
    return quaternion_to_rotation_matrix(q)


def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r


def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result


def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    Turn 6D vectors into rotation matrices. (torch, batch)

    **Warning:** The two 3D vectors of any 6D vector must be linearly independent.

    :param r6d: 6D vector tensor that can reshape to [batch_size, 6].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    r6d = r6d.view(-1, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r


def rotation_matrix_to_r6d(r: torch.Tensor):
    r"""
    Turn rotation matrices into 6D vectors. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: 6D vector tensor of shape [batch_size, 6].
    """
    return r.view(-1, 3, 3)[:, :, :2].transpose(1, 2).clone().view(-1, 6)


def quaternion_to_axis_angle(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into axis-angles. (torch, batch)

    **Warning**: The returned axis angles may have a rotation larger than 180 degrees (in 180 ~ 360).

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    q = normalize_tensor(q.view(-1, 4))
    theta_half = q[:, 0].clamp(min=-1, max=1).acos()
    a = (q[:, 1:] / theta_half.sin().view(-1, 1) * 2 * theta_half.view(-1, 1)).view(-1, 3)
    a[torch.isnan(a)] = 0
    return a


def axis_angle_to_quaternion(a: torch.Tensor):
    r"""
    Turn axis-angles into quaternions. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Quaternion wxyz tensor of shape [batch_size, 4].
    """
    axes, angles = normalize_tensor(a.view(-1, 3), return_norm=True)
    axes[torch.isnan(axes)] = 0
    q = torch.cat(((angles / 2).cos(), (angles / 2).sin() * axes), dim=1)
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    q = normalize_tensor(q.view(-1, 4))
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)


def rotation_matrix_to_euler_angle(r: torch.Tensor, seq='XYZ'):
    r"""
    Turn rotation matrices into euler angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Euler angle tensor of shape [batch_size, 3].
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(r.clone().detach().cpu().view(-1, 3, 3).numpy())
    ret = torch.from_numpy(rot.as_euler(seq)).float().to(r.device)
    return ret


def euler_angle_to_rotation_matrix(q: torch.Tensor, seq='XYZ'):
    r"""
    Turn euler angles into rotation matrices. (torch, batch)

    :param q: Euler angle tensor that can reshape to [batch_size, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler(seq, q.clone().detach().cpu().view(-1, 3).numpy())
    ret = torch.from_numpy(rot.as_matrix()).float().to(q.device)
    return ret


def rotation_matrix_to_euler_angle_np(r, seq='XYZ'):
    r"""
    Turn rotation matrices into euler angles. (numpy, batch)

    :param r: Rotation matrix (np/torch) that can reshape to [batch_size, 3, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Euler angle ndarray of shape [batch_size, 3].
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.array(r).reshape(-1, 3, 3)).as_euler(seq)


def euler_angle_to_rotation_matrix_np(q, seq='XYZ'):
    r"""
    Turn euler angles into rotation matrices. (numpy, batch)

    :param q: Euler angle (np/torch) that can reshape to [batch_size, 3].
    :param seq: 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
                rotations, or {'x', 'y', 'z'} for extrinsic rotations (radians).
                See scipy for details.
    :return: Rotation matrix ndarray of shape [batch_size, 3, 3].
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(seq, np.array(q).reshape(-1, 3)).as_matrix()


def euler_convert_np(q, from_seq='XYZ', to_seq='XYZ'):
    r"""
    Convert euler angles into different axis orders. (numpy, single/batch)

    :param q: An ndarray of euler angles (radians) in from_seq order. Shape [3] or [N, 3].
    :param from_seq: The source(input) axis order. See scipy for details.
    :param to_seq: The target(output) axis order. See scipy for details.
    :return: An ndarray with the same size but in to_seq order.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(from_seq, q).as_euler(to_seq)
