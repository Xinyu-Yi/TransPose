r"""
    Spatial math utils that combine linear and angular calculations for rigid bodies.
    Also contains utils for articulated body kinematics.
"""


__all__ = ['transformation_matrix_np', 'adjoint_transformation_matrix_np', 'transformation_matrix',
           'decode_transformation_matrix', 'inverse_transformation_matrix', 'bone_vector_to_joint_position',
           'joint_position_to_bone_vector', 'forward_kinematics_R', 'inverse_kinematics_R', 'forward_kinematics_T',
           'inverse_kinematics_T', 'forward_kinematics']


from .general import *
import numpy as np
import torch
from functools import partial


def transformation_matrix_np(R, p):
    r"""
    Get the homogeneous transformation matrix. (numpy, single)

    Transformation matrix :math:`T_{sb} \in SE(3)` of shape [4, 4] can convert points or vectors from b frame
    to s frame: :math:`x_s = T_{sb}x_b`.

    :param R: The rotation of b frame expressed in s frame, R_sb, in shape [3, 3].
    :param p: The position of b frame expressed in s frame, p_s, in shape [3].
    :return: The transformation matrix, T_sb, in shape [4, 4].
    """
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = p
    T[3, 3] = 1
    return T


def adjoint_transformation_matrix_np(R, p):
    r"""
    Get the adjoint representation of a transformation matrix. (numpy, single)

    Adjoint matrix :math:`[Ad_{T_{sb}}]` of shape [6, 6] can convert spatial twist/wrench/Jacobian between b/s frames.

        :math:`\mathcal{V}_s = [Ad_{T_{sb}}]\mathcal{V}_b`

        :math:`\mathcal{F}_b = [Ad_{T_{sb}}]^T\mathcal{F}_s`

        :math:`J_s = [Ad_{T_{sb}}]J_b`

    :param R: The rotation of b frame expressed in s frame, R_sb, in shape [3, 3].
    :param p: The position of b frame expressed in s frame, p_s, in shape [3].
    :return: The adjoint representation of the transformation matrix T_sb, in shape [6, 6].
    """
    AdT = np.zeros((6, 6))
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[3:, :3] = np.dot(vector_cross_matrix_np(p), R)
    return AdT


def transformation_matrix(R: torch.Tensor, p: torch.Tensor):
    r"""
    Get the homogeneous transformation matrices. (torch, batch)

    Transformation matrix :math:`T_{sb} \in SE(3)` of shape [4, 4] can convert points or vectors from b frame
    to s frame: :math:`x_s = T_{sb}x_b`.

    :param R: The rotation of b frame expressed in s frame, R_sb, in shape [*, 3, 3].
    :param p: The position of b frame expressed in s frame, p_s, in shape [*, 3].
    :return: The transformation matrix, T_sb, in shape [*, 4, 4].
    """
    Rp = torch.cat((R, p.unsqueeze(-1)), dim=-1)
    OI = torch.cat((torch.zeros(list(Rp.shape[:-2]) + [1, 3], device=R.device),
                    torch.ones(list(Rp.shape[:-2]) + [1, 1], device=R.device)), dim=-1)
    T = torch.cat((Rp, OI), dim=-2)
    return T


def decode_transformation_matrix(T: torch.Tensor):
    r"""
    Decode rotations and positions from the input homogeneous transformation matrices. (torch, batch)

    :param T: The transformation matrix in shape [*, 4, 4].
    :return: Rotation and position, in shape [*, 3, 3] and [*, 3].
    """
    R = T[..., :3, :3].clone()
    p = T[..., :3, 3].clone()
    return R, p


def inverse_transformation_matrix(T: torch.Tensor):
    r"""
    Get the inverse of the input homogeneous transformation matrices. (torch, batch)

    :param T: The transformation matrix in shape [*, 4, 4].
    :return: Matrix inverse in shape [*, 4, 4].
    """
    R, p = decode_transformation_matrix(T)
    invR = R.transpose(-1, -2)
    invp = -torch.matmul(invR, p.unsqueeze(-1)).squeeze(-1)
    invT = transformation_matrix(invR, invp)
    return invT


def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    r"""
    Multiply/Add matrices along the tree branches. x_local [N, J, *]. parent [J].
    """
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global


def _inverse_tree(x_global: torch.Tensor, parent, reduction_fn, inverse_fn):
    r"""
    Inversely multiply/add matrices along the tree branches. x_global [N, J, *]. parent [J].
    """
    x_local = [x_global[:, 0]]
    for i in range(1, len(parent)):
        x_local.append(reduction_fn(inverse_fn(x_global[:, parent[i]]), x_global[:, i]))
    x_local = torch.stack(x_local, dim=1)
    return x_local


def bone_vector_to_joint_position(bone_vec: torch.Tensor, parent):
    r"""
    Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
    in the base frame. (torch, batch)

    Notes
    -----
    bone_vec[:, i] is the vector from parent[i] to i.

    parent[i] should be the parent joint id of joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint position, in shape [batch_size, num_joint, 3].
    """
    bone_vec = bone_vec.view(bone_vec.shape[0], -1, 3)
    joint_pos = _forward_tree(bone_vec, parent, torch.add)
    return joint_pos


def joint_position_to_bone_vector(joint_pos: torch.Tensor, parent):
    r"""
    Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
    in the base frame. (torch, batch)

    Notes
    -----
    bone_vec[:, i] is the vector from parent[i] to i.

    parent[i] should be the parent joint id of joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Bone vector, in shape [batch_size, num_joint, 3].
    """
    joint_pos = joint_pos.view(joint_pos.shape[0], -1, 3)
    bone_vec = _inverse_tree(joint_pos, parent, torch.add, torch.neg)
    return bone_vec


def forward_kinematics_R(R_local: torch.Tensor, parent):
    r"""
    :math:`R_global = FK(R_local)`

    Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_local[:, i], parent[i] should be the local rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
    """
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global


def inverse_kinematics_R(R_global: torch.Tensor, parent):
    r"""
    :math:`R_local = IK(R_global)`

    Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_global[:, i], parent[i] should be the global rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
    """
    R_global = R_global.view(R_global.shape[0], -1, 3, 3)
    R_local = _inverse_tree(R_global, parent, torch.bmm, partial(torch.transpose, dim0=1, dim1=2))
    return R_local


def forward_kinematics_T(T_local: torch.Tensor, parent):
    r"""
    :math:`T_global = FK(T_local)`

    Forward kinematics that computes the global homogeneous transformation of each joint from
    local homogeneous transformations. (torch, batch)

    Notes
    -----
    A joint's *local* transformation is expressed in its parent's frame.

    A joint's *global* transformation is expressed in the base (root's parent) frame.

    T_local[:, i], parent[i] should be the local transformation matrix and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
    """
    T_local = T_local.view(T_local.shape[0], -1, 4, 4)
    T_global = _forward_tree(T_local, parent, torch.bmm)
    return T_global


def inverse_kinematics_T(T_global: torch.Tensor, parent):
    r"""
    :math:`T_local = IK(T_global)`

    Inverse kinematics that computes the local homogeneous transformation of each joint from
    global homogeneous transformations. (torch, batch)

    Notes
    -----
    A joint's *local* transformation is expressed in its parent's frame.

    A joint's *global* transformation is expressed in the base (root's parent) frame.

    T_global[:, i], parent[i] should be the global transformation matrix and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
    """
    T_global = T_global.view(T_global.shape[0], -1, 4, 4)
    T_local = _inverse_tree(T_global, parent, torch.bmm, inverse_transformation_matrix)
    return T_local


def forward_kinematics(R_local: torch.Tensor, p_local: torch.Tensor, parent):
    r"""
    :math:`R_global, p_global = FK(R_local, p_local)`

    Forward kinematics that computes the global rotation and position of each joint from
    local rotations and positions. (torch, batch)

    Notes
    -----
    A joint's *local* rotation and position are expressed in its parent's frame.

    A joint's *global* rotation and position are expressed in the base (root's parent) frame.

    R_local[:, i], p_local[:, i], parent[i] should be the local rotation, local position, and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3, 3] (rotation matrices).
    :param p_local: Joint local position tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3] (zero-pose bone vectors).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation and position, in shape [batch_size, num_joint, 3, 3] and [batch_size, num_joint, 3].
    """
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    p_local = p_local.view(p_local.shape[0], -1, 3)
    T_local = transformation_matrix(R_local, p_local)
    T_global = forward_kinematics_T(T_local, parent)
    return decode_transformation_matrix(T_global)
