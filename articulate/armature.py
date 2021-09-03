r"""
    Joint definition of SMPL/MANO/SMPLH parametric model.
"""


__all__ = ['MANOJoint', 'SMPLJoint', 'SMPLHJoint']


import enum


class MANOJoint(enum.Enum):
    r"""
    W = wrist; I = index; M = middle; L = little; R = ring; T = thumb.
    """
    W = 0
    ROOT = 0
    I0 = 1
    I1 = 2
    I2 = 3
    M0 = 4
    M1 = 5
    M2 = 6
    L0 = 7
    L1 = 8
    L2 = 9
    R0 = 10
    R1 = 11
    R2 = 12
    T0 = 13
    T1 = 14
    T2 = 15


class SMPLJoint(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 0
    PELVIS = 0
    SPINE = 0
    LHIP = 1
    RHIP = 2
    SPINE1 = 3
    LKNEE = 4
    RKNEE = 5
    SPINE2 = 6
    LANKLE = 7
    RANKLE = 8
    SPINE3 = 9
    LFOOT = 10
    RFOOT = 11
    NECK = 12
    LCLAVICLE = 13
    RCLAVICLE = 14
    HEAD = 15
    LSHOULDER = 16
    RSHOULDER = 17
    LELBOW = 18
    RELBOW = 19
    LWRIST = 20
    RWRIST = 21
    LHAND = 22
    RHAND = 23


class SMPLHJoint(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    W = wrist; I = index; M = middle; L = little; R = ring; T = thumb.
    """
    ROOT = 0
    PELVIS = 0
    SPINE = 0
    LHIP = 1
    RHIP = 2
    SPINE1 = 3
    LKNEE = 4
    RKNEE = 5
    SPINE2 = 6
    LANKLE = 7
    RANKLE = 8
    SPINE3 = 9
    LFOOT = 10
    RFOOT = 11
    NECK = 12
    LCLAVICLE = 13
    RCLAVICLE = 14
    HEAD = 15
    LSHOULDER = 16
    RSHOULDER = 17
    LELBOW = 18
    RELBOW = 19
    LWRIST = 20
    LW = 20
    RWRIST = 21
    RW = 21
    LI0 = 22
    LI1 = 23
    LI2 = 24
    LM0 = 25
    LM1 = 26
    LM2 = 27
    LL0 = 28
    LL1 = 29
    LL2 = 30
    LR0 = 31
    LR1 = 32
    LR2 = 33
    LT0 = 34
    LT1 = 35
    LT2 = 36
    RI0 = 37
    RI1 = 38
    RI2 = 39
    RM0 = 40
    RM1 = 41
    RM2 = 42
    RL0 = 43
    RL1 = 44
    RL2 = 45
    RR0 = 46
    RR1 = 47
    RR2 = 48
    RT0 = 49
    RT1 = 50
    RT2 = 51
