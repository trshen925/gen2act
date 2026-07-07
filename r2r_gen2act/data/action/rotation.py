from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Euler convention used everywhere we convert DROID `cartesian_position` rotation
# (obs[..., 3:6] = roll/pitch/yaw, radians) to/from rotation matrices. If a downstream
# consumer interprets euler with a different convention, change this single constant so
# target generation and inference inversion stay consistent.
EULER_CONVENTION = "xyz"


def euler_to_matrix(euler: np.ndarray) -> np.ndarray:
    """DROID cartesian_position euler angles -> rotation matrix."""
    return Rotation.from_euler(EULER_CONVENTION, np.asarray(euler, dtype=np.float64)).as_matrix()


def matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns in the Zhou et al. 6D representation."""
    m = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)


def euler_delta_to_6d(euler_cur: np.ndarray, euler_fut: np.ndarray) -> np.ndarray:
    """Relative rotation (future <- current) as a 6D continuous representation.

    Proper rotation composition `R_delta = R_fut @ R_cur^T` (not euler subtraction), then
    the 6D rep of Zhou et al. 2019: the first two columns of the rotation matrix. Each
    entry is a rotation-matrix element in [-1, 1], so the 6D vector is naturally bounded.
    """
    r_cur = Rotation.from_matrix(euler_to_matrix(euler_cur))
    r_fut = Rotation.from_matrix(euler_to_matrix(euler_fut))
    m = (r_fut * r_cur.inv()).as_matrix()  # 3x3 relative rotation
    return matrix_to_6d(m)


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthonormalization of a (possibly non-orthonormal) 6D rep -> [...,3,3]."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns = basis vectors


def matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix [...,3,3] -> euler angles in EULER_CONVENTION (radians)."""
    return Rotation.from_matrix(np.asarray(matrix)).as_euler(EULER_CONVENTION).astype(np.float32)
