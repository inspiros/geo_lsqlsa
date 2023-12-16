import itertools
from collections import namedtuple
from typing import Sequence, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['binary_assignment']

Array2DLike = Union[np.ndarray, Sequence[Sequence[Union[int, float]]]]
AssignmentResult = namedtuple('AssignmentResult',
                              ['matches', 'free', 'd_star', 'h_star', 't_star'])


def solve_lap(cost_matrix: np.ndarray) -> np.ndarray:
    r"""
    Wrapper around :func:`scipy.optimize.linear_sum_assignment`.
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.stack([x, y]).T


def binary_assignment(camera_coords: Array2DLike,
                      projected_keypoints: Sequence[Array2DLike],
                      d_threshold: float = 0,
                      lstsq_solver: str = 'lstsq',
                      return_free: bool = False,
                      return_position: bool = False) -> AssignmentResult:
    r"""
    Find assignments of object detection results from two cameras.
    This function requires projected positions on the ground of
    centers (or keypoints) of object detection results from both
    views. These can be computed using different methods of camera
    calibration.

    This function optionally returns the estimated ground positions
    using a simple mean of best ground positions from both cameras if
    ``return_position=True``.

    Args:
        camera_coords (np.ndarray): 2D array of 3D world coordinates
            of cameras.
        projected_keypoints (sequence of np.ndarray): List of projected
            positions of detected keypoints from the two cameras,
            represented as a pair of (n, 2) and (m, 2) arrays.
        d_threshold (float): Distance threshold. Defaults to 0.
        lstsq_solver (str): Least Square solver, can be either
            analytical | lstsq. Defaults to 'lstsq'.
        return_free (bool): Return unmatched vertices if true.
            Defaults to False.
        return_position (bool): Return estimated ground positions if
            true. Defaults to False.

    Returns:
        assignment_result: A namedtuple of ``matches``, ``free``, ``d_star``
            (optimal distances), ``h_star``, optimal heights, and ``t_star``
            (optimal estimated ground positions).
    """
    if d_threshold < 0:
        raise ValueError(f'd_threshold must be non-negative. Got {d_threshold}.')
    camera_coords = np.asarray(camera_coords)
    if camera_coords.shape != (2, 3):
        raise ValueError(f'camera_coords\'s shape must be (2, 3). '
                         f'Got camera_coords.shape={camera_coords.shape}.')
    c1c = camera_coords[0]
    c2c = camera_coords[1]
    p1s = np.asarray(projected_keypoints[0])
    if p1s.ndim != 2 or p1s.shape[1] != 2:
        raise ValueError(f'projected_keypoints must be sequence of (n, 2) arrays. '
                         f'Got projected_keypoints[0].shape={p1s.shape}.')
    p2s = np.asarray(projected_keypoints[1])
    if p2s.ndim != 2 or p2s.shape[1] != 2:
        raise ValueError(f'projected_keypoints must be sequence of (n, 2) arrays. '
                         f'Got projected_keypoints[0].shape={p2s.shape}.')

    c1 = c1c[:-1]
    c2 = c2c[:-1]
    z1 = c1c[-1]
    z2 = c2c[-1]

    # ===== Least Squares =====
    n, m = len(p1s), len(p2s)
    N = n * m
    A = np.zeros((2 * N, N), dtype=np.result_type(camera_coords, p1s, p2s))
    b = np.zeros_like(A)

    for p_id, (i, j) in enumerate(itertools.product(range(n), range(m))):
        row_start = 2 * p_id
        p1 = p1s[i]
        p2 = p2s[j]
        A[row_start:row_start + 2, p_id:p_id + 1] = ((c1 - p1) / z1 - (c2 - p2) / z2).reshape(-1, 1)
        b[row_start:row_start + 2, p_id:p_id + 1] = (p2 - p1).reshape(-1, 1)

    if lstsq_solver == 'analytical':
        h = np.linalg.inv((A.T @ A)) @ A.T @ b
    elif lstsq_solver == 'lstsq':
        h = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        raise ValueError(f'lstsq_solver must be either analytical | lstsq. Got {lstsq_solver}.')

    d = np.sqrt(np.power(A @ h - b, 2).sum(0)).reshape(n, m)  # optimal distances
    h = np.diag(h).reshape(n, m)  # optimal heights
    if return_position:
        # estimate ground locations
        t = np.empty((n, m, 2), dtype=np.result_type(p1s, p2s))
        for p_id, (i, j) in enumerate(itertools.product(range(n), range(m))):
            p1 = p1s[i]
            p2 = p2s[j]
            t1 = p1 + h[i, j] * (c1 - p1) / z1
            t2 = p2 + h[i, j] * (c2 - p2) / z2
            t[i, j] = (t1 + t2) / 2  # mean
    else:
        t = None

    # ===== Linear Sum Assignment =====
    # construct a cost matrix C
    C = np.clip(1 - d / d_threshold, a_min=0, a_max=None)
    C = np.where(h > 0, C, 0)  # filter possible undesired negative heights
    matches = np.array([[i, j] for i, j in solve_lap(-C) if C[i, j] != 0])
    free = [list(set(range(n)).difference(matches[:, 0].tolist())),
            list(set(range(m)).difference(matches[:, 1].tolist()))] if return_free else None
    return AssignmentResult(matches=matches, free=free, d_star=d, h_star=h, t_star=t)
