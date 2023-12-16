import numpy as np

import geo_lsqlsa


def test_binary_assignment(dtype=np.float32):
    # camera coordinates
    camera_coords = np.array([
        [-10, -10, 20],  # cam1 world coordinates
        [10, -10, 20],  # cam2 world coordinates
    ], dtype=dtype)

    # projected keypoints from images on the ground
    p1s = np.array([  # cam1 detection results
        [10, 10],
        [4, 5],
        [-11, 8],
        [1000, 1000],
    ], dtype=dtype)
    p2s = np.array([  # cam2 detection results
        [-5, 4],
        [-10, 10],
        [-20, 20],
    ], dtype=dtype)

    result = geo_lsqlsa.binary_assignment(camera_coords,
                                          projected_keypoints=[p1s, p2s],
                                          d_threshold=0.1,
                                          return_free=True,
                                          return_position=True)
    matches = result.matches
    free = result.free
    d_star = result.d_star
    h_star = result.h_star
    t_star = result.t_star

    print(f'Result: n_matches={len(matches)}, n_free={(len(free[0]), len(free[1]))}')

    print('Matched objects:')
    for i, j in matches:
        print(f'\tids=({i}, {j}), '
              f'dist={d_star[i, j]:.03f}, '
              f'height={h_star[i, j]:.03f}, '
              f'location={t_star[i, j].round(decimals=3)}')

    print('Unmatched objects:')
    for i in free[0]:  # Camera 1
        best_distance_ind = d_star[i, :].argmin()
        print(f'\t[cam1] id={i}, '
              f'best_dist={d_star[i, best_distance_ind]:.03f}, '
              f'best_height={h_star[i, best_distance_ind]:.03f}')
    for j in free[1]:  # Camera 2
        best_distance_ind = d_star[:, j].argmin()
        print(f'\t[cam2] id={j}, '
              f'best_dist={d_star[best_distance_ind, j]:.03f}, '
              f'best_height={h_star[best_distance_ind, j]:.03f}')


if __name__ == '__main__':
    test_binary_assignment()
