import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    

    """ YOUR CODE HERE
    """
    P = points.reshape(4, 3).T
    camera = np.linalg.inv(K) @ P
    camera_p = camera * (depth/camera[2,:])
    raw_p = Rt[:3, :3].T @ camera_p - (Rt[:3, :3].T @ Rt[:3, 3]).reshape(3, 1)
    points = raw_p.T
    points = points.reshape(2, 2, 3)
    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """

    projection_matrix = np.dot(K, Rt)

    extra_ones = np.tile([1], points.shape[0]*points.shape[1])
    extra_ones = extra_ones.reshape((points.shape[0], points.shape[1],1))
    h_points = np.concatenate((points, extra_ones), axis=2)
    xs = np.dot(h_points, projection_matrix.T)
    deno = xs[:,:,2][:,:,np.newaxis]
    normalized_xs = xs / deno

    points = normalized_xs[:,:,:2]
    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    src = np.array(((0, 0), (width, 0), (0, height), (width, height),), dtype=np.float32)
    back_project = backproject_corners(K_ref, width, height, depth, Rt_ref)
    dst = project_points(K_neighbor, Rt_neighbor, back_project).reshape(-1, 2)
    H = cv2.findHomography(dst, src)
    H = H[0]
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, dsize=(width, height))
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """

    bar_src = np.mean(src, axis=2, keepdims=True)
    bar_dst = np.mean(dst, axis=2, keepdims=True)
    sig_src = np.std(src, axis=2) + EPS
    sig_dst = np.std(dst, axis=2) + EPS
    
    zncc = np.sum(np.sum((src - bar_src) * (dst - bar_dst), axis = 2) / (sig_src * sig_dst), axis=2)
    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    H, W = dep_map.shape

    xyz_cam = np.zeros((H, W, 3))
    f_0, f_1 = K[0, 0], K[1, 1]

    for v in range(H):
        for u in range(W):
            x = ((u - K[0, 2]) * dep_map[v, u]) / f_0
            y = ((v - K[1, 2]) * dep_map[v, u]) / f_1
            xyz_cam[v, u] = np.array([x, y, dep_map[v, u]])
    """ END YOUR CODE
    """
    return xyz_cam

