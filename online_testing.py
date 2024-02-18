import cv2
import numpy as np
import os
import utils


def load_calibration_results(calibration_input_path="calibrations",
                             calibration_input_filename="calibration_results.npz"):
    """
    Loads results of camera calibration and info from file.
    - err: mean reprojection error
    - mtx: camera matrix
    - dist: distortion coefficient
    - rvecs: rotation vector
    - tvecs: translation vector
    - in_std: standard deviation over intrinsic parameters
    - ex_std: standard deviation over extrinsic parameters
    - per_view_err: per view reprojection error
    - image_points: array of 2D chessboard image points for every image
    - image_shape: shape of the images
    - chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    - chessboard_square_size: chessboard square size in mm
    - run: run number

    :param calibration_input_path: output directory path
    :param calibration_input_filename: output file name (including extension)
    :return: returns calibration results err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err, as well as info
             image_points, image_shape, chessboard_shape, chessboard_square_size, run
    """

    # Load the saved calibration
    data = np.load(os.path.join(calibration_input_path, calibration_input_filename))

    # Retrieve the arrays from the loaded data
    # Calibration results
    err = data["err"]
    mtx = data["mtx"]
    dist = data["dist"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    in_std = data["in_std"]
    ex_std = data["ex_std"]
    per_view_err = data["per_view_err"]
    # Info
    image_points = data["image_points"]
    image_shape = data["image_shape"]
    chessboard_shape = data["chessboard_shape"]
    chessboard_square_size = data["chessboard_square_size"]
    run = data["run"]

    return err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err,\
        image_points, image_shape, chessboard_shape, chessboard_square_size, run


def draw_axes_on_chessboard(image, corners, mtx, dist, rvecs, tvecs, chessboard_square_size=1,
                            chessboard_square_span=1):
    """
    Draws 3D axes with the origin at the center of the world coordinates.

    :param image: image to draw on
    :param corners: array of 2D chessboard image points to take first corner from as origin
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param chessboard_square_size: chessboard square size in mm
    :param chessboard_square_span: chessboard number of squares each axis spans across
    """
    # Axes span a number of chessboard squares
    axis_length = chessboard_square_size * chessboard_square_span
    # 3 axes corners with negative Z value so axis faces the camera
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) * axis_length

    # Project 3D points to image plane
    image_points_axes, _ = cv2.projectPoints(axes, rvecs, tvecs, mtx, dist)

    # First chessboard corner as origin
    origin_corner = corners[0][0]

    # Convert to integer for drawing
    image_points_axes = image_points_axes.astype(np.int32).reshape(-1, 2)
    origin_corner = origin_corner.astype(np.int32)

    # Draw blue arrow for vertical axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[0].ravel()), color=(255, 0, 0), thickness=2)
    # Draw green arrow for horizontal axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[1].ravel()), color=(0, 255, 0), thickness=2)
    # Draw red arrow for Z axis
    cv2.arrowedLine(image, origin_corner, tuple(image_points_axes[2].ravel()), color=(0, 0, 255), thickness=2)


def draw_cube_on_chessboard(image, mtx, dist, rvecs, tvecs, chessboard_square_size=1, chessboard_square_span=1):
    """
    Draws a cube at the origin of the world coordinates.

    :param image: image to draw on
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param chessboard_square_size: chessboard square size in mm
    :param chessboard_square_span: chessboard number of squares the cube base sides span across
    """
    # Cube sides span a number of chessboard squares
    cube_side_length = chessboard_square_size * chessboard_square_span
    # 8 cube corners with negative Z value so cube faces the camera
    cube = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]) * cube_side_length

    # Project 3D points to image plane
    image_points_cube, _ = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
    # Convert to integer for drawing
    image_points_cube = image_points_cube.astype(np.int32).reshape(-1, 2)

    # Draw cube floor
    cv2.drawContours(image, [image_points_cube[:4]], -1, color=(130, 249, 255), thickness=2)

    # Draw cube pillars
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(image, tuple(image_points_cube[i]), tuple(image_points_cube[j]), color=(130, 249, 255), thickness=2)

    # Draw cube top
    cv2.drawContours(image, [image_points_cube[4:]], -1, color=(130, 249, 255), thickness=2)


def test_camera_parameters_with_images(chessboard_shape, chessboard_square_size, mtx, dist,
                                       images_input_path="evaluation_inputs", images_output_path="evaluation_outputs",
                                       more_exact_corners=True, pnp_ransac=True, result_time_visible=1000):
    """
    Tests estimated camera parameters by using them to draw the world 3D axes (XYZ) with the origin at the center of the
    world coordinates and a cube at the origin of the world coordinates on test images. The resulting chessboard images
    with the drawings are saved to files.

    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param chessboard_square_size: chessboard square size in mm
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param images_input_path: testing images directory path
    :param images_output_path: testing output directory path
    :param more_exact_corners: increases corner accuracy with more exact corner positions if set to True, otherwise
                               keeps original corner positions.
    :param pnp_ransac: if True then incorporates RANSAC algorithm for more robust estimation for Perspective-n-Point
                       (PnP) to find rotation and translation vectors
    :param result_time_visible: milliseconds to keep result of corner extraction for an image to screen, 0 to wait for
                                key press
    """
    # Check that images are present
    images_path_list = [os.path.join(images_input_path, file) for file in os.listdir(images_input_path)
                        if file.endswith(".jpg")]
    if not images_path_list:
        utils.show_warning("test_empty")
        return None
    # Sort images
    images_path_list = sorted(images_path_list, key=lambda x: int(x.split("\\test_")[-1].split(".")[0]))

    # Prepare object points [[0,0,0], [1,0,0], [2,0,0] ...,[chessboard_vertical, chessboard_horizontal,0]]
    # Multiply by chessboard square size in mm, keep X, Y (Z=0, stationary XY plane chessboard)
    object_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) \
                           * chessboard_square_size

    # Loop for every image
    for image_path in images_path_list:
        # Read image and resize for viewing
        current_image = cv2.imread(image_path)
        current_image = cv2.resize(current_image, (0, 0), fx=0.5, fy=0.5)

        # Try to detect the chessboard corners automatically using grayscale image and all flags
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                               cv2.CALIB_CB_FILTER_QUADS +
                                                                               cv2.CALIB_CB_FAST_CHECK)

        # Only handling automatically detected corners
        if not ret:
            utils.show_warning("no_automatic_corners_online")
            continue

        # Increase corner accuracy with more exact corner positions
        if more_exact_corners:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # Find the rotation and translation vectors
        # Incorporate RANSAC algorithm for more robust estimation
        if pnp_ransac:
            ret_pnp, rvecs, tvecs, _ = cv2.solvePnPRansac(object_points, corners, mtx, dist)
        else:
            ret_pnp, rvecs, tvecs = cv2.solvePnP(object_points, corners, mtx, dist)

        # Draw axes that span 3 squares on chessboard
        draw_axes_on_chessboard(current_image, corners, mtx, dist, rvecs, tvecs, chessboard_square_size, 3)
        # Draw cube that spans 2 squares on chessboard
        draw_cube_on_chessboard(current_image, mtx, dist, rvecs, tvecs, chessboard_square_size, 2)

        # Show results
        cv2.imshow("Drawn 3D Axes and Cube using Estimated Camera Martix", current_image)
        cv2.waitKey(result_time_visible)

        # Output results to file
        if not os.path.isdir(images_output_path):
            os.makedirs(images_output_path)
        image_output_filename = str(image_path).split("\\")[-1].split(".")[0] + "_result.jpg"
        cv2.imwrite(os.path.join(images_output_path, image_output_filename), current_image)

    cv2.destroyAllWindows()


def test_camera_parameters_with_videos(chessboard_shape, chessboard_square_size, mtx, dist,
                                       videos_input_path="evaluation_inputs", videos_output_path="evaluation_outputs",
                                       more_exact_corners=True, pnp_ransac=True):
    """
    Tests estimated camera parameters by using them to draw the world 3D axes (XYZ) with the origin at the center of the
    world coordinates and a cube at the origin of the world coordinates on test videos. The resulting chessboard videos
    with the drawings on every frame are saved to files.

    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param chessboard_square_size: chessboard square size in mm
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param videos_input_path: testing videos directory path
    :param videos_output_path: testing output directory path
    :param more_exact_corners: increases corner accuracy with more exact corner positions if set to True, otherwise
                               keeps original corner positions.
    :param pnp_ransac: if True then incorporates RANSAC algorithm for more robust estimation for Perspective-n-Point
                       (PnP) to find rotation and translation vectors
    """
    cv2.destroyAllWindows()
    # Check that videos are present
    videos_path_list = [os.path.join(videos_input_path, file) for file in os.listdir(videos_input_path)
                        if file.endswith(".mp4")]
    if not videos_path_list:
        utils.show_warning("test_empty")
        return None

    # Prepare object points [[0,0,0], [1,0,0], [2,0,0] ...,[chessboard_vertical, chessboard_horizontal,0]]
    # Multiply by chessboard square size in mm, keep X, Y (Z=0, stationary XY plane chessboard)
    object_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) \
                           * chessboard_square_size

    for video_path in videos_path_list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            utils.show_warning("video_none")
            continue

        # Get video properties for the output video
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec for MP4 and create VideoWriter object for output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if not os.path.isdir(videos_output_path):
            os.makedirs(videos_output_path)
        video_output_filename = str(video_path).split("\\")[-1].split(".")[0] + "_result.mp4"
        output_video = cv2.VideoWriter(os.path.join(videos_output_path, video_output_filename), fourcc, fps,
                                       np.array([frame_width, frame_height], dtype=np.int32))

        # Loop until all video frames are processed
        skipped_frames = False
        while True:
            # Read video frame
            ret_frame, current_frame = cap.read()
            # Video end
            if not ret_frame:
                break

            # Resize for faster processing
            # current_frame = cv2.resize(current_frame, (0, 0), fx=1, fy=1)

            # Try to detect the chessboard corners automatically using grayscale frame and all flags
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                   cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                                   cv2.CALIB_CB_FILTER_QUADS +
                                                                                   cv2.CALIB_CB_FAST_CHECK)

            # Only handling automatically detected corners
            # Raise flag to inform about frames being discarded from the video after video is processed
            if not ret:
                skipped_frames = True
                continue

            # Increase corner accuracy with more exact corner positions
            if more_exact_corners:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

            # Find the rotation and translation vectors
            # Incorporate RANSAC algorithm for more robust estimation
            if pnp_ransac:
                ret_pnp, rvecs, tvecs, _ = cv2.solvePnPRansac(object_points, corners, mtx, dist)
            else:
                ret_pnp, rvecs, tvecs = cv2.solvePnP(object_points, corners, mtx, dist)

            # Draw axes that span 3 squares on chessboard
            draw_axes_on_chessboard(current_frame, corners, mtx, dist, rvecs, tvecs, chessboard_square_size, 3)
            # Draw cube that spans 2 squares on chessboard
            draw_cube_on_chessboard(current_frame, mtx, dist, rvecs, tvecs, chessboard_square_size, 2)

            # Write the frame into the output file
            output_video.write(current_frame)

            # Show results
            cv2.imshow("Drawn 3D Axes and Cube using Estimated Camera Martix", current_frame)
            # Exit if ESC is pressed
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        output_video.release()

        # Frames skipped from video because automatic corner detection failed
        if skipped_frames:
            utils.show_warning("no_automatic_corners_online_video")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Chessboard vertical and horizontal squares and their size in mm
    chessboard_shape = (9, 6)
    chessboard_square_size = 25

    # Directories
    calibrations_path = "calibration_outputs"
    test_input_path = "evaluation_inputs"
    results_path = "evaluation_outputs"
    results_run_1_path = os.path.join(results_path, "Run 1")
    results_run_2_path = os.path.join(results_path, "Run 2")
    results_run_3_path = os.path.join(results_path, "Run 3")
    results_run_4_path = os.path.join(results_path, "Run 4")

    # Load calibration run 1 parameters
    err_1, mtx_1, dist_1, rvecs_1, tvecs_1,\
        _, _, _, _, _, _, _, _ = load_calibration_results(calibrations_path,
                                                          calibration_input_filename="calibration_run_1_results.npz")

    # Load calibration run 2 parameters
    err_2, mtx_2, dist_2, rvecs_2, tvecs_2, \
        _, _, _, _, _, _, _, _ = load_calibration_results(calibrations_path,
                                                          calibration_input_filename="calibration_run_2_results.npz")

    # Load calibration run 3 parameters
    err_3, mtx_3, dist_3, rvecs_3, tvecs_3, \
        _, _, _, _, _, _, _, _ = load_calibration_results(calibrations_path,
                                                          calibration_input_filename="calibration_run_3_results.npz")

    # Load calibration run 4 parameters (with images discarded from run 1)
    err_4, mtx_4, dist_4, rvecs_4, tvecs_4, \
        _, _, _, _, _, _, _, _ = load_calibration_results(calibrations_path,
                                                          calibration_input_filename="calibration_run_4_results.npz")

    # Evaluate run 1 parameters using test images
    test_camera_parameters_with_images(chessboard_shape, chessboard_square_size, mtx_1, dist_1, test_input_path,
                                       results_run_1_path, more_exact_corners=True, pnp_ransac=False)
    # Evaluate run 1 parameters using test videos
    test_camera_parameters_with_videos(chessboard_shape, chessboard_square_size, mtx_1, dist_1, test_input_path,
                                       results_run_1_path, more_exact_corners=True, pnp_ransac=False)

    # Evaluate run 2 parameters using test images
    test_camera_parameters_with_images(chessboard_shape, chessboard_square_size, mtx_2, dist_2, test_input_path,
                                       results_run_2_path, more_exact_corners=True, pnp_ransac=False)
    # Evaluate run 2 parameters using test videos
    test_camera_parameters_with_videos(chessboard_shape, chessboard_square_size, mtx_2, dist_2, test_input_path,
                                       results_run_2_path, more_exact_corners=True, pnp_ransac=False)

    # Evaluate run 3 parameters using test images
    test_camera_parameters_with_images(chessboard_shape, chessboard_square_size, mtx_3, dist_3, test_input_path,
                                       results_run_3_path, more_exact_corners=True, pnp_ransac=False)
    # Evaluate run 3 parameters using test videos
    test_camera_parameters_with_videos(chessboard_shape, chessboard_square_size, mtx_3, dist_3, test_input_path,
                                       results_run_3_path, more_exact_corners=True, pnp_ransac=False)

    # Evaluate run 4 parameters using test images
    test_camera_parameters_with_images(chessboard_shape, chessboard_square_size, mtx_4, dist_4, test_input_path,
                                       results_run_4_path, more_exact_corners=True, pnp_ransac=False)
    # Evaluate run 4 parameters using test videos
    test_camera_parameters_with_videos(chessboard_shape, chessboard_square_size, mtx_4, dist_4, test_input_path,
                                       results_run_4_path, more_exact_corners=True, pnp_ransac=False)
