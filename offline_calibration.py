import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import random
import utils

# Global variables for manual corner detection callback
manual_corners = []
images_temp = []


def manual_corner_selection(event, x, y, flags, param):
    """
    Callback function to capture user's clicks during manual corner detection. Left-clicking places a corner on the
    window and right-clicking removes the last placed corner. Labels with numbering will be placed at every placed
    corner. When more than 1 corner is placed then the corners will be connected by a line in order of placement.

    :param event: type of event
    :param x: X coordinate of event (click)
    :param y: Y coordinate of event (click)
    :param flags: not used in application
    :param param: not used in application
    """
    global manual_corners, images_temp

    # Left click to add a corner if not all 4 placed
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_corners) < 4:
        # Add corner to list of corners
        manual_corners.append([x, y])

        # Copy last frame to make changes to it
        current_image = deepcopy(images_temp[-1])

        # Add text near click
        cv2.putText(current_image, str(len(manual_corners)), manual_corners[-1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # Draw circle at click
        cv2.circle(current_image, manual_corners[-1],
                   radius=3, color=(0, 255, 0), thickness=-1)

        # Connect corners with line
        if len(manual_corners) > 1:
            cv2.line(current_image, manual_corners[-2], manual_corners[-1],
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if len(manual_corners) == 4:
            cv2.line(current_image, manual_corners[0], manual_corners[-1],
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Save new frame
        images_temp.append(current_image)
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", current_image)
    # Right click to remove a corner if one already is placed
    elif event == cv2.EVENT_RBUTTONDOWN and len(manual_corners) > 0:
        # Removed last placed corner and return to previous frame
        manual_corners.pop()
        images_temp.pop()
        current_image = images_temp[-1]
        cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", current_image)


def sort_corners_clockwise(corners, origin="top-left"):
    """
    Sorts given corner coordinates in clockwise order.

    :param corners: array of corner points ([x, y])
    :param origin: which corner point starts the clockwise order (bottom-left, top-left, top-right, or bottom-right)
    :return: returns array of sorted corners
    """
    # Calculate the centroid of the corners
    centroid = np.mean(corners, axis=0)

    # Sort corners and determine their relative position to the centroid
    top = sorted([corner for corner in corners if corner[1] < centroid[1]], key=lambda point: point[0])
    bottom = sorted([corner for corner in corners if corner[1] >= centroid[1]], key=lambda point: point[0],
                    reverse=True)

    # Sort top and bottom corners depending on first element
    if origin == "top-left":
        return np.array(top + bottom, dtype="float32")
    elif origin == "top-right":
        return np.array([top[1]] + bottom + [top[0]], dtype="float32")
    elif origin == "bottom-right":
        return np.array(bottom + top, dtype="float32")
    else:
        return np.array([bottom[1]] + top + [bottom[0]], dtype="float32")


def interpolate_points_from_manual_corners(corners, chessboard_shape, use_perspective_transform=True):
    """
    Interpolates chessboard points from the 4 manual corners given by the user. Function supports flat interpolation
    and interpolation using perspective transform for enhanced accuracy (use_perspective_transform parameter).

    :param corners: array of 4 corner points ([x, y])
    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param use_perspective_transform: interpolates using homogenous coordinates and perspective transform for enhanced
                                      accuracy if set to True, otherwise uses flat interpolation
    :return: returns array of 2D interpolated image points or None if wrong number of corners given (unequal to 4)
    """
    if len(corners) != 4:
        utils.show_warning("incorrect_num_corners")
        return None

    # Sort corners to (top-left, top-right, bottom-right, bottom-left)
    corners = sort_corners_clockwise(corners, origin="top-left")

    # Calculate the maximum width and height
    max_width = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[2]))
    max_height = max(np.linalg.norm(corners[2] - corners[1]), np.linalg.norm(corners[3] - corners[0]))

    # Horizontal and vertical step calculation using chessboard shape
    horizontal_step = max_width / (chessboard_shape[1] - 1)
    vertical_step = max_height / (chessboard_shape[0] - 1)

    interpolated_row = []
    interpolated_points = []
    # Perform perspective transform for accuracy improvement
    if use_perspective_transform:
        # Use maximum width and height to form destination coordinates for perspective transform
        dest_corners = np.float32([[0, 0], [max_width - 1, 0],
                                   [max_width - 1, max_height - 1], [0, max_height - 1]])
        p_matrix = cv2.getPerspectiveTransform(corners, dest_corners)

        # Get inverse matrix for projecting points from the transformed space back to the original image space
        inverted_p_matrix = np.linalg.inv(p_matrix)

        # Compute each projected point
        for y in range(0, chessboard_shape[0]):
            for x in range(0, chessboard_shape[1]):
                # Calculate the position of the current point relative to the grid using homogenous coordinates
                point = np.array([x * horizontal_step, y * vertical_step, 1])

                # Multiply with inverse matrix to project point from transformed space back to original image space
                point = np.matmul(inverted_p_matrix, point)

                # Divide point by its Z
                point /= point[2]

                # Append the X and Y of point to the list of interpolated points in row
                interpolated_row.append(point[:2])
            # Append interpolated points in row to interpolated points
            interpolated_points.append(interpolated_row)
            interpolated_row = []
    # Flat interpolation
    else:
        for y in range(0, chessboard_shape[0]):
            for x in range(0, chessboard_shape[1]):
                # Calculate the position of the current point relative to the grid
                point = np.array([x * horizontal_step, y * vertical_step])

                # Interpolate the position of the current point between the known corners
                point = corners[0] + point

                # Append the point to the list of interpolated points in row
                interpolated_row.append(point)
            # Append interpolated points in row to interpolated points
            interpolated_points.append(interpolated_row)
            interpolated_row = []

    # If change_point_order is True then point order will start from bottom-left and end on top-right
    # moving through rows before changing column
    # if False then point order will start at top-left and end on bottom-right
    # moving through columns before changing row as already saved
    interpolated_points = np.array(interpolated_points, dtype="float32")
    if chessboard_shape[0] > chessboard_shape[1]:
        interpolated_points = np.flip(interpolated_points, axis=0)
        interpolated_points = np.transpose(interpolated_points, (1, 0, 2))

    # Return (MxN, 1, 2) array to match automatic corner detection output
    return np.reshape(interpolated_points, (-1, 1, 2))


def extract_image_points(chessboard_shape, images_input_path="calibration_inputs", more_exact_corners=True,
                         result_time_visible=1000):
    """
    Parses training images from given directory and detects chessboard corners used for camera calibration. If automatic
    corner detection fails, the user is given instructions to perform manual corner selection by clicking on a prompt
    window of the image.

    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param images_input_path: training images directory path
    :param more_exact_corners: increases corner accuracy with more exact corner positions if set to True, otherwise
                               keeps original corner positions.
    :param result_time_visible: milliseconds to keep result of corner extraction for an image to screen, 0 to wait for
                                key press
    :return: returns array of 2D chessboard image points for every training image, array of booleans of whether each
             training image's points were detected automatically, and shape of the images
    """
    global manual_corners, images_temp

    # Check that images are present
    images_path_list = [os.path.join(images_input_path, file) for file in os.listdir(images_input_path)
                        if file.endswith(".jpg")]
    if not images_path_list:
        utils.show_warning("train_empty")
        return None
    # Sort images
    images_path_list = sorted(images_path_list, key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    # Format and get dimensions for images
    image_shape = utils.uniform_image_dimensions(images_path)

    # Loop for every image to find points
    image_points = []
    automatic_detections = []
    for image_path in images_path_list:
        # Read image and resize for viewing
        current_image = cv2.imread(image_path)
        current_image = cv2.resize(current_image, (0, 0), fx=0.5, fy=0.5)
        if current_image is None:
            utils.show_warning("image_none")
            continue

        # Try to detect the chessboard corners automatically using grayscale image and all flags
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                                               cv2.CALIB_CB_FILTER_QUADS +
                                                                               cv2.CALIB_CB_FAST_CHECK)

        # No automatic corner detection, going to manual corner selection
        if not ret:
            utils.show_warning("no_automatic_corners")

            # Corner selection window, calls callback function
            images_temp.append(deepcopy(current_image))
            cv2.imshow("Manual Corners Selection (Press any key when all corners selected)", current_image)
            cv2.setMouseCallback("Manual Corners Selection (Press any key when all corners selected)",
                                 manual_corner_selection)
            # Loop until 4 corners selected and any key is pressed
            while True:
                cv2.waitKey(0)
                if len(manual_corners) == 4:
                    cv2.destroyAllWindows()
                    break

            # Corner interpolation using selected corners
            corners = interpolate_points_from_manual_corners(manual_corners, chessboard_shape)

            # Reset parameters used for manual corner detection and go back to original image
            current_image = images_temp[0]
            images_temp = []
            manual_corners = []

        # Increase corner accuracy with more exact corner positions
        if more_exact_corners:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # Store extracted image corners
        image_points.append(corners)
        automatic_detections.append(ret)

        # Draw extracted corners on image
        cv2.drawChessboardCorners(current_image, chessboard_shape, corners, True)

        # Show results of extraction
        cv2.imshow("Extracted Chessboard Corners", current_image)
        cv2.waitKey(result_time_visible)
        cv2.destroyAllWindows()

    return image_points, automatic_detections, image_shape


def sample_image_points(image_points, automatic_detections, max_automatic=20, max_manual=5, seed=0):
    """
    Samples a subset of image points by keeping only image points from a subset of the total images.

    :param image_points: 2D chessboard image points for every training image
    :param automatic_detections: array of booleans of whether each training image's points were detected automatically
    :param max_automatic: max images where points were detected automatically
    :param max_manual: max images where points were detected manually
    :param seed: seed for randomization
    :return: returns array of sampled 2D chessboard automatic detection image points, array of sampled 2D chessboard
             manual detection image points, the ids of the images that were sampled in the original image points array
    """
    # Seed for randomization
    random.seed(seed)

    # Split images where points were detected automatically and manually and shuffle each split
    # Keep max automatic detection images
    image_points_automatic = [points for points, automatic in zip(image_points, automatic_detections) if automatic]
    random.shuffle(image_points_automatic)
    image_points_automatic = random.sample(image_points_automatic, min(max_automatic, len(image_points_automatic)))

    # Keep max manual detection images
    image_points_manual = [points for points, automatic in zip(image_points, automatic_detections) if not automatic]
    random.shuffle(image_points_manual)
    image_points_manual = random.sample(image_points_manual, min(max_manual, len(image_points_manual)))

    # Get the image ids of the sampled points in the original array and preserve shuffle order with automatic first
    image_points_tuples = [tuple(points.flatten()) for points in image_points]
    image_points_automatic_tuples = [tuple(points.flatten()) for points in image_points_automatic]
    image_points_manual_tuples = [tuple(points.flatten()) for points in image_points_manual]
    image_points_idx = []
    for points in image_points_automatic_tuples + image_points_manual_tuples:
        for idx, points_original in enumerate(image_points_tuples):
            if points == points_original:
                image_points_idx.append(idx)

    return image_points_automatic, image_points_manual, image_points_idx


def output_extracted_image_points(image_points, automatic_detections, image_shape, output_path="calibration_outputs",
                                  output_filename="extracted_image_points.npz"):
    """
    Outputs image point extraction outputs to file.

    :param image_points: 2D chessboard image points for every training image
    :param automatic_detections: array of booleans of whether each training image's points were detected automatically
    :param image_shape: shape of the images
    :param output_path: output directory path
    :param output_filename: output file name (including extension)
    """
    np.savez(os.path.join(output_path, output_filename),
             image_points=image_points, automatic_detections=automatic_detections, image_shape=image_shape)


def load_extracted_image_points(input_path="calibration_outputs", input_filename="extracted_image_points.npz"):
    """
    Loads image point extraction outputs from file.

    :param input_path: input directory path
    :param input_filename: input file name (including extension)
    :return: returns loaded 2D chessboard image points for every training image, array of booleans of whether each
             training image's points were detected automatically, shape of the images
    """

    # Load the saved image points
    data = np.load(os.path.join(input_path, input_filename))

    # Retrieve the arrays from the loaded data
    image_points = data["image_points"]
    automatic_detections = data["automatic_detections"]
    image_shape = data["image_shape"]

    return image_points, automatic_detections, image_shape


def calibrate_camera(image_points, image_shape, chessboard_shape, chessboard_square_size=1, print_results=True):
    """
    Calibrates camera using extracted image points and calculated object points by estimating the camera parameters.

    :param image_points: array of 2D chessboard image points for every image
    :param image_shape: shape of the images
    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param chessboard_square_size: chessboard square size in mm
    :param print_results: prints results of calibration if True
    :return: returns mean reprojection error, camera matrix, distortion coefficients, rotation vector, translation
             vector, standard deviation over intrinsic parameters, standard deviation over extrinsic parameters,
             per view reprojection error
    """
    # Prepare object points [[0,0,0], [1,0,0], [2,0,0] ...,[chessboard_vertical, chessboard_horizontal,0]]
    # Multiply by chessboard square size in mm, keep X, Y (Z=0, stationary XY plane chessboard)
    object_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) \
                           * chessboard_square_size
    # Repeat object points for every image
    object_points = [object_points for _ in range(len(image_points))]

    # Calibrate camera using object points and image points
    # Get mean reprojection error, camera matrix, distortion coefficients, rotation and translation vectors
    # Extended function also gives std deviation over intrinsic parameters and extrinsic parameters, per view error
    err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err = cv2.calibrateCameraExtended(object_points,
                                                                                             image_points,
                                                                                             image_shape,
                                                                                             None, None)
    # Print results of calibration
    if print_results:
        print("Camera Matrix:")
        print(mtx)
        print("Mean Reprojection Error:", f"{err:.2f}")
        print("Focal Length Fx:", f"{mtx[0][0]:.2f} ± {in_std[0][0]:.2f}")
        print("Focal Length Fy:", f"{mtx[1][1]:.2f} ± {in_std[1][0]:.2f}")
        print("Center Point Cx:", f"{mtx[0][2]:.2f} ± {in_std[2][0]:.2f}")
        print("Center Point Cy:", f"{mtx[1][2]:.2f} ± {in_std[3][0]:.2f}")
        print()

    return err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err


def discard_bad_image_points(image_points, image_shape, chessboard_shape, chessboard_square_size=1,
                             discard_threshold=0.15):
    """
    Discards image points from images that make calibration performance worse. Calibrates camera using all image points
    and uses the mean returned reprojection error as baseline. Then iteratively goes through all possible combinations
    by excluding 1 image's points from the image points set to compare against all image points used. If keeping that
    image's points makes the calibration significantly worse given a threshold, then those points are discarded.

    :param image_points: array of 2D chessboard image points for every image
    :param image_shape: shape of the images
    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param chessboard_square_size: chessboard square size in mm
    :param discard_threshold: threshold of error improvement when an image's points are excluded to discard them
    :return: returns array of 2D chessboard image points for every kept image, array of kept image point indexes,
             array of 2D chessboard image points for every discarded image, and array of discarded image point indexes
    """
    # Calibrate camera using all image points and use the mean returned reprojection error as baseline
    baseline_ret, _, _, _, _, _, _, _ = calibrate_camera(image_points, image_shape, chessboard_shape,
                                                         chessboard_square_size, print_results=False)

    # Go through all possible combinations by excluding 1 image from image set to compare against all images used
    kept_image_points = []
    kept_image_points_idx = []
    discarded_image_points = []
    discarded_image_points_idx = []
    for i in range(len(image_points)):
        image_points_excluding_1 = deepcopy(image_points)
        excluded = image_points_excluding_1.pop(i)
        ret, _, _, _, _, _, _, _ = calibrate_camera(image_points_excluding_1, image_shape, chessboard_shape,
                                                    chessboard_square_size, print_results=False)

        # Made performance worse given a threshold means image points will be discarded, otherwise kept
        if baseline_ret - ret >= discard_threshold:
            discarded_image_points.append(excluded)
            discarded_image_points_idx.append(i)
        else:
            kept_image_points.append(excluded)
            kept_image_points_idx.append(i)

    return kept_image_points, kept_image_points_idx, discarded_image_points, discarded_image_points_idx


def output_calibration_results(err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err, image_points, image_shape,
                               chessboard_shape, chessboard_square_size, run,
                               calibration_output_path="calibration_outputs",
                               calibration_output_filename="calibration_results.npz"):
    """
    Outputs results of camera calibration and info to file.

    :param err: mean reprojection error
    :param mtx: camera matrix
    :param dist: distortion coefficient
    :param rvecs: rotation vector
    :param tvecs: translation vector
    :param in_std: standard deviation over intrinsic parameters
    :param ex_std: standard deviation over extrinsic parameters
    :param per_view_err: per view reprojection error
    :param image_points: array of 2D chessboard image points for every image
    :param image_shape: shape of the images
    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param chessboard_square_size: chessboard square size in mm
    :param run: run number
    :param calibration_output_path: calibration output directory path
    :param calibration_output_filename: calibration output file name (including extension)
    """
    if not os.path.isdir(calibration_output_path):
        os.makedirs(calibration_output_path)
    np.savez(os.path.join(calibration_output_path, calibration_output_filename), err=err, mtx=mtx,
             dist=dist, rvecs=rvecs, tvecs=tvecs, in_std=in_std, ex_std=ex_std, per_view_err=per_view_err,
             image_points=image_points, image_shape=image_shape,
             chessboard_shape=chessboard_shape, chessboard_square_size=chessboard_square_size,
             run=run)


def plot_calibration_results(errs, per_view_errs, mtxs, in_stds, plot_output_path="plots",
                             plot_output_filename="intrinsic_params_runs_comparison.png"):
    """
    Plots results of camera calibration. Supports multiple calibration runs.

    :param errs: array of mean reprojection errors (1 position for every run)
    :param per_view_errs: array of per view reprojection errors (1 position for every run)
    :param mtxs: array of camera matrices (1 position for every run)
    :param in_stds: array of standard deviations over intrinsic parameters (1 position for every run)
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    if len(errs) != len(per_view_errs) != len(mtxs) != len(in_stds):
        utils.show_warning("calibration_results_unequal")
        return

    # Extract intrinsic parameters
    fx = [mtx[0][0] for mtx in mtxs]
    fy = [mtx[1][1] for mtx in mtxs]
    cx = [mtx[0][2] for mtx in mtxs]
    cy = [mtx[1][2] for mtx in mtxs]

    # Extract standard deviations for intrinsic parameters
    fx_stds = [in_std[0][0] for in_std in in_stds]
    fy_stds = [in_std[1][0] for in_std in in_stds]
    cx_stds = [in_std[2][0] for in_std in in_stds]
    cy_stds = [in_std[3][0] for in_std in in_stds]

    # Create calibration run labels
    calibration_runs = ["Calibration " + str(i) for i in range(len(errs))]

    # Plot with 3 rows 2 columns
    fig, ax = plt.subplots(3, 2, figsize=(14, 16))

    # Colormap for plots
    cmap = plt.get_cmap("brg")
    colors = cmap(np.linspace(0, 1, len(errs)))

    # Plot mean reprojection error
    ax[0, 0].set_title("Mean Reprojection Error")
    ax[0, 0].bar(calibration_runs, errs, color=colors, label=[f"{err:.2f}" for err in errs])
    ax[0, 0].legend()

    # Plot reprojection error for every image
    ax[0, 1].set_title("Per View Reprojection Error")
    for run in range(len(errs)):
        ax[0, 1].scatter([str(i) for i in range(len(per_view_errs[run]))], per_view_errs[run],
                         color=colors[run], label=calibration_runs[run])
    ax[0, 1].legend()

    # Plot focal length Fx
    ax[1, 0].set_title("Focal Length Fx")
    labels = [f"{focal_length:.2f} ± {std:.2f}" for focal_length, std in zip(fx, fx_stds)]
    ax[1, 0].set_title("Focal Length Fx")
    for run in range(len(errs)):
        ax[1, 0].errorbar(calibration_runs[run], fx[run], yerr=fx_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[1, 0].legend()

    # Plot focal length Fy
    ax[1, 1].set_title("Focal Length Fy")
    labels = [f"{focal_length:.2f} ± {std:.2f}" for focal_length, std in zip(fy, fy_stds)]
    for run in range(len(errs)):
        ax[1, 1].errorbar(calibration_runs[run], fy[run], yerr=fy_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[1, 1].legend()

    # Plot center point Cx
    ax[2, 0].set_title("Center Point Cx")
    labels = [f"{center:.2f} ± {std:.2f}" for center, std in zip(cx, cx_stds)]
    for run in range(len(errs)):
        ax[2, 0].errorbar(calibration_runs[run], cx[run], yerr=cx_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[2, 0].legend()

    # Plot center point Cy
    ax[2, 1].set_title("Center Point Cy")
    labels = [f"{center:.2f} ± {std:.2f}" for center, std in zip(cy, cy_stds)]
    for run in range(len(errs)):
        ax[2, 1].errorbar(calibration_runs[run], cy[run], yerr=cy_stds[run], fmt="o",
                          color=colors[run], label=labels[run])
    ax[2, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Output plot to file
    if not os.path.isdir(plot_output_path):
        os.makedirs(plot_output_path)
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))

    # Close figure
    plt.close()


if __name__ == "__main__":
    # Chessboard vertical and horizontal squares and their size in mm
    chessboard_shape = (9, 6)
    chessboard_square_size = 25

    # Directories
    images_path = "calibration_inputs"
    plots_path = "calibration_plots"
    calibrations_path = "calibration_outputs"

    # Extract image points from training images
    image_points, automatic_detections, image_shape = extract_image_points(chessboard_shape, images_path,
                                                                           more_exact_corners=True,
                                                                           result_time_visible=1000)

    # Save extracted image points to file
    output_extracted_image_points(image_points, automatic_detections, image_shape, calibrations_path,
                                  output_filename="extracted_image_points.npz")

    # Sort the selected 25 image sets to have images where points were detected automatically first
    image_points_automatic = [points for points, automatic in zip(image_points, automatic_detections) if automatic]
    image_points_manual = [points for points, automatic in zip(image_points, automatic_detections) if not automatic]
    image_points_25 = image_points_automatic + image_points_manual
    # Filter 10 and 5 image sets from images where points were detected automatically for runs 2 and 3
    # Run 3 sets are a subset of run 2 sets which are a subset of run 1 sets for automatic points
    image_points_10 = image_points_automatic[:10]
    image_points_5 = image_points_automatic[:5]

    # Image points for every run
    image_points_runs = [image_points_25, image_points_10, image_points_5]
    # Run calibrations with selected image points
    errs = []
    per_view_errs = []
    mtxs = []
    in_stds = []
    for run in range(3):
        print("Running Calibration Run " + str(run + 1) + " with " + str(len(image_points_runs[run])) + " images.")
        err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err = calibrate_camera(image_points_runs[run],
                                                                                      image_shape,
                                                                                      chessboard_shape,
                                                                                      chessboard_square_size,
                                                                                      print_results=True)
        # Save run results
        errs.append(err)
        per_view_errs.append(per_view_err)
        mtxs.append(mtx)
        in_stds.append(in_std)
        # Output calibration run results to file
        output_calibration_results(err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err,
                                   image_points_runs[run], image_shape, chessboard_shape, chessboard_square_size,
                                   run, calibrations_path,
                                   calibration_output_filename="calibration_run_" + str(run + 1) + "_results.npz")

    # Plot results from all calibration runs
    plot_calibration_results(errs, per_view_errs, mtxs, in_stds, plots_path,
                             plot_output_filename="intrinsic_params_runs_comparison.png")

    # Discarding bad image points from the 25 images of run 1
    print("Discarding images from Run 1 with bad calibration performance.")
    kept_image_points, kept_image_points_idx, \
        discarded_image_points, discarded_image_points_idx = discard_bad_image_points(image_points_25,
                                                                                      image_shape,
                                                                                      chessboard_shape,
                                                                                      chessboard_square_size,
                                                                                      discard_threshold=0.15)

    # Run calibrations with kept image points
    print("Running Calibration Run 4 with " + str(len(kept_image_points)) + "/" + str(len(image_points_runs[0])) +
          " Run 1 images after discarding.")
    err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err = calibrate_camera(kept_image_points,
                                                                                  image_shape,
                                                                                  chessboard_shape,
                                                                                  chessboard_square_size,
                                                                                  print_results=True)

    # Output calibration run results to file
    output_calibration_results(err, mtx, dist, rvecs, tvecs, in_std, ex_std, per_view_err,
                               kept_image_points, image_shape, chessboard_shape, chessboard_square_size,
                               4, calibrations_path, calibration_output_filename="calibration_run_4_results.npz")

    # Map None to per view error where image points were removed from the original 25 images for plotting
    per_view_err_map = np.full((len(image_points_25), 1), [None])
    per_view_err_map[kept_image_points_idx] = per_view_err[:len(kept_image_points_idx)]

    # Plot results with and without discarded image points
    plot_calibration_results([errs[0], err], [per_view_errs[0], per_view_err_map], [mtxs[0], mtx], [in_stds[0], in_std],
                             plots_path, plot_output_filename="intrinsic_params_discard_bad_images.png")
