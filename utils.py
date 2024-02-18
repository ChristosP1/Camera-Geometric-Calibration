import cv2
import numpy as np
import os
import tkinter as tk


def show_warning(message_id):
    """
    Shows warnings to the user during execution of the program. Possible warnings include:
    - train_empty for empty training image folder
    - test_empty for empty testing folder
    - images_need_crop for images without equal dimensions
    - image_none for image that fails to be loaded
    - video_none for video that fails to be played
    - incorrect_num_corners for wrong number of chessboard corners
    - no_automatic_corners for automatic corner detection failure
    - no_automatic_corners_online for automatic corner detection failure in online phase
    - no_automatic_corners_online_video for automatic corner detection failure in online phase during video processing
    - calibration_results_unequal for array length of camera calibration results being unequal

    :param message_id: message ID string to print appropriate message
    """
    # Define the possible messages
    messages = {
        "train_empty": "Calibration image folder is missing the relevant files!",
        "test_empty": "Test folder is missing the relevant files!",
        "images_need_crop": "Not all images have the same dimensions! Images will be cropped!",
        "image_none": "Image could not be loaded and will be skipped!",
        "video_none": "Video could not be played and will be skipped!",
        "incorrect_num_corners": "Incorrect number of corners given!",
        "no_automatic_corners": "Corners not detected automatically! Need to extract manually! " +
                                "Select the 4 corners with left clicks and then press any key to continue. " +
                                "To undo selections in order use right clicks.",
        "no_automatic_corners_online": "Corners not detected automatically! Image will be discarded from testing!",
        "no_automatic_corners_online_video": "Corners not detected automatically for some frames! Frames were skipped",
        "calibration_results_unequal": "Plotting error, array lengths of camera calibration results are not the same!"
    }

    # Fetch the appropriate message
    message = messages.get(message_id, "Unknown Warning")

    # Create the main window
    root = tk.Tk()
    root.title("Warning")

    # Create a label for the message
    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()

    # Start the GUI event loop
    root.mainloop()


def uniform_image_dimensions(directory_path):
    """
    Checks whether all images in given directory have the same width and height dimensions. Crops images to match
    minimum dimensions found if not.

    :param directory_path: directory path
    :return: returns the final shape of the images (original if all the same, otherwise cropped) or None if no images
             present in directory
    """
    # Get list of image file paths
    image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".jpg")]
    if not image_paths:
        return None

    # Initialize variables to store the dimensions
    min_width, min_height = np.inf, np.inf
    dimensions_set = set()

    # First pass to find dimensions and check uniformity
    img = None
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            dimensions_set.add((h, w))
            min_width, min_height = min(min_width, w), min(min_height, h)

    # Check if all images have the same dimensions
    if len(dimensions_set) == 1:
        return img.shape[:2]
    else:
        show_warning("images_need_cropping")

    # Second pass to crop images
    cropped_img = None
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            if h > min_height or w > min_width:
                # Calculate crop dimensions
                top = (h - min_height) // 2
                bottom = h - min_height - top
                left = (w - min_width) // 2
                right = w - min_width - left

                # Crop and save the image
                cropped_img = img[top:h - bottom, left:w - right]
                cv2.imwrite(image_path, cropped_img)

    return cropped_img.shape[:2]