import cv2
import numpy as np
import os
from copy import deepcopy
import tkinter as tk
import glob


VIDEO = False
square_size = 25
chessboard_shape = (9,6)
train_images_path = 'train_images2'
test_images_path = 'test_images'
cube_images_path = 'images_cube'
video_path = 'video/chessboard.mp4'
cube_video_path = 'video_cube/cube_video.mp4'
parameters_path = 'params'



def draw_axes(img, corners, imgpts):
    corner = tuple(int(i) for i in corners[0].ravel())
    img = cv2.arrowedLine(img, corner, tuple(int(i) for i in imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.arrowedLine(img, corner, tuple(int(i) for i in imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.arrowedLine(img, corner, tuple(int(i) for i in imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    img = cv2.drawContours(img, [imgpts[:4]],-1,(130, 249, 255),3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(130, 249, 255),3)
    img = cv2.drawContours(img, [imgpts[4:]],-1,(130, 249, 255),3)
    return img




not_found = []

# Load previously saved data
with np.load('calibrations/calibration_run_1_results.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cube = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],
                   [0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2] ])
    
    
    
if VIDEO == False:
    test_images_path_list = glob.glob(os.path.join(test_images_path,'*.jpg'))
    
    for image_path in test_images_path_list:
        print(image_path)
        # Read cuttenr image
        current_image = cv2.imread(image_path)
        # Reduce all the axes /2 
        current_image = cv2.resize(current_image, (0,0), fx=0.5, fy=0.5)
        # Turn the image into greyscale and try to detect the corners automatically
        gray = cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts_axis, jac_axis = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            imgpts_cube, jac_cube = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
            # Draw cube and axes
            img = draw_cube(current_image,corners2,imgpts_cube)
            img = draw_axes(current_image,corners2,imgpts_axis)

            cv2.imshow('img',img)
            cv2.waitKey(500)
            
            save_image_path = os.path.join(cube_images_path, os.path.basename(image_path))
            cv2.imwrite(save_image_path, img)
            
        else:
            not_found.append(str(image_path).split('\\')[1])
                     
    cv2.destroyAllWindows()
    
else:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*1)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*1)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used here is 'mp4v' for MP4
    out = cv2.VideoWriter(cube_video_path, fourcc, fps, (frame_width, frame_height))


    while True:
        ret, current_frame = cap.read()
        if not ret:
            break  # End of video

        # Resize for faster processing
        current_frame = cv2.resize(current_frame, (0,0), fx=1, fy=1)
        # Convert the picture to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Detect chessboard corners
        found, coords = cv2.findChessboardCorners(gray, chessboard_shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
        
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners2 = cv2.cornerSubPix(gray, coords, (11, 11), (-1, -1), criteria)
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            
            imgpts_axis, jac_axis = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            imgpts_cube, jac_cube = cv2.projectPoints(cube, rvecs, tvecs, mtx, dist)
            # Draw cube and axes
            frame = draw_cube(current_frame,corners2,imgpts_cube)
            frame = draw_axes(current_frame,corners2,imgpts_axis)
            
            # Write the frame into the file 'output_video.mp4'
            out.write(current_frame)

            cv2.imshow('img', current_frame)
            if cv2.waitKey(1) == 27:  # Exit if ESC is pressed
                break
     
    cap.release()
    out.release()       
    cv2.destroyAllWindows()
            

if not_found:
    formatted_filenames = " ,".join(not_found)

    message = f"Corners could not be found automatically in images:\n{formatted_filenames}"

    root = tk.Tk()
    root.title("Warning")

    label = tk.Label(root, text=message, padx=20, pady=20)
    label.pack()

    root.mainloop()

    
    
    
    
    