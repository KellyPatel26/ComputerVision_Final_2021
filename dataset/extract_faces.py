import face_recognition
import os
import glob
import numpy as np 
import cv2
import time

# Resources for cv2 code:
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81 - cv2 image 
# https://stackoverflow.com/questions/11537585/where-can-i-find-haar-cascades-xml-files - extracting haarcascade model 


# Load the Haar Cascade Algorithm, a pre-trained data of faces on opencv
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
trained_face_data = cv2.CascadeClassifier(haar_model)

# User defined settings
BBOX_MAX_WIDTH = 600 
BBOX_MAX_HEIGHT = 600  
NUM_FRAMES = 5 
CROPPED_IMGS_FOLDER_NAME = 'cropped'
DATA_FOLDER = './val/'

def get_mod_coord(coord, coord_dist, bbox_max):
    bbox_length = bbox_max - coord_dist
    half_bbox_length = bbox_length // 2 

    if bbox_length == 1: 
        mod_coord_start = mod_coord_start
        mod_coord_end = coord + 1
    else: 
        # expand bounding box such that mod_coord_end - mod_coord_start == bbox_max 
        mod_coord_start = coord - half_bbox_length 
        mod_coord_end = (coord + coord_dist) + half_bbox_length  

        if ( bbox_length % 2 ) != 0: # if bbox_length is odd, add (-) 1  
            a_pixel = 1 if ( bbox_length >= 0 ) else -1 
            mod_coord_end = mod_coord_end + a_pixel 

    return (mod_coord_start, mod_coord_end)


path = DATA_FOLDER
paths = np.array(glob.glob(os.path.join(path, '*', '*', '*.jpg')))
paths.sort()
paths = paths.reshape(-1, NUM_FRAMES)

for p in paths:
    for i, f in enumerate(p):
        # Choose an image to detect faces in
        img = cv2.imread(f)
    
        # Detect Faces on grayscaled image
        grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        # generate folder to store cropped / processed images
        curr_folder_path = f.rsplit('/', 1)[0][1:]
        new_folder_path = CROPPED_IMGS_FOLDER_NAME + curr_folder_path
        if not (os.path.isdir(new_folder_path)):
            os.makedirs(new_folder_path)

        if len(face_coordinates) == 0: # no face coordinates were found
            print("Face Coordinates Not Found")

        else:
            cropped_imgs = []
            for (x, y, w, h) in face_coordinates:

                # modify the face coordinates such that the face coordinates are BBOX_MAX_WIDTH x BBOX_MAX_HEIGHT 
                (mod_x, mod_x_w) = get_mod_coord(x, w, BBOX_MAX_WIDTH) 
                (mod_y, mod_y_h) = get_mod_coord(y, h, BBOX_MAX_HEIGHT)

                img_h, img_w, _ = img.shape
                
                # pad then crop the image 
                left_padding, right_padding, top_padding, bottom_padding = 0, 0, 0, 0
                if mod_x < 0: 
                    left_padding = 0 - mod_x
                    # shift coordinates 
                    mod_x_w = mod_x_w -left_padding
                    mod_x = 0
                if mod_y < 0:
                    top_padding = 0 - mod_y
                    # shift coordinates up
                    mod_y_h = mod_y_h - top_padding
                    mod_y = 0
                if mod_x_w > img_w : 
                    right_padding = mod_x_w - img_w
                if mod_y_h > img_h : 
                    bottom_padding = mod_y_h - img_h

                padded_img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                cropped_img = padded_img[mod_y:mod_y_h, mod_x:mod_x_w]
                cropped_imgs.append(cropped_img)

                # store cropped img
                num_cropped_imgs = len(cropped_imgs)
                for i in range(num_cropped_imgs):
                    cv2.imwrite(new_folder_path + '/face_' + str(i) + '.jpg', cropped_imgs[i])
