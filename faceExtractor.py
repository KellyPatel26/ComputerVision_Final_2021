import face_recognition
import os
import glob
import numpy as np 
from PIL import Image
import time

path = './val/'
paths = np.array(glob.glob(os.path.join(path, '*', '*', '*.jpg')))
paths.sort()
paths = paths.reshape(-1, 10) # used 5 frames for comparative time performance 
save = "./crop/"

all_times = []
w_max, h_max = 0, 0
for p in paths:
    xmin, ymin, xmax, ymax = float('inf'), float('inf'), -float('inf'), -float('inf')
    num = 0
    notSave = False
    datas = []
    for i, f in enumerate(p):
        print(f)
        image = face_recognition.load_image_file(f)
        h, w, c = image.shape
        start_time = time.time()
        face_locations = face_recognition.face_locations(image, model='cnn')
        end_time = time.time() 
        time_taken = end_time - start_time
        print(time_taken)
        all_times.append(time_taken)

        if i==0:
            num = len(face_locations)
        elif num!=len(face_locations):
            notSave = True
            print("face number mismatch")
            break
        #print(face_locations)
        for face in face_locations:
            height = face[2]-face[0]
            width = face[1]-face[3]
            # enlarge the bounding box by height and width
            ymin = min(max(face[0]-height//2, 0), ymin)
            xmax = max(min(face[1]+width//2, w), xmax)
            ymax = max(min(face[2]+height//2, h), ymax)
            xmin = min(max(face[3]-width//2, 0), xmin)
        if ymax-ymin>700 or xmax-xmin>700:
            notSave = True
            print("Size is too large")
            break
    if notSave:
        continue
    for i, f in enumerate(p):
        new_folder_path = os.path.join(save, os.path.split(f)[0])
        if not (os.path.isdir(new_folder_path)):
            os.makedirs(new_folder_path)
        im = Image.open(f)
        im = np.array(im)
        h, w, _ = im.shape
        yend, xend = min(ymin+700, h), min(xmin+700, w)
        im_ = np.zeros((700, 700, 3), dtype=np.uint8)
        im_[:yend-ymin,:xend-xmin,:] = im[ymin:yend,xmin:xend,:]   
        Image.fromarray(im_).save(os.path.join(save, f))
    w_max, h_max = max(xmax-xmin, w_max), max(ymax-ymin, h_max)
    print(xmax-xmin, ymax-ymin, w_max, h_max)
    #print(xmin, ymin, xmax, ymax)

print("avg time for cnn:", np.mean(np.array(all_times)))
print(len(all_times))