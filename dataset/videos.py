import cv2
import numpy as np
import os
import json


def frame_capture(file, data):
  # Playing video from file:
  # WHEN ADDING DATA, CHANGE LINE BELOW
	path = "data/dfdc_train_part_1/"+file
	cap = cv2.VideoCapture(path)
	retval = os.getcwd()
	print("Current working directory ", retval)
	currentFrame = 0
	i = 0
	# make the folder to put the frames in
	filename = file.split(".")
	type = data[file]["label"]
	os.mkdir('data/train/'+type+'/'+filename[0])

	while(i < 10):
	  # Capture frame by frame
		ret, frame = cap.read()
	  # Only take the every 20 frames
		if currentFrame % 20 == 0:
			name = retval+'\\data\\train\\'+type+"\\"+filename[0]+'\\frame' + str(i) + '.jpg'
			print ('Creating...' + name)
			# find longer side
			if frame.shape[1] > frame.shape[0]:
				longer = 1
				shorter = 0
			else:
				longer = 0
				shorter = 1
			ratio = 224 / frame.shape[longer]
			dim = [1, 2]
			dim[longer] = int(frame.shape[shorter] * ratio)
			dim[shorter] = 224
			dim = tuple(dim)
			resized_img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			resized_img = np.pad(resized_img, ((0,224 - resized_img.shape[0]), (0,224 - resized_img.shape[1]), (0,0)), 'constant', constant_values=0)
			# print(resized_img.shape)
			res = cv2.imwrite(name, resized_img)
			i += 1
		currentFrame += 1

def get_all(data):
	for file in os.listdir("data/dfdc_train_part_1"):
		if file.endswith(".mp4"):
			frame_capture(file, data)


if __name__ == '__main__':
	# When adding data CHANGE LINE BELOW
	with open('./data/dfdc_train_part_1/metadata.json') as f:
		data = json.load(f)
	get_all(data)