import cv2
import numpy as np
import os
import json

max_height = 2560
max_width = 1920

def frame_capture(file, data):
  # Playing video from file:
	global max_width
	global max_height
  # WHEN ADDING DATA, CHANGE LINE BELOW
	path = "data/dfdc_train_part_40/"+file
	print("********************************* ",path)
	cap = cv2.VideoCapture(path)
	retval = os.getcwd()
	# print("Current working directory ", retval)
	currentFrame = 0
	i = 0
	# make the folder to put the frames in
	filename = file.split(".")
	type = data[file]["label"]
	os.mkdir('data/labelled_test/'+type+'/'+filename[0])
	# os.mkdir('data/test/'+filename[0])
	# print("Entered", type, filename[0])
	while(i < 10):
	# Capture frame by frame
		ret, frame = cap.read()
	# Only take the every 20 frames
		if currentFrame % 5 == 0:
			name = retval+'\\data\\labelled_test\\'+type+"\\"+filename[0]+'\\frame' + str(i) + '.jpg'
			# name = retval+'\\data\\test\\'+filename[0]+'\\frame' + str(i) + '.jpg'
			# print ('Creating...' + name)
			# find longer side
			# if frame.shape[1] > frame.shape[0]:
			# 	longer = 1
			# 	shorter = 0
			# else:
			# 	longer = 0
			# 	shorter = 1
			# ratio = 1000 / frame.shape[longer]
			# dim = [1, 2]
			# dim[longer] = int(frame.shape[shorter] * ratio)
			# dim[shorter] = 1000
			# dim = tuple(dim)
			# resized_img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			# resized_img = np.pad(frame, ((0,frame.shape[longer] - frame.shape[0]), (0,frame.shape[longer] - frame.shape[1]), (0,0)), 'constant', constant_values=0)
			# print(resized_img.shape)
			if frame.shape[0] > max_height:
				max_height = frame.shape[0]
			if frame.shape[1] > max_width:
				max_width = frame.shape[1]
			res = cv2.imwrite(name, frame)
			# print(res)
			i += 1
		currentFrame += 1

def get_all(data):
	for file in os.listdir("data/dfdc_train_part_40"):
		if file.endswith(".mp4"):
			frame_capture(file, data)

if __name__ == '__main__':
	# When adding data CHANGE LINE BELOW
	with open('./data/dfdc_train_part_40/metadata.json') as f:
		data = json.load(f)
	get_all(data)
	# res = get_all(None)
	print("Max Height",max_height)
	print("Max Width",max_width)
