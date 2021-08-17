import sys
path_to_darknet = 'E:/MachineLearning/YOLOv4/darknet/build/darknet/x64'
sys.path.insert(1, path_to_darknet)

from network_generator import DarknetNetwork, convertToXYminmax, drawBox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Controller import Controller
from time import perf_counter

configPath = path_to_darknet + "/cfg/yolov4_dw_test.cfg"                                 # Path to cfg
weightPath = path_to_darknet + "/backup/yolov4_dw_train_last.weights"                                 # Path to weights
metaPath = path_to_darknet + "/cfg/drivingWheel.data"    



# display states on the screen
def displayStates(img, states):
	h,w = img.shape[:2]
	cv2.putText(img, 'Angle: {:.2f} deg'.format(states['angle']) , (5, h-10), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                      (255, 241, 43), 2)
	cv2.putText(img, '{}: {}'.format('Brake', 'ON' if states['brake'] else 'OFF') , (5, h-35), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                      (255, 241, 43), 2)
	cv2.putText(img, '{}: {}'.format('Nitro', 'ON' if states['nitro'] else 'OFF') , (5, h-60), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                      (255, 241, 43), 2)
	if states['center'] is not None:
		cv2.circle(img, states['center'], radius=0, color=(242, 0, 36), thickness=2)
		cv2.line(img, (0,states['center'][1]), (w,states['center'][1]), (255, 241, 43), 1)	# horz line
		cv2.line(img, (states['center'][0],0), (states['center'][0],h), (255, 241, 43), 1)	# vert line
		cv2.line(img, states['center'], (states['center'][0]+w,states['center'][1]+int(w*np.tan(np.deg2rad(states['angle'])))), (3, 23, 252), 2) # angle line
		cv2.ellipse(img, states['center'], (30,30), 0, 0, states['angle'], (3, 23, 252), 3)

	return img


network = DarknetNetwork(configPath, weightPath, metaPath)
control = Controller()

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
frame_height = int(cap.get(4))
streaming = True
last_time = 0
cur_time = 0
while streaming:
	cur_time = perf_counter()
	if cur_time - last_time > 0.03: 
		ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
		# For Assertion Failed Error in OpenCV
		if not ret:                                                  # Check if frame present otherwise he break the while loop
			break
		frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
		frame_rgb = cv2.flip(frame_rgb,1) 							 # flip the frame horizontally
		detections = network.get_predictions(frame_rgb)
		frame_rgb = drawBox(frame_rgb, detections)
		states = control.updateStates(detections)
		frame_rgb = displayStates(frame_rgb, states)
		frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

		cv2.imshow('Stream', frame)                                    # Display Image window
		if cv2.waitKey(1) == 13:								 # 'Enter' key
			streaming = False
	last_time = cur_time


cap.release()
control.terminate()
print("Driving wheel emulation completed")



