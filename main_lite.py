import numpy as np
import tensorflow as tf
from PIL import Image
import math
import cv2
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_float.tflite")

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

inp_mean=127.5
inp_std=127.5

#set the path to video file
path_to_video='test2.mp4'

'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Tflite.mp4',fourcc, 10.0, (640,480))
'''
#read a video file
cap = cv2.VideoCapture(path_to_video)
while True:

			#read frames 
	ret, img2 = cap.read()
		
	t_in = cv2.getTickCount()
	if ret:  
		img=cv2.resize(img2,(300,300))
		
		'''
		# convert for int 

		#dont forget to change file
		input_data = np.array(img).astype(np.uint8)
		'''
		
		### convert for float graph
		#convert image from 0:255 to -1 : 1
		input_data = (abs((np.array(img) - inp_mean) / inp_std) -1)  .astype(np.float32) 

		input_data = np.expand_dims(input_data, axis=0)

		# Test model on random input data.
		interpreter.set_tensor(input_details[0]['index'], input_data)
		tin=time.time()
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		predictions = np.squeeze( interpreter.get_tensor(output_details[0]['index']))
		output_classes = np.squeeze( interpreter.get_tensor(output_details[1]['index']))
		confidence_scores = np.squeeze( interpreter.get_tensor(output_details[2]['index']))

		
		for i,newbox in enumerate (predictions):
			if confidence_scores[i] > 0.1 : 
				val=np.asarray(newbox)
				y_min =  int(val[0] *480)
				
				y_max= int(val[2] *480)
				
				x_min= int(val[1] *640) 

				x_max = int(val[3]  *640)
				cv2.rectangle(np.asarray(img2), (x_min,y_min), (x_max,y_max) , (0,255,0), 2, 1)
			
				print(x_min,y_min,x_max,y_max)
		fps=round (cv2.getTickFrequency() / (cv2.getTickCount() - t_in) , 2)				
		cv2.putText(img2,'FPS : {}  '.format(fps),(280,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), lineType=cv2.LINE_AA) 
		cv2.imshow(' ', np.asarray(img2))
		#out.write(img2)		
		cv2.waitKey(1)	
	else :
		break
