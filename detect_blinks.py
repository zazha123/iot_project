# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import datetime
import dlib
import cv2
import matplotlib.pyplot as plt


# TODO: when there is no face in the first second, the program crashes
# TODO: live

SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'

# you should set the time interval here
# 30 frames / s
FRAME_NUM = 100

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
# TODO: you must set the right thresh and frames to get the right number of eyeblinks.
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 8

# ear list for each frame
ear_list = []

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize the frame counters of open and closed of eyes
CLOSED_COUNT = 0
OPEN_COUNT = 0
EYEBLINK_COUNT = 0

# use frame to calculate the perclos
FRAME_COUNT = 0
DETECTED_COUNT = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")

# if use file, please use these lines.
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
# vs = VideoStream(src=0).start()

# if use streaming, please use these lines
vs = VideoStream(usePiCamera=False).start()
fileStream = False

# loop over frames from the video stream

# start = datetime.datetime.now()

while True:

	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	FRAME_COUNT += 1

	# loop over the face detections
	for rect in rects:
		DETECTED_COUNT += 1
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		ear_list.append(ear)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			cv2.putText(frame, 'close eyes', (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			COUNTER += 1
			CLOSED_COUNT += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			OPEN_COUNT += 1
			cv2.putText(frame, 'open eyes', (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				EYEBLINK_COUNT += 1

			# reset the eye frame counter
			COUNTER = 0

	if FRAME_COUNT >= FRAME_NUM :
		if DETECTED_COUNT > FRAME_COUNT // 2:

			result = str(datetime.datetime.now()) + ','
			#PERCLOS
			perclose = CLOSED_COUNT / DETECTED_COUNT
			result += f"{perclose},"
			# BLINK
			# TODO: it may be a bad idea to just use blinks, beacuse there are undetected frames.
			result += f"{EYEBLINK_COUNT}\n"
			print(result)

			# save data
			with open('data.txt', 'a') as fo:
				fo.writelines(result)

		else:

			result = str(datetime.datetime.now())
			result += ",not detected\n"
			print(result)

			# save data
			with open('data.txt', 'a') as fo:
				fo.writelines(result)

		# reset the counter
		OPEN_COUNT = 0
		CLOSED_COUNT = 0
		EYEBLINK_COUNT = 0
		FRAME_COUNT = 0
		DETECTED_COUNT = 0

	# draw the total number of blinks on the frame along with
	# the computed eye aspect ratio for the frame
	try:
		pass
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	except:
		continue

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

x = list(range(len(ear_list)))
y = ear_list
print(x)
print(y)
plt.figure()
plt.plot(x,y)
plt.savefig("easyplot.pdf")
