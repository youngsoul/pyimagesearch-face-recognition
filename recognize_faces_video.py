# import the necessary packages
import face_recognition
import pickle
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time
import os

"""
This file assumes the MacBackup drive is connected.

--encodings-file encodings/jpark.pkl --output output/webcam_output.avi --display 1

To test using laptop video camera:
--encodings-file encodings/jpark.pkl --input camera

ap.add_argument("-i", "--input", type=str, required=False,
                default='/Volumes/MacBackup/PyImageSearch/face-recognition-opencv/videos/lunch_scene.mp4',
                help="path to input video or the word 'camera' to capture video from webcam")

"""
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings-file", required=False, default='encodings/friends_family_encodings.pkl',
                help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default='hog',
                help="face detection model to use: either 'hog' or 'cnn' ")
ap.add_argument("-o", "--output", type=str, required=False, help="path to output video DO NOT ADD EXTENSION.  E.g. output/my_test")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-i", "--input", type=str, required=False,
                default='camera',
                help="path to input video or the word 'camera' to capture video from webcam")

args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args['encodings_file'], "rb").read())

# initialize the video stream and pointer to output video file, then allow the camera
# sensor to warm up
print(f"Input: {args['input']}")
video_file = False
if args['input'] == 'camera':
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
else:
    print("[INFO] using video file...")
    video_file = True
    vs = cv2.VideoCapture(args['input'])

writer = None
time.sleep(0.2)

# loop over frames from the vdeo file stream
while True:
    # grab the frame from the threaded video stream
    if video_file:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
    else:
        frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have a width
    # of 750px (to speedup processing)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = imutils.resize(rgb_image, width=750)
    r = frame.shape[1] / float(rgb_image.shape[1])

    # detect the (x,y)-coordinates of the bounding boxes corresponding to each face in the
    # input frame, then compute the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb_image, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.5)
        name = "Unknown"

        # check to see if we have found any matches
        if True in matches:
            # find the indexes of all matched faces then initialize a dictionary to count
            # the total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face face
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes: (notes: in the event of an unlikely
            # tie, Python will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# close the video file pointers
try:
    if video_file:
        vs.release()
except:
    pass

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()

if args['output']:
    # for some reason, we cannot write the file with the .mp4 extension so we write it without the extension
    # but to get quicktime to play it, the file has to have the .mp4 extension.
    os.rename(args['output'], f"{args['output']}.mp4")
