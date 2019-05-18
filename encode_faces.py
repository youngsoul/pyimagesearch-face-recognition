import time

from imutils import paths
import face_recognition
import pickle
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="/Volumes/MacBackup/friends_family", help="path to input dataset directory")
ap.add_argument("-e", "--encodings-file", required=False, default='encodings/friends_family_encodings.pkl', help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default='cnn', help="face detection model to use: either 'hog' or 'cnn' ")


def get_command_line_args():
    args = vars(ap.parse_args())

    return args['dataset'],args['encodings_file'],args['detection_method']

dataset, encodings_file, detection_method = get_command_line_args()

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
s = time.time()

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    print(f"[INFO] processing image [{name}] {i+1}/{len(imagePaths)}")

    # load the input image and convert from BGR to RGB for dlib
    image = cv2.imread(imagePath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x,y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # we are assuming the the boxes of faces are the SAME FACE or SAME PERSON
    boxes = face_recognition.face_locations(rgb_image, model=detection_method)

    # compute the facial embedding for the face
    # creates a vector of 128 numbers representing the face
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

e = time.time()
print(f"Encoding dataset took: {(e-s)/60} minutes")
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodings_file, "wb")
f.write(pickle.dumps(data))
f.close()