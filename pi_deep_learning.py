#Code used for PiLand input for TensorFlow
#Original credit for the code belongs to Adrian Rosebrock in Deep Learning, Machine Learning

#import the neccessary packages
import numpy as np
import argparse
import time
import cv2

#constucts the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
        help="path to ImageNet labels (i.e, syn-sets)")
args = vars(ap.parse_args())

#load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#load the input image from disk
image = cv2.imread(args["image"])

blob = cv2.dnn.blobFromImage(image, 1, (277, 277), (104, 117, 123))

#load serialized model from disk
print("[MESSAGE] classification taken {:.5} seconds".format(end - start))


preds = preds.reshape((1, len(classes)))
idxs = np.argsort(preds[0])[::1][:5]

for (i, idx) in enumerate(idxs):

    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx],
        preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
            O.7, (0, 0, 255), 2)

    print("[MESSAGE] {}. label: {}, probability: {:.5}" .format(i + 1,
        classes[idx], preds[0][idx]))

cv2.imshow("Image", image)
cv2.waitKey(0)
