import tensorflow as tf
import numpy as np
import imutils
from imutils import paths
import cv2
import random
import pickle
import os

# Load a random image and it's label
imagePaths = list(paths.list_images("dataset"))
imagePath = random.choice(imagePaths)

trueLabel = imagePath.split(os.path.sep)[-2]
image = cv2.imread(imagePath)
orig = image.copy()

# pre-process the image for classification
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)

# Load model
model = tf.keras.models.load_model("model")

# Load labels
lb = pickle.load(open("labels.pickle", "rb"))

# Classify image
preds = model.predict(image)

# find the class label index with the largest probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# build texts
realText = f"Real: {trueLabel}"
predText = "Pred: {} ({:.2f}%)".format(label, preds[0][i] * 100)

# print texts
print(realText)
print(predText)

# draw texts
output = imutils.resize(orig, width=400)
cv2.putText(output, realText, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)
cv2.putText(output, predText, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)