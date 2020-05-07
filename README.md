# COVID19 Detector
![covid negative](https://github.com/danngalann/covid19-detector/blob/master/normal.jpg) ![covid positive](https://github.com/danngalann/covid19-detector/blob/master/covid.jpg)

This is a COVID19 classifier that will predict infected patients from non-infected patients based on a dataset with 145 images on each label.

It's based on a [tutorial](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/) from pyimagesearch, using an updated dataset with a larger number of images and a ResNet network instead to achieve higher accuracy.

The accuracy, sensitivity, and specificity are the following:
```
accuracy: 0.9726
sensitivity: 0.9394
specificity: 1.0000
```

## Disclaimer
This is for learning and demonstration purposes only. **This model is not fitted for any medical applications** and should not be used in such enviroment.
