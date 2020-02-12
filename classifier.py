
#import cv2
from sklearn.externals import joblib     # Save the classifier in a file. 
#from sklearn import datasets
from skimage.feature import hog          #calculating the hog features.
import numpy as np


from sklearn.svm import LinearSVC        #perform prediction after training the classifier.
#import numpy as np
from sklearn.datasets.mldata import fetch_mldata
#data = fetch_mldata('mnist-original')


def TrainingMachine():
    dataset = fetch_mldata('mnist-original')


  #save the images of the digits as respect to corresponding feature and corresponding label using numpy array.
    features = np.array(dataset.data, 'int16') 
    labels = np.array(dataset.target, 'int')
 
  #calculating the HOG feature and saving it in numpy array     
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
     
    #creating an objejct of LinearSVC
    clf = LinearSVC()
       

    clf.fit(hog_features, labels)

    joblib.dump(clf, "digity_cls.pkl", compress=3)
    

TrainingMachine()



