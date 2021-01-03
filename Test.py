import pandas as pd
import skimage.color
from skimage.feature import hog, local_binary_pattern, corner_harris
import numpy as np  # importing numpy
from joblib import dump, load


def ExtractFeaturesHog(inputImage):
    img = (inputImage - np.mean(inputImage)) / np.std(inputImage)
    img = skimage.color.rgb2gray(img)
    #plt.imshow(img)
    #plt.show()
    feature, hog_image = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    #feature = local_binary_pattern(img, P=8, R=1)
    return feature, hog_image


if __name__ == '__main__':
    # Loading the Test Data
    Yts = np.loadtxt("TestData.csv")
    Xtr = np.loadtxt("TrainData.csv")
    Ytr = np.loadtxt("TrainLabels.csv")
    # Loading the Model
    modelKnn = load('MyModelSVM.pkl')


    # performing predictions using KNN

    PredictionsArray = []
    for i in range(len(Yts)):
        image = Yts[i].reshape([28, 28])
        feature, Hog_image = ExtractFeaturesHog(inputImage=image)
        feature = np.array(feature)
        feature = feature.reshape(1, 1296)
        print(feature.shape)
        prediction = modelKnn.predict(feature)
        print("done")
        PredictionsArray.append(prediction)
    PredictionsArray = np.array(PredictionsArray)
    df = pd.DataFrame(data=PredictionsArray, columns=["Predictions"])

    df.to_csv("svmPredictions", sep=',', header=False, index=False)







