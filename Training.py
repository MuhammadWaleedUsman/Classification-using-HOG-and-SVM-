# Script by MUHAMMAD WALEED USMAN
#I19-2140 MSDS-A


import skimage.color
from skimage.feature import hog, local_binary_pattern, corner_harris
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np  # importing numpy
import matplotlib.pyplot as plt  # importing plotting module
import itertools
from scipy.stats import kde
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load


def plotDensity_2d(X, Y):
    nbins = 200
    minx, maxx = np.min(X[:, 0]), np.max(X[:, 0])
    miny, maxy = np.min(X[:, 1]), np.max(X[:, 1])
    xi, yi = np.mgrid[minx:maxx:nbins * 1j, miny:maxy:nbins * 1j]

    def calcDensity(xx):
        k = kde.gaussian_kde(xx.T)
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return zi.reshape(xi.shape)

    pz = calcDensity(X[Y == 1, :])
    nz = calcDensity(X[Y == -1, :])

    c1 = plt.contour(xi, yi, pz, cmap=plt.cm.Greys_r, levels=np.percentile(pz, [75, 90, 95, 97, 99]));
    plt.clabel(c1, inline=1)
    c2 = plt.contour(xi, yi, nz, cmap=plt.cm.Purples_r, levels=np.percentile(nz, [75, 90, 95, 97, 99]));
    plt.clabel(c2, inline=1)
    plt.pcolormesh(xi, yi, 1 - pz * nz, cmap=plt.cm.Blues, vmax=1, vmin=0.99);
    plt.colorbar()
    markers = ('s', 'o')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker=markers[0], c='y', s=30)
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker=markers[1], c='c', s=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.show()


def plotit(X, Y=None, clf=None, markers=('s', 'o'), hold=False, transform=None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    eps = 1e-6
    minx, maxx = np.min(X[:, 0]), np.max(X[:, 0])
    miny, maxy = np.min(X[:, 1]), np.max(X[:, 1])

    if clf is not None:
        npts = 150
        x = np.linspace(minx, maxx, npts)
        y = np.linspace(miny, maxy, npts)
        t = np.array(list(itertools.product(x, y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z, (npts, npts)).T
        extent = [minx, maxx, miny, maxy]
        plt.contour(x, y, z, [-1 + eps, 0, 1 - eps], linewidths=[2], colors=('b', 'k', 'r'), extent=extent,
                    label='f(x)=0')
        # plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z, cmap=plt.cm.Purples, vmin=-2, vmax=+2);
        plt.colorbar()
        plt.axis([minx, maxx, miny, maxy])

    if Y is not None:

        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker=markers[0], c='y', s=30)
        plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker=markers[1], c='c', s=30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

    else:
        plt.scatter(X[:, 0], X[:, 1], marker='.', c='k', s=5)
    if not hold:
        plt.grid()

        plt.show()


def ExtractFeaturesLocalBinary(inputImage):
    img = (inputImage - np.mean(inputImage)) / np.std(inputImage)
    img = skimage.color.rgb2gray(img)
    feature = local_binary_pattern(img, P=10, R=2, method='default')
    im = Image.fromarray(feature)
    im.show()
    return feature


def ExtractFeaturesHog(inputImage):
    img = (inputImage - np.mean(inputImage)) / np.std(inputImage)
    img = skimage.color.rgb2gray(img)
    #plt.imshow(img)
    #plt.show()
    feature, hog_image = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    #feature = local_binary_pattern(img, P=8, R=1)
    return feature, hog_image


def TrainKnn(k, Xtr, Ytr):
    """
    INPUT: K,Testing,Training Arrays
    OUTPUT: Accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=k)  # model for knn classifier
    scores = cross_val_score(knn, Xtr, Ytr, cv=5, scoring='accuracy')
    return scores.mean(), knn


def TrainSVM(C, Xtr, Ytr):
    svm = SVC(C=C, kernel='rbf')
    scores = cross_val_score(svm, Xtr, Ytr, cv=5, scoring='accuracy')
    return scores.mean(), svm


def TrainSGDClassifier(Xtr, Ytr):
    SGD = linear_model.SGDClassifier(loss='hinge')
    scores = cross_val_score(SGD, Xtr, Ytr, cv=5, scoring='accuracy')
    return scores.mean(), SGD


def TrainRandomForest(Xtr, Ytr):
    RandomForest = RandomForestClassifier(criterion='gini', max_depth=100, n_estimators=100)
    scores = cross_val_score(RandomForest, Xtr, Ytr, cv=5, scoring='accuracy')
    return scores.mean(), RandomForest


def MLP(Xtr, Ytr):
    MLPclf = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu')
    scores = cross_val_score(MLPclf, Xtr, Ytr, cv=5, scoring='accuracy')
    return scores.mean(), MLPclf


if __name__ == '__main__':
    Xtr = np.loadtxt("TrainData.csv")
    Ytr = np.loadtxt("TrainLabels.csv")
    Xtr_features = []
    for i in range(len(Ytr)):
        image = Xtr[i].reshape([28, 28])
        feature1, Hog_image = ExtractFeaturesHog(inputImage=image)
        TrainingFeatures = np.array(feature1)
        Xtr_features.append(TrainingFeatures)
    Xtr_features = np.array(Xtr_features)

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
     Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(Hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    print(feature)
    print(TrainingFeatures.shape)'''

    AccuracyKNN, knnModel = TrainKnn(9, Xtr_features, Ytr)
    knnModel.fit(Xtr_features, Ytr)
    print("Training Accuracy for 5 fold cross validation on 9 NN classifier with HOG Feature Extraction: ", AccuracyKNN)

    dump(knnModel, 'knnModel.pkl')

    AccuracyRandomForest, RandomForestModel = TrainSGDClassifier(Xtr_features, Ytr)
    RandomForestModel.fit(Xtr_features, Ytr)
    print("Training Accuracy for 5 fold cross validation on Random Forest classifier with HOG Feature Extraction: ", AccuracyRandomForest)
    dump(RandomForestModel, 'rfModel.pkl')

    AccuracySVM, svmModel = TrainSVM(10, Xtr_features, Ytr)
    svmModel.fit(Xtr_features, Ytr)
    print("Training Accuracy for 5 fold cross validation on RBF Kernelized SVM classifier with HOG Feature Extraction: ", AccuracySVM)
    dump(svmModel, 'svmModel.pkl')

    AccuracySGD, sgdModel = TrainSGDClassifier(Xtr_features, Ytr)
    sgdModel.fit(Xtr_features, Ytr)
    print("Training Accuracy for 5 fold cross validation on SGD classifier with HOG Feature Extraction: ", AccuracySGD)
    dump(sgdModel, 'sgdModel.pkl')

    AccuracyMlp, mlpModel = TrainSGDClassifier(Xtr_features, Ytr)
    mlpModel.fit(Xtr_features, Ytr)
    print("Training Accuracy for 5 fold cross validation on MLP classifier with HOG Feature Extraction: ", AccuracyMlp)
    dump(sgdModel, 'mlpModel.pkl')