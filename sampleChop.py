import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

# yields features and targets for each training sample
def dataGenerator(start ="na"):
    prefix = "C:\Users\Joe\Downloads\sampleChop\data\summed"
    if start == "na":
        reached = True
    else:
        reached = False
    for filename in os.listdir(prefix):
        if filename == start:
            reached = True
        if reached:
            if filename.endswith("features.txt"):
                features = np.loadtxt(prefix + "\\" + filename)
            elif filename.endswith("targets.txt"):
                targets = np.loadtxt(prefix + "\\" + filename)
                yield [features, targets]

# loads and shapes data from one file
# input: filename, boolean for use of tonnetz data
# output: numpy array
def load_features(f, tonnetz = False):
    data = np.loadtxt(f)
    num_pts = len(data) / (54)
    if tonnetz:
        return data.reshape(num_pts, 54, 61)
    else:
        data = data.reshape(num_pts, 54, 61)
        return data[:,:48,:]

# takes 3d matrix and sums columns of datapoints
# input: 3d matrix
# output: returns 2d matrix of summed columns
def sum_columns(data):
    sums = []
    for pt in data:
        newpt = np.array(map(sum,zip(*pt)))
        sums.append((newpt - newpt.mean(axis=0)) / newpt.std(axis=0))
    return np.array(sums)

def trainModel(epochs):
    valid_features = load_features(".\\validation\\"
                                   "02 - 21St Century - The Way We Were 1features.txt", False)
    valid_features = sum_columns(valid_features)
    valid_targets = np.loadtxt(".\\validation\\"
                               "02 - 21St Century - The Way We Were 1targets.txt")

    mlp = MLPClassifier(hidden_layer_sizes=(32,32,4,))
    for i in range(1, epochs + 1):
        dataGen = dataGenerator()
        for tup in dataGen:
            mlp = mlp.partial_fit(tup[0], tup[1], classes = [0.,1.])

        print "epoch: " + str(i)
        pred = mlp.predict(valid_features)
        print "NN classifier:"
        print str(confusion_matrix(valid_targets, pred))
        print str(f1_score(valid_targets, pred))
        joblib.dump(mlp, 'nn32_32_4.pkl')


def main():
    trainModel(epochs=20)

if __name__ == '__main__':
    main()
