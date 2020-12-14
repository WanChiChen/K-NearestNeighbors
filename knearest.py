from sklearn.datasets import load_digits
import numpy as np

# load dataset
digits = load_digits()
labels = digits.target
digits = np.array(digits.data)

# split into train and test
classes = np.zeros(10)
test_labels = list()
train_labels = list()
test = list()
train = list()

# 50 images per class in test
for i in range(labels.size):
    if classes[labels[i]] < 50:
        test_labels.append(labels[i])
        test.append(digits[i])
        classes[labels[i]] += 1
    else:
        train_labels.append(labels[i])
        train.append(digits[i])

# L2-norm of two vectors
def euclidean(v1, v2):
    distance = 0.0
    for i in range(len(v1)-1):
        distance += (v1[i]-v2[i]) **2
    return(np.sqrt(distance))

# get k-nearest vectors
def find_neighbors(train, test, k, labels):
    distance = list()
    i = 0
    for img in train:
        x = euclidean(img, test)
        distance.append((x, labels[i]))
        i = i + 1
    distance.sort(key=lambda tup: tup[0])
    
    neighbors = list()
    for i in range(k):
        neighbors.append(distance[i][1])
    return neighbors

# majority voting to choose prediction
def predict(train, test, k, labels):
    neighbors = find_neighbors(train, test, k, labels)
    counts = np.zeros(10)
    for neighbor in neighbors:
        counts[neighbor] = counts[neighbor] + 1
    prediction = np.argmax(counts)
    return prediction

# make a prediction on images on test given k-nearest neighbors in train
def execute(train, test, k, train_labels, test_labels):
    count = 0
    for i in range(500):
        prediction = predict(train, test[i], k, train_labels)
        if prediction == test_labels[i]:
            count = count + 1

    print("for {} nearest neighbors: {:2.2f} percent accuracy".format(k, ((count / (i+1))*100)))

# run knn algorithm for k = 1, 3, 5, 7
execute(train, test, 1, train_labels, test_labels)
execute(train, test, 3, train_labels, test_labels)
execute(train, test, 5, train_labels, test_labels)
execute(train, test, 7, train_labels, test_labels)
    
