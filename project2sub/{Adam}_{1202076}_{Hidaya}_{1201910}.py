import numpy as np
import csv
import sys
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):

        #self is the training feature
        #convert to numpy to perform what we need
        testing_features = np.array(features)

        # distance between testing and training features
        distance = cdist(testing_features, self.trainingFeatures)

        # k nearest neighbors indices for each testing example
        nearest_indices = np.argsort(distance, axis=1)[:, :k]

        #Labels of the nearest neighbors depending on nearest indices and our labels
        nearest_labels = np.take(self.trainingLabels,nearest_indices)
        #convert nearest_labels to list of integers
        nearest_converted_list = [[int(num) for num in sublist] for sublist in nearest_labels]

        # Predict class labels based on the majority
        predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_converted_list)
        return predicted_labels.tolist()

        raise NotImplementedError


def load_data(filename):
    features = []
    labels = []
  #read csv file
    with open("spambase.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append([float(value) for value in row[:57]]) #features is the first 57 columns
            labels.append(row[57]) #labels is the last column of the file


    return features, labels

    raise NotImplementedError


def preprocess(features):
    # Convert features to numpy to perform what we need
    features_array = np.array(features)

    # Calculate  mean and standard deviation for each feature
    mean_value = np.mean(features_array, axis=0)
    stand_deviation = np.std(features_array, axis=0)

    # Subtracting the mean and divide by the standard deviation
    newUpdated_features = (features_array - mean_value) / stand_deviation
    # Return the normalized features as a list
    return newUpdated_features.tolist()

    raise NotImplementedError

def train_mlp_model(features, labels):
    # Convert to numpy to perform what we need
    features = np.array(features)
    labels = np.array(labels)

    # Normalize the input features
    scale = StandardScaler()
    features = scale.fit_transform(features)

    # Make the MLP classifer
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=5000, tol=1e-4,
                        learning_rate_init=0.001)

    # Start training the model
    mlp.fit(features, labels)

    return mlp
    raise NotImplementedError


def evaluate(labels, predictions):
    # Calculate true positives, true negatives, false positives, false negatives
    true_positive = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    false_positive = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    true_negative = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    false_negative = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

    # Calculate accuracy
    accuracy = (true_positive + true_negative) / len(labels)

    # Calculate precision
    if ((true_positive + false_positive) > 0):
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    # Calculate recall
    if ((true_positive + false_negative) > 0):
        recall = true_positive / (true_positive + false_negative)
    else:
        recall= 0

    # Calculate F1-score
    if ((precision + recall) > 0):
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("True positive:", true_positive, " False positive: ", false_positive, " True negative: ", true_negative,
          " False negative: ", false_negative)

    return accuracy, precision, recall, f1
    raise NotImplementedError


def main():
    # Check command-line arguments
  #  if len(sys.argv) != 2:
   #     sys.exit("Usage: python main.py spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data("spambase.csv")
    features=list(features)
    labels = [int(x) for x in labels]
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)
    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** 3rd-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)



if __name__ == "__main__":
    main()
