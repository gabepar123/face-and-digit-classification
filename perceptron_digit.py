import numpy as np
import math


# Feature Idea: Use 784 (28*28) features for perceptron

DIGIT_TRAIN_IMAGE_PATH = "data/digitdata/trainingimages"
DIGIT_TRAIN_LABEL_PATH = "data/digitdata/traininglabels"
DIGIT_TEST_IMAGE_PATH = "data/digitdata/testimages"
DIGIT_TEST_LABEL_PATH = "data/digitdata/testlabels"


DIMENSIONS = 28
FEATURES = DIMENSIONS * DIMENSIONS # Size of each image 28*28
ITERATIONS = 2 #Number of iterations for perceptron
POSSIBLE_LABELS = 10 #Number of possible predictions, 10 in this case since we have 10 possible digits

#Returns list of each feature for each image
#feature_list[i] returns a list of size 784 for the ith image
def create_feature_list(file):

    list = [] #Temporary master list
    with open(file, "r") as file:
        feature = [] #Feature list for the current image
        for line in file:
            if (len(feature) == FEATURES): #reset every Dimension lines for the next image
                list.append(feature) 
                feature = []

            for c in line: #Covert white-space to 0s and +/# to 1s
                if (c == ' '):
                    feature.append(0)
                elif (c == '+' or c == '#'):
                    feature.append(1)

    list.append(feature) #append final list

    feature_list = np.array(list) #Covert list to numpy array
    return feature_list


#Creates a list of labels for each image
#label_list[i] = label of feature_list[i]
def create_label_list(file):
    return np.loadtxt(file, dtype=int)



def train_perceptron():

    feature_list = create_feature_list(DIGIT_TRAIN_IMAGE_PATH)
    label_list = create_label_list(DIGIT_TRAIN_LABEL_PATH)

    weights = np.zeros((POSSIBLE_LABELS, FEATURES)) # 2d Array of weights, each index corresponds to the weights for that label
    w0 = np.zeros(POSSIBLE_LABELS)

    total_images = len(feature_list)
    
    for _ in range(ITERATIONS):
        for i in range(total_images):
            total_images = len(feature_list)
            feature = feature_list[i]
            label = label_list[i]
            predicted_label = -1
            argmax = -math.inf
            
            for j in range(POSSIBLE_LABELS):
                f = np.dot(weights[j], feature)
                f += w0[j]

                if f > argmax:
                    predicted_label = j
                    argmax = f

            if label != predicted_label: #If we predicted incorrectly
                weights[predicted_label] = np.subtract(weights[predicted_label], feature)
                w0[predicted_label] -= 1

                weights[label] = np.add(weights[label], feature)
                w0[label] += 1


    test_perceptron(weights, w0)
        
        

def test_perceptron(weights, w0):
    
    feature_list = create_feature_list(DIGIT_TEST_IMAGE_PATH)
    label_list = create_label_list(DIGIT_TEST_LABEL_PATH)
    total_images = len(feature_list)
    correct_predicted = 0

    for i in range(total_images):
        feature = feature_list[i]
        label = label_list[i]
        predicted_label = -1
        argmax = -math.inf

        for j in range(POSSIBLE_LABELS):
            f = np.dot(weights[j], feature)
            f += w0[j]
            
            if f > argmax:
                predicted_label = j
                argmax = f

        if label == predicted_label: #If we predicted incorrectly
            correct_predicted += 1

    print ("Digit Test Accuracy: {0:.0%}".format((correct_predicted/total_images)))
    print("Correct:", correct_predicted)
    print("Total", total_images)

train_perceptron()

