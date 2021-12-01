from os import linesep
import numpy as np


# Feature Idea: Use 784 (28*28) features for perceptron

DIGIT_TRAIN_IMAGE_PATH = "data/digitdata/trainingimages"
DIGIT_TRAIN_LABEL_PATH = "data/digitdata/traininglabels"
DIMENSIONS = 28
FEATURES = DIMENSIONS * DIMENSIONS # Size of each image 28*28

POSSIBLE_LABELS = 10 #Number of possible predictions, 10 in this case since we have 10 possible digits

#Returns list of each feature for each image
#feature_list[i] returns a list of size 784 for the ith image

def create_feature_list(file):

    with open(file, "r") as file:
        line_num = 0
        list = [] #Temporary master list
        feature = [] #Feature list for the current image
        for line in file:
            if (line_num == DIMENSIONS): #reset every Dimension lines for the next image
                feature = []
                
                list.append(feature) 
                line_num = 0

            for c in line: #Covert white-space to 0s and +/# to 1s
                if (c == ' '):
                    feature.append(0)
                elif (c == '+' or c == '#'):
                    feature.append(1)
            line_num += 1 

    list.append(feature) #append final list

    feature_list = np.array(list) #Covert list to numpy array
    return feature_list


#Creates a list of labels for each image
#label_list[i] = label of feature_list[i]
def create_label_list(file):
    return np.loadtxt(file, dtype=int)


def is_correct_label(f, label):
    pass

def train_perceptron(feature_list, label_list):
    if (len(feature_list) != len(label_list)):
        print('ERROR')
        return

    weights = np.zeros(FEATURES)
    w0 = 0

    for i in range(len(feature_list)):
        feature = feature_list[i]
        label = label_list[i]

        f = np.dot(feature, weights)
        f += w0

        #TODO
        if f >=0: # label  = false
            weights = np.add(weights, feature)
            w0 += 1
        if f < 0: #and label = true
            weights = np.subtract(weights, feature)
            w0 -= 1
        print(f)



train_perceptron(create_feature_list(DIGIT_TRAIN_IMAGE_PATH),create_label_list(DIGIT_TRAIN_LABEL_PATH))
#create_label_list(DIGIT_TRAIN_LABEL_PATH)

