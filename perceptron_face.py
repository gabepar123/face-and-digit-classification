import numpy as np


FACE_TRAIN_IMAGE_PATH = "data/facedata/facedatatrain"
FACE_TRAIN_LABEL_PATH = "data/facedata/facedatatrainlabels"
FACE_TEST_IMAGE_PATH = "data/facedata/facedatatest"
FACE_TEST_LABEL_PATH = "data/facedata/facedatatestlabels"



HEIGHT = 70 #Dimension of each image
WIDTH = 60 #Dimension of each image
FEATURES = HEIGHT * WIDTH # Size of each image 28*28
ITERATIONS = 5 #Number of iterations for perceptron


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

    feature_list = create_feature_list(FACE_TRAIN_IMAGE_PATH)
    label_list = create_label_list(FACE_TRAIN_LABEL_PATH)

    weights = np.zeros(FEATURES) # 2d Array of weights, each index corresponds to the weights for that label
    w0 = 0

    total_images = len(feature_list)
    
    for _ in range(ITERATIONS):
        for i in range(total_images):
            total_images = len(feature_list)
            feature = feature_list[i]
            label = label_list[i]
            
            f = np.dot(weights, feature)
            f += w0

            if f < 0 and label == True:
                weights = np.add(weights, feature)
                w0 += 1
            if f >= 0 and label == False:
                weights = np.subtract(weights, feature)
                w0 -= 1
            
        
    test_perceptron(weights, w0)

def test_perceptron(weights, w0):
    feature_list = create_feature_list(FACE_TEST_IMAGE_PATH)
    label_list = create_label_list(FACE_TEST_LABEL_PATH)
    correct_predicted = 0

    total_images = len(feature_list)
    
    for i in range(total_images):
        total_images = len(feature_list)
        feature = feature_list[i]
        label = label_list[i]
        
        f = np.dot(weights, feature)
        f += w0
        if (f >= 0 and label == True) or (f < 0 and label == False):
            correct_predicted += 1
    
    print ("Face Test Accuracy: {0:.0%}".format((correct_predicted/total_images)))
    print("Correct:", correct_predicted)
    print("Total", total_images)

train_perceptron()