from os import linesep
import numpy as np

# Feature Idea: Use 4489 (67*67) features for naive bayes face algorithm

FACE_TRAIN_IMAGE_PATH = "data/facedata/facedatatrain"
FACE_TRAIN_LABEL_PATH = "data/facedata/facedatatrainlabels"
DIMENSIONS = 70
FEATURES = DIMENSIONS * DIMENSIONS # Size of each image 67*67

POSSIBLE_LABELS = 2 #Number of possible predictions, 2 in this case since its either a face or not a face

#Returns list of each feature for each image
#feature_list[i] returns a list of size 4489 for the ith image

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


def train_bayes(feature_list, label_list):

    #Calculate P(y=face) as prob_face and P(y=-face) as prob_not_face
    num_faces = 0
    for l in label_list:
        if (l == 1):
            num_faces += 1

    prob_face = num_faces / len(label_list)
    prob_not_face = 1 - prob_face

    print(num_faces)
    print(len(label_list) - num_faces)
    print(prob_face)
    print(prob_not_face)
    print (len(feature_list))

train_bayes(create_feature_list(FACE_TRAIN_IMAGE_PATH), create_label_list(FACE_TRAIN_LABEL_PATH))