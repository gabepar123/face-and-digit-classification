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
                #feature = []
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

    face_feature_table_blanks = [0]*len(feature_list[0])
    face_feature_table_pixels = [0]*len(feature_list[0])

    not_face_feature_table_blanks = [0]*len(feature_list[0])
    not_face_feature_table_pixels = [0]*len(feature_list[0])

    i = 0
    #j = 0
    print("yo")
    while i < len(feature_list):
        if(label_list[i] == 1):
            #print("face")
            # do stuff with face_feature_table
            j = 0
            while j < len(face_feature_table_blanks):
                #print(feature_list[i][j])
                if (feature_list[i][j] == 0):
                    face_feature_table_blanks[j] += 1
                #j += 1
                else:
                    face_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 0):
            #print("not a face")
            # do stuff with not_face_feature_table
            j = 0
            while j < len(not_face_feature_table_blanks):
                #print(feature_list[i][j])
                if (feature_list[i][j] == 0):
                    not_face_feature_table_blanks[j] += 1
                    #print("3")
                else:
                    not_face_feature_table_pixels[j] += 1
                #     #print("4")
                j = j + 1
        i = i + 1

    i = 0
    while i < len(face_feature_table_blanks):
        face_feature_table_blanks[i] /= num_faces
        i += 1

    i = 0
    while i < len(face_feature_table_pixels):
        face_feature_table_pixels[i] /= num_faces
        i += 1

    i = 0
    while i < len(not_face_feature_table_blanks):
        not_face_feature_table_blanks[i] /= (len(label_list) - num_faces)
        i += 1

    i = 0
    while i < len(not_face_feature_table_pixels):
        not_face_feature_table_pixels[i] /= (len(label_list) - num_faces)
        i += 1

    #print(len(feature_list) == len(label_list))
    #print(len(feature_list))
    #print(not_face_feature_table)
    #print(451-num_faces)
    #print(len(feature_list[0]))
    #print(not_face_feature_table_blanks)
    #print(face_feature_table2)
    i = 0
    while i < len(not_face_feature_table_blanks):
        print(not_face_feature_table_blanks[i] + not_face_feature_table_pixels[i])
        i += 1
    # i = 1
    # while i < len(feature_list):
    #     print(np.array_equiv(feature_list[i], feature_list[i-1]))
    #     i += 1
    #print(not_face_feature_table)
    #print(feature_list)



train_bayes(create_feature_list(FACE_TRAIN_IMAGE_PATH), create_label_list(FACE_TRAIN_LABEL_PATH))