from os import linesep
import numpy as np
import time
from matplotlib import pyplot as plt 

# Feature Idea: Use 4489 (67*67) features for naive bayes face algorithm

FACE_TRAIN_IMAGE_PATH = "data/facedata/facedatatrain"
FACE_TRAIN_LABEL_PATH = "data/facedata/facedatatrainlabels"
FACE_TEST_IMAGE_PATH = "data/facedata/facedatatest"
FACE_TEST_LABEL_PATH = "data/facedata/facedatatestlabels"
HEIGHT = 70 #Dimension of each image
WIDTH = 60 #Dimension of each image
FEATURES = HEIGHT * WIDTH # Size of each image 28*28

POSSIBLE_LABELS = 2 #Number of possible predictions, 2 in this case since its either a face or not a face

#Returns list of each feature for each image
#feature_list[i] returns a list of size 4489 for the ith image

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
    #print(len(feature_list))
    return feature_list

#Creates a list of labels for each image
#label_list[i] = label of feature_list[i]
def create_label_list(file):
    return np.loadtxt(file, dtype=int)


def train_bayes(train_percentage):

    start_time = time.time()
    
    feature_list = create_feature_list(FACE_TRAIN_IMAGE_PATH)
    label_list = create_label_list(FACE_TRAIN_LABEL_PATH)

    if (train_percentage != 100):
        #randomly select training sample of percentage
        total_images = len(feature_list)
        sample_size = int((train_percentage/100) * total_images)
        idx = np.random.choice(total_images, size=sample_size, replace=False)

        feature_list = feature_list[idx]
        label_list = label_list[idx]

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
    while i < len(feature_list):
        if(label_list[i] == 1):
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
            # do stuff with not_face_feature_table
            j = 0
            while j < len(not_face_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    not_face_feature_table_blanks[j] += 1
                else:
                    not_face_feature_table_pixels[j] += 1
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

    end_time = time.time()

    #Feature list for testing data
    test_list = create_feature_list(FACE_TEST_IMAGE_PATH)

    #Label list for testing data
    test_label_list = create_label_list(FACE_TEST_LABEL_PATH)

    #list to hold predictions for P(x|y=face)
    face_predictions = [0]*len(test_label_list)

    #list to hold predictions for P(x|y=not face)
    not_face_predictions = [0]*len(test_label_list)

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (face_feature_table_blanks[j] != 0):
                    face_predictions[i] += np.log(face_feature_table_blanks[j])
                else: 
                    face_predictions[i] += 0.001
            else:
                if (face_feature_table_pixels[j] != 0):
                    face_predictions[i] += np.log(face_feature_table_pixels[j])
                else:
                    face_predictions[i] += 0.001
            j += 1
        i += 1

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (not_face_feature_table_blanks[j] != 0):
                    not_face_predictions[i] += np.log(not_face_feature_table_blanks[j])
                else: 
                    not_face_predictions[i] += 0.001
            else:
                if (not_face_feature_table_pixels[j] != 0):
                    not_face_predictions[i] += np.log(not_face_feature_table_pixels[j])
                else:
                    not_face_predictions[i] += 0.001
            j += 1
        i += 1

    i = 0
    while i < len(face_predictions):
        face_predictions[i] += np.log(prob_face)
        i += 1

    i = 0
    while i < len(not_face_predictions):
        not_face_predictions[i] += np.log(prob_not_face)
        i += 1

    final_predictions = [0]*len(test_label_list)
    i = 0
    while i < len(test_label_list):
        if (face_predictions[i] > not_face_predictions[i]):
            final_predictions[i] = 1
        else:
            final_predictions[i] = 0
        i += 1

    num_correct = 0
    i = 0
    while i < len(final_predictions):
        if (final_predictions[i] == test_label_list[i]):
            num_correct += 1
        i += 1

    total_time = end_time - start_time

    print("Naive Bayes for faces accuracy: ", num_correct/len(final_predictions))
    return num_correct/len(final_predictions), total_time

def get_stats():
    # list of averages/std/time for each percentage of data points, 
    #i.e: mean_list[i] is the mean for 10% of the training set
    mean_list = [] 
    std_list = []
    time_list = []
    for train_percentage in range(10, 110, 10):
        acc = []
        total_time = []
        for i in range(5):
            accuracy, time_spent = train_bayes(train_percentage)
            total_time.append(time_spent)
            acc.append(accuracy)

        mean_list.append(np.mean(acc))
        std_list.append(np.std(acc))
        time_list.append(np.mean(total_time))
    return mean_list, std_list, time_list

def graph(mean_list, std_list, time_list, display):
    train_percentage = [.10,.20,.30,.40,.50,.60,.70,.80,.90,1]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Mean Accuracy vs Data points
    if display == 1:
        ax.plot(train_percentage, mean_list, color="blue")
        plt.title("FACE: Accuracy vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Accuracy (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    #Standard Deviation vs Data Points
    if display == 2:
        ax.plot(train_percentage, std_list, color="green")
        plt.title("FACE: Standard Deviation vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Standard Deviation (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    #Time vs Data points
    if display == 3:
        ax.plot(train_percentage, time_list, color="red")
        plt.title("FACE: Time to Train vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Time Needed to Train (s)")
    
    
    ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
    plt.show()

#driver(100)
mean_list, std_list, time_list = get_stats()
graph(mean_list, std_list, time_list, 1)