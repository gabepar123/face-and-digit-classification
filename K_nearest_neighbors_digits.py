import statistics
from os import linesep
import numpy as np
import math



DIGIT_TRAIN_IMAGE_PATH = "data/digitdata/trainingimages"
DIGIT_TRAIN_LABEL_PATH = "data/digitdata/traininglabels"
DIGIT_TEST_IMAGE_PATH = "data/digitdata/testimages"
DIGIT_TEST_LABEL_PATH = "data/digitdata/testlabels"


IMAGE_SIZE = 28
FEATURES = IMAGE_SIZE * IMAGE_SIZE # Size of each image 28*28


def most_frequent(List):
    return max(set(List), key = List.count)

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

#make an array of height*length, "
def train_classifier(feature_list, label_list):
    feature_list = create_feature_list(DIGIT_TRAIN_IMAGE_PATH)
    label_list = create_label_list(DIGIT_TRAIN_LABEL_PATH)

    #every row is an image basically of the feature list

    #label list if what each row is

    #predicator is the amount of 1's in the array

    a = []

    row_index = 0
    feature_list_index = 0
    how_many_pixels = 0

    while (row_index < len(feature_list)):
        calculating_mean_of_row = []
        while (feature_list_index < len(feature_list[row_index])):
            if (feature_list[row_index][feature_list_index] == 1):
                how_many_pixels += 1

            if(feature_list_index != 0 and feature_list_index % 28 == 0):
                if(how_many_pixels!= 0):
                    calculating_mean_of_row.append(how_many_pixels)
                how_many_pixels = 0

            feature_list_index += 1

        which_label = label_list[row_index]
        mean = statistics.mean(calculating_mean_of_row)
        b = [mean, which_label]
        a.append(b)

        row_index += 1
        feature_list_index = 0
        how_many_pixels = 0

    a.sort()
    return a

def test_classifier():
    a = train_classifier(create_feature_list(DIGIT_TRAIN_IMAGE_PATH), create_label_list(DIGIT_TRAIN_LABEL_PATH))
    # Feature list for testing data
    prediction_feature_list = create_feature_list(DIGIT_TEST_IMAGE_PATH)

    # Label list for testing data
    prediction_label_list = create_label_list(DIGIT_TEST_LABEL_PATH)

    predictions = [0] * len(prediction_label_list)

    prediction_row_index = 0
    prediction_list_index = 0
    prediction_how_many_pixels = 0

    while (prediction_row_index < len(prediction_feature_list)):
        calculating_mean_of_row = []
        while (prediction_list_index < len(prediction_feature_list[prediction_row_index])):
            if (prediction_feature_list[prediction_row_index][prediction_list_index] == 1):
                prediction_how_many_pixels += 1

            if(prediction_list_index != 0 and prediction_list_index % 28 == 0):
                if(prediction_how_many_pixels!=0):
                    calculating_mean_of_row.append(prediction_how_many_pixels)
                prediction_how_many_pixels = 0

            prediction_list_index += 1

        count = 0
        mean = statistics.mean(calculating_mean_of_row)

        for x in a:
            if(x[0] >= mean):
                count+=1
                break
            count+=1

        b = []

        upper_bound = count + 5  # 10 nearest neighbors
        if (count - 5 >= 0):
            count -= 5
        else:
            count = 0

        while (count < len(a) and count <= upper_bound):
            b.append(a[count][1])
            count += 1

        predictions[prediction_row_index] = most_frequent(b)

        prediction_row_index += 1
        prediction_list_index = 0
        prediction_how_many_pixels = 0

    num_correct = 0
    x = 0
    while x < len(predictions):
        if predictions[x] == prediction_label_list[x]:
            num_correct += 1
        x += 1

    print("Accuracy: ", num_correct / len(predictions))

test_classifier()


