import statistics
import time
from os import linesep
import numpy as np
import math
from matplotlib import pyplot as plt
import random



FACE_TRAIN_IMAGE_PATH = "data/facedata/facedatatrain"
FACE_TRAIN_LABEL_PATH = "data/facedata/facedatatrainlabels"
FACE_TEST_IMAGE_PATH = "data/facedata/facedatatest"
FACE_TEST_LABEL_PATH = "data/facedata/facedatatestlabels"


HEIGHT = 70 #Dimension of each image
WIDTH = 60 #Dimension of each image
FEATURES = HEIGHT * WIDTH # Size of each image 28*28


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

def train_classifier(x):
    start_time = time.time()
    feature_list = create_feature_list(FACE_TRAIN_IMAGE_PATH)
    label_list = create_label_list(FACE_TRAIN_LABEL_PATH)

    a = []

    random_list = random.sample(range(len(feature_list)), int((len(feature_list)*x*.01)))


    row_index = 0
    feature_list_index = 0
    how_many_pixels = 0

    while (row_index < len(random_list)):
        calculating_mean_of_row = []
        while (feature_list_index < len(feature_list[random_list[row_index]])):
            if (feature_list[random_list[row_index]][feature_list_index] == 1):
                how_many_pixels += 1

            if (feature_list_index != 0 and feature_list_index % 60 == 0):
                if (how_many_pixels != 0):
                    calculating_mean_of_row.append(how_many_pixels)
                how_many_pixels = 0

            feature_list_index += 1

        which_label = label_list[random_list[row_index]]
        mean = statistics.mean(calculating_mean_of_row)
        b = [mean, which_label]
        a.append(b)

        row_index += 1
        feature_list_index = 0
        how_many_pixels = 0

    a.sort()

    end_time = time.time()

    total_time = end_time - start_time

    return a, total_time

def test_classifier(x):

    a, total_time = train_classifier(x)

    # Feature list for testing data
    prediction_feature_list = create_feature_list(FACE_TEST_IMAGE_PATH)

    # Label list for testing data
    prediction_label_list = create_label_list(FACE_TEST_LABEL_PATH)

    predictions = [0] * len(prediction_label_list)

    prediction_row_index = 0
    prediction_list_index = 0
    prediction_how_many_pixels = 0

    while (prediction_row_index < len(prediction_feature_list)):
        calculating_mean_of_row = []
        while (prediction_list_index < len(prediction_feature_list[prediction_row_index])):
            if (prediction_feature_list[prediction_row_index][prediction_list_index] == 1):
                prediction_how_many_pixels += 1

            if (prediction_list_index != 0 and prediction_list_index % 60 == 0):
                if (prediction_how_many_pixels != 0):
                    calculating_mean_of_row.append(prediction_how_many_pixels)
                prediction_how_many_pixels = 0

            prediction_list_index += 1

        count = 0
        mean = statistics.mean(calculating_mean_of_row)

        for x in a:
            if (x[0] >= mean):
                count += 1
                break
            count += 1

        b = []

        upper_bound = count + 20  # 40 nearest neighbors
        if (count - 20 >= 0):
            count -= 20
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

    #print("Accuracy: ", num_correct / len(predictions))

    return (num_correct / len(predictions))

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
            a, time_spent = train_classifier(train_percentage)
            total_time.append(time_spent)
            acc.append(test_classifier(train_percentage))
            print("accuracy: ",i, " ", test_classifier(train_percentage))

        mean_list.append(np.mean(acc))
        std_list.append(np.std(acc))
        time_list.append(np.mean(total_time))
    return mean_list, std_list, time_list


def graph(mean_list, std_list, time_list, display):
    train_percentage = [.10, .20, .30, .40, .50, .60, .70, .80, .90, 1]

    fig = plt.figure()
    ax = fig.add_subplot()

    # Mean Accuracy vs Data points
    if display == 1:
        ax.plot(train_percentage, mean_list, color="blue")
        plt.title("FACE: Accuracy vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Accuracy (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    # Standard Deviation vs Data Points
    if display == 2:
        ax.plot(train_percentage, std_list, color="green")
        plt.title("FACE: Standard Deviation vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Standard Deviation (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    # Time vs Data points
    if display == 3:
        ax.plot(train_percentage, time_list, color="red")
        plt.title("FACE: Time to Train vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Time Needed to Train (s)")

    ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
    plt.show()


mean_list, std_list, time_list = get_stats()
graph(mean_list, std_list, time_list, 1)
graph(mean_list, std_list, time_list, 2)
graph(mean_list, std_list, time_list, 3)

'''
Creates a training list of the average amount of "filled pixels" per row of the image
and then makes a prediction whether it's a face or not based by calculating the average amount of
"filled pixels" in the given image and then finding the 40 nearest neighbors or the 50 closest 
images in the training list that have similar amount of "filled" pixels
'''

'''a = [0] * 10
a[0] = test_classifier(10)
a[1] = test_classifier(20)
a[2] = test_classifier(30)
a[3] = test_classifier(40)
a[4] = test_classifier(50)
a[5] = test_classifier(60)
a[6] = test_classifier(70)
a[7] = test_classifier(80)
a[8] = test_classifier(90)
a[9] = test_classifier(100)
print(*a)'''