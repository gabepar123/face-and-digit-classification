from os import linesep
import numpy as np
import math
import time
from matplotlib import pyplot as plt 

# Feature Idea: Use 4489 (67*67) features for naive bayes face algorithm

DIGITS_TRAIN_IMAGE_PATH = "data/digitdata/trainingimages"
DIGITS_TRAIN_LABEL_PATH = "data/digitdata/traininglabels"
DIGITS_TEST_IMAGE_PATH = "data/digitdata/testimages"
DIGITS_TEST_LABEL_PATH = "data/digitdata/testlabels"
DIMENSIONS = 28
FEATURES = DIMENSIONS * DIMENSIONS # Size of each image 67*67

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
    return feature_list

#Creates a list of labels for each image
#label_list[i] = label of feature_list[i]
def create_label_list(file):
    return np.loadtxt(file, dtype=int)


def train_bayes(train_percentage):

    start_time = time.time()
    
    feature_list = create_feature_list(DIGITS_TRAIN_IMAGE_PATH)
    label_list = create_label_list(DIGITS_TRAIN_LABEL_PATH)

    if (train_percentage != 100):
        #randomly select training sample of percentage
        total_images = len(feature_list)
        sample_size = int((train_percentage/100) * total_images)
        idx = np.random.choice(total_images, size=sample_size, replace=False)

        feature_list = feature_list[idx]
        label_list = label_list[idx]

    #Calculate P(y=face) as prob_face and P(y=-face) as prob_not_face
    num_digits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for l in label_list:
        num_digits[l] += 1

    prob_digits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    i = 0
    while i < len(prob_digits):
        prob_digits[i] = num_digits[i]
        i += 1
    i = 0
    while i < len(prob_digits):
        prob_digits[i] /= len(label_list)
        i += 1

    i = 0

    zero_feature_table_blanks = [0]*len(feature_list[0])
    zero_feature_table_pixels = [0]*len(feature_list[0])

    one_feature_table_blanks = [0]*len(feature_list[0])
    one_feature_table_pixels = [0]*len(feature_list[0])

    two_feature_table_blanks = [0]*len(feature_list[0])
    two_feature_table_pixels = [0]*len(feature_list[0])

    three_feature_table_blanks = [0]*len(feature_list[0])
    three_feature_table_pixels = [0]*len(feature_list[0])

    four_feature_table_blanks = [0]*len(feature_list[0])
    four_feature_table_pixels = [0]*len(feature_list[0])

    five_feature_table_blanks = [0]*len(feature_list[0])
    five_feature_table_pixels = [0]*len(feature_list[0])

    six_feature_table_blanks = [0]*len(feature_list[0])
    six_feature_table_pixels = [0]*len(feature_list[0])

    seven_feature_table_blanks = [0]*len(feature_list[0])
    seven_feature_table_pixels = [0]*len(feature_list[0])

    eight_feature_table_blanks = [0]*len(feature_list[0])
    eight_feature_table_pixels = [0]*len(feature_list[0])

    nine_feature_table_blanks = [0]*len(feature_list[0])
    nine_feature_table_pixels = [0]*len(feature_list[0])

    i = 0
    while i < len(feature_list):
        if(label_list[i] == 0):
            j = 0
            while j < len(zero_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    zero_feature_table_blanks[j] += 1
                else:
                    zero_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 1):
            j = 0
            while j < len(one_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    one_feature_table_blanks[j] += 1
                else:
                    one_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 2):
            j = 0
            while j < len(two_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    two_feature_table_blanks[j] += 1
                else:
                    two_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 3):
            j = 0
            while j < len(three_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    three_feature_table_blanks[j] += 1
                else:
                    three_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 4):
            j = 0
            while j < len(four_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    four_feature_table_blanks[j] += 1
                else:
                    four_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 5):
            j = 0
            while j < len(five_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    five_feature_table_blanks[j] += 1
                else:
                    five_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 6):
            j = 0
            while j < len(six_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    six_feature_table_blanks[j] += 1
                else:
                    six_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 7):
            j = 0
            while j < len(seven_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    seven_feature_table_blanks[j] += 1
                else:
                    seven_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 8):
            j = 0
            while j < len(eight_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    eight_feature_table_blanks[j] += 1
                else:
                    eight_feature_table_pixels[j] += 1
                j = j + 1
        elif(label_list[i] == 9):
            j = 0
            while j < len(nine_feature_table_blanks):
                if (feature_list[i][j] == 0):
                    nine_feature_table_blanks[j] += 1
                else:
                    nine_feature_table_pixels[j] += 1
                j = j + 1
        i = i + 1

    i = 0
    while i < len(zero_feature_table_blanks):
        zero_feature_table_blanks[i] = zero_feature_table_blanks[i] / num_digits[0]
        i += 1

    i = 0
    while i < len(zero_feature_table_pixels):
        zero_feature_table_pixels[i] = zero_feature_table_pixels[i] / num_digits[0]
        i += 1

    i = 0
    while i < len(one_feature_table_blanks):
        one_feature_table_blanks[i] /= num_digits[1]
        i += 1

    i = 0
    while i < len(one_feature_table_pixels):
        one_feature_table_pixels[i] /= num_digits[1]
        i += 1

    i = 0
    while i < len(two_feature_table_blanks):
        two_feature_table_blanks[i] /= num_digits[2]
        i += 1

    i = 0
    while i < len(two_feature_table_pixels):
        two_feature_table_pixels[i] /= num_digits[2]
        i += 1

    i = 0
    while i < len(three_feature_table_blanks):
        three_feature_table_blanks[i] /= num_digits[3]
        i += 1

    i = 0
    while i < len(three_feature_table_pixels):
        three_feature_table_pixels[i] /= num_digits[3]
        i += 1

    i = 0
    while i < len(four_feature_table_blanks):
        four_feature_table_blanks[i] /= num_digits[4]
        i += 1

    i = 0
    while i < len(four_feature_table_pixels):
        four_feature_table_pixels[i] /= num_digits[4]
        i += 1

    i = 0
    while i < len(five_feature_table_blanks):
        five_feature_table_blanks[i] /= num_digits[5]
        i += 1

    i = 0
    while i < len(five_feature_table_pixels):
        five_feature_table_pixels[i] /= num_digits[5]
        i += 1

    i = 0
    while i < len(six_feature_table_blanks):
        six_feature_table_blanks[i] /= num_digits[6]
        i += 1

    i = 0
    while i < len(six_feature_table_pixels):
        six_feature_table_pixels[i] /= num_digits[6]
        i += 1

    i = 0
    while i < len(seven_feature_table_blanks):
        seven_feature_table_blanks[i] /= num_digits[7]
        i += 1

    i = 0
    while i < len(seven_feature_table_pixels):
        seven_feature_table_pixels[i] /= num_digits[7]
        i += 1

    i = 0
    while i < len(eight_feature_table_blanks):
        eight_feature_table_blanks[i] /= num_digits[8]
        i += 1

    i = 0
    while i < len(eight_feature_table_pixels):
        eight_feature_table_pixels[i] /= num_digits[8]
        i += 1

    i = 0
    while i < len(nine_feature_table_blanks):
        nine_feature_table_blanks[i] /= num_digits[9]
        i += 1

    i = 0
    while i < len(nine_feature_table_pixels):
        nine_feature_table_pixels[i] /= num_digits[9]
        i += 1

    end_time = time.time()

    #Feature list for testing data
    test_list = create_feature_list(DIGITS_TEST_IMAGE_PATH)

    #Label list for testing data
    test_label_list = create_label_list(DIGITS_TEST_LABEL_PATH)

    #list to hold predictions for P(x|y=0)
    zero_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=1)
    one_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=2)
    two_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=3)
    three_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=4)
    four_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=5)
    five_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=6)
    six_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=7)
    seven_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=8)
    eight_predictions = [0]*len(test_label_list)
    #list to hold predictions for P(x|y=9)
    nine_predictions = [0]*len(test_label_list)

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (zero_feature_table_blanks[j] != 0):
                    zero_predictions[i] += math.log(zero_feature_table_blanks[j])
                else: 
                    zero_predictions[i] += 0.01
            else:
                if (zero_feature_table_pixels[j] != 0):
                    zero_predictions[i] += math.log(zero_feature_table_pixels[j])
                else:
                    zero_predictions[i] += 0.01
            j += 1
        i += 1

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (one_feature_table_blanks[j] != 0):
                    one_predictions[i] += math.log(one_feature_table_blanks[j])
                else: 
                    one_predictions[i] += 0.01
            else:
                if (one_feature_table_pixels[j] != 0):
                    one_predictions[i] += math.log(one_feature_table_pixels[j])
                else:
                    one_predictions[i] += 0.01
            j += 1
        i += 1

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (two_feature_table_blanks[j] != 0):
                    two_predictions[i] += math.log(two_feature_table_blanks[j])
                else: 
                    two_predictions[i] += 0.01
            else:
                if (two_feature_table_pixels[j] != 0):
                    two_predictions[i] += math.log(two_feature_table_pixels[j])
                else:
                    two_predictions[i] += 0.01
            j += 1
        i += 1

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (three_feature_table_blanks[j] != 0):
                    three_predictions[i] += math.log(three_feature_table_blanks[j])
                else: 
                    three_predictions[i] += 0.01
            else:
                if (three_feature_table_pixels[j] != 0):
                    three_predictions[i] += math.log(three_feature_table_pixels[j])
                else:
                    three_predictions[i] += 0.01
            j += 1
        i += 1  

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (four_feature_table_blanks[j] != 0):
                    four_predictions[i] += math.log(four_feature_table_blanks[j])
                else: 
                    four_predictions[i] += 0.01
            else:
                if (four_feature_table_pixels[j] != 0):
                    four_predictions[i] += math.log(four_feature_table_pixels[j])
                else:
                    four_predictions[i] += 0.01
            j += 1
        i += 1 

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (five_feature_table_blanks[j] != 0):
                    five_predictions[i] += math.log(five_feature_table_blanks[j])
                else: 
                    five_predictions[i] += 0.01
            else:
                if (five_feature_table_pixels[j] != 0):
                    five_predictions[i] += math.log(five_feature_table_pixels[j])
                else:
                    five_predictions[i] += 0.01
            j += 1
        i += 1 

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (six_feature_table_blanks[j] != 0):
                    six_predictions[i] += math.log(six_feature_table_blanks[j])
                else: 
                    six_predictions[i] += 0.01
            else:
                if (six_feature_table_pixels[j] != 0):
                    six_predictions[i] += math.log(six_feature_table_pixels[j])
                else:
                    six_predictions[i] += 0.01
            j += 1
        i += 1 
    
    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (seven_feature_table_blanks[j] != 0):
                    seven_predictions[i] += math.log(seven_feature_table_blanks[j])
                else: 
                    seven_predictions[i] += 0.01
            else:
                if (seven_feature_table_pixels[j] != 0):
                    seven_predictions[i] += math.log(seven_feature_table_pixels[j])
                else:
                    seven_predictions[i] += 0.01
            j += 1
        i += 1 

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (eight_feature_table_blanks[j] != 0):
                    eight_predictions[i] += math.log(eight_feature_table_blanks[j])
                else: 
                    eight_predictions[i] += 0.01
            else:
                if (eight_feature_table_pixels[j] != 0):
                    eight_predictions[i] += math.log(eight_feature_table_pixels[j])
                else:
                    eight_predictions[i] += 0.01
            j += 1
        i += 1 

    i = 0
    while i < len(test_list):
        j = 0
        while j < len(test_list[0]):
            if (test_list[i][j] == 0):
                if (nine_feature_table_blanks[j] != 0):
                    nine_predictions[i] += math.log(nine_feature_table_blanks[j])
                else: 
                    nine_predictions[i] += 0.01
            else:
                if (nine_feature_table_pixels[j] != 0):
                    nine_predictions[i] += math.log(nine_feature_table_pixels[j])
                else:
                    nine_predictions[i] += 0.01
            j += 1
        i += 1 

    i = 0
    while i < len(zero_predictions):
        zero_predictions[i] += math.log(prob_digits[0])
        i += 1

    i = 0
    while i < len(one_predictions):
        one_predictions[i] += math.log(prob_digits[1])
        i += 1

    i = 0
    while i < len(two_predictions):
        two_predictions[i] += math.log(prob_digits[2])
        i += 1
    
    i = 0
    while i < len(three_predictions):
        three_predictions[i] += math.log(prob_digits[3])
        i += 1

    i = 0
    while i < len(four_predictions):
        four_predictions[i] += math.log(prob_digits[4])
        i += 1

    i = 0
    while i < len(five_predictions):
        five_predictions[i] += math.log(prob_digits[5])
        i += 1

    i = 0
    while i < len(six_predictions):
        six_predictions[i] += math.log(prob_digits[6])
        i += 1 

    i = 0
    while i < len(seven_predictions):
        seven_predictions[i] += math.log(prob_digits[7])
        i += 1

    i = 0
    while i < len(eight_predictions):
        eight_predictions[i] += math.log(prob_digits[8])
        i += 1

    i = 0
    while i < len(nine_predictions):
        nine_predictions[i] += math.log(prob_digits[9])
        i += 1


    final_predictions = [0]*len(test_label_list)
    i = 0
    while i < len(test_label_list):
        final_predictions[i] = max(zero_predictions[i], one_predictions[i], two_predictions[i], three_predictions[i], four_predictions[i], five_predictions[i], six_predictions[i], seven_predictions[i], eight_predictions[i], nine_predictions[i])
        if (final_predictions[i] == zero_predictions[i]):
            final_predictions[i] = 0
        elif (final_predictions[i] == one_predictions[i]):
            final_predictions[i] = 1
        elif (final_predictions[i] == two_predictions[i]):
            final_predictions[i] = 2
        elif (final_predictions[i] == three_predictions[i]):
            final_predictions[i] = 3
        elif (final_predictions[i] == four_predictions[i]):
            final_predictions[i] = 4
        elif (final_predictions[i] == five_predictions[i]):
            final_predictions[i] = 5
        elif (final_predictions[i] == six_predictions[i]):
            final_predictions[i] = 6
        elif (final_predictions[i] == seven_predictions[i]):
            final_predictions[i] = 7
        elif (final_predictions[i] == eight_predictions[i]):
            final_predictions[i] = 8
        elif (final_predictions[i] == nine_predictions[i]):
            final_predictions[i] = 9
        i += 1


    num_correct = 0
    i = 0
    while i < len(final_predictions):
        if (final_predictions[i] == test_label_list[i]):
            num_correct += 1
        i += 1

    total_time = end_time - start_time

    print("Naive Bayes for digits accuracy: ", num_correct/len(final_predictions))

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
        plt.title("DIGITS: Accuracy vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Accuracy (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    #Standard Deviation vs Data Points
    if display == 2:
        ax.plot(train_percentage, std_list, color="green")
        plt.title("DIGITS: Standard Deviation vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Average Standard Deviation (%)")
        ax.set_yticklabels(['{:.0%}'.format(x) for x in ax.get_yticks()])
    #Time vs Data points
    if display == 3:
        ax.plot(train_percentage, time_list, color="red")
        plt.title("DIGITS: Time to Train vs Data Points")
        plt.xlabel("Data Points (%)")
        plt.ylabel("Time Needed to Train (s)")
    
    
    ax.set_xticklabels(['{:.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
    plt.show()

#driver(100)
mean_list, std_list, time_list = get_stats()
graph(mean_list, std_list, time_list, 3)