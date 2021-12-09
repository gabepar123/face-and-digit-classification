import numpy as np
import time
from matplotlib import pyplot as plt 

FACE_TRAIN_IMAGE_PATH = "data/facedata/facedatatrain"
FACE_TRAIN_LABEL_PATH = "data/facedata/facedatatrainlabels"
FACE_TEST_IMAGE_PATH = "data/facedata/facedatatest"
FACE_TEST_LABEL_PATH = "data/facedata/facedatatestlabels"



HEIGHT = 70 #Dimension of each image
WIDTH = 60 #Dimension of each image
FEATURES = HEIGHT * WIDTH # Size of each image 28*28
ITERATIONS = 2 #Number of iterations for perceptron


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

def perceptron(weights, w0, feature):
    f = np.dot(weights, feature)
    f += w0
    return f

def train(train_percentage):

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

    weights = np.zeros(FEATURES) # 2d Array of weights, each index corresponds to the weights for that label
    w0 = 0

    total_images = len(feature_list)


    for _ in range(ITERATIONS):
        for i in range(total_images):
            total_images = len(feature_list)
            feature = feature_list[i]
            label = label_list[i]
            
            f = perceptron(weights, w0, feature)

            if f < 0 and label == True:
                weights = np.add(weights, feature)
                w0 += 1
            if f >= 0 and label == False:
                weights = np.subtract(weights, feature)
                w0 -= 1
            
    end_time = time.time()
    total_time = end_time - start_time

    return weights, w0, total_time

def test(weights, w0, print_stats):
    feature_list = create_feature_list(FACE_TEST_IMAGE_PATH)
    label_list = create_label_list(FACE_TEST_LABEL_PATH)
    correct_predicted = 0

    total_images = len(feature_list)
    
    for i in range(total_images):
        total_images = len(feature_list)
        feature = feature_list[i]
        label = label_list[i]
        
        f = perceptron(weights, w0, feature)

        if (f >= 0 and label == True) or (f < 0 and label == False):
            correct_predicted += 1
    accuracy = correct_predicted / total_images
    if print_stats:
        print ("Digit Test Accuracy: {0:.0%}".format((accuracy)))
        print("Correct:", correct_predicted)
        print("Total", total_images)
    return accuracy

def driver(train_percentage):
    weights, w0, _ = train(train_percentage)
    test(weights, w0, True)

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
            weights, w0, time_spent = train(train_percentage)
            total_time.append(time_spent)
            acc.append(test(weights, w0, False))

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
graph(mean_list, std_list, time_list, 3)