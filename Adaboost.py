#!/usr/bin/env python
# coding: utf-8

# In[4]:




#########################################################################
#
# Authors : 	  Eoin Garrigan + Raymond Mc Creesh 
# Student I.Ds:   17187478 + 15211428
#
# Project 4:     AdaBoost Classifier using weak linear classifiers
#
# File Name:    AdaBoost_Classifier.py	  
#
# Description:  An orientation line is created using the weighted means of the
#               positive and negative points. Each point is then projected onto 
#               the line. the midpoint of each pair of projected points is then
#               calculated and used to create a separation boundary.
#               this boundary moved along the line calculating the number of point
#               misclassifications each iteration. The best boundary then becomes the 
#               weak classifier.
#               The weak classifier calculates its error in classification of points
#               and uses this error to update the weights depending on whether the classification
#               was correct. Each weak classifier uses its error to calculate an indivdual alpha 
#               value associated with it.
#               
#               adaboost_strong combines each weak classifier into a
#               strong classifier. Each weak classifier is weighted by their alpha values.
#               The accuracy of each weak classifier
#               and the overall strong classifier is computed and the number of classifiers 
#               used is output.
#               The strong classifer accuracy is output per weak classifer iteration
#               and the final strong classifier output is plotted               
#              
#
# functions:      calc_mean project classifier weak_trainer adaboost_strong plotty
##########################################################################

# Library imports
import numpy as np
import matplotlib.pyplot as plt

# Data imports
train  = np.loadtxt('adaboost-train-20.txt')
labels = train[:,2]
train_sorted = train[np.argsort(train[:, 0])]
data_test = np.loadtxt("adaboost-test-20.txt", dtype = float)
points_test, labels_test = data_test[:, :2], data_test[:, 2]

labels_sorted = train_sorted[:,2].astype(int)
N = len(train)
weights = np.full(N,(1/N))

strong_classifier=[]
iterations = 30


# Function calc_mean takes in points and weights for each classifer iteration
# then calculates the mean for positive and negative points
# Returns the slope of the line between the two mean points and the intercept
def calc_mean(train, weights,N):
    # Initialise means to 0 for every function call
    pos_x_mean = 0
    pos_y_mean = 0
    neg_x_mean = 0
    neg_y_mean = 0
    pos_weights = 0
    neg_weights = 0

    # Sum positive and negative weighted points
    for i in range(N):
        if(train[i][2] == 1):
            pos_x_mean += (train[i][0] * weights[i])
            pos_y_mean += (train[i][1] * weights[i])
            pos_weights += weights[i]
        else:
            neg_x_mean += (train[i][0] * weights[i])
            neg_y_mean += (train[i][1] * weights[i])
            neg_weights += weights[i]

    # Calculate the mean points (1e-10 avoids division by 0)
    pos_x_mean = np.mean(pos_x_mean/pos_weights+1e-10)
    pos_y_mean = np.mean(pos_y_mean/pos_weights+1e-10)
    neg_x_mean = np.mean(neg_x_mean/neg_weights+1e-10)
    neg_y_mean = np.mean(neg_y_mean/neg_weights+1e-10)
    
    # Find slope of line between two points
    m = float((pos_y_mean-neg_y_mean)/(pos_x_mean-neg_x_mean))
    # Get y intercept of line
    c = float(pos_y_mean-(m*pos_x_mean))
    
    return m, c


# Function project: puts all points onto mean orientation line
def project(train, m, b):
    projected = []
    m2 = -1/m
    
    for i in range(len(train)):
        b2 = train[i][1] - (m2*train[i][0])
        a = np.array([[-m,1],[-m2,1]])
        B = np.array([b,b2])
        projected.append(np.linalg.solve(a,B))
    
    # Midpoints are found for all projected points
    midpoints = []
    projected = np.array(projected)
    sort = projected[np.argsort(projected[:,0])]
    for i in range(len(sort)-1):
        x1 = (sort[i][0]+sort[i+1][0])/2
        y1 = (sort[i][1]+sort[i+1][1])/2
        b3 = y1 - m2*x1
        midpoints.append([x1,y1,m2,b3])
    
    return midpoints


# Function classifier: generates two points that define a vector, return sign
def classifier(check_P, pointA, pointB):
    x1 = pointA[0] 
    y1 = pointA[1]
    x2 = pointB[0]
    y2 = pointB[1]
    
    # Check points
    x = check_P[0]
    y = check_P[1]
    a = x-x1
    b = y2-y1
    c = y-y1
    d = x2-x1
    classification = np.sign(((a)*(b))-((c)*(d)))
    
    # Return sign
    return classification


# Function weak_trainer: generate a weighted weak trainer and update the weighted points
def weak_trainer(midpoints, train, weights):
    alpha = 0
    error = 9999999
    weak_classifier = []
    alpha = 0
    
    # Get misclassifications of all points on either side of decision boundary
    for i in range(len(midpoints)):
        misclass = 0
        # Define decision boundary points
        pointA = midpoints[i][:2]
        pointB = train[i][:2]
        # Iterate through points to classify for decision boundary
        for j in range(len(train)):            
            classification = classifier(train[j], pointA, pointB)
            # Weight is the sum of misclassified points
            if(classification != train[j][2]):
                misclass += weights[j]
        # Minimise the error to find best decision boundary       
        if(misclass < error):
            error = misclass
            misclass = 0
            # Best boundary points are stored
            bestA = pointA
            bestB = pointB
    # Check classification is correct for boundary, update weights        
    for i in range(len(weights)):
        best_classification = classifier(train[i], bestA, bestB)
        if (best_classification != train[i][2]):
            weights[i] *= 1/((2*error)+1e-10)
        else:
            weights[i] *= 1/(2*(1-error))
    # Calculate alpha of iterated classifier
    alpha = 0.5*(np.log((1-error)/error))
    weak_classifer = [bestA[0],bestA[1],bestB[0],bestB[1], alpha]
    # Return the weak classifier with the updated weights
    return weak_classifer, weights
           
# Function adaboost_strong: an ensemble of weak learners are made from the combined weak classifiers
# their activation levels and classifications are returned

def adaboost_strong(strong_classifier, data):
    activation = []
    classification = []
    # Iterate through each of the points in data
    for i in range(len(data)):
        level=0
        # Iterate through weak classifiers
        for j in range(len(strong_classifier)):
            # Sum classification values for points in each weak classifier
            level += (strong_classifier[j][4]) * (classifier([data[i][0],data[i][1]], 
                                strong_classifier[j][:2],strong_classifier[j][2:4]))
        # Append activation levels on each of the weak classifiers
        activation.append(level)
        classification.append(np.sign(level))
    # Return activation levels and classified labels
    return activation, classification

# Function plotty: plots results from weak and strong classifiers 
def plotty(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    # Plotting Accuracy V's no. of weak classifiers
    plt.figure(1, [10,10])
    plt.title(' Weak classifiers accuracy per number of runs on testing data ')

    # For each of the weak classifiers get points p
    p = list(range(1,len(strong_classifier)+1))
    plt.plot(p, testing_accuracy, label=f"Accuracy V's {iterations} Weak Classifiers", linewidth=4)
    plt.legend(loc="center right",  prop={"size":20})
    plt.xlabel("Number of weak classifiers")
    plt.ylabel("Accuracy")
    plt.grid(b=True, which='major', linewidth=1)

    # Creat a meshgrid for points and activations
    p, y = np.mgrid[-2.5:3.8:0.05,-2.5:3.8:0.05]
    x_grid = np.reshape(p, (p.size, -1))
    y_grid = np.reshape(y, (y.size, -1))
    grid = np.column_stack((x_grid,y_grid))

    # Get activation levels from strong classifier to populate meshgrid
    z,a=adaboost_strong(strong_classifier, grid)
    # Reshape z for meshgrid
    z = np.reshape(z, (p.shape))
    
    # Plot contour for training data
    plt.figure(2, [10,10])
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title('AdaBoost Classifier')
    # Plot contour for activation levels equalling zero
    plt.contour(p, y, z, levels=[0], colors=["black"])
    # Contour fill either side of plotted line
    plt.contourf(p, y, z, levels=[-1e5, 0, 1e5], colors=["skyblue","aquamarine"])
    plt.scatter(data_test.T[0], data_test.T[1], c=data_test.T[2], cmap='bwr', marker = '.', 
                label="-1 class    <      >          +1  class")
    plt.legend(loc="lower right",  prop={"size":15})
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.grid()
    plt.axis('equal')
    plt.show()

    
# Iterating through classifiers 
for i in range(iterations):
    # Calacuating their weighted means and slope
    m, b = calc_mean(train, weights,N)
    # Find the midpoints between projected points
    midpoints = project(train,m ,b)
    # Train the weak classifier
    weak_classifier, weights_n = weak_trainer(midpoints, train, weights)
    # Update weights array with normalised weights
    weights = weights_n/sum(weights_n)
    # Store weak classifier
    strong_classifier.append(weak_classifier)

# Use the adaboost_strong classifier to classify the training data
train_levels, classifications = adaboost_strong(strong_classifier, train)

# Calculate accuracy
accuracy = np.sum(classifications == labels)
percentage = ((accuracy*100)/len(train))
print(f"- {i+1} weak linear classifiers were used.")
print(f"- Final accuracy was found on the training dataset of: {percentage}%" )

testing_accuracy = []
for i in range(len(strong_classifier)):
    test_levels, test_classifications = adaboost_strong(strong_classifier[:i], data_test)
    num_right = np.sum(test_classifications == labels_test)
    testing_accuracy.append((num_right*100)/len(labels_test))

# Print info for final accuracy on Training dataset
print(f"- The highest accuracy on testing data was: {round(np.max(testing_accuracy))}%")
print(f"- {np.argmax(testing_accuracy)+1} weak learners were used.")

plotty(strong_classifier, testing_accuracy)

