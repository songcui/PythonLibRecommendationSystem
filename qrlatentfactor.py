### This function implements qr latent factor model for the 
### recommendation system. Stochastic gradient descent is 
### used to update the parameters. 


import numpy as np
import math
import random

def qr(k, learningRate, lamda_user, lamda_item, noOfIteration, file_training):
### k is the latent dimension for QR matrix factorization.  learningRate is the 
### rate for parameter updating, lamda_user, lamda_item are regularization 
### parameters for p and q,  noOfIteration is an integer specifying the 
### number of iterations to run, file_training is a string specifying the file 
### directory for training dataset.
    maximumRating = 0
    file = open(file_training, 'r')
    lines = file.readlines()
    numberOfUsers = 0
    numberOfItems = 0
    userID = np.zeros((len(lines)))
    itemID = np.zeros((len(lines)))
    rating = np.zeros((len(lines)))
    count = 0

    for line in lines:
        listOfLine = line.split()
        userID[count] = int(listOfLine[0])-1
        if userID[count]>(numberOfUsers-1):
            numberOfUsers = userID[count]+1
        itemID[count] = int(listOfLine[1])-1
        if itemID[count]>(numberOfItems-1):
            numberOfItems= itemID[count]+1
        rating[count] = float(listOfLine[2])
        if rating[count]>maximumRating:
            maximumRating = rating[count]
        count=count+1

    maximumRating = float(maximumRating)

    #### Inialization for the latent model p and q. We use random initianization method here. Based on my experience,
    #### it will converge faster than zero initialization
    
    p = np.array([[float(random.uniform(0, math.sqrt(maximumRating/k))) for i in range(k)] for j in range (int(numberOfUsers))])
    q = np.array([[float(random.uniform(0, math.sqrt(maximumRating/k))) for i in range(k)] for j in range (int(numberOfItems))])

    #### parameter update by Stochastic Gradient Descent
    
    error = np.zeros((noOfIteration))
    for i in range (noOfIteration):
        for j in range (len(lines)):
            p[userID[j],:] = p[userID[j],:] + learningRate*((rating[j] - np.dot(p[userID[j],:], q[itemID[j],:]))*q[itemID[j],:]-lamda_user*p[userID[j],:])
            q[itemID[j],:] = q[itemID[j],:] + learningRate*((rating[j] - np.dot(p[userID[j],:], q[itemID[j],:]))*p[userID[j],:]-lamda_item*q[itemID[j],:])

        for j in range (len(lines)):
            error[i]= error[i] + math.pow(rating[j] - np.dot(p[userID[j],:], q[itemID[j],:]),2)
        
        error[i] = math.sqrt(error[i])/len(lines)

    return error, p, q
        
def qrPlusBaseline(k, learningRate, lamda_user, lamda_item, lamda_qr_user, lamda_qr_item, noOfIteration, file_training):
### k is the latent dimension for QR matrix factorization.  learningRate is the 
### rate for parameter updating, lamda_qr_user, lamda_qr_item are regularization 
### parameters for p and q,  lamda_user, lamda_item are regularization 
### parameters for user bias and item bias. noOfIteration is an integer specifying the 
### number of iterations to run, file_training is a string specifying the file 
### directory for training dataset.    
    
    
    maximumRating = 0
    file = open(file_training, 'r')
    lines = file.readlines()
    numberOfUsers = 0
    numberOfItems = 0
    userID = np.zeros((len(lines)))
    itemID = np.zeros((len(lines)))
    rating = np.zeros((len(lines)))
    count = 0

    for line in lines:
        listOfLine = line.split()
        userID[count] = int(listOfLine[0])-1
        if userID[count]>(numberOfUsers-1):
            numberOfUsers = userID[count]+1
        itemID[count] = int(listOfLine[1])-1
        if itemID[count]>(numberOfItems-1):
            numberOfItems= itemID[count]+1
        rating[count] = float(listOfLine[2])
        if rating[count]>maximumRating:
            maximumRating = rating[count]
        count=count+1

    maximumRating = float(maximumRating)
    
    #### Compute the global average of the ratings:    
    
    global_average = sum(rating)/len(lines)

    #### Parameter initialization:
    
    #### zero initialization for p and q:
    p = np.array([[float(0.0) for i in range(k)] for j in range (int(numberOfUsers))])
    q = np.array([[float(0.0) for i in range(k)] for j in range (int(numberOfItems))])
    
    ### user bias and item bias are set as the average rating received in the training set for a specific user
    ### or an item minus the global average. This is a good guess as it is the rational behind defining the user bias
    ### and item bias.
    
    b_user = np.zeros((numberOfUsers))
    b_item = np.zeros((numberOfItems))
    for i in range (int(numberOfUsers)):
        b_user_sum = np.array([rating[j] for j in range (int(numberOfUsers)) if int(userID[j])==i])
        if b_user_sum.shape[0]==0:
            b_user[i] = 0
        else:
            b_user[i] = sum(b_user_sum)/b_user_sum.shape[0]-global_average
        del b_user_sum
        
    for i in range (int(numberOfItems)):
        b_item_sum = np.array([rating[j] for j in range (int(numberOfItems)) if int(itemID[j])==i])
        if b_item_sum.shape[0]==0:
            b_item[i] = 0
        else:
            b_item[i] = sum(b_item_sum)/b_item_sum.shape[0]-global_average
        del b_item_sum


    #### parameter update by Stochastic Gradient Descent
    error = np.zeros((noOfIteration))
    for i in range (noOfIteration):
        for j in range (len(lines)):
            b_user[userID[j]] = b_user[userID[j]] + learningRate*((rating[j] - global_average - b_user[userID[j]] - b_item[itemID[j]] - np.dot(p[userID[j],:], q[itemID[j],:]))-lamda_user* b_user[userID[j]])           
            b_item[itemID[j]] = b_item[itemID[j]] + learningRate*((rating[j] - global_average - b_user[userID[j]] - b_item[itemID[j]] - np.dot(p[userID[j],:], q[itemID[j],:]))-lamda_item* b_item[itemID[j]])           
            p[userID[j],:] = p[userID[j],:] + learningRate*((rating[j] - global_average - b_user[userID[j]] - b_item[itemID[j]] - np.dot(p[userID[j],:], q[itemID[j],:]))*q[itemID[j],:]-lamda_qr_user*p[userID[j],:])
            q[itemID[j],:] = q[itemID[j],:] + learningRate*((rating[j] - global_average - b_user[userID[j]] - b_item[itemID[j]] - np.dot(p[userID[j],:], q[itemID[j],:]))*p[userID[j],:]-lamda_qr_item*q[itemID[j],:])

        for j in range (len(lines)):
            error[i]= error[i] + math.pow(rating[j]-global_average-b_user[userID[j]]-b_item[itemID[j]]-np.dot(p[userID[j],:], q[itemID[j],:]),2)
        error[i] = math.sqrt(error[i])/len(lines)

    return error, b_user, b_item, p, q

