
# coding: utf-8

# In[1]:

import numpy as np
import xlrd 

import scipy.stats as sp
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.grid_search import RandomizedSearchCV
# from sklearn.grid_search import GridSearchCV
import time
import pandas as pd
import random


# In[2]:

def letor_data():
    #Read LETOR Data from files
    letor_x_data = np.genfromtxt('Querylevelnorm_X.csv',delimiter = ',', dtype=np.float64)
    letor_t_data = np.genfromtxt('Querylevelnorm_t.csv',delimiter = ',', dtype=np.float64)
    
    #Divide Data into Train, Validate and Test set
    letor_train_limit = int(math.floor(0.8 * len(letor_x_data)))
    letor_validate_limit = int(math.floor(0.1*len(letor_x_data)))

    letor_x_train_set = letor_x_data[0:letor_train_limit,:]
    letor_x_validate_set = letor_x_data[(letor_train_limit+1):(letor_train_limit+letor_validate_limit),:]
    letor_x_test_set = letor_x_data[(letor_train_limit+letor_validate_limit+1):,:]

    letor_t_train_set = letor_t_data[0:letor_train_limit].reshape(-1,1)
    letor_t_validate_set = letor_t_data[(letor_train_limit+1):(letor_train_limit+letor_validate_limit)].reshape(-1,1)
    letor_t_test_set = letor_t_data[(letor_train_limit+letor_validate_limit+1):].reshape(-1,1)
    
    #HyperParameter Values , these are the final optimized values
    M = 10
    ld = 0.1
    learning_rate = 0.5
    epoc = 10000
    p = 20
    print("HyperParameters")
    print("M= ",M,"Lamda= ",ld,"Learning rate= ",learning_rate,"Patience= ",p)
    print("-------------------------------------------------------------------")
    # Letor Closed form main function
    calc_closed_form_solution(letor_x_train_set, letor_t_train_set, letor_x_validate_set, letor_t_validate_set, letor_x_test_set, letor_t_test_set, M, ld)
    print("-------------------------------------------------------------------")
    # Letor Stochastic Gradient Descent main function
    calc_SGD(letor_x_train_set, letor_t_train_set, letor_x_validate_set, letor_t_validate_set, letor_x_test_set, letor_t_test_set, M, ld, learning_rate, epoc, p)
        


# In[3]:

def syn_data():
    # Read Syntehtic data from files
    dataset_input= pd.read_csv('input.csv',delimiter = ',', header=None)
    dataset_output = pd.read_csv('output.csv',delimiter = ',', header=None)

    syn_x_data = dataset_input.as_matrix()
    syn_t_data = dataset_output.as_matrix()

    #Divide data into Train, Validate and Test Set
    syn_train_limit = int(math.floor(0.8 * len(syn_x_data)))
    syn_validate_limit = int(math.floor(0.1*len(syn_x_data)))

    syn_x_train_set = syn_x_data[0:syn_train_limit,:]
    syn_x_validate_set = syn_x_data[(syn_train_limit+1):(syn_train_limit+syn_validate_limit),:]
    syn_x_test_set = syn_x_data[(syn_train_limit+syn_validate_limit+1):,:]

    syn_t_train_set = syn_t_data[0:syn_train_limit].reshape(-1,1)
    syn_t_validate_set = syn_t_data[(syn_train_limit+1):(syn_train_limit+syn_validate_limit)].reshape(-1,1)
    syn_t_test_set = syn_t_data[(syn_train_limit+syn_validate_limit+1):].reshape(-1,1)
    
    #HyperParameter Values , these are the final optimized values
    M = 6
    ld = 0.1
    learning_rate = 0.5
    epoc = 10000
    p = 20

    print("HyperParameters")
    print("M= ",M,"Lamda= ",ld,"Learning rate= ",learning_rate,"Patience= ",p)
    print("-------------------------------------------------------------------")
    #Synthetic Closed form Main Function
    calc_closed_form_solution(syn_x_train_set, syn_t_train_set, syn_x_validate_set, syn_t_validate_set, syn_x_test_set, syn_t_test_set, M, ld)
    print("-------------------------------------------------------------------")
    #Synthetic Stochastic Gradient Descent Main Function
    calc_SGD(syn_x_train_set, syn_t_train_set, syn_x_validate_set, syn_t_validate_set, syn_x_test_set, syn_t_test_set, M, ld, learning_rate, epoc, p)


# In[4]:
# Input: data_set
# Output: Sigma
def calculateSigmaInverse(data):
    temp = []   
    row,col=data.shape     
    for i in range(0,col):
        variance = np.var(data[:,i])
        temp.append(variance)    
    dig = np.diag(temp) 
    sigmainv = np.linalg.pinv(dig)  
    return sigmainv


# In[5]:
# Input: data_set, M
# Output: mean
def calculateMean(M,data):
        sigma_inverse = []
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(data)
        centroid = kmeans.cluster_centers_
        labels=kmeans.labels_
        return centroid


# In[6]:

# Input: data_set, M
# Output: mean,sigma
# Performs k Means clustering and returns M clusters, using which mean and spread are calculated
def calculateMeanAndVariance(M,data):
        sigma_inverse = []
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(data)
        centroid = kmeans.cluster_centers_
        labels=kmeans.labels_
        sigma_inverse = []
        for i in range(0, M):
            sigma_inverse.append(np.eye(data.shape[1]))
        return centroid,np.asarray(sigma_inverse)


# In[7]:

# Input: data_set, mean, spread
# Output: basis function matrix
def compute_design_matrix(X, centers, spreads): 
    basis_func_outputs = np.exp(
    np. sum(np.matmul(X - centers, spreads) * (X - centers), axis=2) / (-2) ).T
    return np.insert(basis_func_outputs, 0, 1, axis=1) # insert ones to the 1st col


# In[8]:

# Input: Design Matrix, t_data_set
# Output: Weights
def closed_form_sol(L2_lambda, design_matrix, output_data): 
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),np.matmul(design_matrix.T, output_data) ).flatten()


# In[9]:

# Input: Design Matrix, weights, t_data_set
# Output: Error
def calculateErrorWithReg(phi,w,t,lamb): 
    w_t = np.transpose(w)
    sum = 0
    e_d=0
    row,col=phi.shape
    for i in range(0,row):
        temp = np.dot(w_t,np.transpose(phi[i,:]))
        squr_term = np.square(np.subtract(t[i],temp))
        sum = sum + (squr_term) 
    e_d = sum/2 
    #e_w = lamb*(np.dot(w_t,w))/2 
    #error=e_d+e_w  
    return e_d    


# In[10]:

# Input: Error, weights, t_data_set
# Output: Error RMS value
def calculateRMSError(error, shape):  

    e_rms = np.round(np.sqrt((2*error)/shape),5)  
    return e_rms 


# In[11]:

#Input: train_set, validate_set, test_set, Hyperparemeters
#Output: Calculating and Displaying ERMS values for data sets
def calc_closed_form_solution(x_train_set, t_train_set, x_validate_set, t_validate_set, x_test_set, t_test_set, M, ld):
    print("********* Closed Form Solution *********")
    N, D = x_train_set.shape

#     centers = calculateMean(M,x_train_set)
#     spreads = []
#     for i in range(0,M):
#         spreads.append(np.identity(D) * 0.5)
    centers, spreads = calculateMeanAndVariance(M,x_train_set)
    centers = centers[:, np.newaxis, :]
    X = x_train_set[np.newaxis, :, :]
    design_matrix = compute_design_matrix(X, centers, spreads) #Computing design matrix for train_set
    weights = closed_form_sol(L2_lambda=ld, design_matrix=design_matrix,output_data=t_train_set)
    print("***** Train Set closed form *****")
    print ("Weights = ")
    print (weights)
    error=calculateErrorWithReg(design_matrix,weights,t_train_set,ld)
    error_rms=calculateRMSError(error,N)
    print("Error ",error)
    print("Error RMS",error_rms)

    print("***** Validation Set closed form *****")
    N, D = x_validate_set.shape
    X = x_validate_set[np.newaxis, :, :]
    design_matrix = compute_design_matrix(X, centers, spreads) #Computing design matrix for validate_set
    error=calculateErrorWithReg(design_matrix,weights,t_validate_set,ld)
    error_rms=calculateRMSError(error,N)
    print("Error ",error)
    print("Error RMS",error_rms)

    print("***** Test Set closed form *****")
    N, D = x_test_set.shape
    X = x_test_set[np.newaxis, :, :]
    design_matrix = compute_design_matrix(X, centers, spreads) #Computing design matrix for test_set
    error=calculateErrorWithReg(design_matrix,weights,t_test_set,ld)
    error_rms=calculateRMSError(error,N)
    print("Error ",error)
    print("Error RMS",error_rms)

# In[13]:

#Input: train_set, validate_set, design_Mtarix, Hyperparemeters
#Output: Weights for validation data set based on Early stop crieria 'P'
def SGD_sol_early(learning_rate, minibatch_size,num_epochs, L2_lambda,train_design_matrix,train_output_data,validate_design_matrix,validate_output_data, p, M):
    N, _ = train_design_matrix.shape
    prev = 999;
    counter = 0
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, M+1])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary

    #Graph Plotting
    #plot_validation = []

    for epoch in range(num_epochs):
        #print("ITERATIONS", epoch)
        size_N = int(N / minibatch_size)
        for i in range(size_N):
            lower_bound = i * minibatch_size 
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = train_design_matrix[lower_bound : upper_bound, :]
            t = train_output_data[lower_bound : upper_bound, :]
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E 
        error=calculateErrorWithReg(validate_design_matrix,np.transpose(weights),validate_output_data,L2_lambda)
        error_rms=calculateRMSError(error,validate_design_matrix.shape[0])
        #plot_validation.append(error_rms)
        #print ("Previous ", prev)
        #print ("Current ", error_rms)
        if(prev < error_rms):  #Condition to check slope
            counter = counter + 1;
            #print('--------------------------------------------------')
            #print('counter', counter)
            if(p == counter): #Early stop criteria
                #Graph Ploting
                # fig = plt.figure()
                # fig.suptitle("Validation Erms vs Epoch")
                # plt.plot(plot_validation)
                # plt.ylabel('Validation Error (Erms)')
                # plt.xlabel('Epoch')
                # plt.show()
                print("Convergence Achieved At Iteration= ", epoch)
                print("***** Validation Set Gradient Descent *****")
                print("Error RMS",prev)
                return weights.flatten()
        if(prev>error_rms):
            counter = 0
            prev = error_rms[0]
        #print ("Error ",np.linalg.norm(E))
        #print ("Weights ",weights)
    return weights.flatten()


# In[14]:

#Input: train_set, validate_set, test_set, design_Mtarix, Hyperparemeters
#Output: Calculating and Displaying ERMS for test set data
def calc_SGD(x_train_set, t_train_set, x_validate_set, t_validate_set, x_test_set, t_test_set, M, lambd, learning_rate, epoc, p):
    N, D = x_train_set.shape
    print("********* Gradient descent Solution *********")
    #Used for only mean calculation using KMeans and constant Sigma
#     centers = calculateMean(M,x_train_set)
#     spreads = []
#     for i in range(0,M):
#         spreads.append(np.identity(D) * 0.5)

    centers, spreads = calculateMeanAndVariance(M,x_train_set) #Caluclaute Mean & Spread
    spreads = np.asarray(spreads)
    centers = centers[:, np.newaxis, :]
    X = x_train_set[np.newaxis, :, :]
    train_design_matrix = compute_design_matrix(X, centers, spreads) #Computing design matrix for train_set
    X = x_validate_set[np.newaxis, :, :]
    validate_design_matrix = compute_design_matrix(X, centers, spreads) #Computing design matrix for validate_set
    w = SGD_sol_early(learning_rate, N,epoc, lambd,train_design_matrix,t_train_set,validate_design_matrix,t_validate_set, p, M)
    print ("Weights = ")
    print (w)
    print("***** Test Set Gradient Descent *****")
    N, D = x_test_set.shape
    X = x_test_set[np.newaxis, :, :]
    design_matrix = compute_design_matrix(X, centers, spreads)
    error=calculateErrorWithReg(design_matrix,np.transpose(w),t_test_set,lambd)
    error_rms=calculateRMSError(error,N)
    print("Error ",error)
    print("Error RMS",error_rms)


# In[15]:
if __name__ == "__main__":
    print("********************************************************************************")
    print("UBitName:prasadde")
    print("personNumber:50207353")
    print("UBitName:veerappa")
    print("personNumber:50247314")
    print("UBitName:sarahmul")
    print("personNumber:34508498")
    print("*********************************** LETOR **************************************")
    letor_data()
    print("*********************************** SYN ****************************************")
    syn_data()


