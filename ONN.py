#------------------------------------------------------------------------------+
#
#   Anastasios Papathanasopoulos & Pavlos A. Apostolopoulos
#   Optimization Assisted by Neural Networks (ONN) Algorithm
#   Last Editted, December, 2022
#
#------------------------------------------------------------------------------+

#------------------------------------------------------------------------------+
#
#   Import Libraries and Path
#
#------------------------------------------------------------------------------+

import statistics
from json import load
import benchmark_functions as bf
import numpy as np
import csv
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import time
timestr = time.strftime("%Y-%m-%d %H %M") # Current time and date for the output optimization file
print(timestr)
import os
path = os.getcwd()
path = path +'/ONN_Optimization_Results/' # Create a folder under the same directory as the optimization code where the optimization results will be saved

#------------------------------------------------------------------------------+
#
#   Optimization Definitions and Hyperparameters
#
#------------------------------------------------------------------------------+

n_dimensions = 4 # n_dimensions of optimization benchmark function
s_init = 4 # number of initial samples (initial dataset)
num_candidates = n_dimensions # number of candidate solutions (it has to be smaller than s_init)
num_new_candidates = 1 # new samples from the candidate solutions
num_predictors = 5 # number of neural network predictor functions f^(x)
n_iterations = 200 # number of maximum optimization iterations
alpha_mutation_var = 0.4 # initial mutation ratio of the n_dimensions
n_mutations = int(alpha_mutation_var*n_dimensions)+1 # number of dimensions to be mutated
n_opt = 10 # number of total optimization runs

#--- For Neural Network -------------------------------------------------------+

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODEL_DIM = 10*n_dimensions
EPOCHS = 2*int(s_init/BATCH_SIZE)+1

#------------------------------------------------------------------------------+
#
#   Cost Function
#
#------------------------------------------------------------------------------+

func = bf.Griewank(n_dimensions)
var_bounds = func.suggested_bounds()

#------------------------------------------------------------------------------+
#
#   Functions Definitions
#
#------------------------------------------------------------------------------+

def saving_predictors_optimizers(predictors,optimizers):
    for i in range(len(predictors)):
        predictor = predictors[i]
        optimizer = optimizers[i]
        predictor_state = {
            "state_dict": predictor.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        name = '/Opt'+str(i_opt+1)+"_"+ str(n_dimensions) + "dim_" + str(num_new_candidates) + "newcand_" + 'predictor_'+str(i)+'.pt'
        torch.save(predictor_state,path+name)

def loading_predictors_optimizers(num_predictors=num_predictors):
    predictors = []
    optimizers = []
    for i in range(num_predictors):
        name = '/Opt'+str(i_opt+1)+"_"+ str(n_dimensions) + "dim_" + str(num_new_candidates) + "newcand_" + 'predictor_'+str(i)+'.pt'
        predictor_state = torch.load(path+name)
        predictor = torch.nn.Sequential( 
            torch.nn.Linear(n_dimensions, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, 1),
        )
        predictor.load_state_dict(predictor_state['state_dict'])
        optimizer = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(predictor_state['optimizer'])
        predictors.append(predictor)
        optimizers.append(optimizer)
    return predictors, optimizers

def generate_inital_Xsamples(n_dimensions, s_init, var_bounds):
    # Inputs: Variable's Dimensions, Initial Sample Population, Variable Bounds
    # Output: Initial Sample X to be trained by the neural predictor
    X = [] # 
    for i in range(n_dimensions):
        X.append(random.uniform(var_bounds[0][i], var_bounds[1][i]))
    for j in range(s_init-1):
        X_loop = []
        for i in range(n_dimensions):
            X_loop.append(random.uniform(var_bounds[0][i], var_bounds[1][i]))
        X = np.vstack([X,X_loop])
        X = np.array(X)  
    return X

def generate_inital_Ysamples(X):
    # Inputs: Initial Sample X, Evaluation Function
    # Output: Y(X) Function Evaluation
    Y = []
    for i in range(len(X)):
        Y.append(func(X[i]))
    return Y

def generate_initial_predictors(num_predictors, dim_in):
    predictors = []
    optimizers = []
    for _ in range(num_predictors):
        net = torch.nn.Sequential( 
            torch.nn.Linear(dim_in, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, 1),
        )
        optimizers.append(torch.optim.Adam(net.parameters(), lr=LEARNING_RATE))
        predictors.append(net)
    return predictors, optimizers

def update_predictors(predictors, optimizers, X, Y):
    x_tensor = torch.from_numpy(np.array(X))
    y_tensor = torch.from_numpy(np.array(Y))
    loss_func = torch.nn.MSELoss()
    errors = []
    for i in range(len(predictors)):
        net = predictors[i]
        optimizer = optimizers[i]
        #optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        prediction = net(x_tensor.float())
        loss = loss_func(prediction, y_tensor.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(loss.detach())
    avg_error = torch.mean(torch.stack(errors))
    return predictors, optimizers, avg_error
        
def generate_predictors(X, Y, num_predictors): # Step 1
    # the X is list of lists, the Y is list
    # make them tensors
    BATCH_SIZE = int(len(X))
    x_tensor = torch.from_numpy(np.array(X))
    y_tensor = torch.from_numpy(np.array(Y))
    x_train, y_train = Variable(x_tensor), Variable(y_tensor)
    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
    all_errors = []
    predictors = []
    for _ in range(num_predictors):
        net = torch.nn.Sequential( 
            torch.nn.Linear(x_tensor.shape[1], MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, 1),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        loss_func = torch.nn.MSELoss()
        errors = []
        temp_errors = []
        for epoch in range(EPOCHS):
            for step, (batch_x, batch_y) in enumerate(loader):
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                prediction = net(b_x.float())
                loss = loss_func(prediction, b_y.float().view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp_errors.append(loss.detach())
            errors.append(torch.mean(torch.stack(temp_errors)))
        
        all_errors.append(errors)
        predictors.append(net)
    
    return all_errors, predictors

def gen_candidates(X, Y, num_candidates): # Step 2
    idx = np.argpartition(Y,num_candidates-1) #It does not sort the entire array. It only guarantees that the kth element is in sorted position and all smaller elements will be moved before it. Thus the first k elements will be the k-smallest elements.
    X_cand = []
    Y_cand = []
    for i in range(num_candidates):
        X_cand.append(X[idx[i]])
        Y_cand.append(Y[idx[i]])
    X_cand = np.array(X_cand)
    Y_cand = np.array(Y_cand)
    low = X_cand.min() # This is the low/high of all variables and works for benchmark functions with variables with same bounds
    high = X_cand.max()
    # low = np.amin(X_cand, axis=0) # This is the low/high of each variable and works best for  functions with variables with different bounds (ex. antennas)
    # high = np.amax(X_cand, axis=0)
    # find the num_candidates best out of Y->Y_cand and their corresponding X -> X_cand
    # find the lowest and highest value of X_cand to be used in the mutation
    X_mutated_candidate = []
    for candidate in X_cand:
        X_mutated_candidate.append(mutate(candidate, low, high, n_mutations))
    X_mutated_candidate = np.array(X_mutated_candidate)
    return X_mutated_candidate

def mutate(candidate, low, high, n_mutations):
    indx = random.sample(range(0, len(candidate)), n_mutations)
    # choose randomly from (0, len(x)-1) - indx
    # index = random.randint(0, len(candidate)-1)
    # generate a random value between low, high - random_number
    for i in range(n_mutations):
        random_number = random.uniform(low, high) # This is the low/high of all variables and works best for benchmark functions with variables with same bounds
        # random_number = random.uniform(low[indx[i]], high[indx[i]]) # This is the low/high of each variable and works for functions with variables with different bounds (ex. antennas)
        mutated_candidate = candidate
        mutated_candidate[indx[i]] = random_number
    return mutated_candidate

def find_new_candidates(X_mutated_candidate, predictors, num_predictors, num_new_candidates):
    # choose randomly from (0, len(num_predictors)-1) - predictor_indx
    new_candidates = []
    predictor_indx = random.randint(0, num_predictors-1)
    predictor = predictors[predictor_indx]
    predictor.eval()
    with torch.no_grad():
        pred = predictor(torch.from_numpy(np.array(X_mutated_candidate)).float()) # opou x candidate
        pred = pred.numpy()
        # min_value = pred.min()
        # min_index = np.where(pred == pred.min()) # index of element with min phi
    min_index = (np.argpartition(pred.transpose(), num_new_candidates-1)).transpose()
    for i in range(num_new_candidates):
        new_candidates.extend(X_mutated_candidate[min_index[i]])
    new_candidates = np.array(new_candidates)
    predictor.train()
    return new_candidates

    # the X is list of lists, the Y is list
    # make them tensors
    x_tensor = torch.from_numpy(np.array(X))
    y_tensor = torch.from_numpy(np.array(Y))
    x_train, y_train = Variable(x_tensor), Variable(y_tensor)
    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # all_errors = []
    # predictors = []
    for _ in range(num_predictors):
        net = torch.nn.Sequential( 
            torch.nn.Linear(x_tensor.shape[1], MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, MODEL_DIM),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(MODEL_DIM, 1),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        loss_func = torch.nn.MSELoss()
        errors = []
        temp_errors = []
        for epoch in range(EPOCHS):
            for step, (batch_x, batch_y) in enumerate(loader):
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                prediction = net(b_x.float())
                loss = loss_func(prediction, b_y.float().view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp_errors.append(loss.detach())
            errors.append(torch.mean(torch.stack(temp_errors)))
        
        all_errors.append(errors)
        predictors.append(net)
    
    return all_errors, predictors

#------------------------------------------------------------------------------+
#
#   Main Code
#
#------------------------------------------------------------------------------+

opt_duration = []
for i_opt in range(n_opt):
    torch.manual_seed(i_opt+1)
    random.seed(i_opt+1)
    
    start = time.perf_counter()

    # if i_opt==0:
    predictors, optimizers = generate_initial_predictors(num_predictors, n_dimensions)
    # else:
    # predictors, optimizers = loading_predictors_optimizers()

    #--- FILE WRITE ---------------------------------------------------------------+
    X = generate_inital_Xsamples(n_dimensions, s_init, var_bounds)
    Y = generate_inital_Ysamples(X)
    
    bestY_history = [min(Y)]
    min_index = np.where(Y == min(Y))
    bestX_history = X[min_index[0]]
    new_candidates, y_new_candidates = X, Y

    for iter in range(n_iterations):
        predictors, optimizers, error = update_predictors(predictors, optimizers, new_candidates, y_new_candidates)
        # if iter % 50 == 0:
        print('Iter ' + str(iter) + ' error: ' +  str(error))
        X_mutated_candidate = gen_candidates(X ,Y, num_candidates) # Create Mutation
        
        # Find new Candidate with min phi
        new_candidates = find_new_candidates(X_mutated_candidate, predictors, num_predictors, num_new_candidates) 
        y_new_candidates = []
        for i in range(num_new_candidates):
            y_new_candidates.append(func(new_candidates[i]))
        
        # Add the new candidate to X and Y  --- X, Y are not yet used from predictors, only the new candidates and their y values
        X = np.vstack([X,new_candidates])
        for i in range(num_new_candidates):
            Y.append(func(list(new_candidates[i])))
        bestY_history.append(min(Y))
        min_index = np.where(Y == min(Y))
        bestX_history = np.vstack([X,X[min_index[0]]]) 

        alpha_mutation = (1-iter/(n_iterations))*(alpha_mutation_var)
        n_mutations = int(alpha_mutation*n_dimensions)+1
        
        with open(path + str(n_iterations) + "iter_" + str(n_dimensions) + "dim_" + str(num_new_candidates) + "NewCand_" + str(MODEL_DIM) + "ModelDim_" + str(0.4*100) + "alphamutation_" + "Opt" + str(i_opt+1) + ".csv", mode='a') as Y_History_file:
            Y_History_writer = csv.writer(Y_History_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row =[str(iter+1)]
            for i in range(n_dimensions):
                row.append("{:.4f}".format(bestX_history[-1][i]) + " ")
            row.append("{:.4f}".format(min(Y)))
            Y_History_writer.writerow(row)
        print(min(Y))

    end = time.perf_counter()
    opt_duration.append(end-start) 

    saving_predictors_optimizers(predictors, optimizers)

with open(path + str(n_iterations) + "iter_" + str(n_dimensions) + "dim_" + str(num_new_candidates) + "NewCand_" + str(MODEL_DIM) + "ModelDim_" + str(0.4*100) + "alphamutation_Duration" + ".csv", mode='a') as Y_History_file:
    Y_History_writer = csv.writer(Y_History_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row =[]
    for i in range(len(opt_duration)):
        row.append("{:.4f}".format(opt_duration[i]) + " ")
    row.append("{:.4f}".format(statistics.mean(opt_duration)) + " ")
    Y_History_writer.writerow(row)


