# -*- coding: utf-8 -*-
import random
from math import exp
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 


# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     


def matrixDotProduct (matrx1,matrx2):
    dotProduct = np.dot(matrx1,matrx2)
    
    return dotProduct


def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum           
    return weight


def initializeWeightArray (numBottomNodes,numUpperNodes):    
    #numBottomNodes = weightArraySizeList[0]
    #numUpperNodes = weightArraySizeList[1]

# Initialize the weight variables with random weights    
    weightArray = np.zeros((numUpperNodes,numBottomNodes)) 
    for row in range(numUpperNodes):
        for col in range(numBottomNodes):
            weightArray[row,col] = InitializeWeight ()
                     
    return weightArray  


def initializeBiasWeightArray (numBiasNodes):

    biasWeightArray = np.zeros(numBiasNodes)    
    for node in range(numBiasNodes): 
        biasWeightArray[node] = InitializeWeight ()                  
    
    return biasWeightArray  


# Function to obtain a specified neural network
def obtainNeuralNetwork(n_input,n_hidden,n_output):
                
    # The node-to-node connection weights are stored in a 2-D array
    wWeightArray = initializeWeightArray(n_input,n_hidden)
    vWeightArray = initializeWeightArray(n_hidden,n_output)      
    biasHiddenWeightArray = initializeBiasWeightArray(n_hidden)
    biasOutputWeightArray = initializeBiasWeightArray(n_output)
    
    return wWeightArray,vWeightArray,biasHiddenWeightArray,biasOutputWeightArray


def obtainRandomAlphaTrainingValues(trainingDataSet):
    # We are starting with five letters in the training set: X, M, N, H, and A
    # Thus there are five choices for training data, which we'll select on random basis
    
    trainingDataSetNum = random.randint(0, len(trainingDataSet.keys())-1)

    return trainingDataSet[trainingDataSetNum]

            
# Perform a single feedforward pass
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
def ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray):

    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray = np.zeros(hiddenArrayLength)    
    hiddenArray = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray = matrixDotProduct (wWeightArray,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput=sumIntoHiddenArray[node]+biasHiddenWeightArray[node]
        hiddenArray[node] = computeTransferFnctn(hiddenNodeSumInput, alpha)
             
    return hiddenArray


####################################################################################################
#
# Function to compute the output node activations, given the hidden node activations, the hidden-to
#   output connection weights, and the output bias weights.
# Function returns the array of output node activations.
#
####################################################################################################

def ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray, vWeightArray, biasOutputWeightArray):

    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray = matrixDotProduct (vWeightArray,hiddenArray)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput=sumIntoOutputArray[node]+biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)
                                                                                                   
    return outputArray


def BackpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):
# Unpack array lengths
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)

                        
    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row]*transferFuncDerivArray[row]*hiddenArray[col]
            deltaVWtArray[row,col] = -eta*partialSSE_w_V_Wt
            newVWeightArray[row,col] = vWeightArray[row,col] + deltaVWtArray[row,col]                                                                                     
                                                                                                                             
    return newVWeightArray

            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-output connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

    outputArrayLength = arraySizeList [2]

    deltaBiasOutputArray = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput = -errorArray[node]*transferFuncDerivArray[node]
        deltaBiasOutputArray[node] = -eta*partialSSE_w_BiasOutput  
        newBiasOutputWeightArray[node] =  biasOutputWeightArray[node] + deltaBiasOutputArray[node]           
                                                                                                                                                
    return (newBiasOutputWeightArray);     


    ####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the input-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]              
                                          
    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)
        
    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] \
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
             
    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):
        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row]*inputArray[col]*weightedErrorArray[row]
            deltaWWtArray[row,col] = -eta*partialSSE_w_W_Wts
            newWWeightArray[row,col] = wWeightArray[row,col] + deltaWWtArray[row,col]
                                                                    
    return newWWeightArray
    
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def BackpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]                         

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode]
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
            

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode]*weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta*partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return newBiasHiddenWeightArray
    
            

def eval_training(training_data,mod,outputArray,verbose=False):
    """Mod should be number of output nodes - 1"""
    desiredOutputArray = np.zeros(outputArray)
    desiredOutputArray[training_data['target']] = 1
    
    if verbose:
        print "The target letter is %s"%(training_data['letter'])
        print "The input data looks like:"
        
        print_list = []
        for i,d in enumerate(training_data['data']):
            print_list.append(d)
            if i!=0 and (i+1)% mod == 0:
                print print_list
                print_list = []
        print
        print ' The desired output array values are: '
        print desiredOutputArray  
        print
        
    return desiredOutputArray

def main(trainingData, numInputNodes, numHiddenNodes, numOutputNodes,alpha=1.0,eta=.5,maxNumIterations=5000,epsilon=.05,verbose=False):

    iteration=0
    avg_sse=0
    threshold_count=0
    letter_errors = {trainingData[k]['letter']:[] for k in trainingData.keys()}
    letter_errors['avg_sse']=[]

    arraySizeList = (numInputNodes,numHiddenNodes,numOutputNodes)
    wWeightArray,vWeightArray,biasHiddenWeightArray,biasOutputWeightArray = obtainNeuralNetwork(numInputNodes,numHiddenNodes,numOutputNodes) 
    
    while iteration < maxNumIterations:           
        inputDataList = []                                
                                                                                          
        training_data = obtainRandomAlphaTrainingValues (trainingData) 
        inputDataList = training_data['data']        

        desiredOutputArray = eval_training(training_data,arraySizeList[0]**.5,numOutputNodes)
                
        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray)
        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, arraySizeList, hiddenArray, vWeightArray, biasOutputWeightArray)
 
        # Initialize the error array
        errorArray = np.zeros(numOutputNodes) 
    
        # Determine the error between actual and desired outputs
        newSSE = 0.0
        for node in range(numOutputNodes):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE += errorArray[node]*errorArray[node]        
        
        avg_sse = ((avg_sse*iteration) + newSSE)/float(iteration+1.0)
        if verbose:
        
            print ' '
            print ' The error values are:'
            print errorArray       
            # Print the Summed Squared Error  
            print 'Average SSE = %.6f' % avg_sse
            print 'New SSE = %.6f' % newSSE 
            print ' '
            print 'Iteration number ', iteration
        iteration = iteration + 1
        
        #capture errors
        letter_errors['avg_sse'].append(avg_sse)
        letter_errors[training_data['letter']].append(newSSE)
        
        if newSSE < epsilon:
            threshold_count+=1
            if threshold_count >=25:      
                #package up the nn into a dictionary
                trained_nn = {'i_to_h_weights':wWeightArray,
                    'i_to_h_bias':biasHiddenWeightArray,
                    'h_to_o_weights':vWeightArray,
                    'h_to_o_bias': biasOutputWeightArray}                
                break
        else:
            threshold_count=0

        #Backprop  
        newVWeightArray = BackpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray)
        newBiasOutputWeightArray = BackpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray) 

        newWWeightArray = BackpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)

        newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)  
    
        vWeightArray = newVWeightArray[:]
        biasOutputWeightArray = newBiasOutputWeightArray[:]
    
        wWeightArray = newWWeightArray[:]  
        biasHiddenWeightArray = newBiasHiddenWeightArray[:] 
    
    else:
        #max iterations reached
        trained_nn = {'i_to_h_weights':wWeightArray,
                    'i_to_h_bias':biasHiddenWeightArray,
                    'h_to_o_weights':vWeightArray,
                    'h_to_o_bias': biasOutputWeightArray}
                    
    print '\nTraining Finished'
    print "Average SSE: %.6f"%avg_sse
    print "Total Iterations: %d\n"%iteration
    return trained_nn,letter_errors
    

###### outside of main class
### training data
trainingData = {0:{"data":[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
                    "target":0,
                    "letter":'X'},
                1: {"data":[1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
                    "target":1,
                    "letter":'M'},
                2: {"data":[1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
                    "target":2,
                    "letter":'N'},
                3: {"data":[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
                    "target":3,
                    "letter":'H'},
                4: {"data":[0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
                    "target":4,
                    "letter":'A'},
                5: {"data":[1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                    "target":5,
                    "letter":"P"},
                6: {"data":[1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
                    "target":6,
                    "letter":"R"},
                7: {"data":[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1],
                    "target":7,
                    "letter":"I"},
                8: {"data":[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                    "target":8,
                    "letter":"T"},
                9: {"data":[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
                    "target":9,
                    "letter":"U"},
                10: {"data": [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],
                     "target":10,
                     "letter":"W"},
                11: {"data":[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                     "target":11,
                     "letter":"V"},
                12: {"data":[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                     "target":12,
                     "letter":"Y"},
                13: {"data":[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                     "target":13,
                     "letter":"Z"},
                14: {"data":[1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0],
                     "target":14,
                     "letter":"B"},
                15: {"data":[0,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
                     "target":15,
                     "letter":"C"},
                16: {"data":[1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,0],
                     "target":16,
                     "letter":"D"},
                17: {"data":[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                     "target":17,
                     "letter":"E"},
                18: {"data":[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                     "target":18,
                     "letter":"F"},
                19: {"data":[0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0],
                     "target":19,
                     "letter":"G"},
                20: {"data":[1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0],
                     "target":20,
                     "letter":"J"},
                21: {"data":[1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],
                     "target":21,
                     "letter":"K"},
                22: {"data":[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                     "target":22,
                     "letter":"L"},
                23: {"data":[0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0],
                     "target":23,
                     "letter":"O"},
                24: {"data":[0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1],
                     "target":24,
                     "letter":"Q"},
                25: {"data":[0,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0],
                     "target":25,
                     "letter":"S"}
                }
                
### limit the data set the network sees
for b in range(19,20):
    print b
    n=b
    subset_train = np.random.choice(trainingData.keys(),n,False)
    mini_train = {}

    for i,j in enumerate(subset_train):
        temp = trainingData[j]
        temp['target']=i
        mini_train[i] = temp
        
    
    nn,letter_errors = main(mini_train,81,6,len(mini_train.keys()),alpha=1.0,eta=.5,maxNumIterations=5000,epsilon=.05)
    alpha = 1.0
    arraySizeList = [len(nn['i_to_h_weights'][0]),len(nn['i_to_h_weights']),len(nn['h_to_o_weights'])]
    
    hidden_layers = {}
    outputs = {}
    targets = {}
    y_true = []
    y_pred = []
    sse= 0.0
    network_sse = {}
    
    for k in mini_train.keys():
        targets[mini_train[k]['letter']]=eval_training(mini_train[k],arraySizeList[0]**.5,arraySizeList[2],verbose=False)
        y_true.append(mini_train[k]['letter'])
        
        #this part tells me the activation weights
        hidden = ComputeSingleFeedforwardPassFirstStep(alpha,arraySizeList,mini_train[k]['data'],nn["i_to_h_weights"],nn["i_to_h_bias"])
        hidden_layers[mini_train[k]['letter']] = hidden
        
        #this grabs the predicted output
        output = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hidden, nn["h_to_o_weights"],nn["h_to_o_bias"])
        outputs[mini_train[k]['letter']]= output
        sse = sum((targets[mini_train[k]['letter']]-outputs[mini_train[k]['letter']])**2)
        class_prediction = mini_train[np.argmax(output)]['letter']
        y_pred.append(class_prediction)
        network_sse[mini_train[k]['letter']] = [sse,class_prediction]
        
    df_sse = pd.DataFrame(network_sse).T        
    df_hidden = pd.DataFrame(hidden_layers)
    df_outputs = pd.DataFrame(outputs)
    
    #### Plot errors over training session
    
    max_length = len(letter_errors['avg_sse'])
    max_letter_len = 0
    for i in letter_errors.keys():
        vector_length = len(letter_errors[i])
        
        if  vector_length < max_length:
            if vector_length > max_letter_len:
                max_letter_len = vector_length
            for j in range(max_length - vector_length):
                letter_errors[i].append(np.nan)
            
    df_errs = pd.DataFrame(letter_errors)
    
    #SSE per training 
    plt.figure()
    ax = df_errs[[col for col in df_errs.columns if col!='avg_sse']].plot()
    ax.set_title("Letter SSE Per Iteration: "+str(b))
    ax.set_xlim(0,max_letter_len)
    ax.set_xlabel("Ordered Frequency of Training")
    ax.set_ylabel("Sum of Squares Error")
    plt.show()
    
    #Avg SSE over iterations
    plt.figure()
    ax = df_errs['avg_sse'].plot()
    ax.set_title("Average SSE per Iteration: "+str(b))
    ax.set_xlim(0,len(df_errs))
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Sum of Squares Error")
    plt.show()
    
    # Heatplot of Hidden Layer
    plt.figure()
    ax = sns.heatmap(df_hidden,vmin=0,vmax=1,linewidths=.2)
    ax.set_title("Hidden Layer Weights Heatmap: "+str(b))
    ax.set_xlabel("Letter")
    ax.set_ylabel("Hidden Node")
    plt.show()
    
    #confusion matrix
    c_mat = confusion_matrix(y_true,y_pred)
    class_labels = [mini_train[l]['letter'] for l in range(len(y_true))]
    plt.figure()
    ax = sns.heatmap(c_mat,vmin=0,vmax=1,linewidths=.1,cbar=False,xticklabels=sorted(y_true),yticklabels=sorted(y_true))
    ax.set_title("Confusion Matrix: "+str(b))
    ax.set_ylabel("Actual Classes")
    ax.set_xlabel("Predicted Classes")
    plt.show()

    #SSE for each letter
    plt.figure()
    ax = sns.barplot(x=df_sse.index,y=df_sse[0])
    ax.set_title("Trained Network SSE per Letter"+str(b))
    ax.set_ylabel("Sum of Squares Error")
    plt.show()
