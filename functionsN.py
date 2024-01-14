import numpy as np
from numba import njit

data = np.loadtxt(('zipcombo.dat'))

#load kernel matrices to be used
poly_mat = np.load('data/poly_mat.npy')
gauss_mat = np.load('data/gauss_mat.npy')
    
#kernel matrices used to massively speed up computation, as well as remove redundancy of calculating kernels repeatedly between questions
#also allows the polynomial and gaussian kernel questions to use the same training function

#using indices instead of data matrices helps:
# facilitate tracking of individual mistakes
# instantiate and track train/test splits
# access rows
# use of kernel matrices

@njit
def class_pred(mat, idx, alphas):
    '''
    Computes the predictions made by each perceptron for a given vector by multiplying the alphas matrix with the kernel values of that vector.
    INPUTS:
    mat:        (#points,#points) kernel matrix
    idx:        index of vector to evaluate
    alphas:     (#labels,#points) matrix of values of alpha in the dual problem
    OUTPUT:
    preds:      (#labels,) vector of predictions for each label 
    '''
    kervals = np.ascontiguousarray(mat[idx])        #made contiguous to work with numba
    preds = alphas @ kervals.reshape(-1,1)          
    return preds.ravel()                            #ensures result is 1D

def init_alphas(data_idx, rows=10):
    '''
    Instantiate the alphas matrix, controlling rows and columns.
    '''
    alphas = np.zeros((rows,int(len(data_idx))))
    return alphas

# # # ONE-VERSUS-ALL IMPLEMENTATION # # #

#training algorithm:
#have num_label individual classifiers which each classify whether it is or isnt that label
#on each point, we see how the classifier predicts, and update each classifier's coefficients which made a mistake
#the overall classifier's prediction is the given by the classifier with the largest prediction 

def train_perceptron(data_idx, alphas, kernel_mat, exponent):
    '''
    Trains a multiclass perceptron by simultaneously training 10 one-vs-all binary perceptrons.
    INPUTS:
    data_idx:    index vector
    alphas:      (#labels,#points) matrix of values of alpha in the dual problem
    kernel_mat:  kernel matrix (polynomial or gaussian)
    OUTPUT:
    error_rate, alphas
    '''
    data = np.loadtxt(('zipcombo.dat'))

    #both the polynomial and gaussian kernel matrices just need to be raised to a power
    kernel_mat_exp = (kernel_mat ** exponent)[:,data_idx]

    mistakes = 0

    #for each training point
    for i, idx in enumerate(data_idx):

        label = data[idx,0]

        #obtain prediction made by each classifier
        preds = class_pred(kernel_mat_exp, idx, alphas)
        
        preds_binary = np.where(preds <= 0, -1, 1)

        #check which classifier made a mistake
        truth_col = -np.ones(10)
        truth_col[int(label)] = 1
        is_pred_wrong = (preds_binary != truth_col).astype(np.int32)          #a vector which has a 1 if the kth classifier was wrong

        #update the classifier according to the update step given in the problem sheet by adding 1/-1 to the appropriate alphas
        alphas[:,i] -= is_pred_wrong*preds_binary

        #mistake is made if the classifier most confident in its prediction is wrong
        #add mistake
        if np.argmax(preds) != label:
            mistakes += 1
    
    error_rate = mistakes/len(data)

    results = {'error_rate':error_rate, 'alphas':alphas}

    return results

#testing algorithm
def test_perceptron(train_idx, test_idx, kernel_mat, alphas, exponent, calc_conf=False, keep_mistakes=True):
    '''
    Tests a multiclass perceptron by checking the most confident classifier against the true label.
    INPUTS:
    train_idx:    train index vector
    test_idx:     test index vector
    kernel_mat:   kernel matrix (polynomial or gaussian)
    alphas:       (#labels,#points) matrix of values of alpha in the dual problem
    exponent:     number to raise the kernel matrix to; d or c
    calc_conf:    boolean; whether to return the confusion matrix
    keep_mistakes:boolean; whether to return the mistake idxs
    OUTPUT:
    error_rate, alphas, (conf_mat, mistakes_idx) 
    '''
    mistakes = 0
    conf_mat = np.zeros((10,10))

    #allows us to track the idx which are predicted wrong
    mistakes_idx = np.zeros(len(data))

    #apply exponent and restrict to training data
    kernel_mat_exp = (kernel_mat ** exponent)[:,train_idx]

    for i in test_idx:
        label = data[i,0]

        preds = class_pred(kernel_mat_exp, i, alphas)

        if int(np.argmax(preds)) != int(label):
            mistakes += 1
            mistakes_idx[i] += 1
            if calc_conf:
                conf_mat[int(label),int(np.argmax(preds))] += 1

    #normalise conf_mat
    if calc_conf:
        label_counts = np.bincount(data[test_idx,0].astype(int),minlength=10)
        label_counts[label_counts==0] = 1
        label_counts = label_counts.reshape(-1,1)

        #turn it into a rate
        conf_mat = conf_mat / label_counts

    error_rate = mistakes/len(test_idx)

    results = {'error_rate':error_rate}

    if calc_conf:
        results['conf_mat'] = conf_mat 
    if keep_mistakes:
        results['mistakes_idx'] = mistakes_idx

    return results 

#allows us to split indices into training and testing for regular use and cross-validation
def train_test_index_split(data_idx, k=5, CV=False):
    '''
    Shuffles data indices and splits into k bins, where the testing data takes 1 bin, and the training data takes k-1.
    INPUT:
    data_indices:               a 1D array of indices to be shuffled
    OUTPUT:
    if CV=False:
    train_idx, test_idx
    if CV=True
    '''
    n = len(data_idx)

    fold_sizes = np.full(k, n // k, dtype=int)  
    fold_sizes[:n % k] += 1 

    shuffled_indices = np.random.permutation(data_idx)

    #regular - returns train and test indices
    if not CV:
        train_idx = shuffled_indices[fold_sizes[0]:]
        test_idx = shuffled_indices[:fold_sizes[0]]
        return train_idx, test_idx
    else:
    #CV - returns dictionary of train and test indices    
        train_idx = {i: [] for i in range(k)}
        test_idx = {i: [] for i in range(k)}
        for i in range(k):
            #go through each fold and take the test indices as the indices belonging to that fold, then take the rest as train
            test_ = np.arange(sum(fold_sizes[:i]),sum(fold_sizes[:i+1]))
            test_idx[i] = shuffled_indices[test_]
            train_idx[i] = np.setdiff1d(shuffled_indices, test_idx[i])

    return train_idx, test_idx


#automates the train and test process for ease of notebook reading, as well as adding easy access to epochs
def train_test_basic(train_idx, test_idx, kernel_mat, exponent, epochs=3, calc_conf=False, keep_mistakes=False):
    '''
    Runs the train and test methods.
    '''
    alphas = init_alphas(train_idx)

    train_results = train_perceptron(train_idx, alphas, kernel_mat, exponent)
    for e in range(epochs-1):
        train_results = train_perceptron(train_idx, train_results['alphas'], kernel_mat, exponent)    #uses the alphas just trained

    test_results = test_perceptron(train_idx, test_idx, kernel_mat, train_results['alphas'], exponent, calc_conf, keep_mistakes)

    return train_results, test_results

#automates the cross validation
def cross_val(data_idx, kernel_mat, exponent, epochs=3, k=5):
    '''
    Returns the average error rate by cross-validating over a data set.
    '''
    train_idx, test_idx, = train_test_index_split(data_idx, k, CV=True)

    score = np.zeros(k)
    
    for i in range(k):
        _, r_test = train_test_basic(train_idx[i], test_idx[i], kernel_mat, exponent, epochs)

        score[i] = r_test['error_rate']
    
    return score.mean()

#automates question 2 by combining cross validation and training and testing
def find_best_val(data_idx, kernel_mat, candidates, epochs=3, k=5, calc_conf=True, keep_mistakes=True):

    train_idx, test_idx = train_test_index_split(data_idx)

    scores = np.array([cross_val(train_idx, kernel_mat, candidates[i], epochs) for i in range(len(candidates))])

    v_star = candidates[np.argmin(scores)]

    #train perceptron with v_star
    r_train, r_test = train_test_basic(train_idx, test_idx, kernel_mat, v_star, calc_conf, keep_mistakes)

    return v_star, r_train, r_test


# # # ONE-VERSUS-ONE IMPLEMENTATION # # #
#instead of 10 classifiers giving confidence to one label, we have 45 classifiers comparing labels one-on-one
#we treat it as 45 binary classification problems; each classifier is only trained on the 2 labels it compares
#this can be implemented easily by slightly changing the first training algorithm:
#   each classifier has a different number of relevant training points, so each classifier (in its dual form) has a different elements in its kernel sum
#   to preserve the fast implementation we have before storing the alphas in a matrix, we simply embed these classifiers in a 45 x (#training points) matrix
#   then as we move along the training points, and along the alphas, we only predict and train with classifiers relevant to the current label
#   this forces the alphas for irrelevant training points to be 0 (their starting point)


def train_perceptron_ovo(data_idx, alphas, kernel_mat, exponent):
    #same general idea as train_perceptron, but with OVO implementation
    class_pairs = np.array([[i, j] for i in range(10) for j in range(i+1, 10)])

    kernel_mat_exp = (kernel_mat ** exponent)[:,data_idx]

    mistakes = 0

    #for each training point
    for i, idx in enumerate(data_idx):

        label = int(data[idx,0])

        #only train the perceptrons containing that label
        valid_pairs_idx = np.where((class_pairs[:,0] == label) | (class_pairs[:,1] == label))[0]

        #obtain prediction made by each relevant classifier
        preds = class_pred(kernel_mat_exp, idx, alphas[valid_pairs_idx,:])      #prevents unnecessary computation
        
        preds_binary = np.where(preds <= 0, -1, 1)

        #check which classifier made a mistake
        truth_col = -np.ones(9)        #there are only ever 9 relevant classifiers (comparing one of the 10 labels with the other 9)
        truth_col[label:] = 1          #due to the way the class pairs were instantiated, this makes the truth cols correct
        is_pred_wrong = (preds_binary != truth_col).astype(np.int32)          #a vector which has a 1 if the kth classifier was wrong

        #update the alpha of the relevant classifiers
        alphas[valid_pairs_idx,i] -= is_pred_wrong*preds_binary

        # MAYBE WRONG - this isnt the way we predict using OVO. Really, we should have made predictions with every classifier and taken the most voted, but then we are increasing the prediction calculations (most expensive part) by 400%
        #add mistake?
        if sum(is_pred_wrong)>4:         #if the majority of the classifiers including 'label' get it wrong
            mistakes += 1

    error_rate = mistakes/len(data)

    results = {'error_rate':error_rate, 'alphas':alphas}

    return results

#testing algorithm
def test_perceptron_ovo(train_idx, test_idx, kernel_mat, alphas, exponent, keep_mistakes=True):
    '''
    Tests a multiclass perceptron by checking the label with the most votes against the true label.
    INPUTS:
    train_idx:    train index vector
    test_idx:     test index vector
    kernel_mat:   kernel matrix (polynomial or gaussian)
    alphas:       (#labels,#points) matrix of values of alpha in the dual problem
    exponent:     number to raise the kernel matrix to; d or c
    calc_conf:    boolean; whether to return the confusion matrix
    keep_mistakes:boolean; whether to return the mistake idxs
    OUTPUT:
    error_rate, alphas, (conf_mat, mistakes_idx) 
    '''
    class_pairs = np.array([[i, j] for i in range(10) for j in range(i+1, 10)])

    mistakes = 0

    mistakes_idx = np.zeros(len(data))

    kernel_mat_exp = (kernel_mat ** exponent)[:,train_idx]

    for i in test_idx:
        label = data[i,0]

        #obtain predictions
        preds = class_pred(kernel_mat_exp, i, alphas)
        preds_binary = np.where(preds < 0, 1, 0)       #1 (2nd col) if <0, 0 (1st col) otherwise
        preds_classes = [class_pairs[i,idx] for i, idx in enumerate(preds_binary)]      #takes the prediction made by each classifier
        final_pred = np.argmax(np.bincount(preds_classes))           #counts the # of votes for each label and returns the largest

        if int(final_pred) != int(label):
            mistakes += 1
            mistakes_idx[i] += 1

    error_rate = mistakes/len(test_idx)

    results = {'error_rate':error_rate}

    if keep_mistakes:
        results['mistakes_idx'] = mistakes_idx

    return results 


#reimplementing the train_test_basic algorithm with OVO
def train_test_basic_ovo(train_idx, test_idx, kernel_mat, exponent, epochs=3, keep_mistakes=False):
    '''
    Runs the train and test methods

    
    '''
    alphas = init_alphas(train_idx, rows=45)

    train_results = train_perceptron_ovo(train_idx, alphas, kernel_mat, exponent)
    for e in range(epochs-1):
        train_results = train_perceptron_ovo(train_idx, train_results['alphas'], kernel_mat, exponent)

    test_results = test_perceptron_ovo(train_idx, test_idx, kernel_mat, train_results['alphas'], exponent, keep_mistakes)

    return train_results, test_results

#same, but with OVO
def cross_val_ovo(data_idx, kernel_mat, exponent, epochs=3, k=5):
    '''
    Returns the average error rate by cross-validating over a data set.

    
    '''
    train_idx, test_idx, = train_test_index_split(data_idx, k, CV=True)

    score = np.zeros(k)
    
    for i in range(k):
        _, r_test = train_test_basic_ovo(train_idx[i], test_idx[i], kernel_mat, exponent, epochs)

        score[i] = r_test['error_rate']
    
    return score.mean()

#same, but with OVO
def find_best_val_ovo(data_idx, kernel_mat, candidates, epochs=3, k=5, keep_mistakes=True):
    '''
    Implements the method in question 2.
    
    '''
    train_idx, test_idx = train_test_index_split(data_idx)

    scores = np.array([cross_val_ovo(train_idx, kernel_mat, candidates[i], epochs) for i in range(len(candidates))])

    v_star = candidates[np.argmin(scores)]

    #train perceptron with v_star
    r_train, r_test = train_test_basic_ovo(train_idx, test_idx, kernel_mat, v_star, keep_mistakes)

    return v_star, r_train, r_test
