import numpy as np
from numba import njit

data = np.loadtxt(('zipcombo.dat'))
poly_mat = np.load('results/poly_mat.npy')
gauss_mat = np.load('results/gauss_mat.npy')

@njit
def class_pred(mat, idx, alphas):
    '''
    INPUTS:
    mat:        (#points,#points) kernel matrix
    idx:        index of vector to evaluate
    alphas:     (#labels,#points) matrix of values of alpha in the dual problem
    OUTPUT:
    preds:      (#labels,) vector of predictions for each label 
    '''
    kervals = np.ascontiguousarray(mat[idx])
    preds = alphas @ kervals.reshape(-1,1)
    return preds.ravel()

def init_alphas(data_idx):
    alphas = np.zeros((10,int(len(data_idx))))
    return alphas

#training algorithm:
#have num_label individual classifiers which each classify whether it is or isnt that label
#on each point, we see how the classifier predicts, and update each classifier's coefficients which made a mistake
#the overall classifier's prediction is the classifier with the largest prediction 

def train_perceptron(data_idx, alphas, kernel_mat, exponent):
    '''
    INPUTS:
    data_idx:    index vector
    alphas:      (#labels,#points) matrix of values of alpha in the dual problem
    kernel_mat:  kernel matrix (polynomial or gaussian)
    OUTPUT:
    error_rate, alphas
    '''
    data = np.loadtxt(('zipcombo.dat'))

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

        #update alpha
        alphas[:,i] -= is_pred_wrong*preds_binary

        #add mistake
        if np.argmax(preds) != label:
            mistakes += 1
    
    error_rate = mistakes/len(data)

    results = {'error_rate':error_rate, 'alphas':alphas}

    return results

#testing algorithm
def test_perceptron(train_idx, test_idx, kernel_mat, alphas, exponent, calc_conf=False, keep_mistakes=True):
    '''

    '''
    mistakes = 0
    conf_mat = np.zeros((10,10))

    mistakes_idx = np.zeros(len(data))

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

    if not CV:
        train_idx = shuffled_indices[fold_sizes[0]:]
        test_idx = shuffled_indices[:fold_sizes[0]]
        return train_idx, test_idx
    else:
        train_idx = {i: [] for i in range(k)}
        test_idx = {i: [] for i in range(k)}
        for i in range(k):
            test_ = np.arange(sum(fold_sizes[:i]),sum(fold_sizes[:i+1]))
            test_idx[i] = shuffled_indices[test_]
            train_idx[i] = np.setdiff1d(shuffled_indices, test_idx[i])

    return train_idx, test_idx

def train_test_basic(train_idx, test_idx, kernel_mat, exponent, epochs=3, calc_conf=False, keep_mistakes=False):
    '''
    Runs the train and test methods

    
    '''
    alphas = init_alphas(train_idx)

    train_results = train_perceptron(train_idx, alphas, kernel_mat, exponent)
    for e in range(epochs-1):
        train_results = train_perceptron(train_idx, train_results['alphas'], kernel_mat, exponent)

    test_results = test_perceptron(train_idx, test_idx, kernel_mat, train_results['alphas'], exponent, calc_conf, keep_mistakes)

    return train_results, test_results

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

def find_best_val(data_idx, kernel_mat, candidates, epochs=3, k=5, calc_conf=True, keep_mistakes=True):
    '''
    Implements the method in question 2.
    
    '''
    train_idx, test_idx = train_test_index_split(data_idx)

    scores = np.array([cross_val(train_idx, kernel_mat, candidates[i], epochs) for i in range(len(candidates))])

    v_star = candidates[np.argmin(scores)]

    #train perceptron with v_star
    r_train, r_test = train_test_basic(train_idx, test_idx, kernel_mat, v_star, calc_conf, keep_mistakes)

    return v_star, r_train, r_test


