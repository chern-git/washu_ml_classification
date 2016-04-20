import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

def strip_punctuation(s):
    return ''.join([i for i in s if i not in string.punctuation])

def get_numpy_data(df, features, label):
    df['constant'] = 1
    features = ['constant'] + features
    features_frame = df[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = df[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)

def predict_probability(feature_matrix, coefs):
    '''
    produces probablistic estimate for P(y_i = +1 | x_i, w).
    estimate ranges between 0 and 1.
    '''
    return 1/(1+np.exp(-np.dot(feature_matrix, coefs)))

def feature_derivative(errors, feature):
    return np.dot(errors,feature)

def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)
    return lp

def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    """
    * Create an empty list called log_likelihood_all
    * Initialize coefficients to initial_coefficients
    * Set random seed = 1
    * Shuffle the data before starting the loop below
    * Set i = 0, the index of current batch

    * Run the following steps max_iter times, performing linear scans over the data:
      * Predict P(y_i = +1|x_i,w) using your predict_probability() function
        Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,:]
      * Compute indicator value for (y_i = +1)
        Make sure to slice the i-th entry with [i:i+batch_size]
      * Compute the errors as (indicator - predictions)
      * For each coefficients[j]:
        - Compute the derivative for coefficients[j] and save it to derivative.
          Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,j]
        - Compute the product of the step size, the derivative, and (1./batch_size).
        - Increment coefficients[j] by the product just computed.
      * Compute the average log likelihood over the current batch.
        Add this value to the list log_likelihood_all.
      * Increment i by batch_size, indicating the progress made so far on the data.
      * Check whether we made a complete pass over data by checking
        whether (i+batch_size) exceeds the data size. If so, shuffle the data. If not, do nothing.

    * Return the final set of coefficients, along with the list log_likelihood_all.
    """

    log_likelihood_all = []
    coefficients = np.array(initial_coefficients)

    # Shuffling data
    np.random.seed(seed = 1)
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]

    i = 0
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix[i:i+batch_size,:], coefficients)
        indicator = (sentiment[i:i+batch_size] == +1)
        errors = indicator - predictions

        for j in range(len(coefficients)):
            derivative = feature_derivative(errors,feature_matrix[i:i+batch_size,j])
            coefficients[j] += (1./batch_size) * step_size * derivative

        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:],
                                        sentiment[i:i+batch_size], coefficients)
        log_likelihood_all.append(lp)
        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
         or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print('Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp))

        # if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0
    return coefficients, log_likelihood_all

def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')
    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})

# Pre-processing - Questions 1-8
products = pd.read_csv('data/2/amazon_baby_subset.csv')
products['review'] = products['review'].fillna('')
products['review_clean'] = products['review'].apply(strip_punctuation)

im_json = pd.read_json('data/7/important_words.json')
im_words = [str(s) for s in im_json[0]]
for w in im_words:
    products[w] = products['review_clean'].apply(lambda x: x.split().count(w))

# Splitting data
train_idx = pd.read_json('data/7/module-10-assignment-train-idx.json')
val_idx = pd.read_json('data/7/module-10-assignment-validation-idx.json')
train_data = products.iloc[train_idx[0]]
val_data = products.iloc[val_idx[0]]
feature_matrix_train, sentiment_train  = get_numpy_data(train_data, im_words , 'sentiment')
feature_matrix_val, sentiment_val  = get_numpy_data(val_data, im_words , 'sentiment')

# Modifying derivative for SGA - Questions 11
j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which we are computing the gradient.
predictions = predict_probability(feature_matrix_train[i:i+1,:], coefficients)
indicator = (sentiment_train[i:i+1]==+1)
errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i+1,j])
print("Gradient single data point: %s" % gradient_single_data_point)
print("           --> Should print 0.0")

# Modifying the derivative for using batch of Data points - Questions 12
j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which we are computing the gradient.
predictions = predict_probability(feature_matrix_train[i:i+B,:], coefficients)
indicator = (sentiment_train[i:i+B]==+1)
errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i+B,j])
print("Gradient mini-batch data points: %s" % gradient_mini_batch)
print("                --> Should print 1.0")

# Checker
sample_feature_matrix = np.array([[1.,2.,-1.], [1.,0.,1.]])
sample_sentiment = np.array([+1, -1])
coefficients, log_likelihood = logistic_regression_SG(sample_feature_matrix, sample_sentiment, np.zeros(3),
                                                  step_size=1., batch_size=2, max_iter=2)
print('-------------------------------------------------------------------------------------')
print('Coefficients learned                 :', coefficients)
print('Average log likelihood per-iteration :', log_likelihood)
if np.allclose(coefficients, np.array([-0.09755757,  0.68242552, -0.7799831]), atol=1e-3)\
  and np.allclose(log_likelihood, np.array([-0.33774513108142956, -0.2345530939410341])):
    # pass if elements match within 1e-3
    print('-------------------------------------------------------------------------------------')
    print('Test passed!')
else:
    print('-------------------------------------------------------------------------------------')
    print('Test failed')

# Implementing SGA  - Questions 14-17
# Running SGA (setup 1)
initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = 1
max_iter = 10
sga_1_coeff, sga_1_lla = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                                initial_coefficients, step_size, batch_size, max_iter)
# Avg LL fluctuates

# Running SGA (setup 2)
initial_coefficients = np.zeros(194)
step_size = 5e-1
batch_size = len(feature_matrix_train)
max_iter = 200
sga_2_coeff_, sga_2_lla = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                                 initial_coefficients, step_size, batch_size, max_iter)
# Avg LL increases

# Making passes over data set - Questions 18
# 2 passes, size of data set = 50K
# number of points touched = 2 pass * data set size = 100K
# with batch size = 100, 100K/100 = 1K grad updates performed

# With 10 passes, size of data set = 10 * data set size
# Number of grad updates performed = .1 * data set size

# Log-likelihood plots for SGA - Questions 19, 20
initial_coefficients = np.zeros(194)
batch_size = 100
step_size = 1e-1
num_passes = 10
num_iter = num_passes * int(len(feature_matrix_train)/batch_size)
coeffs, lla = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                     initial_coefficients, step_size, batch_size, max_iter = num_iter)
make_plot(lla, len(feature_matrix_train) , batch_size, label='Step size = 1e-1, Smoothing window = 1')
# Smoothing the SGA curve
make_plot(lla, len(feature_matrix_train) , batch_size, smoothing_window= 30,
          label = 'Step size = 1e-1, Smoothing window = 30')
plt.savefig('plots\ml-class7-plot.png')
# SGA vs BGA - Questions 21, 22
# SGA part
step_size = 1e-1
initial_coefficients = np.zeros(194)
num_passes = 200
num_iter = num_passes * int(len(feature_matrix_train)/batch_size)
coeffs_sga, lla_sga = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients,
                                     step_size, batch_size = 100, max_iter = num_iter)
make_plot(lla_sga, len(feature_matrix_train) , batch_size = 100, smoothing_window= 30,
          label= 'Stochastic gradient, step size = 1e-1')

# BGA part
coeffs_bga, lla_bga = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients,
                                     step_size = 0.5, batch_size = len(feature_matrix_train), max_iter = 200)
make_plot(lla_bga, len(feature_matrix_train) , batch_size = len(feature_matrix_train),
          smoothing_window = 1, label = 'Batch gradient, step_size=5e-1')
plt.savefig('plots\ml-class7-stoch_vs_batch_passes.png')
# More than 150 passes needed

# Exploring effects of step size on SGA - Questions 23, 24
step_size_arr = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
step_sz_lla = []
num_passes = 10
num_iter = num_passes * int(len(feature_matrix_train)/batch_size)
for i in range(len(step_size_arr)):
    coeffs, lla = logistic_regression_SG(feature_matrix_train, sentiment_train,
                                         initial_coefficients = np.zeros(194), step_size = step_size_arr[i],
                                         batch_size = 100, max_iter = num_iter)
    step_sz_lla.append(lla)
for i in range(len(step_size_arr)):
    print(step_size_arr[i],np.mean(step_sz_lla[i]))
    # Charting
    make_plot(step_sz_lla[i], len(feature_matrix_train) , batch_size = 100,
              smoothing_window = 30, label = 'batch, step_size=%.1e'%step_size_arr[i])
plt.savefig('plots\ml-class7-stepsize_comparison.png')