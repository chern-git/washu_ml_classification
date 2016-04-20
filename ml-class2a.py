import pandas as pd
import numpy as np
import string

def count_words(s):
    '''
    creates a dict of words and their respective counts from a string stripped of punctuation
    :param s: input string
    :return: dict with individual words as keys, and their respective counts as output
    '''
    word_list = s.split()
    word_count = [word_list.count(i) for i in set(word_list) if i in word_list]
    return dict(zip(word_list,word_count))

def strip_punctuation(s):
    return ''.join([i for i in s if i not in string.punctuation])

products = pd.read_csv('data/2/amazon_baby_subset.csv')
products['review'] = products['review'].fillna('')
products['review_clean'] = products['review'].apply(strip_punctuation)

im_json = pd.read_json('data/2/important_words.json')
im_words = [str(s) for s in im_json[0]]
for w in im_words:
    products[w] = products['review_clean'].apply(lambda x: x.split().count(w))

# products2 = products.copy()
# for w in im_words:
#     products2['words'] = products2['review_clean'].apply(lambda x: x.split().count(w))

# Counting for perfect
products['contains_perfect'] = products['perfect'].apply(lambda x: +1 if x >= 1 else 0)
sum(products['contains_perfect']) #2955

# sum(im_words[0].apply(lambda x: x == 'perfect'))
# sum(im_words[0].str.contains('perfect'))

# Converting to multidimensional arr
products.columns

def get_numpy_data(df, features, label):
    df['constant'] = 1
    features = ['constant'] + features
    features_frame = df[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = df[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)

feature_matrix, sentiment  = get_numpy_data(products, im_words , 'sentiment')
feature_matrix.shape


# Estimating conditional probability with link function
def predict_probability(feature_matrix, coefs):
    '''
    produces probablistic estimate for P(y_i = +1 | x_i, w).
    estimate ranges between 0 and 1.
    '''
    return 1/(1+np.exp(-np.dot(feature_matrix, coefs)))

def feature_derivative(errors, feature):
    return np.dot(errors,feature)

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients)
    for itr in range(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix,coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions

        for j in range(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient
            coefficients[j] += derivative * step_size

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients

init_coefs = np.zeros(194)
step_size = 1e-7
max_iter = 301

coefficients = logistic_regression(feature_matrix, sentiment, init_coefs, step_size, max_iter)

# Predicting sentiments
num_pos = sum(np.dot(feature_matrix, coefficients)>0) #25126

# Measuring accuracy
products['probability'] = predict_probability(feature_matrix,coefficients)
products['prediction'] = products['probability'].apply(lambda x: 1 if x > 0.5 else -1)
sum(products['prediction'] == products['sentiment']) / len(products['sentiment'])  # 0.75

word_coef_df = pd.DataFrame({'word' : im_words, 'coef' : coefficients[1:]})
word_coef_df.sort(ascending=True, columns='coef')[:10]
word_coef_df.sort(ascending=False, columns='coef')[:10]
word_coef_df.sort(ascending=True, columns='coef')[:-10:-1]

