import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

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

def feature_derivative_with_L2(errors, feature, coef, l2_pen, feat_is_const):
    deriv = np.dot(errors,feature)
    if not feat_is_const:
        deriv += l2_pen
    return deriv

def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    return lp

def logistic_regression_with_L2(feat_mat, sentiment, init_coefs, step_sz, l2_pen, max_iter):
    coefficients = np.array(init_coefs)
    for itr in range(max_iter):
        # preliminary calcs
        predictions = predict_probability(feat_mat,coefficients)
        indicator = (sentiment==+1)
        errors = indicator - predictions
        # compute derivatives
        for j in range(len(coefficients)):
            derivative = feature_derivative_with_L2(errors, feat_mat[:,j], init_coefs, l2_pen, j==0)
            coefficients[j] += step_sz * derivative
        # print results
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feat_mat, sentiment, coefficients, l2_pen)
            print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')

    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')

    table_positive_words = table[table['word'].isin(positive_words)]
    table_negative_words = table[table['word'].isin(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']

    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)

    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)

    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()

# def add_coefficients_to_table(coefficients, column_name):
#     table[column_name] = coefficients
#     return table

# Pre-processing
products = pd.read_csv('data/2/amazon_baby_subset.csv')
products['review'] = products['review'].fillna('')
products['review_clean'] = products['review'].apply(strip_punctuation)
im_json = pd.read_json('data/2/important_words.json')
im_words = [str(s) for s in im_json[0]]
for w in im_words:
    products[w] = products['review_clean'].apply(lambda x: x.split().count(w))

# Training-validation split
train_idx = pd.read_json('data/2/module-4-assignment-train-idx.json')
val_idx = pd.read_json('data/2/module-4-assignment-validation-idx.json')
train_data = products.iloc[train_idx[0]]
val_data = products.iloc[val_idx[0]]

# Convert DF to multi-dim array
feature_matrix_train, sentiment_train = get_numpy_data(train_data.copy(), im_words, 'sentiment')
feature_matrix_val, sentiment_val = get_numpy_data(val_data.copy(), im_words, 'sentiment')

# LogReg args to use
l2_pen_list = [0, 4, 10, 1e2, 1e3, 1e5]
l2_pen_str = ['0','4','10','1e2','1e3','1e5']
init_coefs = np.zeros(194)
step_sz = 5e-6
max_iter = 501
l2_pen_coef = {}

for i in range(len(l2_pen_list)):
    print('\nRunning LogReg using l2_pen:', l2_pen_list[i])
    l2_pen_coef['coef_{0}_pen'.format(l2_pen_str[i])] = \
        logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                    init_coefs, step_sz, l2_pen_list[i],
                                    max_iter)

# Comparing coefficients
word_coef_tbl = pd.DataFrame({'words': ['INTERCEPT'] + im_words})
for k in l2_pen_coef.keys():
    word_coef_tbl[k] = l2_pen_coef[k]
positive_words = word_coef_tbl.sort(ascending=False, columns='coef_0_pen')[:5]
negative_words = word_coef_tbl.sort(ascending=True, columns='coef_0_pen')[:5]

# Plotting

# %matplotlib inline
# plt.rcParams['figure.figsize'] = 10, 6
# make_coefficient_plot(im_words,positive_words.index, negative_words.index, l2_pen_str)
# plt.show()

# Accuracy on data sets
dict_acc = {}
# train_acc = {}
for k in l2_pen_coef.keys():
    train_data['probability'] = np.dot(feature_matrix_train,word_coef_tbl[k])
    train_data['prediction'] = train_data['probability'].apply(lambda x: 1 if x > 0 else -1)
    val_data['probability'] = np.dot(feature_matrix_val, word_coef_tbl[k])
    val_data['prediction'] = val_data['probability'].apply(lambda x: 1 if x > 0 else -1)
    dict_acc[k] = [sum(train_data['sentiment'] == train_data['prediction']) / \
                len(train_data['sentiment']),
                sum(val_data['sentiment'] == val_data['prediction']) / \
                len(val_data['sentiment'])]
dict_acc
# val_acc = {}
# for k in l2_pen_coef.keys():
#     val_data['probability'] = np.dot(feature_matrix_val, word_coef_tbl[k])
#     val_data['prediction'] = val_data['probability'].apply(lambda x: 1 if x > 0 else -1)
#     val_acc[k] = sum(val_data['sentiment'] == val_data['prediction']) / \
#                 len(val_data['sentiment'])

# Quiz Answers:
# intercept term not regularized
# decreases ll(w)
