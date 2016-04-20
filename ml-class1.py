import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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

products = pd.read_csv('data/1/amazon_baby.csv')

# Pre-processing
products['review'] = products['review'].fillna('')
products['review'] = products['review'].astype('str')
products = products[products['rating'] != 3]
products['review_clean'] = products['review'].apply(strip_punctuation)
products['word_count'] = products['review_clean'].apply(count_words)
products['sentiment'] = products['rating'].apply(lambda r: +1 if  r > 3 else -1)

# Splitting data into training and test sets
train_idx = pd.read_json('data/1/module-2-assignment-train-idx.json')
test_idx = pd.read_json('data/1/module-2-assignment-test-idx.json')
train_data = products.iloc[train_idx[0],:]
test_data = products.iloc[test_idx[0],:]
# train_data = products[products.index.to_series().isin(train_idx[0])]
# test_data = products[products.index.to_series().isin(test_idx[0])]

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])

# Specifying and fitting the model
logreg = LogisticRegression()
sentiment_model = logreg.fit(X = train_matrix, y = train_data['sentiment'])
np.sum([sentiment_model.coef_ >=0]) #85886

# Making predictions with logistic regression
sample_test_data = test_data[10:13]
print(sample_test_data)

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)

def predict_labels(score):
    if score > 0:
        label = 1
    else:
        label = -1
    return label

for i in range(len(scores)):
    print(predict_labels(scores[i]))
# Checker
sentiment_model.predict(sample_test_matrix)

def get_prob_prediction(score):
    return 1/(1+np.exp(-score))

for i in range(len(scores)):
    print(get_prob_prediction(scores[i]))
# Checker
sentiment_model.predict_proba(sample_test_matrix)

# Finding the most positive/negative review
test_data_scores = sentiment_model.decision_function(test_matrix)
test_product_scored = pd.DataFrame({'name': test_data['name'], 'scores': test_data_scores})
test_product_scored.sort(ascending=False,columns='scores')[:20]
test_product_scored.sort(ascending=True,columns='scores')[:20]

# Computing accuracy of classifier
np.sum(sentiment_model.predict(test_matrix) == test_data['sentiment']) / len(test_data['sentiment'])

# Learning another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

# Building simple_model
logreg_simple = LogisticRegression()
simple_model = logreg_simple.fit(X = train_matrix_word_subset, y = train_data['sentiment'])
simple_model_coef = pd.DataFrame({'word': significant_words, 'coef':simple_model.coef_.flatten()})
print(simple_model_coef.sort(columns='coef',ascending=True))
print(sum([1 for x in simple_model_coef['coef'] if x > 0]))

# significant words in sentiment model
vectorizer_senti_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_senti_subset = vectorizer_senti_subset.fit_transform(train_data['review_clean'])
sentiment_model_subset = logreg.fit(X = train_matrix_senti_subset, y = train_data['sentiment'])
sentiment_model_subset_coef = pd.DataFrame({'word': significant_words,
                                  'coef':sentiment_model_subset.coef_.flatten()})
print(sentiment_model_subset_coef.sort(columns='coef',ascending=True))

# Comparing models
# Accuracy of sentimental model on train and test
sentiment_model = logreg.fit(X = train_matrix, y = train_data['sentiment'])
np.sum(sentiment_model.predict(train_matrix) == train_data['sentiment']) / len(train_data['sentiment'])
np.sum(sentiment_model.predict(test_matrix) == test_data['sentiment']) / len(test_data['sentiment'])

# Accuracy of simple model on train and test
simple_model = logreg_simple.fit(X = train_matrix_word_subset, y = train_data['sentiment'])
np.sum(simple_model.predict(train_matrix_word_subset) == train_data['sentiment']) / len(train_data['sentiment'])
np.sum(simple_model.predict(test_matrix_word_subset) == test_data['sentiment']) / len(test_data['sentiment'])

# Majority classifier
num_positive  = (test_data['sentiment'] == +1).sum()
num_negative = (test_data['sentiment'] == -1).sum()
print(num_positive/(num_positive+num_negative))
