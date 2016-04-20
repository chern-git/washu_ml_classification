import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def strip_punctuation(s):
    return ''.join(i for i in s if i not in string.punctuation)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})

def apply_threshold(prob, threshold):
    return [1 if i >= threshold else -1 for i in prob]

# Load data
products = pd.read_csv('data/6/amazon_baby.csv')

# Pre-processing
products['review'] = products['review'].fillna('')
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products['review_clean'] = products['review'].apply(strip_punctuation)
train_idx = pd.read_json('data/6/module-9-assignment-train-idx.json')
test_idx = pd.read_json('data/6/module-9-assignment-test-idx.json')
train_data = products.iloc[train_idx[0]]
test_data = products.iloc[test_idx[0]]

# Sparsifying data
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])

# Logistic regression and calculating accuracy
log_reg = LogisticRegression()
model = log_reg.fit(X = train_matrix, y = train_data['sentiment'])
accuracy_score(y_true=test_data['sentiment'], y_pred = model.predict(test_matrix))

# Baseline: Majority class pred
baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print("Baseline accuracy (majority class classifier): %s" % baseline)

# Confusion matrix
cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))

# Cost of mistakes
print(1451*100 + 809)

# Precision & Recall
precision = precision_score(y_true=test_data['sentiment'],
                            y_pred=model.predict(test_matrix))
print("Precision on test data: %s" % precision)
print("FP/(tot. P):",1-precision)

recall = recall_score(y_true=test_data['sentiment'],
                      y_pred=model.predict(test_matrix))
print("Recall on test data: %s" % recall)

# Varying threshold
test_prob = model.decision_function(test_matrix)
# output_prob = model.predict_proba(test_matrix).flatten()[1::2]
precision_score(y_true = test_data['sentiment'], y_pred = apply_threshold(test_prob, 0.5))
precision_score(y_true = test_data['sentiment'], y_pred = apply_threshold(test_prob, 0.9))

# Prec-recall curve
threshold_values = np.linspace(0.5, 1, num=100)
precision_all = np.zeros(100)
recall_all = np.zeros(100)
for i in range(len(threshold_values)):
    precision_all[i] = precision_score(y_true = test_data['sentiment'],
                                       y_pred = apply_threshold(test_prob, threshold_values[i]))
    recall_all[i] = recall_score(y_true = test_data['sentiment'],
                                       y_pred = apply_threshold(test_prob, threshold_values[i]))
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
threshold_values[sum(precision_all < 0.965)]
# Checker
precision_score(y_true = test_data['sentiment'], y_pred = apply_threshold(test_prob, 0.879))

# Number of false negatives (Approach 1)
sum((test_data['sentiment'] == 1) & [i == -1 for i in apply_threshold(test_prob, 0.98)])
# Number of false negatives (Approach 2)
cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=apply_threshold(test_prob, 0.98),
                        labels=model.classes_)    # use the same order of class as the LR model.
print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))

# Prec-recall on all 'baby' related items
baby_reviews = test_data[test_data['name'].apply(lambda x:'baby' in str(x).lower())]
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.decision_function(baby_matrix)
threshold_values = np.linspace(0.5,  1, num=100)
bb_precision_all = np.zeros(100)
bb_recall_all = np.zeros(100)
for i in range(len(threshold_values)):
    bb_precision_all[i] = precision_score(y_true = baby_reviews['sentiment'],
                                       y_pred = apply_threshold(probabilities, threshold_values[i]))
    bb_recall_all[i] = recall_score(y_true = baby_reviews['sentiment'],
                                       y_pred = apply_threshold(probabilities, threshold_values[i]))
plot_pr_curve(bb_precision_all, bb_recall_all, "Precision-Recall (Baby)")
threshold_values[sum(bb_precision_all <0.965)]

# Checker
precision_score(y_true = baby_reviews['sentiment'], y_pred = apply_threshold(probabilities, 0.995))