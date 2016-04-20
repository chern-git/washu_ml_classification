import pandas as pd
import numpy as np
from math import log
from math import exp
import matplotlib.pyplot as plt

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    weighted_mistakes_all_negative = total_weight_positive
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    weighted_mistakes_all_positive = total_weight_negative

    if weighted_mistakes_all_positive <= weighted_mistakes_all_negative:
        return (weighted_mistakes_all_positive,+1)
    else:
        return (weighted_mistakes_all_negative,-1)

def best_splitting_feature(data, features, target, data_weights):
    best_feature = None # Keep track of the best feature
    best_error = float('+inf')     # Keep track of the best error so far
    num_points = float(len(data))

    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]

        left_data_weights = data_weights[data[feature] == 0]
        right_data_weights = data_weights[data[feature] == 1]

        left_weighted_mistakes, left_class = \
            intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = \
            intermediate_node_weighted_mistakes(right_split[target], right_data_weights)

        error = (left_weighted_mistakes + right_weighted_mistakes) / \
                (sum(left_data_weights) + sum(right_data_weights))

        if error < best_error:
            best_feature = feature
            best_error = error
    return best_feature

def create_leaf(target_values, data_weights):
    leaf = {'splitting_feature' : None,
            'is_leaf': True    }
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    return leaf

def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    # Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print("Stopping condition 1 reached.")
        return create_leaf(target_values, data_weights)

    # Stopping condition 2. No more features.
    if remaining_features == []:
        print("Stopping condition 2 reached.")
        return create_leaf(target_values, data_weights)

    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values, data_weights)

    # If all the datapoints are the same, splitting_feature will be None. Create a leaf
    splitting_feature = best_splitting_feature(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)

    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]

    print("Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split)))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target], data_weights)

    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)

    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def classify(tree, x, annotate = False):
    # we use classify code from class-4
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value.get_values() == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data):
    prediction = data.apply(lambda x: classify(tree, x))
    return (prediction != data[target]).sum() / float(len(data))

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = pd.Series(np.ones(len(data)), index = train_data.index)
    weights = []
    tree_stumps = []
    target_values = data[target]

    for t in range(num_tree_stumps):
        print('=====================================================')
        print('Adaboost Iteration %d' % t)
        print('=====================================================')
        tree_stump = weighted_decision_tree_create(data, features, target, alpha, max_depth=1)
        tree_stumps.append(tree_stump)

        # Make predictions
        predictions = np.zeros(len(data))
        for i in range(len(predictions)):
            predictions[i] = classify(tree_stump, data[i:i+1])
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values

        # Compute weighted error
        weighted_error = sum(alpha[is_wrong]) / sum(alpha)

        # Compute model coefficient using weighted error
        weight = 1/2 * np.log( (1- weighted_error) / weighted_error)
        weights.append(weight)

        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))

        # Scale alpha by multiplying by adjustment, then normalize data points weights
        alpha = alpha * adjustment
        alpha = alpha / sum(alpha)
    return weights, tree_stumps

def predict_adaboost(stump_weights, tree_stumps, data):
    score = 0
    for i, tree_stump in enumerate(tree_stumps):
        pred = classify(tree_stump, data)
        score += stump_weights[i] * pred
    if score > 0:
        return +1
    else:
        return -1

loans = pd.read_csv('data/3/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
del loans['bad_loans']
target = 'safe_loans'
features = ['grade', 'term', 'home_ownership', 'emp_length']
loans_subset = loans[features + [target]]
loans_subset = loans_subset.dropna()

# One-hot implementation
categorical_variables = []
for feat_name, feat_type in zip(loans_subset.columns, loans_subset.dtypes):
    if feat_type == 'O':
        categorical_variables.append(feat_name)
for feature in categorical_variables:
    tmp = pd.DataFrame({'key': loans_subset[feature]})
    # use periods as prefix_sep to mimic GL
    loans_data_unpacked = pd.get_dummies(tmp['key'], prefix_sep='.', prefix=feature)
    loans_subset.drop(feature, axis = 1, inplace = True)
    loans_subset = loans_subset.join(loans_data_unpacked)

train_idx = pd.read_json('data/5/module-8-assignment-2-train-idx.json')
test_idx = pd.read_json('data/5/module-8-assignment-2-test-idx.json')
train_data = loans_subset.iloc[train_idx[0]]
test_data = loans_subset.iloc[test_idx[0]]

# Creating weight vec
vec = np.zeros(len(train_data))
vec[:10] = 1
vec[-10:] = 1
weight_vec = pd.Series(vec, index = train_data.index)

# ce = (tot mistakes) / (tot data pts)
# we = w * mist / (tot w) = mist * (w / tot w) = mist * ce
# last equality possible if all error weights = 1

# Fitting a DT with max-depth = 2
one_hot_features = train_data.columns.values.tolist()
one_hot_features.remove('safe_loans')
subset_20 = weighted_decision_tree_create(train_data, one_hot_features,
                                                       target, weight_vec, max_depth= 2)
train_subset_20 = pd.concat([train_data[:10],train_data[-10:]])

# Implementing Adaboost
ada_weight_vec, ada_stump = adaboost_with_tree_stumps(train_data, one_hot_features, target, 10)
print(ada_weight_vec)
# Component weights are neither increasing or decreasing

# Making predictions
for i in range(10):
    print(predict_adaboost(ada_weight_vec, ada_stump, train_data[i:i+1]))

ada_weight_30, ada_stump_30 = \
    adaboost_with_tree_stumps(train_data, one_hot_features, target, 30)

# Calculating train, test error and plotting
error_train = []
for n in range(1, 31):
    train_preds = np.zeros(len(train_data))
    for i in range(len(train_data)):
        train_preds[i] = \
            predict_adaboost(ada_weight_30[:n], ada_stump_30[:n], train_data[i:i+1])
    error = 1.0 - sum(train_data[target] == train_preds) / len(train_preds)
    error_train.append(error)
    print("Iteration %s, training error = %s" % (n, error_train[n-1]))

error_test = []
for n in range(1, 31):
    test_preds = np.zeros(len(test_data))
    for i in range(len(test_data)):
        test_preds[i] = \
            predict_adaboost(ada_weight_30[:n], ada_stump_30[:n], test_data[i:i+1])
    error = 1.0 - sum(test_data[target] == test_preds) / len(test_preds)
    error_test.append(error)
    print("Iteration %s, training error = %s" % (n, error_test[n-1]))
    
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_train, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), error_test, '-', linewidth=4.0, label='Test error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()
plt.savefig('ml-class5b-train-test-error')

# Training error fluctuates, but general downwards trend as stumps increases
# No massive overfitting as iterations increase

