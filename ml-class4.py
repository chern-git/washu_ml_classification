import pandas as pd

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size:
        return True
    else:
        return False

def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split)

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    num_pos = sum(i > 0 for i in labels_in_node)
    num_neg = sum(i < 0 for i in labels_in_node)
    # Return the number of mistakes that the majority classifier makes.
    if num_pos > num_neg:
        return num_neg
    else:
        return num_pos

def best_splitting_feature(data, features, target):
    best_feature = None # Keep track of the best feature
    best_error = 10     # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]

        # Calculate the number of misclassified examples in both splits:
        left_mistakes = intermediate_node_num_mistakes(left_split[target])
        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        # Compute the classification error of this split.
        error = (left_mistakes + right_mistakes) / num_data_points

        if error < best_error:
            best_feature = feature
            best_error = error
    return best_feature

def create_leaf(target_values):
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True    }   ## YOUR CODE HERE

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1         ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1        ## YOUR CODE HERE
    return leaf

def decision_tree_create(data, features, target, current_depth = 0,
                         max_depth = 10, min_node_size=1,
                         min_error_reduction=0.0):

    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))


    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached. All data points have the same target value.")
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if len(remaining_features) == 0:
        print("Stopping condition 2 reached. No remaining features.")
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print("Early stopping condition 1 reached. Reached maximum depth.")
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    if reached_minimum_node_size(target_values, min_node_size):
        print("Early stopping condition 2 reached. Reached minimum node size.")
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes =    intermediate_node_num_mistakes(left_split[target])
    right_mistakes =   intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))

    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)
    right_tree = decision_tree_create(right_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)

    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}

def classify(tree, x, annotate = False):
       # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value.get_values() == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

loans = pd.read_csv('data/3/lending-club-data.csv')     # Uses Week 3 lending club data
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
del loans['bad_loans']
features = ['grade', 'term', 'home_ownership', 'emp_length']
target = 'safe_loans'
loans_subset = loans[features + [target]]

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

train_idx = pd.read_json('data/4/module-6-assignment-train-idx.json')
val_idx = pd.read_json('data/4/module-6-assignment-validation-idx.json')
train_data = loans_subset.iloc[train_idx[0]]
val_data = loans_subset.iloc[val_idx[0]]

# Tree building
max_depth = 6
min_node_size = 100,
min_error_reduction = 0.0
one_hot_features = train_data.columns.values.tolist()
one_hot_features.remove('safe_loans')
my_decision_tree_new = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 100, min_error_reduction=0.0)
my_decision_tree_old = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                            min_node_size = 0, min_error_reduction=-1)

# Making predictions
print(val_data[0:1])
classify(my_decision_tree_new,val_data[0:1], annotate=True)  # -1
classify(my_decision_tree_old,val_data[0:1], annotate=True)  # -1

# Classification error
ce = 0
for i in range(len(val_data)):
    if val_data['safe_loans'][i:i+1].get_values() != classify(my_decision_tree_new,val_data[i:i+1]):
        ce += 1
print(ce/len(val_data['safe_loans']))

ce = 0
for i in range(len(val_data)):
    if val_data['safe_loans'][i:i+1].get_values() != classify(my_decision_tree_old,val_data[i:i+1]):
        ce += 1
print(ce/len(val_data['safe_loans']))

# Exploring max depth
mdl_1 = decision_tree_create(train_data, one_hot_features, target, max_depth = 2,
                                min_node_size = 0, min_error_reduction=-1)
mdl_2 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)
mdl_3 = decision_tree_create(train_data, one_hot_features, target, max_depth = 14,
                                min_node_size = 0, min_error_reduction=-1)

for i in [1,2,3]:
    mdl_name  = 'mdl_' + str(i)
    possibles = locals().copy()
    mdl = possibles.get(mdl_name)
    ce = 0
    for j in range(len(train_data)):
        if train_data['safe_loans'][j:j+1].get_values() != classify(mdl,train_data[j:j+1]):
            ce += 1
    print(mdl_name, ce/len(train_data['safe_loans']))

# Measuring tree complexity
print(count_leaves(mdl_1))
print(count_leaves(mdl_2))
print(count_leaves(mdl_3))

# Exploring effect of min_error
mdl_4 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)
mdl_5 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 0, min_error_reduction=+0)
mdl_6 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 0, min_error_reduction=+5)

for i in [4,5,6]:
    mdl_name  = 'mdl_' + str(i)
    possibles = locals().copy()
    mdl = possibles.get(mdl_name)
    ce = 0
    for j in range(len(val_data)):
        if val_data['safe_loans'][j:j+1].get_values() != classify(mdl,val_data[j:j+1]):
            ce += 1
    print(mdl_name, ce/len(val_data['safe_loans']))

print(count_leaves(mdl_4))
print(count_leaves(mdl_5))
print(count_leaves(mdl_6))

# Exploring min node size
mdl_7 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)
mdl_8 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 2000, min_error_reduction=-1)
mdl_9 = decision_tree_create(train_data, one_hot_features, target, max_depth = 6,
                                min_node_size = 50000, min_error_reduction=-1)

for i in [4,5,6]:
    mdl_name  = 'mdl_' + str(i)
    possibles = locals().copy()
    mdl = possibles.get(mdl_name)
    ce = 0
    for j in range(len(val_data)):
        if val_data['safe_loans'][j:j+1].get_values() != classify(mdl,val_data[j:j+1]):
            ce += 1
    print(mdl_name, ce/len(val_data['safe_loans']))

print(count_leaves(mdl_7))
print(count_leaves(mdl_8))
print(count_leaves(mdl_9))