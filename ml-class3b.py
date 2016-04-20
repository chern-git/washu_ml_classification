
import pandas as pd

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

def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    if intermediate_node_num_mistakes(target_values) == 0:  ## YOUR CODE HERE
        print("Stopping condition 1 reached.")
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if len(remaining_features) == 0:   ## YOUR CODE HERE
        print("Stopping condition 2 reached.")
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, features, target)
    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target])

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)
    right_tree =decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

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

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    return sum(data[target] == prediction)/len(data[target])

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('.')
    print('                       %s' % name)
    print('         |---------------|----------------|')
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree')))

loans = pd.read_csv('data/3/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
del loans['bad_loans']

features = ['grade', 'term', 'home_ownership', 'emp_length']
target = 'safe_loans'
loans_subset = loans[features + [target]]

# Building One-hot
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

train_idx = pd.read_json('data/3/module-5-assignment-2-train-idx.json')
test_idx = pd.read_json('data/3/module-5-assignment-2-test-idx.json')
train_data = loans_subset.iloc[train_idx[0]]
test_data = loans_subset.iloc[test_idx[0]]

# Checker
intermediate_node_num_mistakes([-1, -1, 1, 1, 1, 1, 1])

one_hot_features = train_data.columns.values.tolist()
one_hot_features.remove('safe_loans')
my_decision_tree = decision_tree_create(train_data,one_hot_features,target,0,6)

classify(my_decision_tree, test_data[:1],annotate=True)

# Classification error
ce = 0
for i in range(len(test_data)):
    if test_data['safe_loans'][i:i+1].get_values() != classify(my_decision_tree,test_data[i:i+1]):
        ce += 1
print(ce/len(test_data['safe_loans']))

# Print stump
print_stump(my_decision_tree)
print_stump(my_decision_tree['left'],my_decision_tree['left']['splitting_feature'])
print_stump(my_decision_tree['left']['left'],my_decision_tree['left']['left']['splitting_feature'])
print_stump(my_decision_tree['right'],my_decision_tree['right']['splitting_feature'])

