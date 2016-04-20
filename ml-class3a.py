import pandas as pd
import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

loans = pd.read_csv('data/3/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
del loans['bad_loans']

# Subsetting data
features = ['grade','sub_grade', 'short_emp', 'emp_length_num', 'home_ownership', 'dti',
            'purpose', 'term', 'last_delinq_none', 'last_major_derog_none', 'revol_util',
            'total_rec_late_fee']
target = 'safe_loans'
loans_subset = loans[features + [target]].copy()

# Building One-hot
categorical_variables = []
for feat_name, feat_type in zip(loans_subset.columns, loans_subset.dtypes):
    if feat_type == 'O':
        categorical_variables.append(feat_name)
for feature in categorical_variables:
    tmp = pd.DataFrame({'key': loans_subset[feature]})
    loans_data_unpacked = pd.get_dummies(tmp['key'], prefix=feature)
    loans_subset.drop(feature, axis = 1, inplace = True)
    loans_subset = loans_subset.join(loans_data_unpacked)

# Data sub-setting
train_idx = pd.read_json('data/3/module-5-assignment-1-train-idx.json')
val_idx = pd.read_json('data/3/module-5-assignment-1-validation-idx.json')
train_data = loans_subset.iloc[train_idx[0]]
val_data = loans_subset.iloc[val_idx[0]]

one_hot_features = [i for i in train_data.columns if i is not target]
# one_hot_features = train_data.columns.values.tolist()   # Alternate approach
# one_hot_features = one_hot_features.remove(target)
big_tree = DecisionTreeClassifier(max_depth = 6)
decision_tree_model = big_tree.fit(X = train_data[one_hot_features], y = train_data[target])
small_tree = DecisionTreeClassifier(max_depth = 2)
small_model = small_tree.fit(train_data[one_hot_features], train_data[target])

# Making predictions
val_safe_loans = val_data[val_data[target] == 1]
val_risky_loans = val_data[val_data[target] == -1]
sample_val_data_risky = val_risky_loans[0:2]
sample_val_data_safe = val_safe_loans[0:2]
sample_val_data = sample_val_data_safe.append(sample_val_data_risky)
sample_val_data

decision_tree_model.predict(sample_val_data[one_hot_features]) # OK

# Exploring probability predictions
decision_tree_model.predict_proba(sample_val_data[one_hot_features])

# Tricky predictions
small_model.predict(sample_val_data[one_hot_features])
small_model.predict_proba(sample_val_data[one_hot_features])
small_model.predict(sample_val_data[0:1][one_hot_features])

# Tree visualization
visualize_tree(decision_tree_model, one_hot_features)

# Evaluating accuracy
decision_tree_model.score(train_data[one_hot_features], train_data[target])
small_model.score(train_data[one_hot_features], train_data[target])

decision_tree_model.score(val_data[one_hot_features], val_data[target])
small_model.score(val_data[one_hot_features], val_data[target])

# Evaluating accuracy on a complex decision tree model
big_model = DecisionTreeClassifier(max_depth = 10)
big_model = big_model.fit(X = train_data[one_hot_features], y = train_data[target])
big_model.score(train_data[one_hot_features], train_data[target])
big_model.score(val_data[one_hot_features], val_data[target])

# Quantifying cost of mistakes
act_v_pred = pd.DataFrame({'p': decision_tree_model.predict(val_data[one_hot_features]),
                           'a': val_data[target].as_matrix()})
false_pos = len(act_v_pred[(act_v_pred['a'] == -1) & (act_v_pred['p'] == 1)])
false_neg = len(act_v_pred[(act_v_pred['a'] == 1) & (act_v_pred['p'] == -1)])
print(false_neg * 10 + false_pos * 20,"K")















