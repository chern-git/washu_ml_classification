import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import pylab

def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

loans = pd.read_csv('data/3/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
del loans['bad_loans']
target = 'safe_loans'
features = ['grade', 'sub_grade_num', 'short_emp', 'emp_length_num', 'home_ownership',
            'dti', 'purpose','payment_inc_ratio', 'delinq_2yrs', 'delinq_2yrs_zero',
            'inq_last_6mths', 'last_delinq_none', 'last_major_derog_none', 'open_acc',
            'pub_rec', 'pub_rec_zero', 'revol_util', 'total_rec_late_fee', 'int_rate',
            'total_rec_int', 'annual_inc', 'funded_amnt', 'funded_amnt_inv', 'installment']
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

train_idx = pd.read_json('data/5/module-8-assignment-1-train-idx.json')
val_idx = pd.read_json('data/5/module-8-assignment-1-validation-idx.json')
train_data = loans_subset.iloc[train_idx[0]]
val_data = loans_subset.iloc[val_idx[0]]

one_hot_features = [i for i in train_data.columns if i is not 'safe_loans']
grad_boost = GradientBoostingClassifier(max_depth=6, n_estimators=5)
mdl_5 = grad_boost.fit(X = train_data[one_hot_features], y = train_data[target])

# Making predictions
val_safe_loans = val_data[val_data[target] == 1]
val_risky_loans = val_data[val_data[target] == -1]
sample_val_data_risky = val_risky_loans[0:2]
sample_val_data_safe = val_safe_loans[0:2]
sample_val_data = sample_val_data_safe.append(sample_val_data_risky)

mdl_5.predict(sample_val_data[one_hot_features])
mdl_5.predict_proba(sample_val_data[one_hot_features])

# Evaluating on val data, comparison with DT
mdl_5.score(val_data[one_hot_features], val_data[target])
false_pos = np.sum((mdl_5.predict(val_data[one_hot_features]) == 1) & (val_data[target].values == -1))
false_neg = np.sum((mdl_5.predict(val_data[one_hot_features]) == -1) & (val_data[target].values == 1))
print(false_pos * 20 + false_neg * 10,"K")

# Most pos and neg loans
val_data_ranked = val_data.copy()
pos_preds = mdl_5.predict_proba(val_data[one_hot_features])
pos_preds = pos_preds.flatten()[1::2]
pos_preds_df = pd.DataFrame({'preds': pos_preds}, index= val_data_ranked.index)
val_data_ranked = val_data_ranked.join(pos_preds_df)
grade_filt = [i for i in val_data_ranked.columns if ('grade.' in i or 'preds' in i)]
val_data_ranked[grade_filt].sort_values(by='preds', ascending=False)[:5]

# Effect of more trees
train_error = np.zeros(6)
val_error = np.zeros(6)
for i, j in enumerate([5, 10, 50, 100, 200, 500]):
    gb = GradientBoostingClassifier(max_depth=6, n_estimators = j)
    gb_mdl = gb.fit(X = train_data[one_hot_features], y = train_data[target])
    print(j,'trees on val data:',gb_mdl.score(val_data[one_hot_features], val_data[target]))
    # for plotting of training / error curves
    train_error[i] = 1 - gb_mdl.score(train_data[one_hot_features], train_data[target])
    val_error[i] = 1 - gb_mdl.score(val_data[one_hot_features], val_data[target])

# Plotting training vs val errors
plt.plot([5, 10, 50, 100, 200, 500], train_error, linewidth=4.0, label='Training error')
plt.plot([5, 10, 50, 100, 200, 500], val_error, linewidth=4.0, label='Validation error')
make_figure(dim=(10,6), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
plt.savefig('ml-class5a-ensemble-train-val-plot.png')