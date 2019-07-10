from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import datetime


# Function get_data reads and preprocesses data from the file f_name
# also showing gaps if needed.
# Function returns dataframe of features features_f
# and array of goal variable y_f (if possible)
def get_data(f_name, show_gaps=False):
    features_f = pd.read_csv(f_name, index_col='match_id')
    # try to delete final related features
    # they are in train and absent in test data
    try:
        features_f.drop([
            'duration', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire'],
            axis='columns', inplace=True)
    except KeyError:
        pass
    # count gaps in data
    rows_n = len(features_f.index)
    gaps = rows_n - features_f.count()
    gaps = gaps[gaps > 0]
    # show features with gaps if show_gaps flag is True
    if show_gaps:
        print('\nFeatures with gaps:')
        for f in gaps.index:
            print(f)
    # interpolate gaps
    # features_f = features_f.interpolate()
    # fill gaps in features by zeros
    features_f = features_f.fillna(0)
    # goal variable column: 'radiant_win'
    # try to select y_f from features_f if possible
    try:
        y_f = features_f['radiant_win'].values
        features_f.drop(['radiant_win'], axis='columns', inplace=True)
    except KeyError:
        y_f = None
    return features_f, y_f


# This function gets dataframe object features_e with features,
# transforms categorial heroes identifiers into bag-of-words
# and returns array of transformed features x_f
# Parameter use_bag is for bag applying
# Parameter drop_features is for dropping categorial features
def bag_of_words(features_e, use_bag=True, drop_features=True):
    # create copy of features_e
    features_f = features_e.copy()
    # apply bag if use_bag=True
    if use_bag:
        # init empty set
        z = set()
        # create teams members list
        lst = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
               'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
        # cycle through teams members
        # and add selected heroes identifiers to the set
        for h in lst:
            z = z.union(features_f[h].unique())
        # save the number of heroes identities
        hi_n = len(z)
        # save different heroes identifiers in Series object
        z1 = pd.Series(list(z))
        # create dict from Series with inverse pairs: value - its index
        z2 = {}
        for i in z1.index:
            z2[z1[i]] = i
        # create the bag-of-words using dict above
        x_pick = np.zeros((features_f.shape[0], hi_n))
        for i, match_id in enumerate(features_f.index):
            for p in range(1, 6):
                # get r(p)_hero identity from current data row
                h_i = features_f['r%d_hero' % p][match_id]
                # find its index in dict
                j = z2[h_i]
                # put 1 to the corresponding cell of the bag-of-words
                x_pick[i][j] = 1
                # get d(p)_hero identity from current data row
                h_i = features_f['d%d_hero' % p][match_id]
                # find its index in dict
                j = z2[h_i]
                # put -1 to the corresponding cell of the bag-of-words
                x_pick[i][j] = -1
    # drop features if drop_features=True
    if drop_features:
        # drop categorial features
        features_f.drop([
            'lobby_type',
            'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
            'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'],
            axis='columns', inplace=True)
    # build x_f
    x_f = features_f.values
    if use_bag:
        # add bag-of-words
        x_f = np.hstack((x_f, x_pick))
    return x_f


# This function gets feautures x_f  and returns standardized result
def standardize(x_f):
    ss = StandardScaler()
    ss.fit(x_f)
    x_f = ss.transform(x_f)
    return x_f


# Stage One
# create X and y from train file
features, y = get_data('features.csv', show_gaps=True)
X = features.values
# KFold init
kf = KFold(n_splits=5, shuffle=True, random_state=241)
# cycle trough different tree estimators for GradientBoostingClassifier
trees_lst = [10, 20, 30]
print('\nGradient Boosting:')
for n_estimators in trees_lst:
    # GradientBoostingClassifier init
    gbc = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=241, max_depth=3)
    start_time = datetime.datetime.now()
    # cross validate
    z = np.mean(cross_val_score(estimator=gbc, cv=kf, X=X, y=y,
                                scoring='roc_auc'))
    t = (datetime.datetime.now() - start_time).total_seconds()
    print('№ of trees: %3d' % n_estimators, 'Time elapsed: %3dс' % t,
          ' Score: %.3f' % z)
# Stage Two
# This stage uses the bag-of-words matching
# different heroes identifiers existing in data
# -------------------------------------
# First create list to cycle 3 variants:
# 1. without bag, all features
# 2. without bag, drop categorial features
# 3. create bag, drop categorial features
lst = [[False, False], [False, True], [True, True]]
# cycle through 3 variants above
for i in range(3):
    X = bag_of_words(features, use_bag=lst[i][0],
                     drop_features=lst[i][1])
    # standardize X
    X = standardize(X)
    # init Logistic Regression
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    best_C = grid['C'][0]
    best_z = 0
    print()
    print('\nLogistic Regression: ', '\nuse_bag - ', lst[i][0],
          '\ndrop_categorial_features -', lst[i][1])
    for C in grid['C']:
        lr = LogisticRegression(random_state=241, C=C, solver='lbfgs')
        start_time = datetime.datetime.now()
        # cross validate
        z = np.mean(cross_val_score(
            estimator=lr, cv=kf, X=X, y=y, scoring='roc_auc'))
        t = (datetime.datetime.now() - start_time).total_seconds()
        print('C: %e' % C, 'Time elapsed: %3dс' % t, ' Score: %.3f' % z)
        if best_z < z:
            best_z = z
            best_C = C
    print('Best C: %e' % best_C, 'Best score: %.3f' % best_z)
# init Logistic Regression with best C
lr = LogisticRegression(random_state=241, C=best_C, solver='lbfgs')
# fit Logistic Regression on train data
lr.fit(X, y)
# repeat all the above preprocessing with test data
features, y_test = get_data('features_test.csv')
X_test = bag_of_words(features, use_bag=True, drop_features=True)
X_test = standardize(X_test)
# predict and show Radiant win results for test data
print('\nPrediction for Radiant win on test data:')
y_pred = lr.predict_proba(X_test)
y_min = round(min(y_pred[:, 1]), 3)
y_max = round(max(y_pred[:, 1]), 3)
print(y_min, y_max)
