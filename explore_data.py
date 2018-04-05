import warnings

import seaborn as sns
import pandas as pd

import sys

from datetime import timedelta

from mlxtend.preprocessing import minmax_scaling
from scipy.stats import pearsonr, spearmanr, stats
from sklearn import preprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, forest, \
    RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, \
    GenericUnivariateSelect
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def calculate_mean(series):
    suma = 0
    for entry in series:
        if entry is not "NAN":
            suma = suma + float(entry)
    return suma / len(series)


def compute_covariance_matrix(train_x, test_x):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(train_x)
    X_test_std = sc.fit_transform(test_x)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('\nEigenvalues \n%s' % eigen_vals)

    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    print(len(eigen_vals))
    import matplotlib.pyplot as plt
    plt.bar(range(1, len(eigen_vals) + 1), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, len(eigen_vals) + 1), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()


def merge_non_param(df, series_name, param):
    vals = df[series_name].values
    non_us_vals = []
    for value in vals:
        if value != param:
            non_us_vals.append("NON-" + param)
        else:
            non_us_vals.append(param)
    df[series_name] = pd.Series(np.array(non_us_vals).tolist())
    df.dropna(inplace=True)
    ax = sns.countplot(x=series_name, data=df)
    plt.show()
    return df


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    print(str(pvalues))
    return pvalues


def calculate_pvalues_spearman(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    print(str(pvalues))
    return pvalues


def create_boxplot(df, name_series):
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.xlim(df[name_series].min(), df[name_series].max() * 1.1)
    ax = df[name_series].plot(kind='kde')

    plt.subplot(212)
    plt.xlim(df[name_series].min(), df[name_series].max() * 1.1)
    sns.boxplot(x=df[name_series])
    plt.show()


def clear_boxplot(df, name_series):
    q75 = np.percentile(df[name_series], 75)
    q25 = np.percentile(df[name_series], 25)

    iqr = q75 - q25

    min = q25 - (iqr * 1.5)
    max = q75 + (iqr * 1.5)
    print("Before clean ->" + str(len(df)) + ", name of feature: " + name_series)
    df['Outlier'] = 0

    df.loc[df[name_series] < min, 'Outlier'] = 1
    df.loc[df[name_series] > max, 'Outlier'] = 1

    tmp = df[df.Outlier != 1]
    print("After clean ->" + str(len(tmp)) + ", name of feature: " + name_series)
    return tmp


df = pd.read_csv('ks_with_days_201612.csv', encoding="ISO-8859-1", sep='\t',
                 na_values=['NAN'])
df.head()
df.info()
df.dropna(inplace=True)
# DROP UNUSEFUL COLUMNS AND RENAME THEM
# df.drop(df.columns[[13, 14, 15, 16]], axis=1, inplace=True)
df.rename(columns={'country ': 'country', 'main_category ': 'main_category',
                   'ID ': 'ID', 'name ': 'name', 'category ': 'category',
                   'currency ': 'currency', 'deadline ': 'deadline',
                   'goal ': 'goal', 'launched ': 'launched',
                   'pledged ': 'pledged', 'state ': 'state',
                   'backers ': 'backers', 'usd pledged ': 'usd_pledged'}
          , inplace=True)
# print(df.columns)
# print(df.head())
d = defaultdict(preprocessing.LabelEncoder)
onehot = defaultdict(preprocessing.OneHotEncoder)

import numpy as np

df["log_goal"] = np.log(df["goal"] + 1)

# create_boxplot(df, "log_goal")
df = clear_boxplot(df, "log_goal")
# create_boxplot(df, "log_goal")

# df.drop(columns=['goal'], inplace=True)

# PLEDGED
df["pledged"] = df["pledged"].astype(np.float)
df["log_pledged"] = np.log(df["pledged"] + 1)

# create_boxplot(df, "log_pledged")
df = clear_boxplot(df, "log_pledged")
# create_boxplot(df, "log_pledged")

# df.drop(columns=['pledged'], inplace=True)

# BACKERS
df["log_backers"] = np.log(df["backers"] + 1)

# create_boxplot(df, "log_backers")
df = clear_boxplot(df, "log_backers")
# create_boxplot(df, "log_pledged")

# df.drop(columns=['backers'], inplace=True)

# NUMDAYS
df["log_num_days"] = np.log(df["num_days"] + 1)
ax = sns.distplot(df["log_num_days"])
plt.show()
# create_boxplot(df, "log_num_days")

# DEFINE PATTERNS
patternFloatDet = "[+-]?([0-9]*[.])?[0-9]+"
patternDelDate = "^\d{4}\-(0?[1-9]|1[012])\-(0?[1-9]|[12][0-9]|3[01]) "
patternStringDel = "[A-z]"

# # CLEAN STATE COLUMN
# date_filter = df.state.str.contains(patternDelDate)
# float_filter = df.state.str.contains(patternFloatDet)
# df = df[~date_filter]
# df = df[~float_filter]
# state = df.state.value_counts(dropna=True)
# # state.plot.pie(figsize=(6, 6))
# #print(state.to_string())
#
# # CLEAN USD_PLEDGED COLUMN
# date_usd_filter = df.usd_pledged.str.contains(patternDelDate, na=False)
# string_usd_filter = df.usd_pledged.str.contains(patternStringDel, na=False)
# # float_usd_filter = df.usd_pledged.str.contains(patternFloatDet, na=False)
# df = df[~date_usd_filter]
# df = df[~string_usd_filter]
#
# # CLEAN PLEDGED COLUMN
# date_pledged_filter = df.pledged.str.contains(patternDelDate, na=False)
# # float_usd_filter = df.usd_pledged.str.contains(patternFloatDet, na=False)
# df = df[~date_pledged_filter]
# #print(calculate_mean(df.pledged.dropna()))
#
# # CLEAN CURRENCY - CURRENCY IS OK
# #print(df.currency.value_counts(dropna=True).to_string())
#
# # CLEAN GOAL - REMOVE DATE
# date_goal_filter = df.goal.str.contains(patternDelDate, na=False)
# # float_goal_filter = df.goal.str.contains(patternFloatDet, na=False)
# df = df[~date_goal_filter]
# #print(calculate_mean(df.goal))
#
# # CLEAN COUNTRY - RENAME NAN
# df.country.replace(['N,"0'], 'NAN', inplace=True)
#
# # CLEAN CATEGORY
# #print(df.category.value_counts(dropna=True))
#
# # CLEAN MAIN_CATEGORY
# #print(df.main_category.value_counts(dropna=True))
#
# # CLEAN LAUNCH
# date_launched_filter = df.launched.str.contains(patternDelDate)
# df = df[date_launched_filter]
# #print(len(df.launched))
#
# # CLEAN DEADLINE
# date_deadline_filter = df.deadline.str.contains(patternDelDate)
# df = df[date_deadline_filter]
# #print(len(df.deadline))
#
# # CLEAN GOAL
# date_goal_filter = df.goal.str.contains(patternDelDate, na=False)
# string_goal_filter = df.goal.str.contains(patternStringDel, na=False)
# df = df[~date_goal_filter]
# df = df[~string_goal_filter]
# #print(len(df.goal))
#
# # CLEAN BACKERS
# date_backers_filter = df.backers.str.contains(patternDelDate, na=False)
# string_backers_filter = df.backers.str.contains(patternStringDel, na=False)
# df = df[~date_backers_filter]
# df = df[~string_backers_filter]
# #print(len(df.backers))

# DELETE ROWS WHICH WE DONT WANT FROM STATE
df = df[df.state.str.contains("undefined") == False]
# print(len(df.state))

df = df[df.state.str.contains("live") == False]
# print(len(df.state))

df = df[df.state.str.contains("suspended") == False]
merge_non_param(df, "country", "US")
merge_non_param(df, "currency", "USD")
# WE USE LABEL ENCODER TO ENCODE CATEGORICAL ENUM DATA INTO NUMERICAL
df_simple = df[
    ["goal", "pledged", "country", "main_category", "category", "currency",
     "state", "backers", "num_days"]]

# boxplot from goal
df_simple["country"] = df["country"].astype(str)
df_transformed = df_simple[
    ["main_category", "category", "currency", "country", "state"]].apply(
    lambda x: d[x.name].fit_transform(x))
import numpy as np

df_transformed['goal'] = df_simple.goal.astype(np.float64)
df_transformed['pledged'] = df_simple.pledged.astype(np.float64)
df_transformed['backers'] = df_simple.backers.astype(np.float64)
df_transformed['state'] = df_transformed.state.astype(np.int)
df_transformed['num_days'] = df_simple['num_days'].astype(np.float64)
print(len(df_transformed))
df_transformed.dropna(inplace=True)
print(len(df_transformed))

df_transformed_data = df_transformed[
    ["goal", "pledged", "country", "main_category", "category", "currency",
     "backers", "num_days"]]

pp = sns.jointplot(x=df["log_pledged"], y=df_transformed.state, bins=5, kind="hex")
plt.show()

pp = sns.jointplot(x=df["log_goal"], y=df_transformed.state, bins=5, kind="hex")
plt.show()

pp = sns.jointplot(x=df["log_backers"], y=df_transformed.state, bins=5, kind="hex")
plt.show()

# ADDED PLEDGED IN PERCENTAGE - FLOAT
df_transformed_data["percentage_of_pledged"] = round(df_transformed_data['pledged'] / df_transformed_data['goal'] * 100,
                                                     2)
df_transformed_data["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"].astype(np.float64)
df_transformed_data["log_percentage_of_pledged"] = np.log(df_transformed_data["percentage_of_pledged"] + 1)
ax = sns.distplot(df_transformed_data["log_percentage_of_pledged"])
plt.show()
spec = df_transformed_data[
    ["goal", "pledged", "backers", "num_days", "percentage_of_pledged", "country", "main_category", "category",
     "currency"]]
spec["state"] = df_transformed["state"]

calculate_pvalues(spec)
sns.heatmap(spec.corr(method="pearson"), xticklabels=spec.columns.values, yticklabels=spec.columns.values)
plt.show()

df_transformed["state"] = df_simple["state"]

train_x, test_x, train_y, test_y = train_test_split(
    df_transformed_data[["goal", "pledged", "backers", "percentage_of_pledged", "num_days"]], df_transformed["state"],
    test_size=0.2, random_state=0)

knn_tt = KNeighborsClassifier(n_neighbors=20)
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier using the pledged percentage only float features: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
# SWITCH BACK TO ALL DATA
train_x, test_x, train_y, test_y = train_test_split(df_transformed_data, df_transformed["state"], test_size=0.2,
                                                    random_state=0)
compute_covariance_matrix(train_x, test_x)

clf = RandomForestClassifier(n_estimators=250, random_state=0, max_depth=3)
clf = clf.fit(train_x, train_y)
print(clf.feature_importances_)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

feat_labels = df_transformed_data.columns[1:]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_x.shape[1] - 1):
    print("%d. feature %s (%f)" % (f + 1, feat_labels[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])
plt.show()

print('Numb of mismatch RandomForestClassifier: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# WITHOUT PLEDGED IN PERCENTAGE - FLOAT
knn_tt = KNeighborsClassifier()
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# GAUSSIAN WITH ALL
gnb = GaussianNB()
gnb.fit(train_x, train_y)
predicted_y = gnb.predict(test_x)

print('Numb of mismatch GaussianNB: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITHOUT PLEDGED PERCENTAGE
gausian_df = pd.get_dummies(df[["country", "main_category", "currency", "category"]])
gausian_df["pledged"] = df["pledged"].astype(np.float64)
gausian_df["goal"] = df["goal"].astype(np.float64)
gausian_df['backers'] = df.backers.astype(np.float64)
gausian_df.dropna(inplace=True)
gausian_df['num_days'] = df["num_days"].astype(np.int)

print(gausian_df.info())

train_gausian_x, test_gausian_x, train_y, test_y = train_test_split(gausian_df,
                                                                    df_transformed["state"],
                                                                    test_size=0.2, random_state=0)
gnb_hot = GaussianNB()
gnb_hot.fit(train_gausian_x, train_y)
predicted_y = gnb_hot.predict(test_gausian_x)

print('Numb of mismatch GaussianNB with hot encoder: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITH PLEDGED PERCENTAGE
gausian_df["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"]

train_gausian_x_pledgedP, test_gausian_x_pledgedP, train_y, test_y = train_test_split(gausian_df,
                                                                                      df_transformed["state"],
                                                                                      test_size=0.2, random_state=0)
gnb_hotP = GaussianNB()
gnb_hotP.fit(train_gausian_x_pledgedP, train_y)
predicted_yP = gnb_hotP.predict(test_gausian_x_pledgedP)

print('Numb of mismatch GaussianNB with hot encoder with percentage pledged: ' + str((test_y != predicted_yP).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_yP))

lg = LogisticRegression()
lg.fit(train_x, train_y)
predicted_y = lg.predict(test_x)

print('Numb of mismatch LogisticRegression: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

lg = LogisticRegression()
lg.fit(train_gausian_x, train_y)
predicted_y = lg.predict(test_gausian_x)

print('Numb of mismatch LogisticRegression with hot encoder: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

lg = LogisticRegression()
lg.fit(train_gausian_x_pledgedP, train_y)
predicted_y = lg.predict(test_gausian_x_pledgedP)

print('Numb of mismatch LogisticRegression with hot encoder with percentage pledged: : ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

knn_tt = KNeighborsClassifier()
knn_tt.fit(train_gausian_x, train_y)
predicted_y = knn_tt.predict(test_gausian_x)

print('Numb of mismatch KNeighborsClassifier with hot encoder: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

knn_tt = KNeighborsClassifier()
knn_tt.fit(train_gausian_x_pledgedP, train_y)
predicted_y = knn_tt.predict(test_gausian_x_pledgedP)

print('Numb of mismatch KNeighborsClassifier with hot encoder using the pledged percentage: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# NOW WITH SCALING DATA - KN
scaled_data = minmax_scaling(df_transformed_data, columns=["goal", "pledged", "backers", "percentage_of_pledged", "num_days", "country", "main_category", "category", "currency"])

train_scaled_x, test_scaled_x, train_scaled_y, test_scaled_y = train_test_split(scaled_data,
                                                                                df_transformed["state"],
                                                                                test_size=0.2, random_state=0)

knn_tt = KNeighborsClassifier()
knn_tt.fit(train_scaled_x, train_scaled_y)
predicted_y = knn_tt.predict(test_scaled_x)

print('Numb of mismatch KNeighborsClassifier scaled (all scaled) all data: ' + str(
    (test_scaled_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_scaled_y, predicted_y))
