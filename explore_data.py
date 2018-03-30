import warnings

import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import sys

from datetime import timedelta
from sklearn import preprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
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
    plt.bar(range(1, len(eigen_vals)+1), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, len(eigen_vals)+1), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()



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
#print(df.columns)
#print(df.head())
d = defaultdict(preprocessing.LabelEncoder)
onehot = defaultdict(preprocessing.OneHotEncoder)

import numpy as np
df["log_goal"] = np.log(df["goal"]+1)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df["log_goal"].min(), df["log_goal"].max()*1.1)
ax = df["log_goal"].plot(kind='kde')

plt.subplot(212)
plt.xlim(df["log_goal"].min(), df["log_goal"].max()*1.1)
sns.boxplot(x=df["log_goal"])
plt.show()

q75 = np.percentile(df["log_goal"], 75)
q25 = np.percentile(df["log_goal"], 25)

iqr = q75 - q25

min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(len(df))
df['Outlier'] = 0

df.loc[df["log_goal"] < min, 'Outlier'] = 1
df.loc[df["log_goal"] > max, 'Outlier'] = 1

df = df[df.Outlier != 1]
print(len(df))
plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df["log_goal"].min(), df["log_goal"].max()*1.1)
ax = df["log_goal"].plot(kind='kde')

plt.subplot(212)
plt.xlim(df["log_goal"].min(), df["log_goal"].max()*1.1)
sns.boxplot(x=df["log_goal"])
plt.show()

#PLEDGED
df["pledged"] = df["pledged"].astype(np.float)
df["log_pledged"] = np.log(df["pledged"]+1)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df["log_pledged"].min(), df["log_pledged"].max()*1.1)
ax = df["log_pledged"].plot(kind='kde')

plt.subplot(212)
plt.xlim(df["log_pledged"].min(), df["log_pledged"].max()*1.1)
sns.boxplot(x=df["log_pledged"])
plt.show()

#BACKERS
df["log_backers"] = np.log(df["backers"]+1)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df["log_backers"].min(), df["log_backers"].max()*1.1)
ax = df["log_backers"].plot(kind='kde')

plt.subplot(212)
plt.xlim(df["log_backers"].min(), df["log_backers"].max()*1.1)
sns.boxplot(x=df["log_backers"])
plt.show()

df["log_num_days"] = np.log(df["num_days"]+1)

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df["log_num_days"].min(), df["log_num_days"].max()*1.1)
ax = df["log_num_days"].plot(kind='kde')

plt.subplot(212)
plt.xlim(df["log_num_days"].min(), df["log_num_days"].max()*1.1)
sns.boxplot(x=df["log_num_days"])
plt.show()

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
#print(len(df.state))

df = df[df.state.str.contains("live") == False]
#print(len(df.state))

df = df[df.state.str.contains("suspended") == False]
#print(len(df.state))
# linear
# plt.subplot(221)
# df.main_category.value_counts(dropna=True).plot.bar()
# plt.title('main_category')
# plt.grid(True)


# log
# plt.subplot(222)
# df.country.value_counts(dropna=True).head(10).plot.pie(subplots=True)
# plt.title('country')
# plt.grid(True)

# symmetric log
# plt.subplot(223)
# df.currency.value_counts(dropna=True).head(6).plot.pie(subplots=True)
# plt.title('currency')
# plt.grid(True)

# logit
# plt.subplot(224)
# df.category.value_counts(dropna=True).head(8).plot.bar()
# plt.title('category')
# plt.grid(True)

# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
# wspace=0.35)

# plt.show()

# WE USE LABEL ENCODER TO ENCODE CATEGORICAL ENUM DATA INTO NUMERICAL
df_simple = df[
    ["goal", "pledged", "country", "main_category", "category", "currency",
     "state", "backers", "num_days"]]
df_simple["country"] = df["country"].astype(str)
df_transformed = df_simple[
    ["main_category", "category", "currency", "country"]].apply(
    lambda x: d[x.name].fit_transform(x))
import numpy as np
df_transformed['goal'] = df_simple.goal.astype(np.float64)
df_transformed['pledged'] = df_simple.pledged.astype(np.float64)
df_transformed['backers'] = df_simple.backers.astype(np.float64)
df_transformed['state'] = df_simple.state
df_transformed['num_days'] = df_simple['num_days'].astype(np.float64)
print(len(df_transformed))
df_transformed.dropna(inplace=True)
print(len(df_transformed))

df_transformed_data = df_transformed[
    ["goal", "pledged", "country", "main_category", "category", "currency",
     "backers", "num_days"]]

# ADDED PLEDGED IN PERCENTAGE - FLOAT
df_transformed_data["percentage_of_pledged"] = round(df_transformed_data['pledged'] / df_transformed_data['goal'] * 100,2)
df_transformed_data["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"].astype(np.float64)

train_y = df_transformed[250000:]["state"]
train_x = df_transformed_data[250000:][["goal", "pledged", "backers", "percentage_of_pledged", "num_days"]]
test_y = df_transformed[:250000]["state"]
test_x = df_transformed_data[:250000][["goal", "pledged", "backers", "percentage_of_pledged", "num_days"]]

knn_tt = KNeighborsClassifier(n_neighbors=20)
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier using the pledged percentage only float features: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))


train_y = df_transformed[250000:]["state"]
train_x = df_transformed_data[250000:]
test_y = df_transformed[:250000]["state"]
test_x = df_transformed_data[:250000]

compute_covariance_matrix(train_x, test_x)

# WITHOUT PLEDGED IN PERCENTAGE - FLOAT
knn_tt = KNeighborsClassifier()
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

print(df_transformed_data.info())

clf = RandomForestClassifier(n_estimators=250, random_state=0)
clf = clf.fit(train_x, train_y)
print(clf.feature_importances_)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(train_x.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(train_x.shape[1]), indices)
# plt.xlim([-1, train_x.shape[1]])
# plt.show()

predicted_y = clf.predict(test_x)

print('Numb of mismatch RandomForestClassifier: ' + str(
    (test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

#GAUSSIAN WITH ALL
gnb = GaussianNB()
gnb.fit(train_x, train_y)
predicted_y = gnb.predict(test_x)

print('Numb of mismatch GaussianNB: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITHOUT PLEDGED PERCENTAGE
gausian_df = pd.get_dummies(df[["country", "main_category", "currency"]])
gausian_df["pledged"] = df["pledged"].astype(np.float64)
gausian_df["goal"] = df["goal"].astype(np.float64)
gausian_df['backers'] = df.backers.astype(np.float64)
gausian_df.dropna(inplace=True)
gausian_df['num_days'] = df["num_days"].astype(np.int)

train_gausian_x = gausian_df[250000:]
test_gausian_x = gausian_df[:250000]

gnb_hot = GaussianNB()
gnb_hot.fit(train_gausian_x, train_y)
predicted_y = gnb_hot.predict(test_gausian_x)

print('Numb of mismatch GaussianNB with hot encoder: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

# GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITH PLEDGED PERCENTAGE
gausian_df["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"]

train_gausian_x_pledgedP = gausian_df[250000:]
test_gausian_x_pledgedP = gausian_df[:250000]

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

print('Numb of mismatch LogisticRegression with hot encoder with percentage pledged: : ' + str((test_y != predicted_y).sum()))
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
# df[["country", "main_category", "category", "currency", "state", "country"]] = df[
#     ["country", "main_category", "category", "currency", "state", "country"]].apply(
#     lambda x: d[x.name].fit_transform(x))
#
# pp = sns.jointplot(x= df.main_category, y = df.state, bins=5, kind="hex")
# plt.show()
#
# # convert pledged to int
# df.pledged = df.pledged.astype(float)
# df.goal = df.goal.astype(float)
# import numpy as np
# df["pledge_log"] = np.log(df["pledged"]+ 1)
# df["goal_log"] = np.log(df["goal"]+ 1)
# ax = sns.distplot(df["pledge_log"])
# plt.show()
#
# # boxplot from goal
# ax = sns.distplot(df["goal_log"])
# plt.show()
#
# spec = df[["goal_log","pledge_log"]]
#
# sns.heatmap(spec.corr(method="pearson"), xticklabels=spec.columns.values, yticklabels = spec.columns.values)
# plt.show()
#
# #jointplot from country and state
# sns.jointplot(x=df.country, y=df.state, kind="hex", bins=5)
# print("")
