import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def calculate_mean(series):
    suma = 0
    for entry in series:
        if entry is not "NAN":
            suma = suma + float(entry)
    return suma / len(series)


df = pd.read_csv('ks-projects-201612.csv', encoding="ISO-8859-1",
                 na_values=['NAN'])
df.head()
df.info()
# DROP UNUSEFUL COLUMNS AND RENAME THEM
df.drop(df.columns[[13, 14, 15, 16]], axis=1, inplace=True)
df.rename(columns={'country ': 'country', 'main_category ': 'main_category',
                   'ID ': 'ID', 'name ': 'name', 'category ': 'category',
                   'currency ': 'currency', 'deadline ': 'deadline',
                   'goal ': 'goal', 'launched ': 'launched',
                   'pledged ': 'pledged', 'state ': 'state',
                   'backers ': 'backers', 'usd pledged ': 'usd_pledged'}
          , inplace=True)
print(df.columns)
print(df.head())
d = defaultdict(preprocessing.LabelEncoder)

# DEFINE PATTERNS
patternFloatDet = "[+-]?([0-9]*[.])?[0-9]+"
patternDelDate = "^\d{4}\-(0?[1-9]|1[012])\-(0?[1-9]|[12][0-9]|3[01]) "
patternStringDel = "[A-z]"

# CLEAN STATE COLUMN
date_filter = df.state.str.contains(patternDelDate)
float_filter = df.state.str.contains(patternFloatDet)
df = df[~date_filter]
df = df[~float_filter]
state = df.state.value_counts(dropna=True)
# state.plot.pie(figsize=(6, 6))
print(state.to_string())

# CLEAN USD_PLEDGED COLUMN
date_usd_filter = df.usd_pledged.str.contains(patternDelDate, na=False)
string_usd_filter = df.usd_pledged.str.contains(patternStringDel, na=False)
# float_usd_filter = df.usd_pledged.str.contains(patternFloatDet, na=False)
df = df[~date_usd_filter]
df = df[~string_usd_filter]

# CLEAN PLEDGED COLUMN
date_pledged_filter = df.pledged.str.contains(patternDelDate, na=False)
# float_usd_filter = df.usd_pledged.str.contains(patternFloatDet, na=False)
df = df[~date_pledged_filter]
print(calculate_mean(df.pledged.dropna()))

# CLEAN CURRENCY - CURRENCY IS OK
print(df.currency.value_counts(dropna=True).to_string())

# CLEAN GOAL - REMOVE DATE
date_goal_filter = df.goal.str.contains(patternDelDate, na=False)
# float_goal_filter = df.goal.str.contains(patternFloatDet, na=False)
df = df[~date_goal_filter]
print(calculate_mean(df.goal))

# CLEAN COUNTRY - RENAME NAN
df.country.replace(['N,"0'], 'NAN', inplace=True)

# CLEAN CATEGORY
print(df.category.value_counts(dropna=True))

# CLEAN MAIN_CATEGORY
print(df.main_category.value_counts(dropna=True))

# CLEAN LAUNCH
date_launched_filter = df.launched.str.contains(patternDelDate)
df = df[date_launched_filter]
print(len(df.launched))

# CLEAN DEADLINE
date_deadline_filter = df.deadline.str.contains(patternDelDate)
df = df[date_deadline_filter]
print(len(df.deadline))

# CLEAN GOAL
date_goal_filter = df.goal.str.contains(patternDelDate, na=False)
string_goal_filter = df.goal.str.contains(patternStringDel, na=False)
df = df[~date_goal_filter]
df = df[~string_goal_filter]
print(len(df.goal))

# CLEAN BACKERS
date_backers_filter = df.backers.str.contains(patternDelDate, na=False)
string_backers_filter = df.backers.str.contains(patternStringDel, na=False)
df = df[~date_backers_filter]
df = df[~string_backers_filter]
print(len(df.backers))

#DELETE ROWS WHICH WE DONT WANT FROM STATE
df = df[df.state.str.contains("undefined") == False]
print(len(df.state))

df = df[df.state.str.contains("live") == False]
print(len(df.state))

df = df[df.state.str.contains("suspended") == False]
print(len(df.state))
# linear
#plt.subplot(221)
#df.main_category.value_counts(dropna=True).plot.bar()
#plt.title('main_category')
#plt.grid(True)


# log
#plt.subplot(222)
#df.country.value_counts(dropna=True).head(10).plot.pie(subplots=True)
#plt.title('country')
#plt.grid(True)

# symmetric log
#plt.subplot(223)
#df.currency.value_counts(dropna=True).head(6).plot.pie(subplots=True)
#plt.title('currency')
#plt.grid(True)

# logit
#plt.subplot(224)
#df.category.value_counts(dropna=True).head(8).plot.bar()
#plt.title('category')
#plt.grid(True)

# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    #wspace=0.35)

#plt.show()

# WE USE LABEL ENCODER TO ENCODE CATEGORICAL ENUM DATA INTO NUMERICAL
df_simple = df[["goal","pledged","country", "main_category", "category", "currency", "state","backers"]]
df_transformed = df_simple[
    ["country", "main_category", "category", "currency","country"]].apply(
    lambda x: d[x.name].fit_transform(x))
import numpy as np
df_transformed['goal'] = df_simple.goal.astype(np.float64)
df_transformed['pledged'] = df_simple.pledged.astype(np.float64)
df_transformed['backers'] = df_simple.backers.astype(np.float64)
df_transformed['state'] = df_simple.state

df_transformed_data = df_transformed[["goal","pledged","country", "main_category", "category", "currency","backers"]]

print(df_transformed_data.head())
print(df_transformed_data.info())
print(df_transformed["state"].head())

train_y = df_transformed[250000:]["state"]
train_x = df_transformed_data[250000:]
test_y = df_transformed[:250000]["state"]
test_x = df_transformed_data[:250000]

knn_tt = KNeighborsClassifier()
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

knn_tt = KNeighborsClassifier(n_neighbors=20)
knn_tt.fit(train_x, train_y)
predicted_y = knn_tt.predict(test_x)

print('Numb of mismatch KNeighborsClassifier: ' + str((test_y != predicted_y).sum()))
print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))


gnb = GaussianNB()
gnb.fit(train_x, train_y)
predicted_y = gnb.predict(test_x)

print('Numb of mismatch GaussianNB: ' + str((test_y != predicted_y).sum()))
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
