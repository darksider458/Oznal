import scipy
import warnings

import seaborn as sns
import pandas as pd

import sys

from datetime import timedelta

from mlxtend.preprocessing import minmax_scaling
from scipy.stats import pearsonr, spearmanr, stats, boxcox
from sklearn import preprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, forest, \
    RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, chi2, \
    GenericUnivariateSelect
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def chi2_squared_test(df1, df2, df3, series):
    counted_df1 = df1[series].value_counts()
    counted_df2 = df2[series].value_counts()
    counted_df3 = df3[series].value_counts()
    arr = np.array([counted_df1.values, counted_df2.values, counted_df3.values])
    chi2, p, dof, expected = scipy.stats.chi2_contingency(arr)
    print('P-value of dataset supplied for series: %.8f' % p)


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
    # ax = sns.countplot(x=series_name, data=df)
    # plt.show()
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


def calculateKFoldvalidation(y_train, X_train, classifier):
    kfold = StratifiedKFold(y=y_train.values, n_folds=10, random_state=1)
    scores = []
    for k, (train, test) in enumerate(kfold):
        classifier.fit(X_train.values[train], y_train.values[train])
        score = classifier.score(X_train.values[test], y_train.values[test])
        scores.append(score)
        print('Fold: %s' % (k + 1))
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def upsampleDataset(df_transformed):
    # UPSAMPLE
    # Separate classes for upsampling
    df_successful = df_transformed[df_transformed.state == 2]
    df_cancelled = df_transformed[df_transformed.state == 0]
    df_failed = df_transformed[df_transformed.state == 1]

    # Upsample minority class
    df_cancelled_upsampled = resample(df_cancelled,
                                      replace=True,  # sample with replacement
                                      n_samples=151627)

    df_successful_upsampled = resample(df_successful,
                                       replace=True,  # sample with replacement
                                       n_samples=151627)

    # Combine majority class with upsampled minority class
    return pd.concat([df_failed, df_cancelled_upsampled, df_successful_upsampled])


def downsampleDataset(df_transformed):
    # DOWNSAMPLE
    # Separate classes for downsampling
    df_successful = df_transformed[df_transformed.state == 2]
    df_cancelled = df_transformed[df_transformed.state == 0]
    df_failed = df_transformed[df_transformed.state == 1]

    # Upsample minority class
    df_failed_upsample = resample(df_failed,
                                  replace=True,  # sample with replacement
                                  n_samples=28700)

    df_successful_upsampled = resample(df_successful,
                                       replace=True,  # sample with replacement
                                       n_samples=28700)

    # Combine majority class with upsampled minority class
    return pd.concat([df_cancelled, df_failed_upsample, df_successful_upsampled])


if __name__ == '__main__':
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
    # ax = sns.distplot(df["log_num_days"])
    # plt.show()
    # create_boxplot(df, "log_num_days")
    # DEFINE PATTERNS
    patternFloatDet = "[+-]?([0-9]*[.])?[0-9]+"
    patternDelDate = "^\d{4}\-(0?[1-9]|1[012])\-(0?[1-9]|[12][0-9]|3[01]) "
    patternStringDel = "[A-z]"
    # DELETE ROWS WHICH WE DONT WANT FROM STATE
    df = df[df.state.str.contains("undefined") == False]
    # print(len(df.state))
    df = df[df.state.str.contains("live") == False]
    # print(len(df.state))
    df = df[df.state.str.contains("suspended") == False]
    # WE DO CHISQUARE TEST WITH CATEGORICAL DATA
    # df_onlyS = df[df.state.str.contains("successful") == True]
    # df_onlyF = df[df.state.str.contains("failed") == True]
    # df_onlyC = df[df.state.str.contains("canceled") == True]
    #
    # chi2_squared_test(df_onlyS, df_onlyF, df_onlyC, "main_category")
    # chi2_squared_test(df_onlyS, df_onlyF, df_onlyC, "country")
    # chi2_squared_test(df_onlyS, df_onlyF, df_onlyC, "currency")
    # chi2_squared_test(df_onlyS, df_onlyF, df_onlyC, "category")
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
    # df_transformed["percentage_of_pledged"] = round(df_transformed_data['pledged'] / df_transformed_data['goal'] * 100,
    #                                                 2)
    # df_transformed["percentage_of_pledged"] = df_transformed["percentage_of_pledged"].astype(np.float64)
    # # WE ARE GONNA DO THE WILCOXON TEST
    # df_onlyS = df_transformed[df_transformed.state.str.contains("successful") == True]
    #
    # first, second, train_y, test_y = train_test_split(df_onlyS["pledged"], df_onlyS["state"], test_size=0.5,
    #                                                   random_state=0)
    # print(scipy.stats.wilcoxon(df_onlyS["pledged"], df_onlyS["goal"]))
    #
    # first, second, train_y, test_y = train_test_split(df_onlyS["goal"], df_onlyS["state"], test_size=0.5,
    #                                                   random_state=0)
    # print(scipy.stats.wilcoxon(df_onlyS["num_days"], df_onlyS["backers"]))
    #
    # first, second, train_y, test_y = train_test_split(df_onlyS["backers"], df_onlyS["state"], test_size=0.5,
    #                                                   random_state=0)
    # print(scipy.stats.wilcoxon(df_onlyS["pledged"], df_onlyS["percentage_of_pledged"]))
    #
    # first, second, train_y, test_y = train_test_split(df_onlyS["num_days"], df_onlyS["state"], test_size=0.5,
    #                                                   random_state=0)
    # print(scipy.stats.wilcoxon(df_onlyS["pledged"], df_onlyS["goal"]))
    # pp = sns.jointplot(x=df["log_pledged"], y=df_transformed.state, bins=5, kind="hex")
    # plt.show()
    #
    # pp = sns.jointplot(x=df["log_goal"], y=df_transformed.state, bins=5, kind="hex")
    # plt.show()
    #
    # pp = sns.jointplot(x=df["log_backers"], y=df_transformed.state, bins=5, kind="hex")
    # plt.show()
    # UPSAMPLE
    df_transformed = downsampleDataset(df_transformed)
    df_transformed["goal"] = boxcox(df_transformed["goal"] + 0.001)[0]
    df_transformed["pledged"] = boxcox(df_transformed["pledged"] + 0.001)[0]
    df_transformed["num_days"] = boxcox(df_transformed["num_days"] + 0.001)[0]
    # df_transformed.goal = minmax_scaling(df_transformed.goal, columns=[0])
    # df_transformed.pledged = minmax_scaling(df_transformed.pledged, columns=[0])
    # df_transformed.num_days = minmax_scaling(df_transformed.num_days, columns=[0])
    df_transformed["percentage_of_pledged"] = round(df_transformed['pledged'] / df_transformed['goal'] * 100,
                                                    4)
    df_transformed["percentage_of_pledged"] = df_transformed["percentage_of_pledged"].astype(np.float64)
    df_transformed["pledge_per_backer"] = round(df_transformed['pledged'] / (df_transformed['backers'] + 1),
                                                4)
    df_transformed["pledge_per_backer"] = df_transformed["pledge_per_backer"].astype(np.float64)
    df_transformed["pledge_per_day"] = round(df_transformed['pledged'] / (df_transformed['num_days'] + 1),
                                             4)
    df_transformed["pledge_per_day"] = df_transformed["pledge_per_day"].astype(np.float64)
    df_transformed.dropna(inplace=True)
    # spec = df_transformed_data[
    #     ["goal", "pledged", "backers", "num_days", "percentage_of_pledged", "pledge_per_backer", "pledge_per_day"]]
    # Display new class counts
    print(df_transformed.state.value_counts())
    df_transformed_data = df_transformed[
        ["goal", "pledged", "country", "main_category", "category", "currency",
         "backers", "num_days"]]
    # END UPSAMPLE
    # calculate_pvalues(spec)
    # sns.heatmap(spec.corr(method="pearson"), xticklabels=spec.columns.values, yticklabels=spec.columns.values)
    # plt.show()
    # df_for_validationY = df_transformed["state"]
    # df_transformed["state"] = df_simple["state"]
    #
    # train_x, test_x, train_y, test_y = train_test_split(
    #     df_transformed_data[["goal", "pledged", "backers", "percentage_of_pledged", "num_days", "pledge_per_backer",
    #                          "pledge_per_day"]], df_transformed["state"],
    #     test_size=0.2, random_state=0)
    #
    # knn_tt = KNeighborsClassifier(n_neighbors=20)
    # knn_tt.fit(train_x, train_y)
    # predicted_y = knn_tt.predict(test_x)
    #
    # print('Numb of mismatch KNeighborsClassifier using the pledged percentage only float features: ' + str(
    #     (test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # calculateKFoldvalidation(df_for_validationY, df_transformed_data[["goal", "pledged", "backers", "percentage_of_pledged", "num_days"]], knn_tt)
    # SWITCH BACK TO ALL DATA
    df_transformed["state"] = label_binarize(df_transformed["state"], classes=[0, 1, 2])
    train_x, test_x, train_y, test_y = train_test_split(df_transformed_data, df_transformed["state"], test_size=0.2,
                                                        random_state=0)
    #
    # compute_covariance_matrix(train_x, test_x)
    clf = RandomForestClassifier(n_estimators=250, random_state=0, max_depth=3)
    clf = clf.fit(train_x, train_y)
    print(clf.feature_importances_)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    feat_labels = df_transformed_data.columns[1:]
    # # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(train_x.shape[1] - 1):
    #     print("%d. feature %s (%f)" % (f + 1, feat_labels[f], importances[indices[f]]))
    #
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(train_x.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(train_x.shape[1]), indices)
    # plt.xlim([-1, train_x.shape[1]])
    # plt.show()
    predicted_y = clf.predict(test_x)
    print('Numb of mismatch RandomForestClassifier: ' + str(
        (test_y != predicted_y).sum()))
    print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # calculateKFoldvalidation(df_transformed["state"], df_transformed_data, clf)
    # WITHOUT PLEDGED IN PERCENTAGE - FLOAT
    knn_ttALL = KNeighborsClassifier()
    knn_ttALL.fit(train_x, train_y)
    predicted_y = knn_ttALL.predict(test_x)
    print('Numb of mismatch KNeighborsClassifier: ' + str(
        (test_y != predicted_y).sum()))
    print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))

    # myList = list(range(1, 50))
    #
    # # subsetting just the odd ones
    # neighbors = filter(lambda x: x % 2 != 0, myList)
    #
    # # empty list that will hold cv scores
    # cv_scores = []
    # nums = []
    # # perform 10-fold cross validation
    # for k in neighbors:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, train_x, train_y, cv=10, scoring='accuracy')
    #     cv_scores.append(scores.mean())
    #     nums.append(k)
    #
    # # changing to misclassification error
    # MSE = [1 - x for x in cv_scores]
    #
    # # plot misclassification error vs k
    # plt.plot(nums, cv_scores)
    # plt.xlabel('Number of Neighbors K')
    # plt.ylabel('Accuracy')
    # plt.show()
    # calculateKFoldvalidation(df_transformed["state"], df_transformed_data, knn_tt)
    # GAUSSIAN WITH ALL
    gnbALL = GaussianNB()
    gnbALL.fit(train_x, train_y)
    predicted_y = gnbALL.predict(test_x)
    print('Numb of mismatch GaussianNB: ' + str((test_y != predicted_y).sum()))
    print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # calculateKFoldvalidation(df_transformed["state"], df_transformed_data, gnb)
    # GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITHOUT PLEDGED PERCENTAGE
    # gausian_df = pd.get_dummies(df[["country", "main_category", "currency", "category"]])
    # gausian_df["pledged"] = df["pledged"].astype(np.float64)
    # gausian_df["goal"] = df["goal"].astype(np.float64)
    # gausian_df['backers'] = df.backers.astype(np.float64)
    # gausian_df.dropna(inplace=True)
    # gausian_df['num_days'] = df["num_days"].astype(np.int)
    #
    # print(gausian_df.info())
    #
    # train_gausian_x, test_gausian_x, train_y, test_y = train_test_split(gausian_df,
    #                                                                     df_transformed["state"],
    #                                                                     test_size=0.2, random_state=0)
    # gnb_hot = GaussianNB()
    # gnb_hot.fit(train_gausian_x, train_y)
    # predicted_y = gnb_hot.predict(test_gausian_x)
    #
    # print('Numb of mismatch GaussianNB with hot encoder: ' + str((test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    #
    # # calculateKFoldvalidation(df_transformed["state"], gausian_df, gnb_hot)
    #
    # # GAUSSIAN WITH ONE HOT ENCODER WITHOUT SUB CATEGORY WITH PLEDGED PERCENTAGE
    # gausian_df["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"]
    # gausian_df["pledge_per_day"] = df_transformed_data["pledge_per_day"]
    #
    # train_gausian_x_pledgedP, test_gausian_x_pledgedP, train_y, test_y = train_test_split(gausian_df,
    #                                                                                       df_transformed["state"],
    #                                                                                       test_size=0.2, random_state=0)
    # gnb_hotP = GaussianNB()
    # gnb_hotP.fit(train_gausian_x_pledgedP, train_y)
    # predicted_yP = gnb_hotP.predict(test_gausian_x_pledgedP)
    #
    # print('Numb of mismatch GaussianNB with hot encoder with percentage pledged: ' + str((test_y != predicted_yP).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_yP))
    # calculateKFoldvalidation(df_transformed["state"], gausian_df, gnb_hotP)
    lgALL = LogisticRegression()
    lgALL.fit(train_x, train_y)
    predicted_y = lgALL.predict(test_x)
    print('Numb of mismatch LogisticRegression: ' + str((test_y != predicted_y).sum()))
    print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # calculateKFoldvalidation(df_transformed["state"], df_transformed_data, lg)
    # lg = LogisticRegression()
    # lg.fit(train_gausian_x, train_y)
    # predicted_y = lg.predict(test_gausian_x)
    #
    # print('Numb of mismatch LogisticRegression with hot encoder: ' + str((test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # gausian_df.drop(columns=["percentage_of_pledged"], inplace=True)
    # calculateKFoldvalidation(df_transformed["state"], gausian_df, lg)
    # lg = LogisticRegression()
    # lg.fit(train_gausian_x_pledgedP, train_y)
    # predicted_y = lg.predict(test_gausian_x_pledgedP)
    #
    # gausian_df["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"]
    # gausian_df["pledge_per_backer"] = df_transformed_data["pledge_per_backer"]
    # gausian_df["pledge_per_day"] = df_transformed_data["pledge_per_day"]
    # print('Numb of mismatch LogisticRegression with hot encoder with percentage pledged: : ' + str(
    #     (test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # calculateKFoldvalidation(df_transformed["state"], gausian_df, lg)
    # knn_tt = KNeighborsClassifier()
    # knn_tt.fit(train_gausian_x, train_y)
    # predicted_y = knn_tt.predict(test_gausian_x)
    #
    # print('Numb of mismatch KNeighborsClassifier with hot encoder: ' + str(
    #     (test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    # gausian_df.drop(columns=["percentage_of_pledged"], inplace=True)
    # calculateKFoldvalidation(df_transformed["state"], gausian_df, knn_tt)
    # knn_tt = KNeighborsClassifier()
    # knn_tt.fit(train_gausian_x_pledgedP, train_y)
    # predicted_y = knn_tt.predict(test_gausian_x_pledgedP)
    #
    # print('Numb of mismatch KNeighborsClassifier with hot encoder using the pledged percentage: ' + str(
    #     (test_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_y, predicted_y))
    #
    # gausian_df["percentage_of_pledged"] = df_transformed_data["percentage_of_pledged"]
    # gausian_df["pledge_per_backer"] = df_transformed_data["pledge_per_backer"]
    # gausian_df["pledge_per_day"] = df_transformed_data["pledge_per_day"]
    # calculateKFoldvalidation(df_transformed["state"], gausian_df, knn_tt)
    # NOW WITH SCALING DATA - KN
    # scaled_data = minmax_scaling(df_transformed_data,
    #                              columns=["goal", "pledged", "backers", "percentage_of_pledged", "pledge_per_day",
    #                                       "num_days", "country",
    #                                       "main_category", "category", "currency"])
    #
    # train_scaled_x, test_scaled_x, train_scaled_y, test_scaled_y = train_test_split(scaled_data,
    #                                                                                 df_transformed["state"],
    #                                                                                 test_size=0.2, random_state=0)
    #
    # knn_tt = KNeighborsClassifier()
    # knn_tt.fit(train_scaled_x, train_scaled_y)
    # predicted_y = knn_tt.predict(test_scaled_x)
    #
    # print('Numb of mismatch KNeighborsClassifier scaled (all scaled) all data: ' + str(
    #     (test_scaled_y != predicted_y).sum()))
    # print('Accuracy: %.4f' % accuracy_score(test_scaled_y, predicted_y))
    # calculateKFoldvalidation(df_transformed["state"], scaled_data, knn_tt)
    # MAJORITY VOTE CLASSIFIER WITH ALL DATA
    # ADA BOOST
    print("DecisionTree")
    tree = DecisionTreeClassifier(random_state=11, max_features="auto", class_weight="balanced", max_depth=None)

    param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                  "base_estimator__splitter": ["best", "random"],
                  "learning_rate": [0.1, 0.01, 0.001],
                  "n_estimators": [100, 250, 500]
                  }

    print("AdaBoost")
    ada = AdaBoostClassifier(base_estimator=tree, learning_rate=0.01)

    # run grid search
    grid_search_ABC = GridSearchCV(ada, param_grid=param_grid, scoring='accuracy')

    grid_search_ABC.fit(train_x, train_y)

    print('Best parameter set: %s ' % grid_search_ABC.best_params_)
    print('CV Accuracy: %.3f' % grid_search_ABC.best_score_)

    clf = grid_search_ABC.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(test_x, test_y))

    # scores = cross_val_score(estimator=ada,
    #                          X=train_x,
    #                          y=train_y,
    #                          cv=10,
    #                          scoring='accuracy')
    #
    #
    # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), "ada"))

    param_grid = {"max_depth": [3, 5, 10],
                  "max_features": [None, "log2", "auto"],
                  "n_estimators": [250, 500]
                  }

    print("GradientBoostingClassifier")
    gbc = GradientBoostingClassifier(subsample=0.5, learning_rate=0.01,
                                     random_state=3, min_samples_leaf=1)

    grid_search_GBC = GridSearchCV(gbc, param_grid=param_grid, scoring='accuracy')

    grid_search_GBC.fit(train_x, train_y)

    print('Best parameter set: %s ' % grid_search_GBC.best_params_)
    print('CV Accuracy: %.3f' % grid_search_GBC.best_score_)
    # scores = cross_val_score(estimator=clf,
    #                          X=train_x,
    #                          y=train_y,
    #                          cv=10,
    #                          scoring='accuracy')
    # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), "GBC"))

    param_grid = {"weights": [[1, 1, 1, 2], [1, 1, 1, 3], [1, 1, 1, 4]]
                  }

    eclf1 = VotingClassifier(estimators=[('lr', lgALL), ('rf', clf), ('gnb', gnbALL), ('knn', knn_ttALL)],
                             voting='soft')

    grid_search_eclf1 = GridSearchCV(eclf1, param_grid=param_grid, scoring='accuracy')

    grid_search_eclf1.fit(train_x, train_y)

    print('Best parameter set: %s ' % grid_search_eclf1.best_params_)
    print('CV Accuracy: %.3f' % grid_search_eclf1.best_score_)

    #
    # clf_labels = ['VotingClassifier', 'Logistic Regression with ALL', 'Random Forest with ALL', 'GNB with ALL',
    #               'KNN with ALL', 'DecisionTree', 'AdaBoost']
    # all_clf = [eclf1, lgALL, clf, gnbALL, knn_ttALL, tree, ada, eclf1]
    # for clf, label in zip(all_clf, clf_labels):
    #     scores = cross_val_score(estimator=clf,
    #                              X=train_x,
    #                              y=train_y,
    #                              cv=10,
    #                              scoring='accuracy', n_jobs=2)
    #     print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
