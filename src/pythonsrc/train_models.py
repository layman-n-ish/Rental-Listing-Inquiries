import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

df = pd.read_json("data/train.json")
X = pd.read_csv("data/X.csv")
y = np.asarray(pd.read_csv("data/y.csv")).ravel()

df_test = pd.read_json("data/test.json")
X_test = pd.read_csv("data/X_test.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)

def pred_to_csv(model_name, y_test):
    y_pred = pd.DataFrame()
    y_pred['listing_id'] = df_test['listing_id']
    for i in range(len(list(y_test))):
        [y_pred.loc[y_pred.index[i], 'high'], y_pred.loc[y_pred.index[i], 'low'], y_pred.loc[y_pred.index[i], 'medium']] = list(y_test[i])

    print(y_pred.head())

    y_pred.to_csv("results/pred_"+ model_name +".csv", index=False)

    print("\n\nDone! CSV for "+model_name+"'s predictions created!\n")

def plot_learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=None):
    plt.figure()
    plt.title("Leaning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='neg_log_loss')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def train_log_reg():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.fit_transform(X_test)

    model = LogisticRegression(C=0.01)
    print("\nLogistic Regresion: Training...\n")
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict_proba(X_train_scaled)
    print("Logistic Regression: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val_scaled)
    print("\nLogistic Regression: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test_scaled)

    pred_to_csv("lr", y_test)

def train_svm():
    model = SVC(kernel='rbf', probability=True)
    print("\nSupport Vector Machine: Training...\n")
    model.fit(X_train, y_train)
    print("Done training!\n")

    y_train_pred = model.predict_proba(X_train)
    print("Support Vector Machine: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val)
    print("\nSupport Vector Machine: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test)

    pred_to_csv("svm", y_test)

def train_gnb():
    model = GaussianNB()
    print("\nSupport Vector Machine: Training...\n")
    model.fit(X_train, y_train)
    print("Done training!\n")

    y_train_pred = model.predict_proba(X_train)
    print("Support Vector Machine: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val)
    print("\nSupport Vector Machine: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test)

    pred_to_csv("gnb", y_test)


def train_random_forest():
    model = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='auto'
    , min_samples_leaf=70, min_samples_split=50, n_jobs=-1)
    print("\nRandom Forest: Training...\n")
    model.fit(X_train, y_train)

    y_train_pred = model.predict_proba(X_train)
    print("Random Forest: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val)
    print("\nRandom Forest: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test)

    pred_to_csv("rf", y_test)

def train_xgboost():
    model = XGBClassifier(n_estimators=700, learning_rate=0.1, early_stopping_rounds=10)
    print("\nXGBoost: Training...\n")
    model.fit(X_train, y_train)

    y_train_pred = model.predict_proba(X_train)
    print("XGBoost: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val)
    print("\nXGBoost: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test)

    pred_to_csv("xg", y_test)

    # plot_learning_curve(model, X = X, y = y, cv=cv)

def train_bag():
    model = BaggingClassifier(n_estimators=500, n_jobs=-1, max_features=1, max_samples=1)
    print("\nBagging Classifier: Training...\n")
    model.fit(X_train, y_train)
    print("Done training!\n")

    y_train_pred = model.predict_proba(X_train)
    print("Bagging Classifier: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val)
    print("\nBagging Classifier: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test)

    pred_to_csv("bag", y_test)

def train_voting():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.fit_transform(X_test)

    model1 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=70, n_jobs=-1)
    model2 = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=70, n_jobs=-1)
    model3 = GradientBoostingClassifier()
    model4 = XGBClassifier(n_estimators=500, learning_rate=1)

    model = VotingClassifier(estimators=[('rf1', model1), ('rf2', model2), ('gb', model3), ('xgb', model3)]
    , voting='soft', n_jobs=-1, weights=[1, 1, 5, 10])
    print("\nVoting Classifier: Training...\n")
    model.fit(X_train_scaled, y_train)
    print("Done training!\n")

    y_train_pred = model.predict_proba(X_train_scaled)
    # print(y_train_pred)
    # print(y_train)
    print("\nVoting Classifier: Log loss on training set: %f" %log_loss(y_train, y_train_pred))

    y_val_pred = model.predict_proba(X_val_scaled)
    print("\nVoting Classifier: Log loss on validation set: %f" %log_loss(y_val, y_val_pred))

    print("\nPredicting on test set...\n\n\n")
    y_test = model.predict_proba(X_test_scaled)

    pred_to_csv("vote", y_test)


if __name__ == "__main__":

    # train_svm()
    train_random_forest()
    # train_voting()
    # train_log_reg()
    # train_gnb()
    
