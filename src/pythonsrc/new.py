import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("train_all_feat.csv")
test  = pd.read_csv("test_all_feat.csv")

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = pd.read_csv("y.csv")
train_y = np.array(train_y['interest_level'].apply(lambda x: target_num_map[x]))


features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price", "num_features", 
    "len_description","created_year", "created_month", "created_day", "created_hour", "created_weekday", 
    "is_night", "is_weekend", "price_per_bed", "price_per_bath", "total_rooms", "price_per_room",
    "manager_skill", "building_popularity", "logprice", "density"]
count_features = CountVectorizer(stop_words='english', max_features=500,ngram_range=(1,4))
count_features_train_sparse = count_features.fit_transform(train["features"])
count_features_test_sparse = count_features.transform(test["features"])

train_X = sparse.hstack([train[features_to_use], count_features_train_sparse]).tocsr()
test_X = sparse.hstack([test[features_to_use], count_features_test_sparse]).tocsr()

tfidf_features = TfidfVectorizer(stop_words='english', max_features=500,ngram_range=(1,4))
tfidf_features_train_sparse = count_features.fit_transform(train_df["features"])
tfidf_features_test_sparse = count_features.transform(test_df["features"])

train_X = sparse.hstack([train_X, tfidf_features_train_sparse]).tocsr()
test_X = sparse.hstack([test_X, tfidf_features_test_sparse]).tocsr()

count_desc = CountVectorizer(stop_words='english', max_features=500,ngram_range=(1,4))
count_desc_train_sparse = count_desc.fit_transform(train_df["description"])
count_desc_test_sparse = count_desc.transform(test_df["description"])

train_X = sparse.hstack([train_X, count_desc_train_sparse]).tocsr()
test_X = sparse.hstack([test_X, count_desc_test_sparse]).tocsr()

tfidf_desc = CountVectorizer(stop_words='english', max_features=500,ngram_range=(1,4))
tfidf_desc_train_sparse = tfidf_desc.fit_transform(train_df["description"])
tfidf_desc_test_sparse = tfidf_desc.transform(test_df["description"])

train_X = sparse.hstack([train_X, tfidf_desc_train_sparse]).tocsr()
test_X = sparse.hstack([test_X, tfidf_desc_test_sparse]).tocsr()