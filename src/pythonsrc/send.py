def get_train_clusters (train_df):
    latlong = train_df.copy () [['latitude', 'longitude']]
    n_cluster = 15
    clf = KMeans(n_clusters=n_cluster)
    clf.fit(latlong)
    train_df['cluster_id'] = clf.labels_
    
    return train_df, clf

train_df, clf = get_train_clusters (train_df)

def get_test_clusters (test_df, clf):
    latlong = test_df.copy () [['latitude', 'longitude']]
    testLabels = [clf.predict([x]) for x in test_df.copy()[['latitude','longitude']].values]
    test_df["cluster_id"] = np.asarray(testLabels)
    
    return test_df
