#"place" : (latitude, longitude)
geo = {"airport":(40.64912697, -73.78692627), 
        "central park": (40.783661, -73.96536827),
        "washington sq. park": (40.73083612, -73.99749041), 
        "financial district": (40.705628, -74.010278)}

    #40.849209,-73.888508 #crotona aven
    #40.747844,-73.901731 #61st 
    #40.678722,-73.951174 #atlantic aven
    #40.688788,-73.870111 #lincoln aven
    #40.624861,-73.967846 #j aven

def get_dist(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance

def get_dist_features(df):
    df["cp_dist"] = list(map(lambda lat2, lon2: get_dist(40.783661, -73.96536827, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["jfk_dist"] = list(map(lambda lat2, lon2: get_dist(40.64912697, -73.78692627, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["wsp_dist"] = list(map(lambda lat2, lon2: get_dist(40.73083612, -73.99749041, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["fd_dist"] = list(map(lambda lat2, lon2: get_dist(40.705628, -74.010278, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["ca_dist"] = list(map(lambda lat2, lon2: get_dist(40.849209,-73.888508, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["61_dist"] = list(map(lambda lat2, lon2: get_dist(40.747844,-73.901731, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["aa_dist"] = list(map(lambda lat2, lon2: get_dist(40.678722,-73.951174, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["la_dist"] = list(map(lambda lat2, lon2: get_dist(40.688788,-73.870111, lat2, lon2), 
    df["latitude"], df["longitude"]))
    df["ja_dist"] = list(map(lambda lat2, lon2: get_dist(40.624861,-73.967846, lat2, lon2), 
    df["latitude"], df["longitude"]))
    

    features = ["cp_dist", "jfk_dist", "wsp_dist", "fd_dist", "ca_dist", "61_dist", "aa_dist", "la_dist", "ja_dist"]
    
    return df[features]

def get_count_features(df, df_test):
    m_count = df.groupby(['manager_id']).count().iloc[:, 1].to_dict() 
    m_test_count = df_test.groupby(['manager_id']).count().iloc[:, 1].to_dict()
    for manager in m_test_count:
        if manager not in m_count:
            m_count[str(manager)] = m_test_count[str(manager)]
        else:
            m_count[str(manager)] += m_test_count[str(manager)]

    b_count = df.groupby(['building_id']).count().iloc[:, 1].to_dict() 
    b_test_count = df_test.groupby(['building_id']).count().iloc[:, 1].to_dict()
    for building in b_test_count:
        if building not in b_count:
            b_count[str(building)] = b_test_count[str(building)]
        else:
            b_count[str(building)] += b_test_count[str(building)]
    
    da_count = df.groupby(['display_address']).count().iloc[:, 1].to_dict() 
    da_test_count = df_test.groupby(['display_address']).count().iloc[:, 1].to_dict()
    for d_address in da_test_count:
        if d_address not in da_count:
            da_count[str(d_address)] = da_test_count[str(d_address)]
        else:
            da_count[str(d_address)] += da_test_count[str(d_address)]

    sa_count = df.groupby(['street_address']).count().iloc[:, 1].to_dict() 
    sa_test_count = df_test.groupby(['street_address']).count().iloc[:, 1].to_dict()
    for s_address in sa_test_count:
        if s_address not in sa_count:
            sa_count[str(s_address)] = sa_test_count[str(s_address)]
        else:
            sa_count[str(s_address)] += sa_test_count[str(s_address)]

    for i in range(df.shape[0]):
        df.loc[df.index[i], 'manager_listings_count'] = m_count[df.loc[df.index[i], 'manager_id']]
        df.loc[df.index[i], 'building_listings_count'] = b_count[df.loc[df.index[i], 'building_id']]
        df.loc[df.index[i], 's_address_listings_count'] = sa_count[df.loc[df.index[i], 'street_address']]
        df.loc[df.index[i], 'd_address_listings_count'] = da_count[df.loc[df.index[i], 'display_address']]
    
    for i in range(df_test.shape[0]):
        df_test.loc[df_test.index[i], 'manager_listings_count'] = m_count[df_test.loc[df_test.index[i], 'manager_id']]
        df_test.loc[df_test.index[i], 'building_listings_count'] = b_count[df_test.loc[df_test.index[i], 'building_id']]
        df_test.loc[df_test.index[i], 's_address_listings_count'] = sa_count[df_test.loc[df_test.index[i], 'street_address']]
        df_test.loc[df_test.index[i], 'd_address_listings_count'] = da_count[df_test.loc[df_test.index[i], 'display_address']]
        
    count_features = ['manager_listings_count', 'building_listings_count', 's_address_listings_count', 
    'd_address_listings_count']
    
    X_count = df[count_features]
    X_test_count = df_test[count_features]

    return [X_count, X_test_count]

def feature_engineering(df, flag, b_dict={}, m_dict={}):

    ulimit = np.percentile(df.price.values, 99)
    df['price'].loc[df.price > ulimit] = ulimit

    llimit = np.percentile(df.latitude.values, 1)
    ulimit = np.percentile(df.latitude.values, 99)
    df['latitude'].loc[df.latitude < llimit] = llimit
    df['latitude'].loc[df.latitude > ulimit] = ulimit

    llimit = np.percentile(df.longitude.values, 1)
    ulimit = np.percentile(df.longitude.values, 99)
    df['longitude'].loc[df.longitude < llimit] = llimit
    df['longitude'].loc[df.longitude > ulimit] = ulimit
    
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    df["is_night"] = 1*(df["created_hour"] <= 7)
    df["created_weekday"] = df["created"].dt.weekday
    df["is_weekend"] = 1*(df["created_weekday"] > 5)
    
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df['price_per_bed'] = df['price']/(df['bedrooms']+1)
    df['price_per_bath'] = df['price']/(df['bathrooms']+1)
    df["price_per_room"] = df['price']/(df['total_rooms']+1)
    
    df['description'] = df['description'].str.replace('<[^<>]+>', ' ', regex=True)
    df['description'] = df['description'].str.replace('[0-9]+', 'num', regex=True)
    df['description'] = df['description'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
    for i in df.index:
        df.loc[i, 'len_description'] = len(df.loc[i, 'description'])
        df.loc[i, 'num_features'] = len(df.loc[i, 'features'])

    if not flag:
        interest_dummies = pd.get_dummies(df.interest_level)
        df_dumm = pd.concat([df,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)

        df_n = df_dumm[['building_id', 'manager_id', 'low', 'high', 'medium']]
        df_n_m = df_n.groupby(df_n.manager_id).mean()
        df_n_b = df_n.groupby(df_n.building_id).mean()

        df_n_m['manager_skill'] = df_n_m['low']*0+ df_n_m['medium']*1 + df_n_m['high']*2
        df_n_b['build_popularity'] = df_n_b['low']*0+ df_n_b['medium']*1 + df_n_b['high']*2

        df_n_m.drop(['low', 'high', 'medium'], axis=1)
        df_n_b.drop(['low', 'high', 'medium'], axis=1)
        b_dict = df_n_b.to_dict()
        m_dict = df_n_m.to_dict() 

    for i in range(df.shape[0]):
        if df.loc[df.index[i], 'building_id'] in b_dict['build_popularity']:
            df.loc[df.index[i], 'building_popularity'] = b_dict['build_popularity'][df.loc[df.index[i], 'building_id']]
        else:
            df.loc[df.index[i], 'building_popularity'] = 0.33453255354249495 #bp_mean
    
        if df.loc[df.index[i], 'manager_id'] in m_dict['manager_skill']:
            df.loc[df.index[i], 'manager_skill'] = m_dict['manager_skill'][df.loc[df.index[i], 'manager_id']]
        else:
            df.loc[df.index[i], 'manager_skill'] = 0.3666338439324288 #ms_mean


    num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "num_features", 
    "len_description","created_year", "created_month", "created_day", "created_hour", "created_weekday", 
    "is_night", "is_weekend", "price_per_bed", "price_per_bath", "total_rooms", "price_per_room",
    "manager_skill", "building_popularity"]
    X = df[num_feats]
    
    return [X, b_dict, m_dict]

if __name__ == "__main__":
    
    import sys
    import json
    import numpy as np
    import pandas as pd 
    from math import sin, cos, sqrt, atan2, radians

    df = pd.read_json(sys.argv[1])
    df_test = pd.read_json(sys.argv[2])
    # build_df = pd.read_csv("data/train_new.csv", index_col='Unnamed: 0')

    # print(build_df["building_id"].head())
    # df["building_id"] = build_df["building_id"]
    # print(df["building_id"].head())

    X_dist = get_dist_features(df)
    X_test_dist = get_dist_features(df_test)
# 
    # print(X_dist.head())
    # print(X_test_dist.head()) 
# 
    X_dist.to_csv("data/X_dist.csv", index=True)
    X_test_dist.to_csv("data/X_test_dist.csv", index=True)

    # [X_count, X_test_count] = get_count_features(df, df_test)

    # print(X_count.head())
    # print(X_test_count.head()) 
 
    # X_count.to_csv("data/X_count.csv", index=True)
    # X_test_count.to_csv("data/X_test_count.csv", index=True)

    # print("\nFeature engineering on train set...\n\n\n")
    # [X, b_dict, m_dict] = feature_engineering(df, flag=0)
    # y = pd.DataFrame(df["interest_level"], index=df.index, columns=['interest_level'])

    # X = pd.concat([X, X_count], axis=1)
    # print(X.head())

    # print("\nWriting to CSV...\n")
    # X.to_csv("data/X.csv")
    # y.to_csv("data/y.csv", index=False)
    # print("Created the CSV!\n\n")

    # print("\nFeature engineering on test set...\n\n\n")
    # [X_test, b_dict, m_dict] = feature_engineering(df_test, flag=1, b_dict=b_dict, m_dict=m_dict)

    # X_test = pd.concat([X_test, X_test_count], axis=1)
    # print(X_test.head())

    # print("\nWriting to CSV...\n")
    # X_test.to_csv("data/X_test.csv")
    # print("Created the CSV!\n\n")
