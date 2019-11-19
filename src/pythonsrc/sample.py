import json
import pandas as pd
import numpy as np

df = pd.read_json('test.json')

df_new = pd.DataFrame([df['listing_id']], index=df.index, columns=['listing_id'])
df_new['listing_id'] = df['listing_id']
leng = len(df_new['listing_id'])

for i in df_new.index:
   [[df_new.loc[i, 'high'], df_new.loc[i, 'low'], df_new.loc[i, 'medium']]] = np.random.dirichlet(np.ones(3), 1) 
   
df_new.to_csv('sample_sub.csv')