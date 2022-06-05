import pandas as pd 
# new branch
import pickle
pd.set_option('display.max_columns',None)
from sasrec.util import filter_k_core
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,use_memory_fs=False)
import os
  

def make_maps(df):  
    # user, item label encoding dictionary 생성
    user_set, item_set = set(df['userID'].unique()), set(df['itemID'].unique())
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1

    return (user_map, item_map)   


def make_sasrec_df(df, maps):
    # assert maps
    user_map = maps[0]
    item_map = maps[1]

    # labelEncoding
    df.rename({'datetime':'time'},inplace=True,axis=1)
    df["userID"] = df["userID"].apply(lambda x: user_map[x])
    df["itemID"] = df["itemID"].apply(lambda x: item_map[x])
    
    # sort by time
    df = df.sort_values(by=["userID", "time"])
    final_df = df[['userID','itemID']]

    return final_df


def make_sasrecData(df, filter_k=10):

    df_temp = df.copy() # copy data

    # simple preprocessing
    df_temp['datetime_shift'] = df_temp['datetime'].shift()
    df_temp['고객번호_shift'] = df_temp['고객번호'].shift()
    df_temp = df_temp.loc[~((df_temp['datetime']==df_temp['datetime_shift'])&(df_temp['고객번호']==df_temp['고객번호_shift']))]

    # make userID and itemID
    df_temp['itemID'] = df_temp[['new_cat','상품중분류명','상품소분류명','브랜드명']].parallel_apply(lambda row: row['new_cat']+'_'+row['상품중분류명']+'_'+row['상품소분류명']+'_'+row['브랜드명'],axis=1)
    df_temp.rename({'고객번호':'userID'},axis=1,inplace=True)

    # filter by k
    filtered_df = filter_k_core(df_temp,filter_k) #userID, itemID가 모두 filter_k회 이상 등장하도록 filtering

    # make maps
    maps = make_maps(filtered_df)

    # make sasrec df
    sasrec_df = make_sasrec_df(filtered_df, maps)

    return sasrec_df, maps


def save_data(sasrec_df, maps, save_path, exp_name):
    
    # make dir
    if not os.path.exists(f'{save_path}/{exp_name}'):
        os.mkdir(f'{save_path}/{exp_name}')
    
    # save sasrec data    
    sasrec_df.to_csv(f'{save_path}/{exp_name}/SASRec_dataset_{exp_name}.txt', sep="\t", header=False, index=False)

    # save maps
    with open(f'{save_path}/{exp_name}/SASRec_user_item_map_{exp_name}.pkl','wb') as f:
        pickle.dump(maps, f)
    
    # save item_map.txt
    with open(f'{save_path}/{exp_name}/SASRec_item_map_{exp_name}.txt','w') as f:
        f.write(f'{maps[1]}')  

    return