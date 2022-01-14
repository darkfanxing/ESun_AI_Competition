import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_data_and_label_encoder(need_columns):
    chunks = pd.read_csv("src/data/tbrain_cc_training_48tags_hash_final.csv", chunksize=1000000, iterator=True)

    data = []
    for (_, chunk) in enumerate(chunks):
        chunk = chunk[need_columns]
        chunk = _data_preprocessing(chunk, need_columns)
        data.append(chunk)

    dataframe = pd.concat(data)
    
    label_encoder = LabelEncoder()
    dataframe["shop_tag"] = dataframe["shop_tag"].astype(str)
    dataframe["shop_tag"] = label_encoder.fit_transform(dataframe["shop_tag"]) 

    return dataframe, label_encoder

def _data_preprocessing(data, need_columns):
    data.dropna(inplace=True)
    data["shop_tag"] = data["shop_tag"].replace("other", 49)

    if "txn_cnt" in data.columns:
        data = data[data["txn_cnt"] > 0]

    integer_column_names = ["age", "naty", "cuorg", "masts", "educd", "trdtp", "poscd", "gender_code"]
    for interger_column_name in integer_column_names:
        if interger_column_name in need_columns:
            data[interger_column_name] = data[interger_column_name].apply(pd.to_numeric, errors="coerce")

    data.dropna(inplace=True)

    # scale amount data
    # reference: https://blog.csdn.net/bbbeoy/article/details/70170455
    data["txn_amt"] = data["txn_amt"].apply(np.log)

    return data