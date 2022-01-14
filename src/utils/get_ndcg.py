import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

def get_ndcg(dataframe, feature_names, model, label_encoder, k=3):
    total_ndcg = 0
    for user_id in tqdm(dataframe["chid"].unique(), desc=f"ndcg@{k}"):
        # chid: customer ID
        temp_dataframe = dataframe[dataframe["chid"] == user_id]
        dataframe.drop(dataframe[dataframe["chid"] == user_id].index, inplace=True)

        # txn_amt: amount of consumption
        temp_dataframe = temp_dataframe.sort_values(by=['txn_amt'], ascending=False)
        temp_dataframe = temp_dataframe.drop_duplicates(subset=['shop_tag'])
        temp_dataframe = temp_dataframe[temp_dataframe["shop_tag"] != 0]
        
        temp_dataframe_input = { feature_name: temp_dataframe[feature_name].values for feature_name in feature_names }
        
        if len(temp_dataframe_input["shop_tag"]) > 0:
            temp_dataframe_prediction = model.predict(temp_dataframe_input, batch_size=128)
        else:
            continue
        
        label = temp_dataframe["shop_tag"].values.astype(str)
        if len(label) < 3:
            continue
        
        encoded_label = label_encoder.transform(list(label))
        rearranged_label = encoded_label[np.argsort(temp_dataframe_prediction, axis=0)[::-1]]
        rearranged_label = label_encoder.inverse_transform(rearranged_label.ravel()) # Transform labels back to original encoding
        score = ndcg_score(label.reshape(1, -1).astype(float), rearranged_label.reshape(1, -1).astype(float), k=k)
        
        total_ndcg += score

    return total_ndcg / len(dataframe["chid"].unique())