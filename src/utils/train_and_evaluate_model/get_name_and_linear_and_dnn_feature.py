# reference: https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr

from sklearn.preprocessing import MinMaxScaler
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

def get_name_and_linear_and_dnn_feature(data, sparse_features, dense_features):
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
            for i,feat in enumerate(sparse_features)
    ] + [
        DenseFeat(feat, 1, ) for feat in dense_features
    ]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    
    return data, feature_names, linear_feature_columns, dnn_feature_columns