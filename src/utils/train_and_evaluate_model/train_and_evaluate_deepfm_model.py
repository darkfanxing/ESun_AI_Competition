from sklearn.model_selection import train_test_split
from models import deepfm
from .get_name_and_linear_and_dnn_feature import get_name_and_linear_and_dnn_feature
from ..get_ndcg import get_ndcg

def train_and_evaluate_deepfm_model(data, sparse_features, dense_features, target_name, label_encoder):
    data, feature_names, linear_feature_columns, dnn_feature_columns = get_name_and_linear_and_dnn_feature(data, sparse_features, dense_features)

    training_data, test_data = train_test_split(data, test_size=0.2)
    training_input = { feature_name: training_data[feature_name].values for feature_name in feature_names}

    model = deepfm(linear_feature_columns, dnn_feature_columns)
    model.fit(
        training_input,
        training_data[target_name].values.ravel(),
        batch_size=256,
        epochs=15,
        verbose=2,
        validation_split=0.1
    )

    print(get_ndcg(training_data, feature_names, model, label_encoder))
    print(get_ndcg(test_data, feature_names, model, label_encoder))
