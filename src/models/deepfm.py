from deepctr.models import DeepFM

def deepfm(linear_feature_columns, dnn_feature_columns):
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task="regression")
    model.compile("adam", "mse", metrics=["mse"])

    return model