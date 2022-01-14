from utils import get_data_and_label_encoder, train_and_evaluate_deepfm_model, check_and_use_gpus

check_and_use_gpus(memory_limit=8192)

sparse_features = ["shop_tag", "masts"]
dense_features = ["dt"]
target_name = ["txn_amt"]
data, label_encoder = get_data_and_label_encoder(sparse_features + dense_features + target_name + ["chid"])
train_and_evaluate_deepfm_model(data, sparse_features, dense_features, target_name, label_encoder)