import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

prefix = "processed"


def get_data_dim(dataset):
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(
    dataset,
    max_train_size=None,
    max_test_size=None,
    print_log=True,
    do_preprocess=True,
    train_start=0,
    test_start=0,
):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """returns normalized and standardized data."""

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be a 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains null values. Will be replaced with 0")
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print("Data normalized")

    return df
