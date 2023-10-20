import numpy as np
import pickle
from matplotlib import pyplot as plt, image as im
import os
from tqdm import tqdm
from utils import *
from network import architectures


def model(training_data, testing_data):
    [x_train, y_train] = training_data
    config = get_config()
    lr = config["lr"]
    network = architectures.CNN(learning_rate=lr)
    network.train(x_train, y_train, config["batch_size"])
    # image = np.asarray([x_train[0]])
    # print(x_train.shape)
    # y_hat = network.predict(image)
    return None


def dataset():
    config = get_config()
    home_dir = config["home_dir"]
    if not (os.path.exists("data/test.pkl")) and not (os.path.exists("data/train.pkl")):
        get_data(home_dir)
    if not (os.path.exists("data/standardized_train.pkl")) and not (os.path.exists("data/standardized_test.pkl")):
        standardize(home_dir)
    x_train, y_train, x_test, y_test = load_data(home_dir)
    [x_train, y_train], [x_test, y_test] = format_data(x_train, y_train), format_data(x_test, y_test)
    return shuffle(x_train, y_train), shuffle(x_test, y_test)


def get_data(home_dir):
    dirs = {"train": os.path.join(home_dir, "train"), "test": os.path.join(home_dir, "test")}
    data = {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}}
    emotions = [folder for folder in (os.listdir(dirs["train"])) and (os.listdir(dirs["test"]))]
    for train_test in dirs.keys():
        for expression in emotions:
            directory = os.path.join(home_dir, train_test, expression)
            label = np.zeros(len(emotions))
            label[emotions.index(expression)] = 1
            for image in tqdm(os.listdir(directory), desc=f"{train_test}; {expression}"):
                item = im.imread(os.path.join(directory, image))
                data[train_test]["data"].append(item)
                data[train_test]["labels"].append(label)
        with open(os.path.join(home_dir, train_test+".pkl"), "wb") as file:
            pickle.dump(data[train_test], file)
        print("\n", f"Saved {train_test} dataset.")
        return None


def standardize(home_dir):
    split = ["train", "test"]
    for item in split:
        with open(os.path.join(home_dir, f"{item}.pkl"), "rb") as file:
            contents = pickle.load(file)
            data = np.asarray(contents["data"], dtype=np.float32)
            labels = np.asarray(contents["labels"], dtype=np.float32)
        data_std = {"data": (data - data.mean())/(data.std(axis=0)), "labels": labels}
        with open(os.path.join(home_dir, f"standardized_{item}.pkl"), "wb") as file:
            pickle.dump(data_std, file)
        print(f"Saved standardized {item} dataset.")


def load_data(home_dir):
    with open(os.path.join(home_dir, "standardized_train.pkl"), "rb") as file:
        train_data = pickle.load(file)
    x_tr, y_tr = train_data["data"], train_data["labels"]
    with open(os.path.join(home_dir, "standardized_test.pkl"), "rb") as file:
        test_data = pickle.load(file)
    x_te, y_te = test_data["data"], test_data["labels"]
    return x_tr, y_tr, x_te, y_te


def format_data(x_data, y_data):
    idxs = []
    for i, _ in enumerate(y_data.T):
        idxs.append(np.where(y_data[:, i] == 1)[0])
    min_num_images = min([len(images) for images in idxs])
    sample_idxs = []
    for j, k in enumerate(idxs):
        sample_idxs.append(np.random.choice(k, min_num_images))
    sample_idxs = np.asarray(sample_idxs).flatten()
    return np.moveaxis(
        np.asarray([x_data[sample_idxs]]), source=[1, 2, 3, 0], destination=[0, 1, 2, 3]
    ), y_data[sample_idxs]


def shuffle(x_data, y_data):
    assert len(x_data) == len(y_data), f"x and y arrays do not match. x: {len(x_data)}, y: {len(y_data)}"
    indexes = np.random.randint(0, len(x_data), len(x_data))
    return x_data[indexes], y_data[indexes]
