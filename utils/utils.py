import yaml
import torch
import random
import numpy as np
from optuna import Trial
from optuna.integration import TorchDistributedTrial
from matplotlib import pyplot as plt
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlInit


def read_config(config_path):
    if config_path is None:
        config_path = './configs/configs/train_sciplex_drug_optuna_DP.yaml'
    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        data = list(data)[0]

    return data


def parse_config(trial: Trial | TorchDistributedTrial, config_path):
    config = read_config(config_path)

    res = {}

    for key in config.keys():
        if isinstance(config[key], dict):
            if config[key]["type"] == "int":
                res[key] = trial.suggest_int(key, config[key]["min"], config[key]["max"])
            elif config[key]["type"] == "float":
                res[key] = trial.suggest_float(
                    key, config[key]["min"], config[key]["max"]
                )
            elif config[key]["type"] == "choices":
                res[key] = trial.suggest_categorical(key, config[key]["options"])
        else:
            res[key] = config[key]

    return res


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


def get_num_workers():
    num_workers = 6
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    if "4090" in nvmlDeviceGetName(handle):
        num_workers = 8
    return num_workers


def get_train_batch_data(batch, dataset_name="sciplex3"):
    controls_left = batch[0].cuda()
    controls_right = batch[1].cuda()
    treats_left = batch[2].cuda()
    treats_right = batch[3].cuda()
    de_idx_left = batch[4].cuda()
    de_idx_right = batch[5].cuda()
    pert_names_left = batch[6]
    pert_names_right = batch[7]

    res = {
        "sciplex3": (
            controls_left,
            controls_right,
            treats_left,
            treats_right,
            de_idx_left,
            de_idx_right,
            pert_names_left,
            pert_names_right,
        ),
        "gears": (
            controls_left,
            controls_right,
            treats_left,
            treats_right,
            de_idx_left,
            de_idx_right,
            pert_names_left,
            pert_names_right,
        ),
        "gene_pert": (
            controls_left,
            controls_right,
            treats_left,
            treats_right,
            de_idx_left,
            de_idx_right,
            pert_names_left,
            pert_names_right,
        ),
    }

    result = res.get(dataset_name)

    if result is None:
        raise ValueError(dataset_name + " not exist!")

    return result


def line_chart(x, y, label, xlabel, ylabel, title: str, filename: str):
    x = np.arange(x)
    plt.plot(
        x,
        y,
        'red',
        label=label,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(filename)
    plt.cla()
    plt.clf()

def line_chart_2(x, y1, y2, label1, label2, xlabel, ylabel, title: str, filename: str):
    x = np.arange(x)
    plt.plot(
        x,
        y1,
        'red',
        label=label1,
    )
    plt.plot(
        x,
        y2,
        'blue',
        label=label2,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(filename)
    plt.cla()
    plt.clf()
