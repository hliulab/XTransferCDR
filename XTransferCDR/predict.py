import os
import sys
sys.path.append(os.getcwd())
import torch
import yaml
import random
from torch.utils.data import DataLoader
from XTransferCDR.model import XTransferCDR, XTransferCDR2
from XTransferCDR.dataset.predict_data import load_dataset_splits
from XTransferCDR.utils.predict_evaluate import double_gene_evaluate
from XTransferCDR.utils.predict_evaluate import evaluate
import hashlib

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_config():
    config_path = './configs/predict_sciplex_drug_optuna.yaml'
    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        data = list(data)[0]
    return data

def load_and_predict(model_path, config, test_dataloader, save_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["dataset"] == "gears_double_gene":
        model = XTransferCDR2(config).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            test_res = double_gene_evaluate(
                model, test_dataloader, "Test", config["dataset"], save_file
            )
        return test_res
    else:
        model = XTransferCDR(config).to(device)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            test_res = evaluate(
                model, test_dataloader, "Test", config["dataset"], save_file
            )
        return test_res

if __name__ == "__main__":
    config = parse_config()
    
    set_seed(config["seed"])

    datasets = load_dataset_splits(config)
    
    test_dataloader = DataLoader(
        dataset=datasets["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    
    model_path = config['model_path']

    save_file = hashlib.md5(str(config).encode()).hexdigest()
    output = load_and_predict(model_path, config, test_dataloader, save_file)