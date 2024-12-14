import os
import sys
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.getcwd())
from XTransferCDR.utils import Trainer
from XTransferCDR.utils import double_gene_trainer
from XTransferCDR.model import XTransferCDR
from XTransferCDR.model import XTransferCDR2
from XTransferCDR.utils import get_num_workers
from XTransferCDR.dataset import load_dataset_splits
torch.set_float32_matmul_precision('high')
import yaml
import random
# os.environ["OMP_NUM_THREADS"] = "1"
def parse_config():
    config_path = './configs/predict_sciplex_drug_optuna.yaml'
    with open(config_path, 'rb') as f:
        data = yaml.safe_load_all(f)
        data = list(data)[0]
    return data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    config = parse_config()
    num_workers = get_num_workers()
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0")

    set_seed(config["seed"])
    datasets = load_dataset_splits(config)

    if(config["dataset"]=="gears_double_gene"):
        model = XTransferCDR2(config).cuda()
    else:
        model = XTransferCDR(config).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=float(config["weight_decay"]))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs! device: ", local_rank)
        train_sampler = DistributedSampler(datasets["train"], shuffle=True)
        valid_sampler = DistributedSampler(datasets["valid"], shuffle=True)
        test_sampler = DistributedSampler(datasets["test"], shuffle=False)
        train_dataloader = DataLoader(
            dataset=datasets["train"],
            batch_size=config["batch_size"],
            shuffle=False,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=num_workers,
        )
        valid_dataloader = DataLoader(
            dataset=datasets["valid"],
            batch_size=config["batch_size"],
            shuffle=False,
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_dataloader = DataLoader(
            dataset=datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            sampler=test_sampler,
            pin_memory=True,
            num_workers=num_workers,
        )

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
        if(config["dataset"]=="gears_double_gene"):
            trainer = double_gene_trainer(
                model,
                config["num_epoch"],
                optimizer,
                scheduler,
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                config,
                config["dataset"],
                train_sampler,
                valid_sampler,
                is_test=True,
            )
        else:
            trainer = Trainer(
                model,
                config["num_epoch"],
                optimizer,
                scheduler,
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                config,
                config["dataset"],
                train_sampler,
                valid_sampler,
                is_test=True,
            )
    else:
        train_dataloader = DataLoader(
            dataset=datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        valid_dataloader = DataLoader(
            dataset=datasets["valid"],
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_dataloader = DataLoader(
            dataset=datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        if(config["dataset"]=="gears_double_gene"):
            trainer = double_gene_trainer(
                model,
                config["num_epoch"],
                optimizer,
                scheduler,
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                config,
                config["dataset"],
            )
        else:
            trainer = Trainer(
                model,
                config["num_epoch"],
                optimizer,
                scheduler,
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                config,
                config["dataset"],
            )

    trainer.fit()
    trainer.plot(str(model.device))
    print(config)