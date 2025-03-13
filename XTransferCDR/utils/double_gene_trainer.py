import os
import json
import time
import torch
import pickle
import hashlib
from tqdm import tqdm
import datetime
from numpy import mean

from .utils import line_chart
from .utils import line_chart_2
from .evaluate import evaluate
from .evaluate import double_gene_evaluate

from .evaluate import compute_train_r2
from .evaluate import compute_double_gene_train_r2
# from fastpearsonr import compute_pearsonr as fast_compute_pearsonr
# from torch.cuda.amp import autocast as autocast
# from torch.cuda.amp import GradScaler as GradScaler


def get_train_batch_data(batch, dataset_name="sciplex3"):
    controls_left = batch[0].cuda()
    controls_right = batch[1].cuda()
    treats_left = batch[2].cuda()
    treats_right = batch[3].cuda()
    treats_double = batch[4].cuda()
    # cross_left = batch[4].cuda()
    # cross_right = batch[5].cuda()
    de_idx_left = batch[5].cuda()
    de_idx_right = batch[6].cuda()
    de_idx_double = batch[7].cuda()
    pert_names_left = batch[8]
    pert_names_right = batch[9]
    pert_names_double = batch[10]

    res = {
        "sciplex3": (
            controls_left,
            controls_right,
            treats_left,
            treats_right,
            # cross_left,
            # cross_right,
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
            # cross_left,
            # cross_right,
            de_idx_left,
            de_idx_right,
            pert_names_left,
            pert_names_right,
        ),
         "gears_double_gene": (
            controls_left,
            controls_right,
            treats_left,
            treats_right,
            treats_double,
            de_idx_left,
            de_idx_right,
            de_idx_double,
            pert_names_left,
            pert_names_right,
            pert_names_double,
        ),
    }

    return res.get(dataset_name)


class Trainer2:
    def __init__(
        self,
        model,
        num_epoch,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        config,
        dataset_name="sciplex3",
        train_sampler=None,
        valid_sampler=None,
        is_test=False,
    ):
        self.model = model
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.dataset_name = dataset_name
        self.config = config
        self.config_hash = hashlib.md5(str(config).encode()).hexdigest()
        self.is_test = is_test

    def fit(self):
        self.model.train()

        self.lr = []
        self.loss_train = []
        self.loss_1_train = []
        self.loss_2_train = []
        self.loss_3_train = []
        self.loss_4_train = []
        self.loss_5_train = []
        self.loss_base_left_train = []
        self.loss_base_right_train = []
        self.dis_train = []
        self.orthogonal_loss = []
        self.valid_all_gene_eval_r2 = []
        self.valid_deg_gene_eval_r2 = []
        self.valid_all_gene_eval_r2_median = []
        self.valid_deg_gene_eval_r2_median = []
        self.valid_all_gene_eval_explained_variance = []
        self.valid_deg_gene_eval_explained_variance = []
        self.train_all_gene_r2 = []
        self.train_deg_gene_r2 = []

        for epoch in range(self.num_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            loss_train = []
            loss_1_train = []
            loss_2_train = []
            loss_3_train = []
            loss_4_train = []
            loss_5_train = []
            loss_base_left_train = []
            loss_base_right_train = []
            dis_train = []
            orthogonal_loss = []
            embedding_dict = {}
            train_all_gene_r2 = []
            train_deg_gene_r2 = []

            self.model.train()
            bar = tqdm(self.train_dataloader)
            # scaler = GradScaler()

            # with autocast():
            for batch in bar:
                # with torch.no_grad():
                (
                    controls_left,
                    controls_right,
                    treats_left,
                    treats_right,
                    treats_double,
                    # cross_left,
                    # cross_right,
                    de_idx_left,
                    de_idx_right,
                    de_idx_double,
                    _,
                    _,
                    _,
                ) = get_train_batch_data(batch, self.dataset_name)

                (
                    loss,
                    loss_1,
                    loss_2,
                    loss_3,
                    loss_4,
                    loss_5,
                    loss_base_left,
                    loss_base_right,
                    dis,
                    ort_loss,
                    embedding,
                ) = self.model(controls_left, controls_right, treats_left, treats_right, treats_double)

                if len(loss_5_train) > 0:
                    all_gene_r2, deg_gene_r2 = compute_double_gene_train_r2(
                        embedding, treats_left, treats_right, treats_double, de_idx_left, de_idx_right, de_idx_double
                    )
                else:
                    all_gene_r2, deg_gene_r2 = compute_train_r2(
                        embedding, treats_left, treats_right, de_idx_left, de_idx_right
                    )
                train_all_gene_r2.extend(all_gene_r2)
                train_deg_gene_r2.extend(deg_gene_r2)

                self.optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                loss.backward()
                self.optimizer.step()

                loss_train.append(loss.item())
                loss_1_train.append(loss_1.item())
                loss_2_train.append(loss_2.item())
                loss_3_train.append(loss_3.item())
                loss_4_train.append(loss_4.item())
                loss_5_train.append(loss_5.item())
                loss_base_left_train.append(loss_base_left.item())
                loss_base_right_train.append(loss_base_right.item())
                dis_train.append(dis.item())
                orthogonal_loss.append(ort_loss.item())

                desc = f"Epoch: {epoch+1}/{self.num_epoch} train loss: {mean(loss_train):.6f}"
                desc += f" dis: {mean(dis_train):.6f}"
                desc += f" ort_loss: {mean(orthogonal_loss):.6f}"
                desc += f" lr: {self.scheduler.get_last_lr()[0]:.6f}"
                bar.set_description(desc)

                if "pert_left" not in embedding_dict.keys():
                    embedding_dict["pert_left"] = (
                        embedding[0].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["pert_right"] = (
                        embedding[1].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["base_left"] = (
                        embedding[2].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["base_right"] = (
                        embedding[3].cpu().detach().numpy().tolist()
                    )
                elif len(embedding_dict["pert_left"]) < 50000:
                    embedding_dict["pert_left"].extend(
                        embedding[0].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["pert_right"].extend(
                        embedding[1].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["base_left"].extend(
                        embedding[2].cpu().detach().numpy().tolist()
                    )
                    embedding_dict["base_right"].extend(
                        embedding[3].cpu().detach().numpy().tolist()
                    )

            self.lr.append(self.scheduler.get_last_lr()[0])
            self.scheduler.step()
            self.loss_train.append(mean(loss_train))
            self.loss_1_train.append(mean(loss_1_train))
            self.loss_2_train.append(mean(loss_2_train))
            self.loss_3_train.append(mean(loss_3_train))
            self.loss_4_train.append(mean(loss_4_train))
            self.loss_5_train.append(mean(loss_5_train))
            self.loss_base_left_train.append(mean(loss_base_left_train))
            self.loss_base_right_train.append(mean(loss_base_right_train))
            self.dis_train.append(mean(dis_train))
            self.orthogonal_loss.append(mean(orthogonal_loss))
            self.train_embedding_dict = embedding_dict

            self.train_all_gene_r2.append(mean(train_all_gene_r2))
            self.train_deg_gene_r2.append(mean(train_deg_gene_r2))
            print(
                f"train_all_gene_r2: {mean(train_all_gene_r2)}, train_deg_gene_r2: {mean(train_deg_gene_r2)}"
            )

            self.model.eval()
            with torch.no_grad():
                if self.valid_sampler is not None:
                    self.valid_sampler.set_epoch(epoch)
                if(self.dataset_name == "gears_double_gene"):
                    valid_res = double_gene_evaluate(
                        self.model, self.valid_dataloader, "Valid", self.dataset_name
                    )
                else:
                    valid_res = evaluate(
                        self.model, self.valid_dataloader, "Valid", self.dataset_name
                    )

            self.valid_res = valid_res
            self.valid_all_gene_eval_r2.append(valid_res["all_gene_eval_r2_mean"])
            self.valid_deg_gene_eval_r2.append(valid_res["deg_gene_eval_r2_mean"])
            self.valid_all_gene_eval_r2_median.append(valid_res["all_gene_eval_r2_median"])
            self.valid_deg_gene_eval_r2_median.append(valid_res["deg_gene_eval_r2_median"])
            self.valid_all_gene_eval_explained_variance.append(
                valid_res["all_gene_eval_explained_variance"]
            )
            self.valid_deg_gene_eval_explained_variance.append(
                valid_res["deg_gene_eval_explained_variance"]
            )

        if self.is_test:
            self.model.eval()
            with torch.no_grad():

                if self.dataset_name == "gears_double_gene":
                    test_res = double_gene_evaluate(
                        self.model, self.test_dataloader, "Test", self.dataset_name
                    )
                else:
                    test_res = evaluate(
                        self.model, self.test_dataloader, "Test", self.dataset_name
                    )

            self.test_res = test_res

        self.save_data(str(self.model.device))

    def save_data(self, device):
        module_path = f'./results/modules/{self.dataset_name}/{self.config_hash}'
        if os.path.exists(module_path) is False and device == "cuda:0":
            os.mkdir(module_path)

        while os.path.exists(module_path) is False:
            time.sleep(1)

        module_name = (
            f'{module_path}/XTransferCDR_' + str(self.dataset_name) + '_state_dict.pkl'
        )
        if device == "cuda:0":
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), module_name)
            else:
                torch.save(self.model.state_dict(), module_name)

        data = {
            'loss_train': self.loss_train,
            'loss_1_train': self.loss_1_train,
            'loss_2_train': self.loss_2_train,
            'loss_3_train': self.loss_3_train,
            'loss_4_train': self.loss_4_train,
            'loss_5_train': self.loss_5_train,
            'loss_base_left_train': self.loss_base_left_train,
            'loss_base_right_train': self.loss_base_right_train,
            'dis_train': self.dis_train,
            'orthogonal_loss': self.orthogonal_loss,
            'valid_res': self.valid_res,
            'valid_all_gene_eval_r2': self.valid_all_gene_eval_r2,
            'valid_all_gene_eval_r2_median': self.valid_all_gene_eval_r2_median,
            'valid_deg_gene_eval_r2': self.valid_deg_gene_eval_r2,
            'valid_deg_gene_eval_r2_median': self.valid_deg_gene_eval_r2_median,
            'valid_all_gene_eval_explained_variance': self.valid_all_gene_eval_explained_variance,
            'valid_deg_gene_eval_explained_variance': self.valid_deg_gene_eval_explained_variance,
            'lr': self.lr,
            "train_embedding_dict": self.train_embedding_dict,
        }

        if self.is_test:
            data["test_res"] = self.test_res

        plot_data_path = f'./results/plot_data/{self.dataset_name}/{self.config_hash}'
        if os.path.exists(plot_data_path) is False and device == "cuda:0":
            os.mkdir(plot_data_path)

        while os.path.exists(plot_data_path) is False:
            time.sleep(1)

        with open(
            f'{plot_data_path}/XTransferCDR_{str(self.dataset_name)}_{str(device)}.pkl',
            'wb',
        ) as f:
            pickle.dump(data, f)

        if device == "cuda:0":
            with open(
                f'./results/plot_data/{self.dataset_name}/{self.config_hash}/config.json',
                'w',
            ) as f:
                json.dump(self.config, f)

    def plot(self, device):
        if device != "cuda:0":
            return
        if (
            os.path.exists(f'./results/plots/{self.dataset_name}/{self.config_hash}')
            is False
        ):
            os.mkdir(f'./results/plots/{self.dataset_name}/{self.config_hash}')
        print(datetime.datetime.now())
        line_chart(
            x=self.num_epoch,
            y=self.loss_train,
            label=f' loss = {min(self.loss_train):.4f}',
            xlabel='epoch',
            ylabel='loss',
            title='generator loss',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.lr,
            label=f' lr = {max(self.lr):.4f}',
            xlabel='epoch',
            ylabel='lr',
            title='lr',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/lr.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_1_train,
            label=f' loss_1 = {min(self.loss_1_train):.4f}',
            xlabel='epoch',
            ylabel='loss_1',
            title='loss_1',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_1.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_2_train,
            label=f' loss_2 = {min(self.loss_2_train):.4f}',
            xlabel='epoch',
            ylabel='loss_2',
            title='loss_2',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_2.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_3_train,
            label=f' loss_3 = {min(self.loss_3_train):.4f}',
            xlabel='epoch',
            ylabel='loss_3',
            title='loss_3',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_3.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_4_train,
            label=f' loss_4 = {min(self.loss_4_train):.4f}',
            xlabel='epoch',
            ylabel='loss_4',
            title='loss_4',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_4.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_4_train,
            label=f' loss_5 = {min(self.loss_5_train):.4f}',
            xlabel='epoch',
            ylabel='loss_5',
            title='loss_5',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_5.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_base_left_train,
            label=f' loss_base_left_train = {min(self.loss_base_left_train):.4f}',
            xlabel='epoch',
            ylabel='loss_base_left_train',
            title='loss_base_left_train',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_base_left_train.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.loss_base_right_train,
            label=f' loss_4 = {min(self.loss_base_right_train):.4f}',
            xlabel='epoch',
            ylabel='loss_base_right_train',
            title='loss_base_right_train',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/loss_base_right_train.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.dis_train,
            label=f' dis = {min(self.dis_train):.4f}',
            xlabel='epoch',
            ylabel='dis',
            title='dis',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/cos_dis.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.orthogonal_loss,
            label=f' orthogonal_loss = {min(self.orthogonal_loss):.4f}',
            xlabel='epoch',
            ylabel='orthogonal_loss',
            title='orthogonal_loss',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/orthogonal_loss.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.valid_all_gene_eval_explained_variance,
            label=f' explained_variance = {max(self.valid_all_gene_eval_explained_variance):.4f}',
            xlabel='epoch',
            ylabel='explained_variance',
            title='valid_all_gene_eval_explained_variance',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/valid_all_gene_eval_explained_variance.png',
        )

        line_chart(
            x=self.num_epoch,
            y=self.valid_deg_gene_eval_explained_variance,
            label=f' explained_variance = {max(self.valid_deg_gene_eval_explained_variance):.4f}',
            xlabel='epoch',
            ylabel='explained_variance',
            title='valid_deg_gene_eval_explained_variance',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/valid_deg_gene_eval_explained_variance.png',
        )

        line_chart_2(
            x=self.num_epoch,
            y1=self.train_all_gene_r2,
            y2=self.valid_all_gene_eval_r2,
            label1=f' train_r2 = {max(self.train_all_gene_r2):.4f}',
            label2=f' valid_r2 = {max(self.valid_all_gene_eval_r2):.4f}',
            xlabel='epoch',
            ylabel='r2',
            title='train_valid_all_gene_r2',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/train_valid_all_gene_r2.png',
        )

        line_chart_2(
            x=self.num_epoch,
            y1=self.train_deg_gene_r2,
            y2=self.valid_deg_gene_eval_r2,
            label1=f' train_r2 = {max(self.train_deg_gene_r2):.4f}',
            label2=f' valid_r2 = {max(self.valid_deg_gene_eval_r2):.4f}',
            xlabel='epoch',
            ylabel='r2',
            title='train_valid_deg_gene_r2',
            filename=f'./results/plots/{self.dataset_name}/{self.config_hash}/train_valid_deg_gene_r2.png',
        )
