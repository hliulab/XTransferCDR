import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import mean
from typing import Dict
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import r2_score, explained_variance_score
from numpy import median
import scipy.stats as stats

sys.path.append(os.getcwd())
from XTransferCDR.model import XTransferCDR, XTransferCDR2

def get_test_batch_data(batch, dataset_name="sciplex3"):
    if dataset_name == "gears_double_gene":
        treats_left = batch[0].cuda()
        treats_right = batch[1].cuda()
        treats_double = batch[2].cuda()
        controls_left = batch[3].cuda()
        controls_right = batch[4].cuda()
        de_idx_left = batch[5].cuda()
        de_idx_right = batch[6].cuda()
        de_idx_double = batch[7].cuda()
        pert_names_left = batch[8]
        pert_names_right = batch[9]
        pert_names_double = batch[10]
        return (
                treats_left,
                treats_right,
                treats_double,
                de_idx_left,
                de_idx_right,
                de_idx_double,
                pert_names_left,
                pert_names_right,
                pert_names_double
            )
    else:
        treats_cell_a_with_pert_1 = batch[0].cuda()
        treats_cell_b_with_pert_1 = batch[1].cuda()
        control_cell_b = batch[2].cuda()

        if dataset_name in ["sciplex3", "gears", "gene_pert"]:
            de_idx_cell_a_pert_1 = batch[3].cuda()
            de_idx_cell_b_pert_1 = batch[4].cuda()
            cov_drug_a_1 = batch[5]

            # print(cov_drug_a_1)

            cov_drug_b_1 = batch[6]
            return (
                treats_cell_a_with_pert_1,
                treats_cell_b_with_pert_1,
                control_cell_b,
                de_idx_cell_a_pert_1,
                de_idx_cell_b_pert_1,
                cov_drug_a_1,
                cov_drug_b_1
            )
        else:
            raise ValueError(dataset_name + " not exist!")


def compute_r2(y_true: Tensor, y_pred: Tensor):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    t = y_true.cpu().numpy().tolist()
    p = y_pred.cpu().numpy().tolist()
    res = r2_score(t, p)
    return res


def compute_explained_variance_score(y_true: Tensor, y_pred: Tensor):
    y_pred = torch.clamp(y_pred, -3e12, 3e12)
    t = y_true.cpu().numpy().tolist()
    p = y_pred.cpu().numpy().tolist()
    res = explained_variance_score(t, p)
    return res


def compute_pearsonr(y_true: Tensor, y_pred: Tensor):
    p = np.corrcoef(y_true.cpu().numpy(), y_pred.cpu().numpy())

    if math.isnan(p[0, 1]):
        return 0.0
    return p[0, 1]

def compute_spearmanr(y_true: Tensor, y_pred: Tensor):
    correlation, _= stats.spearmanr(y_true.cpu().numpy(), y_pred.cpu().numpy())

    if np.isnan(correlation):
        return 0.0
    return correlation


def compute(true_input: Dict, pred_input: Dict, de_gene_idxs_dict: Dict):
    treats_explained_variance_dict = {}
    treats_explained_variance_de_dict = {}
    treats_r2_dict = {}
    treats_r2_de_dict = {}
    treats_pearsonr_dict = {}
    treats_pearsonr_de_dict = {}
    treats_spearman_dict = {}
    treats_spearman_de_dict = {}

    for cov_pert in true_input.keys():
        temp_true_treats = (
            torch.stack(true_input[cov_pert]).mean(0).detach().cuda().squeeze(0)
        )
        temp_pred_treats = (
            torch.stack(pred_input[cov_pert]).mean(0).detach().cuda().squeeze(0)
        )
        treats_explained_variance = compute_explained_variance_score(
            temp_true_treats, temp_pred_treats
        )
        treats_explained_variance_dict[cov_pert] = treats_explained_variance

        treats_r2 = compute_r2(temp_true_treats, temp_pred_treats)
        treats_r2_dict[cov_pert] = treats_r2

        treats_pearsonr = compute_pearsonr(temp_true_treats, temp_pred_treats)
        treats_pearsonr_dict[cov_pert] = treats_pearsonr

        treats_spearman = compute_spearmanr(temp_true_treats, temp_pred_treats)
        treats_spearman_dict[cov_pert] = treats_spearman

        if len(de_gene_idxs_dict.keys()) > 0 and -1 not in de_gene_idxs_dict[cov_pert]:
            # print(temp_true_treats[de_gene_idxs_dict[pert]])
            # print(temp_pred_treats[de_gene_idxs_dict[pert]])
            # # print(len(temp_true_treats[de_gene_idxs_dict[pert]]))
            # exit()
            treats_explained_variance_de = compute_explained_variance_score(
                temp_true_treats[de_gene_idxs_dict[cov_pert]],
                temp_pred_treats[de_gene_idxs_dict[cov_pert]],
            )
            treats_explained_variance_de_dict[cov_pert] = treats_explained_variance_de
            treats_r2_de = compute_r2(
                temp_true_treats[de_gene_idxs_dict[cov_pert]],
                temp_pred_treats[de_gene_idxs_dict[cov_pert]],
            )
            treats_r2_de_dict[cov_pert] = treats_r2_de

            treats_pearsonr_de = compute_pearsonr(
                temp_true_treats[de_gene_idxs_dict[cov_pert]],
                temp_pred_treats[de_gene_idxs_dict[cov_pert]],
            )
            treats_pearsonr_de_dict[cov_pert] = treats_pearsonr_de

            treats_spearman_de = compute_spearmanr(
                temp_true_treats[de_gene_idxs_dict[cov_pert]],
                temp_pred_treats[de_gene_idxs_dict[cov_pert]],
            )
            treats_spearman_de_dict[cov_pert] = treats_spearman_de

    return (
        treats_explained_variance_dict,
        treats_explained_variance_de_dict,
        treats_r2_dict,
        treats_r2_de_dict,
        treats_pearsonr_dict,
        treats_pearsonr_de_dict,
        treats_spearman_dict,
        treats_spearman_de_dict,
    )


def handle(
    true_treat,
    pred_treat,
    de_gene_idxs_dict,
    cov_drug,
    de_idx_pert,
    treats_cell_true,
    treats_cell_pred,
):
    for cell_pert in set(cov_drug):
        idx = [i for i, x in enumerate(cov_drug) if x == cell_pert]

        if cell_pert not in pred_treat.keys():
            true_treat[cell_pert] = []
            pred_treat[cell_pert] = []

        for id in idx:
            true_treat[cell_pert].append(treats_cell_true[id].cpu())
            pred_treat[cell_pert].append(treats_cell_pred[id].cpu())

        if de_idx_pert is not None:
            de_gene_idxs_dict[cell_pert] = de_idx_pert[idx[0]]

    return true_treat, pred_treat, de_gene_idxs_dict


def evaluate(
    model: XTransferCDR,
    dataloader,
    method="Valid",
    dataset_name="sciplex3",
    save_file="temp"
):
    true_b_1 = {}
    pred_b_1 = {}

    de_gene_idxs_dict_cell_a_pert_1 = {}
    de_gene_idxs_dict_cell_b_pert_1 = {}

    embedding_dict = {}

    bar = tqdm(dataloader)
    bar.set_description(f"{method}:")
    predict_result = None
    deg_predict_result = None
    cov_drug_b_1_all = None
    for batch in bar:
        (
            treats_cell_a_with_pert_1,
            treats_cell_b_with_pert_1,
            control_cell_b,
            de_idx_cell_a_pert_1,
            de_idx_cell_b_pert_1,
            cov_drug_a_1,
            cov_drug_b_1,
        ) = get_test_batch_data(batch, dataset_name)

        if "predict" in dir(model):
            (
                treats_cell_b_with_pert_1_cross,
                embedding,
            ) = model.predict(
                treats_cell_a_with_pert_1,
                control_cell_b,
            ) # type: ignore
        else:
            (
                treats_cell_b_with_pert_1_cross,
                embedding,
            ) = model.module.predict( # type: ignore
                treats_cell_a_with_pert_1,
                control_cell_b,
            )

        if "pert_1_for_cell_a" not in embedding_dict.keys():
            embedding_dict["pert_1_for_cell_a"] = embedding[0].cpu().numpy().tolist()
            embedding_dict["base_control_cell_b"] = embedding[1].cpu().numpy().tolist()
        else:
            embedding_dict["pert_1_for_cell_a"].extend(embedding[0].cpu().numpy().tolist())
            embedding_dict["base_control_cell_b"].extend(embedding[1].cpu().numpy().tolist())

        true_b_1, pred_b_1, de_gene_idxs_dict_cell_b_pert_1 = handle(
            true_b_1,
            pred_b_1,
            de_gene_idxs_dict_cell_b_pert_1,
            cov_drug_b_1,
            de_idx_cell_b_pert_1,
            treats_cell_b_with_pert_1,
            treats_cell_b_with_pert_1_cross,
        )
        
        if method=="Test":
            print(len(treats_cell_b_with_pert_1_cross))
            print(len(treats_cell_b_with_pert_1))
            print(len(cov_drug_b_1))
            if cov_drug_b_1_all is None:
                cov_drug_b_1_all = cov_drug_b_1
            else:
                cov_drug_b_1_all = np.concatenate((cov_drug_b_1_all, cov_drug_b_1))

            if predict_result is None:
                predict_result = treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()
            else:
                predict_result = np.concatenate((predict_result, treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()), axis=0)

        if deg_predict_result is None:
            deg_predict_result = treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()
        else:
            deg_predict_result = np.concatenate((deg_predict_result, treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()), axis=0)

    (
        treats_explained_variance_dict_b_1,
        treats_explained_variance_de_dict_b_1,
        treats_r2_dict_b_1,
        treats_r2_de_dict_b_1,
        treats_pearsonr_dict_b_1,
        treats_pearsonr_de_dict_b_1,
        treats_spearman_dict_b_1,
        treats_spearman_de_dict_b_1,
    ) = compute(true_b_1, pred_b_1, de_gene_idxs_dict_cell_b_pert_1)

    if method=="Test":
        df = pd.DataFrame({
        'treats_explained_variance': treats_explained_variance_dict_b_1,
        'treats_explained_variance_de': treats_explained_variance_de_dict_b_1,
        'treats_r2': treats_r2_dict_b_1,
        'treats_r2_de': treats_r2_de_dict_b_1,
        'treats_pearsonr': treats_pearsonr_dict_b_1,
        'treats_pearsonr_de': treats_pearsonr_de_dict_b_1,
        'treats_spearman': treats_spearman_dict_b_1,
        'treats_spearman_de': treats_spearman_de_dict_b_1,
        })

        
        predict_result_path = f"./results/predict_data/{dataset_name}/{save_file}"
        if os.path.exists(predict_result_path) is False:
            os.mkdir(predict_result_path)

        df.to_csv(f"{predict_result_path}/predict_score.csv", index=True)

    temp_r2 = list(treats_r2_dict_b_1.values())
    all_gene_eval_r2 = mean(temp_r2)
    print(f"{method} all gene r2 mean:", all_gene_eval_r2)
    all_gene_eval_r2_median = median(temp_r2)
    print(f"{method} all gene r2 median:", all_gene_eval_r2_median)

    temp_r2_de = list(treats_r2_de_dict_b_1.values())
    deg_gene_eval_r2 = mean(temp_r2_de)
    print(f"{method} degs gene r2 mean:", deg_gene_eval_r2)
    deg_gene_eval_r2_median = median(temp_r2_de)
    print(f"{method} degs gene r2 median:", deg_gene_eval_r2_median)


    temp_pearsonr = list(treats_pearsonr_dict_b_1.values())
    all_gene_eval_pearsonr = mean(temp_pearsonr)
    print(f"{method} all gene pearsonr:", all_gene_eval_pearsonr)

    temp_pearsonr_de = list(treats_pearsonr_de_dict_b_1.values())
    deg_gene_eval_pearsonr = mean(temp_pearsonr_de)
    print(f"{method} degs gene pearsonr:", deg_gene_eval_pearsonr)

    temp_spearman = list(treats_spearman_dict_b_1.values())
    all_gene_eval_spearman = mean(temp_spearman)
    print(f"{method} all gene spearman:", all_gene_eval_spearman)

    temp_spearman_de = list(treats_spearman_de_dict_b_1.values())
    deg_gene_eval_spearman = mean(temp_spearman_de)
    print(f"{method} degs gene spearman:", deg_gene_eval_spearman)


    temp_explained_variance = list(treats_explained_variance_dict_b_1.values())
    all_gene_eval_explained_variance = mean(temp_explained_variance)
    print(f"{method} all gene explained_variance:", all_gene_eval_explained_variance)

    temp_explained_variance_de = list(treats_explained_variance_de_dict_b_1.values())
    deg_gene_eval_explained_variance = mean(temp_explained_variance_de)
    print(f"{method} degs gene explained_variance:", deg_gene_eval_explained_variance)

    res = {
        "treats_r2_dict_b_1": treats_r2_dict_b_1,
        "treats_r2_de_dict_b_1": treats_r2_de_dict_b_1,
        "treats_pearsonr_dict_b_1": treats_pearsonr_dict_b_1,
        "treats_pearsonr_de_dict_b_1": treats_pearsonr_de_dict_b_1,
        "treats_spearman_dict_b_1": treats_spearman_dict_b_1,
        "treats_spearman_de_dict_b_1": treats_spearman_de_dict_b_1,
        "treats_explained_variance_dict_b_1": treats_explained_variance_dict_b_1,
        "treats_explained_variance_de_dict_b_1": treats_explained_variance_de_dict_b_1,
        "all_gene_eval_r2_mean": all_gene_eval_r2,
        "all_gene_eval_r2_median": all_gene_eval_r2_median,
        "deg_gene_eval_r2_mean": deg_gene_eval_r2,
        "deg_gene_eval_r2_median": deg_gene_eval_r2_median,

        "all_gene_eval_explained_variance": all_gene_eval_explained_variance,
        "deg_gene_eval_explained_variance": deg_gene_eval_explained_variance,
        "embedding_dict": embedding_dict,
    }
    
    if method=="Test":
        print(len(predict_result))
        
        
        predict_result_path = f"./results/predict_data/{dataset_name}/{save_file}"
        if os.path.exists(predict_result_path) is False:
            os.mkdir(predict_result_path)

        
        predict_result_df = pd.DataFrame(predict_result)
        def add_underscore(s):
            return s.replace('BRD-', '_BRD-')
        predict_result_df["condition_name"] = np.vectorize(add_underscore)(cov_drug_b_1_all)
        predict_result_df.to_csv(f"{predict_result_path}/predict_expression.csv", index=True)


    return res


def compute_train_r2(
    embedding_dict,
    true_treats_left,
    true_treats_right,
    de_idx_left,
    de_idx_right,
):
    all_gene_r2 = []
    deg_gene_r2 = []

    de_idx_left = de_idx_left.cpu()
    de_idx_right = de_idx_right.cpu()

    pred_treats_left: torch.Tensor = embedding_dict[4]
    pred_treats_right: torch.Tensor = embedding_dict[5]

    for i in range(true_treats_left.shape[0]):
        all_gene_r2.append(
            compute_r2(true_treats_left[i].detach(), pred_treats_left[i].detach())
        )
        all_gene_r2.append(
            compute_r2(true_treats_right[i].detach(), pred_treats_right[i].detach())
        )
        if len(de_idx_left) > 0:
            deg_gene_r2.append(
                compute_r2(
                    true_treats_left[i][de_idx_left[i]].detach(),
                    pred_treats_left[i][de_idx_left[i]].detach(),
                )
            )
            deg_gene_r2.append(
                compute_r2(
                    true_treats_right[i][de_idx_right[i]].detach(),
                    pred_treats_right[i][de_idx_right[i]].detach(),
                )
            )

    return all_gene_r2, deg_gene_r2


def double_gene_evaluate(
    model: XTransferCDR2,
    dataloader,
    method="Valid",
    dataset_name="sciplex3",
    save_file="temp"
):
    true_left = {}
    pred_left = {}

    true_right = {}
    pred_right = {}

    true_double = {}
    pred_double = {}

    de_gene_idxs_dict_cell_left = {}
    de_gene_idxs_dict_cell_right = {}
    de_gene_idxs_dict_cell_double = {}

    embedding_dict = {}

    bar = tqdm(dataloader)
    bar.set_description(f"{method}:")
    predict_result = None
    deg_predict_result = None
    cov_drug_right_all = None
    for batch in bar:
        (
            treats_left,
            treats_right,
            treats_double,
            de_idx_left,
            de_idx_right,
            de_idx_double,
            cov_drug_left,
            cov_drug_right,
            cov_drug_double,
        ) = get_test_batch_data(batch, dataset_name)

        if "predict" in dir(model):
            (
                treats_cell_b_with_pert_1_cross,
                embedding,
            ) = model.double_gene_predict(
                treats_left,
                treats_right
            ) # type: ignore
        else:
            (
                treats_cell_b_with_pert_1_cross,
                embedding,
            ) = model.module.double_gene_predict( # type: ignore
                treats_left,
                treats_right,
            )

        if "pert_left" not in embedding_dict.keys():
            embedding_dict["pert_left"] = embedding[0].cpu().numpy().tolist()
            embedding_dict["base_left"] = embedding[1].cpu().numpy().tolist()
            embedding_dict["pert_right"] = embedding[2].cpu().numpy().tolist()
        else:
            embedding_dict["pert_left"].extend(embedding[0].cpu().numpy().tolist())
            embedding_dict["base_left"].extend(embedding[1].cpu().numpy().tolist())
            embedding_dict["pert_right"].extend(embedding[2].cpu().numpy().tolist())

        true_double, pred_double, de_gene_idxs_dict_cell_double = handle(
            true_double,
            pred_double,
            de_gene_idxs_dict_cell_double,
            cov_drug_double,
            de_idx_double,
            treats_double,
            treats_cell_b_with_pert_1_cross,
        )
       
        if method=="Test":
            print(len(treats_cell_b_with_pert_1_cross))
            print(len(cov_drug_double))
            if cov_drug_right_all is None:
                cov_drug_right_all = cov_drug_double
            else:
                cov_drug_right_all = np.concatenate((cov_drug_right_all, cov_drug_double))

            if predict_result is None:
                predict_result = treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()
            else:
                predict_result = np.concatenate((predict_result, treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()), axis=0)

        if deg_predict_result is None:
            deg_predict_result = treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()
        else:
            deg_predict_result = np.concatenate((deg_predict_result, treats_cell_b_with_pert_1_cross.cpu().numpy().tolist()), axis=0)

    (
        treats_explained_variance_dict_b_1,
        treats_explained_variance_de_dict_b_1,
        treats_r2_dict_b_1,
        treats_r2_de_dict_b_1,
        treats_pearsonr_dict_b_1,
        treats_pearsonr_de_dict_b_1,
        treats_spearman_dict_b_1,
        treats_spearman_de_dict_b_1,
    ) = compute(true_double, pred_double, de_gene_idxs_dict_cell_double)

    if method=="Test":
        df = pd.DataFrame({
        'treats_explained_variance': treats_explained_variance_dict_b_1,
        'treats_explained_variance_de': treats_explained_variance_de_dict_b_1,
        'treats_r2': treats_r2_dict_b_1,
        'treats_r2_de': treats_r2_de_dict_b_1,
        'treats_pearsonr': treats_pearsonr_dict_b_1,
        'treats_pearsonr_de': treats_pearsonr_de_dict_b_1,
        'treats_spearman': treats_spearman_dict_b_1,
        'treats_spearman_de': treats_spearman_de_dict_b_1
        })

        predict_result_path = f"./results/predict_data/{dataset_name}/{save_file}"
        if os.path.exists(predict_result_path) is False:
            os.mkdir(predict_result_path)

        df.to_csv(f"{predict_result_path}/predict_score.csv", index=True)

    temp_r2 = list(treats_r2_dict_b_1.values())
    all_gene_eval_r2 = mean(temp_r2)
    print(f"{method} all gene r2 mean:", all_gene_eval_r2)
    all_gene_eval_r2_median = median(temp_r2)
    print(f"{method} all gene r2 median:", all_gene_eval_r2_median)

    temp_r2_de = list(treats_r2_de_dict_b_1.values())
    deg_gene_eval_r2 = mean(temp_r2_de)
    print(f"{method} degs gene r2 mean:", deg_gene_eval_r2)
    deg_gene_eval_r2_median = median(temp_r2_de)
    print(f"{method} degs gene r2 median:", deg_gene_eval_r2_median)


    temp_pearsonr = list(treats_pearsonr_dict_b_1.values())
    all_gene_eval_pearsonr = mean(temp_pearsonr)
    print(f"{method} all gene pearsonr:", all_gene_eval_pearsonr)

    temp_pearsonr_de = list(treats_pearsonr_de_dict_b_1.values())
    deg_gene_eval_pearsonr = mean(temp_pearsonr_de)
    print(f"{method} degs gene pearsonr:", deg_gene_eval_pearsonr)

    temp_spearman = list(treats_spearman_dict_b_1.values())
    all_gene_eval_spearman = mean(temp_spearman)
    print(f"{method} all gene spearman:", all_gene_eval_spearman)

    temp_spearman_de = list(treats_spearman_de_dict_b_1.values())
    deg_gene_eval_spearman = mean(temp_spearman_de)
    print(f"{method} degs gene spearman:", deg_gene_eval_spearman)

    temp_explained_variance = list(treats_explained_variance_dict_b_1.values())
    all_gene_eval_explained_variance = mean(temp_explained_variance)
    print(f"{method} all gene explained_variance:", all_gene_eval_explained_variance)

    temp_explained_variance_de = list(treats_explained_variance_de_dict_b_1.values())
    deg_gene_eval_explained_variance = mean(temp_explained_variance_de)
    print(f"{method} degs gene explained_variance:", deg_gene_eval_explained_variance)

    res = {
        "treats_r2_dict_b_1": treats_r2_dict_b_1,
        "treats_r2_de_dict_b_1": treats_r2_de_dict_b_1,
        "treats_pearsonr_dict_b_1": treats_pearsonr_dict_b_1,
        "treats_pearsonr_de_dict_b_1": treats_pearsonr_de_dict_b_1,
        "treats_spearman_dict_b_1": treats_spearman_dict_b_1,
        "treats_spearman_de_dict_b_1": treats_spearman_de_dict_b_1,
        "treats_explained_variance_dict_b_1": treats_explained_variance_dict_b_1,
        "treats_explained_variance_de_dict_b_1": treats_explained_variance_de_dict_b_1,
        "all_gene_eval_r2_mean": all_gene_eval_r2,
        "all_gene_eval_r2_median": all_gene_eval_r2_median,
        "deg_gene_eval_r2_mean": deg_gene_eval_r2,
        "deg_gene_eval_r2_median": deg_gene_eval_r2_median,

        "all_gene_eval_explained_variance": all_gene_eval_explained_variance,
        "deg_gene_eval_explained_variance": deg_gene_eval_explained_variance,
        "embedding_dict": embedding_dict,
    }

    if method=="Test":
        print(len(predict_result))
        
        predict_result_path = f"./results/predict_data/{dataset_name}/{save_file}"
        if os.path.exists(predict_result_path) is False:
            os.mkdir(predict_result_path)

        predict_result_df = pd.DataFrame(predict_result)
        def add_underscore(s):
            return s.replace('BRD-', '_BRD-')
        predict_result_df["condition_name"] = cov_drug_right_all
        predict_result_df.to_csv(f"{predict_result_path}/predict_expression.csv", index=False)

    return res

def compute_double_gene_train_r2(
    embedding_dict,
    true_treats_left,
    true_treats_right,
    true_treats_double,
    de_idx_left,
    de_idx_right,
    de_idx_double,
):
    all_gene_r2 = []
    deg_gene_r2 = []

    de_idx_left = de_idx_left.cpu()
    de_idx_right = de_idx_right.cpu()
    de_idx_double = de_idx_double.cpu()

    pred_treats_left: torch.Tensor = embedding_dict[4]
    pred_treats_right: torch.Tensor = embedding_dict[5]
    pred_treats_double: torch.Tensor = embedding_dict[6]

    for i in range(true_treats_left.shape[0]):
        all_gene_r2.append(
            compute_r2(true_treats_left[i].detach(), pred_treats_left[i].detach())
        )
        all_gene_r2.append(
            compute_r2(true_treats_right[i].detach(), pred_treats_right[i].detach())
        )
        all_gene_r2.append(
            compute_r2(true_treats_double[i].detach(), pred_treats_double[i].detach())
        )
        if len(de_idx_left) > 0:
            deg_gene_r2.append(
                compute_r2(
                    true_treats_left[i][de_idx_left[i]].detach(),
                    pred_treats_left[i][de_idx_left[i]].detach(),
                )
            )
            deg_gene_r2.append(
                compute_r2(
                    true_treats_right[i][de_idx_right[i]].detach(),
                    pred_treats_right[i][de_idx_right[i]].detach(),
                )
            )
            deg_gene_r2.append(
                compute_r2(
                    true_treats_double[i][de_idx_double[i]].detach(),
                    pred_treats_double[i][de_idx_double[i]].detach(),
                )
            )

    return all_gene_r2, deg_gene_r2