import torch
import pandas as pd
from .utils import split_df
from torch.utils.data import Dataset


class GEARS_TrainDataset(Dataset):
    def __init__(
        self,
        control_left_path,
        control_right_path,
        treat_left_path,
        treat_right_path,
        de_gene_path,
        # cross_left_path,
        # cross_right_path,
        dtype=torch.float32,
    ):
        self.split_size = 2000

        self.control_cell_left = pd.read_csv(control_left_path)
        self.control_cell_right = pd.read_csv(control_right_path)
        self.control_cell_left = self.control_cell_left.set_index("cell_type")
        self.control_cell_right = self.control_cell_right.set_index("cell_type")

        self.control_cell_left = split_df(self.control_cell_left, self.split_size)
        self.control_cell_right = split_df(self.control_cell_right, self.split_size)

        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")

        self.de_gene_idx = self.de_gene_idx[
            ~self.de_gene_idx.index.duplicated(keep='first')
        ]

        treat_train_cell_left = pd.read_csv(treat_left_path)
        treat_train_cell_right = pd.read_csv(treat_right_path)

        # cross_train_cell_left = pd.read_csv(cross_left_path)
        # cross_train_cell_right = pd.read_csv(cross_right_path)

        self.len = treat_train_cell_left.shape[0]

        self.treat_train_cell_left = split_df(treat_train_cell_left, self.split_size)
        self.treat_train_cell_right = split_df(treat_train_cell_right, self.split_size)
        # self.cross_train_cell_left = split_df(cross_train_cell_left, self.split_size)
        # self.cross_train_cell_right = split_df(cross_train_cell_right, self.split_size)

        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        index = i - ((i // self.split_size) * self.split_size)

        temp_train_left = self.treat_train_cell_left[i // self.split_size].iloc[
            index, :
        ]
        temp_train_right = self.treat_train_cell_right[i // self.split_size].iloc[
            index, :
        ]
        # temp_cross_train_left = self.cross_train_cell_left[i // self.split_size].iloc[
        #     index, :
        # ]
        # temp_cross_train_right = self.cross_train_cell_right[i // self.split_size].iloc[
        #     index, :
        # ]
        # cell_id_left = temp_train_left["cell_type"]
        pert_id_left = temp_train_left["condition_name"]
        # cell_id_right = temp_train_right["cell_type"]
        pert_id_right = temp_train_right["condition_name"]

        # pert_id_cross_left = temp_cross_train_left["condition_name"]
        # pert_id_cross_right = temp_cross_train_right["condition_name"]

        cov_drug_left = temp_train_left["condition_name"]
        cov_drug_right = temp_train_right["condition_name"]

        # cov_drug_cross_left = temp_cross_train_left["condition_name"]
        # cov_drug_cross_right = temp_cross_train_right["condition_name"]

        temp_train_left = temp_train_left.drop(["cell_type", "condition_name"])
        temp_train_right = temp_train_right.drop(["cell_type", "condition_name"])

        # temp_cross_train_left = temp_cross_train_left.drop(["cell_type", "condition_name"])
        # temp_cross_train_right = temp_cross_train_right.drop(["cell_type", "condition_name"])

        treat_left = torch.tensor(temp_train_left.to_list(), dtype=self.dtype)
        treat_right = torch.tensor(temp_train_right.to_list(), dtype=self.dtype)
        # cross_left = torch.tensor(temp_cross_train_left.to_list(), dtype=self.dtype)
        # cross_right = torch.tensor(temp_cross_train_right.to_list(), dtype=self.dtype)

        if pert_id_left in self.de_gene_idx.index:
            de_idx_left = self.de_gene_idx.loc[pert_id_left].to_numpy()
            if de_idx_left.shape[0] != 50:
                de_idx_left = de_idx_left[0]
        else:
            raise (f"cov_drug {pert_id_left} not exists.")

        de_idx_left = torch.tensor(de_idx_left, dtype=torch.long)

        if pert_id_right in self.de_gene_idx.index:
            de_idx_right = self.de_gene_idx.loc[pert_id_right].to_numpy()
            if de_idx_right.shape[0] != 50:
                de_idx_right = de_idx_right[0]
        else:
            raise (f"cov_drug {pert_id_right} not exists.")

        de_idx_right = torch.tensor(de_idx_right, dtype=torch.long)

        control_left = self.control_cell_left[i // self.split_size].iloc[index, :]
        control_left = torch.tensor(control_left.to_list(), dtype=self.dtype)
        control_right = self.control_cell_right[i // self.split_size].iloc[index, :]
        control_right = torch.tensor(control_right.to_list(), dtype=self.dtype)

        return (
            control_left,
            control_right,
            treat_left,
            treat_right,
            # cross_left,
            # cross_right,
            de_idx_left,
            de_idx_right,
            cov_drug_left,
            cov_drug_right,
        )


class GEARS_TestDataset(Dataset):
    def __init__(
        self,
        treats_cell_a_with_pert_1_path,
        treats_cell_b_with_pert_1_path,
        control_cell_b_path,
        de_gene_path,
        dtype=torch.float32,
    ):
        print("start load data")
        self.split_size = 2000
        self.treats_cell_a_with_pert_1 = pd.read_csv(treats_cell_a_with_pert_1_path)
        self.treats_cell_b_with_pert_1 = pd.read_csv(treats_cell_b_with_pert_1_path)
        self.control_cell_b = pd.read_csv(control_cell_b_path)
        self.control_cell_b = self.control_cell_b.set_index("cell_type")

        # self.control_cell_a = split_df(self.control_cell_a, self.split_size)
        self.de_gene_idx = pd.read_csv(de_gene_path)
        self.de_gene_idx = self.de_gene_idx.set_index("Unnamed: 0")
        # print(self.de_gene_idx)

        self.de_gene_idx = self.de_gene_idx[
            ~self.de_gene_idx.index.duplicated(keep='first')
        ]
        print("finish load data")
        self.dtype = dtype

    def __len__(self):
        return len(self.treats_cell_a_with_pert_1)

    def __getitem__(self, i):
        temp_treats_cell_a_with_pert_1 = self.treats_cell_a_with_pert_1.iloc[i, :]
        temp_treats_cell_b_with_pert_1 = self.treats_cell_b_with_pert_1.iloc[i, :]
        temp_control_cell_b = self.control_cell_b.iloc[i, :]
        
        cov_drug_a_1 = temp_treats_cell_a_with_pert_1["condition_name"]
        cov_drug_b_1 = temp_treats_cell_b_with_pert_1["condition_name"]
        temp_treats_cell_a_with_pert_1 = temp_treats_cell_a_with_pert_1.drop(
            ["cell_type",  "condition_name"]
        )
        temp_treats_cell_b_with_pert_1 = temp_treats_cell_b_with_pert_1.drop(
            ["cell_type",  "condition_name"]
        )

        treats_cell_a_with_pert_1 = torch.tensor(temp_treats_cell_a_with_pert_1.to_list(), dtype=self.dtype)
        treats_cell_b_with_pert_1 = torch.tensor(temp_treats_cell_b_with_pert_1.to_list(), dtype=self.dtype)
        control_cell_b = torch.tensor(temp_control_cell_b.to_list(), dtype=self.dtype)

        if cov_drug_a_1 in self.de_gene_idx.index:
            de_idx_cell_a_pert_1 = self.de_gene_idx.loc[cov_drug_a_1].to_numpy()
            if de_idx_cell_a_pert_1.shape[0] != 50:
                de_idx_cell_a_pert_1 = de_idx_cell_a_pert_1[0]
        else:
            de_idx_cell_a_pert_1 = [-1 for _ in range(50)]
            raise (f"cov_drug {cov_drug_a_1} not exists.")
        
        if cov_drug_b_1 in self.de_gene_idx.index:
            de_idx_cell_b_pert_1 = self.de_gene_idx.loc[cov_drug_b_1].to_numpy()
            if de_idx_cell_b_pert_1.shape[0] != 50:
                de_idx_cell_b_pert_1 = de_idx_cell_b_pert_1[0]
        else:
            de_idx_cell_b_pert_1 = [-1 for _ in range(50)]
            raise (f"cov_drug {cov_drug_b_1} not exists.")


        de_idx_cell_a_pert_1 = torch.tensor(de_idx_cell_a_pert_1, dtype=torch.long)
        de_idx_cell_b_pert_1 = torch.tensor(de_idx_cell_b_pert_1, dtype=torch.long)

        return (
            treats_cell_a_with_pert_1,
            treats_cell_b_with_pert_1,
            control_cell_b,
            de_idx_cell_a_pert_1,
            de_idx_cell_b_pert_1,
            cov_drug_a_1,
            cov_drug_b_1
        )