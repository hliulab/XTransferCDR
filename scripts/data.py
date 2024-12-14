from .sciplex3 import SC_TrainDataset, SC_TestDataset
from .gears import GEARS_TrainDataset, GEARS_TestDataset
from .gears_double_gene import GEARS_double_gene_TrainDataset, GEARS_double_gene_TestDataset

def load_dataset_splits(config):
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    if config["dataset"] == "sciplex3":
        train_dataset = SC_TrainDataset(
            config["sciplex3_control_train_left"],
            config["sciplex3_control_train_right"],
            config["sciplex3_treat_train_left"],
            config["sciplex3_treat_train_right"],
            config["sciplex3_de_gene"],
            # config["sciplex3_cross_train_left"],
            # config["sciplex3_cross_train_right"],
        )
        valid_dataset = SC_TestDataset(
            config["sciplex3_treat_valid_cell_a_with_pert_1_path"],
            # config["sciplex3_treat_valid_cell_b_with_pert_2_path"],
            # config["sciplex3_treat_valid_cell_a_with_pert_2_path"],
            # config["sciplex3_treat_valid_cell_b_with_pert_1_path"],
            config["sciplex3_treat_valid_cell_b_with_pert_1_path"],
            config["sciplex3_control_valid_cell_b_path"],
            config["sciplex3_de_gene"],
        )
        test_dataset = SC_TestDataset(
            config["sciplex3_treat_test_cell_a_with_pert_1_path"],
            # config["sciplex3_treat_test_cell_b_with_pert_2_path"],
            # config["sciplex3_treat_test_cell_a_with_pert_2_path"],
            # config["sciplex3_treat_test_cell_b_with_pert_1_path"],
             config["sciplex3_treat_test_cell_b_with_pert_1_path"],
            config["sciplex3_control_test_cell_b_path"],
            config["sciplex3_de_gene"],
        )
    elif config["dataset"] == "gears_double_gene":
        train_dataset = GEARS_double_gene_TrainDataset(
            config["gears_control_train_left"],
            config["gears_control_train_right"],
            config["gears_treat_train_left"],
            config["gears_treat_train_right"],
            config["gears_treat_train_double"],
            config["gears_de_gene"],
        )
        valid_dataset = GEARS_double_gene_TestDataset(
            config["gears_trapnell_treat_valid_left_cell_path"],
            config["gears_trapnell_treat_valid_right_cell_path"],
            config["gears_trapnell_treat_valid_double_cell_path"],
            config["gears_trapnell_control_valid_left_cell_path"],
            config["gears_trapnell_control_valid_right_cell_path"],
            config["gears_de_gene"],
        )
        test_dataset = GEARS_double_gene_TestDataset(
            config["gears_trapnell_treat_test_left_cell_path"],
            config["gears_trapnell_treat_test_right_cell_path"],
            config["gears_trapnell_treat_test_double_cell_path"],
            config["gears_trapnell_control_test_left_cell_path"],
             config["gears_trapnell_control_test_right_cell_path"],
            config["gears_de_gene"],
        )
    elif config["dataset"] == "gears":
        train_dataset = GEARS_TrainDataset(
            config["gears_control_train_left"],
            config["gears_control_train_right"],
            config["gears_treat_train_left"],
            config["gears_treat_train_right"],
            config["gears_de_gene"],
            # config["gears_cross_train_left"],
            # config["gears_cross_train_right"],
        )
        valid_dataset = GEARS_TestDataset(
            config["gears_treat_valid_cell_a_with_pert_1_path"],
            # config["gears_treat_valid_cell_b_with_pert_2_path"],
            # config["gears_treat_valid_cell_a_with_pert_2_path"],
            # config["gears_treat_valid_cell_b_with_pert_1_path"],
            config["gears_treat_valid_cell_b_with_pert_1_path"],
            config["gears_control_valid_cell_b_path"],
            config["gears_de_gene"],
        )
        test_dataset = GEARS_TestDataset(
            config["gears_treat_test_cell_a_with_pert_1_path"],
            # config["gears_treat_test_cell_b_with_pert_2_path"],
            # config["gears_treat_test_cell_a_with_pert_2_path"],
            # config["gears_treat_test_cell_b_with_pert_1_path"],
             config["gears_treat_test_cell_b_with_pert_1_path"],
            config["gears_control_test_cell_b_path"],
            config["gears_de_gene"],
        )

    splits = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

    return splits
