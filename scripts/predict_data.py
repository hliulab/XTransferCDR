from .sciplex3 import SC_TrainDataset, SC_TestDataset
from .gears import GEARS_TrainDataset, GEARS_TestDataset
from .gears_double_gene import GEARS_double_gene_TestDataset

def load_dataset_splits(config):
    test_dataset = None

    if config["dataset"] == "sciplex3":
        test_dataset = SC_TestDataset(
            config["sciplex3_treat_test_cell_a_with_pert_1_path"],
            # config["sciplex3_treat_test_cell_b_with_pert_2_path"],
            # config["sciplex3_treat_test_cell_a_with_pert_2_path"],
            # config["sciplex3_treat_test_cell_b_with_pert_1_path"],
             config["sciplex3_treat_test_cell_b_with_pert_1_path"],
            config["sciplex3_control_test_cell_b_path"],
            config["sciplex3_de_gene"],
        )
    elif config["dataset"] == "gears":
        test_dataset = GEARS_TestDataset(
            config["gears_treat_test_cell_a_with_pert_1_path"],
            # config["gears_treat_test_cell_b_with_pert_2_path"],
            # config["gears_treat_test_cell_a_with_pert_2_path"],
            # config["gears_treat_test_cell_b_with_pert_1_path"],
             config["gears_treat_test_cell_b_with_pert_1_path"],
            config["gears_control_test_cell_b_path"],
            config["gears_de_gene"],
        )
    elif config["dataset"] == "gears_double_gene":
        test_dataset = GEARS_double_gene_TestDataset(
            config["gears_trapnell_treat_test_left_cell_path"],
            config["gears_trapnell_treat_test_right_cell_path"],
            config["gears_trapnell_treat_test_double_cell_path"],
            config["gears_trapnell_control_test_left_cell_path"],
            config["gears_trapnell_control_test_right_cell_path"],
            config["gears_de_gene"],
        )
    splits = {
        "test": test_dataset,
    }

    return splits
