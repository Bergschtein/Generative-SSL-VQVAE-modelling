import copy
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_ssl_vqvae import Exp_NC_VQVAE
from experiments.exp_vqvae import Exp_VQVAE

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import (
    load_yaml_param_settings,
    save_model,
    get_root_dir,
    model_filename,
)
import torch

from train_vqvae import train_vqvae
from train_ssl_vqvae import train_ssl_vqvae

UCR_SUBSET = [
    "ElectricDevices",
    # "StarLightCurves",
    # "Wafer",
    # "ECG5000",
    # "TwoPatterns",
    # "FordA",
    # "UWaveGestureLibraryAll",
    # "FordB",
    # "ChlorineConcentration",
    # "ShapesAll",
]

STAGE1_METHODS = ["", "barlowtwins", "vicreg"]
SSL_WEIGHTS = {"barlowtwins": 1.0, "vicreg": 0.01, "": 0}


def run_experiments():
    config_dir = get_root_dir().joinpath("configs", "config.yaml")
    config = load_yaml_param_settings(config_dir)
    config["trainer_params"]["max_epochs"]["stage1"] = 1
    batch_size = config["dataset"]["batch_sizes"]["stage1"]

    project_name = "SSL_VQVAE-STAGE1-IDUN"

    for dataset in UCR_SUBSET:
        c = config.copy()
        c["dataset"]["dataset_name"] = dataset

        dataset_importer = UCRDatasetImporter(**config["dataset"])

        train_data_loader_no_aug = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="train"
        )
        train_data_loader_aug = build_data_pipeline(
            batch_size, dataset_importer, config, augment=True, kind="train"
        )
        test_data_loader = build_data_pipeline(
            batch_size, dataset_importer, config, augment=False, kind="test"
        )

        for method in STAGE1_METHODS:
            c["SSL"]["stage1_method"] = method
            c["SSL"]["stage1_weight"] = SSL_WEIGHTS[method]

            if method == "":
                train_vqvae(
                    config=c,
                    train_data_loader=train_data_loader_no_aug,
                    test_data_loader=test_data_loader,
                    do_validate=True,
                    gpu_device_idx=0,
                    wandb_run_name=f"{model_filename(c, 'stage1')}-{dataset}",
                    wandb_project_name=project_name,
                )

            else:
                train_ssl_vqvae(
                    config=c,
                    train_data_loader=train_data_loader_aug,
                    test_data_loader=test_data_loader,
                    do_validate=True,
                    gpu_device_idx=0,
                    wandb_run_name=f"{model_filename(c, 'stage1')}-{dataset}",
                    wandb_project_name=project_name,
                )


if __name__ == "__main__":
    run_experiments()
