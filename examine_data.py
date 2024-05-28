from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.preprocess_ucr import UCRDataset
from preprocessing.preprocess_ucr import AugUCRDataset
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.augmentations import Augmenter
from models.stage2.maskgit import MaskGIT
from models.stage2.full_embedding_maskgit import Full_Embedding_MaskGIT

from models.stage2.sample import unconditional_sample, conditional_sample, plot_generated_samples
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import os
from pathlib import Path
import tempfile

from utils import (
    compute_downsample_rate,
    get_root_dir,
    freeze,
    timefreq_to_time,
    load_yaml_param_settings,
    time_to_timefreq,
    quantize,
    model_filename,
)
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.svm import SVC

from utils import time_to_timefreq, quantize
import torch.nn.functional as F
import torch
import wandb
import pandas as pd
import seaborn as sns
import plotly.express as px

class Examiner():
    def __init__(self, 
                 datasets = [
                    "ElectricDevices",
                    "StarLightCurves",
                    "Wafer",
                    "ECG5000",
                    "TwoPatterns",
                    "FordA",
                    "UWaveGestureLibraryAll",
                    "FordB",
                    "ShapesAll",
                    'SonyAIBORobotSurface1', 
                    'SonyAIBORobotSurface2', 
                    'Symbols',
                    'Mallat'
                ],
                    augment = True,        
                    ):
        self.augment = augment
        self.datasets = datasets
        self.config = load_yaml_param_settings("configs/config.yaml")
        self.loader_dict = {}
        for dataset in self.datasets:
            self.config['dataset']['dataset_name'] = dataset
            self.config["VQVAE"]["n_fft"] = 8

            # data pipeline
            dataset_importer = UCRDatasetImporter(**self.config["dataset"])
            batch_size = self.config["dataset"]["batch_sizes"]["stage1"]
            train_data_loader = build_data_pipeline(
                batch_size, dataset_importer, self.config, "train", augment= self.augment
            )
            test_data_loader = build_data_pipeline(batch_size, dataset_importer, self.config, "test")
            self.loader_dict[dataset] = (train_data_loader, test_data_loader)

    def update_config(self):
        self.config = load_yaml_param_settings("configs/config.yaml")

    def load_augmenter(self, time_augs, timefreq_augs,aug_params, use_all_methods, **kwargs):
        self.augmenter = Augmenter(time_augs, timefreq_augs,aug_params, use_all_methods, **kwargs)
        # return self.augmenter

    def augment_samples(self,n_samples):
        dataset = self.current_dataset

        X = self.loader_dict[dataset][0].dataset.X
        Y = self.loader_dict[dataset][0].dataset.Y
        random_samples = np.random.choice(X.shape[0], n_samples)
        X = X[random_samples,]
        Y = Y[random_samples]

        data_list = []
        for i in range(n_samples):
            X_aug, augs = self.augmenter.augment(X[i,], True)
            data_list.append([X[i,], X_aug.numpy(), augs, Y[i]])

        for x, x_aug, augs, lable in data_list:
            plt.plot(x,color = 'blue', alpha = 0.5 ,label = "Original")
            plt.plot(x_aug,color = 'orange', alpha = 0.5, label = "Augmented")
            plt.legend()
            plt.suptitle(f"Dataset: {dataset}")
            plt.title(f"Augmentations used: {augs}, Class: {lable}")
            
            plt.show()
        # return data_list


    def get_config(self):
        return self.config
    
    def loader_dict(self):
        return self.loader_dict
    
    def datasets(self):
            return self.datasets

    def generate_samples(self, n_samples, label = None):
        generative_model = self.maskgit
        if label != None:
            x_new = conditional_sample(
                    generative_model = generative_model,
                    n_samples = n_samples,
                    device = "cpu",
                    class_index = label,
                    batch_size=256,
                    return_representations=False,
                    guidance_scale=1.0,
                )
        else:
            x_new = unconditional_sample(
                    generative_model = generative_model,
                    n_samples = n_samples,
                    device = "cpu",
                    class_index = None,
                    batch_size=256,
                    return_representations=False,
                    guidance_scale=1.0,
                )
        return np.squeeze(x_new)
        # plot_generated_samples(x_new, title = f"Sample from class: {label}", max_len=20)        


    def plot_datasets(self, train = True):

        colors = ['blue','red', 'green', 'black', 'brown', 'purple','pink'] #, 'brown', 'pink'

        if train:
            for dataset in self.datasets:
                
                train_data_loader = self.loader_dict[dataset][0]
                X_train = train_data_loader.dataset.X
                Y_train = train_data_loader.dataset.Y
                labels = np.unique(Y_train)

                f, a = plt.subplots(len(labels), 1,figsize=(8, 2*len(labels)), sharex='col', sharey='row')
                for i in range(len(labels)):
                    mask = np.squeeze(Y_train == i)
                    x_conditional = X_train[mask, :]
                    nr_of_samples = x_conditional.shape[0]

                    a[i].set_title(f"Samples from class: {i}")
                    a[i] = plot_new(a[i],x_conditional, color= colors[i%len(colors)])
                    # for x in x_conditional:
                    #     plt.plot(x, color = colors[i%len(colors)], alpha = 0.05)        
                    # plt.title(f"Class: {i}, Nr of samples: {nr_of_samples}")
                plt.suptitle(f"Dataset name: {dataset}")
                plt.show()


            

        else:
            for dataset in self.datasets:
                test_data_loader = self.loader_dict[dataset][1]
                X_test = test_data_loader.dataset.X
                Y_test = test_data_loader.dataset.Y
                lables = np.unique(Y_test)

                for i in range(len(lables)):
                    mask = np.squeeze(Y_test == i)
                    x_conditional = X_test[mask, :]
                    nr_of_samples = x_conditional.shape[0]
                    for x in x_conditional:
                        plt.plot(x, color = colors[i%8], alpha = 0.1)        
                    plt.title(f"Class: {i}, Nr of samples: {nr_of_samples}")
                    plt.suptitle(f"Dataset name: {dataset}")
                    plt.show()


    def PCA_latent(self, z, y, train = True):
        # Label to color dict (automatic)
        label_color_dict = {label:idx for idx,label in enumerate(np.unique(y))}
        z = F.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)
        # Color vector creation
        cvec = [label_color_dict[label.tolist()] for label in y]

        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z)
        plt.figure(figsize=(4, 4))
        plt.scatter(
            z_pca[:, 0], z_pca[:, 1],c=cvec, alpha=0.1
        )

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def TSNE_latents(self, z,y):
        z = F.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)
        z_tsne = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(z)

        plt.figure(figsize=(4, 4))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, alpha=0.1)
        plt.legend()
        plt.show()
        plt.close()

    def load_models(self, dataset, ssl_method = "", aug = None, decorr = False):
            if ssl_method == "":
                self.config['ID'] = "Y3A7B9"
                self.config['seed'] = 1
            else:
                if aug == "gauss":
                    self.config['ID'] = "DQZWZT"
                    self.config['seed'] = 1
                if aug == "warp":
                    self.config['ID'] = "P6MCKO"
                    self.config['seed'] = 3
                if aug == "warp":
                    self.config['ID'] = "GDAQEN"
                    self.config['seed'] = 2
                
            self.current_dataset = dataset
            self.config['dataset']['dataset_name'] = dataset
            self.config["SSL"]["stage1_method"] = ssl_method
            if decorr:
                self.config["VQVAE"]["orthogonal_reg_weight"] = 10

            input_length = self.loader_dict[dataset][0].dataset.X.shape[-1]
            n_classes = len(np.unique(self.loader_dict[dataset][0].dataset.Y))

            if ssl_method == "":
                self.maskgit = MaskGIT(
                input_length,
                **self.config["MaskGIT"],
                config=self.config,
                n_classes=n_classes,
                )
                fname = f"maskgit-seed-1-Y3A7B9-{dataset}.ckpt"
                try:
                    ckpt_fname = os.path.join("saved_models", fname)
                    self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
                except FileNotFoundError:
                    ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
                    self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
                print(fname)    
                print("maskgit loaded")
            else:
                self.maskgit = Full_Embedding_MaskGIT(
                input_length,
                **self.config["MaskGIT"],
                config=self.config,
                n_classes=n_classes,
                finetune_codebook = True, 
                load_finetuned_codebook = True,
                device = "cpu"
                )
                fname = f"{ssl_method}-fullembed-maskgit-finetuned-seed-1-DQZWZT-{dataset}.ckpt"
                try:
                    ckpt_fname = os.path.join("saved_models", fname)
                    self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
                except FileNotFoundError:
                    ckpt_fname = Path(tempfile.gettempdir()).joinpath(fname)
                    self.maskgit.load_state_dict(torch.load(ckpt_fname), strict=False)
                print(fname)      
                print("maskgit loaded")
            self.encoder = self.maskgit.encoder
            self.decoder = self.maskgit.decoder
            self.vq_model = self.maskgit.vq_model

    def get_latents(self, dataset, train = True):
        train_dataloader = self.loader_dict[dataset][0]
        test_dataloader = self.loader_dict[dataset][1]

        n_fft = self.config['VQVAE']['n_fft'] 

        z_tr, y_tr, counts = encode_data(train_dataloader, self.encoder, n_fft)
        z_te, y_te, counts = encode_data(test_dataloader, self.encoder, n_fft)
        z_tr = z_tr.squeeze()
        y_tr = y_tr.squeeze() 
        z_te = z_te.squeeze()
        y_te = y_te.squeeze() 
        # z_encoded = z_encoded.squeeze()
        # ys = ys.squeeze()

        return z_tr, y_tr, z_te, y_te


def encode_data(
    dataloader,
    encoder,
    n_fft = 8,
    vq_model=None,
    avg_pooling=False,
    num_tokens=32,
    #device="cuda",
):
    """
    Function to encode the data using the encoder and optionally the quantizer.
    It encodes to continous latent variables by default (vq_model=False).
    ---
    returns
    """

    z_list = []  # List to hold all the encoded representations
    y_list = []  # List to hold all the labels/targets

    # Iterate over the entire dataloader
    counts = torch.zeros(num_tokens)

    for batch in dataloader:
        x, y = batch  # Unpack the batch.
        if len(x) == 2:
            x = x[0]  # discard the potential augmented view

        # Perform the encoding
        C = x.shape[1]
        xf = time_to_timefreq(x, n_fft, C)  # Convert time domain to frequency domain
        z = encoder(xf).detach()  # Encode the input

        if vq_model is not None:
            z, s, _, _ = quantize(z, vq_model)
            counts += torch.bincount(s.flatten(), minlength=32)

        # Convert the tensors to lists and append to z_list and y_list
        z_list.extend(z.detach().tolist())
        y_list.extend(
            y.detach().tolist()
        )  # Make sure to detach y and move to CPU as well

    # Convert lists of lists to 2D tensors
    z_encoded = torch.tensor(z_list)
    ys = torch.tensor(y_list)

    if avg_pooling:
        z_encoded = F.adaptive_avg_pool2d(z_encoded, (1, 1)).squeeze(-1).squeeze(-1)

    return z_encoded, ys, counts

def probes(x_tr, x_ts, y_tr, y_ts):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_ts = scaler.transform(x_ts)
    y_tr = y_tr.flatten()
    y_ts = y_ts.flatten()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_tr, y_tr)
    preds_knn = knn.predict(x_ts)

    svm = SVC(kernel="linear")
    svm.fit(x_tr, y_tr)
    preds_svm = svm.predict(x_ts)

    scores = {
        "knn_accuracy": metrics.accuracy_score(y_ts, preds_knn),
        "svm_accuracy": metrics.accuracy_score(y_ts, preds_svm),
    }
    return scores


def plot_new(ax, x_new, color = "red", alpha = 0.1):
    # nr_of_samples = x_new.shape[0]
    for x in x_new:
        ax.plot(x, color = color, alpha = alpha) 
    return ax