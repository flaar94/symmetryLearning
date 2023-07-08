import torch
import numpy as np
import matplotlib.pyplot as plt
from core import SymmetryFinder
from numpy.random import default_rng
import torchvision
import logging
import torchvision.transforms.functional as funct
import torchvision.transforms as transforms
import itertools
import pandas as pd
import random
import dill
from pathlib import Path
DATA_PATH = Path("..") / "data"

logging.basicConfig(filename="../logs/experimental_semi_synthetic.log", level=logging.INFO)





def ground_truth_loss(mat1, mat2):
    return np.sqrt(np.mean((mat1 - mat2) ** 2))


def horiz_flip_gt_accuracy(model, dim=(28, 28)):
    # flip = transforms.RandomHorizontalFlip(p=1.)
    # eigenvector1 = model.eigenvectors_.reshape(dim[0], dim[1], -1)

    eigenvectors = model.eigenvectors_.T.reshape(-1, dim[0], dim[1])
    diff = np.array(
        [eigenvectors[i].reshape(-1, dim[0] * dim[1]) @ eigenvectors[i, :, ::-1].reshape(-1, dim[0] * dim[1]).T for i in
         range(eigenvectors.shape[0])])
    diff = diff.reshape(-1)
    diff = np.arccos(diff) * 180 / np.pi
    diff[diff > 120] = -1
    diff[(120 >= diff) & (diff >= 60)] = 0
    diff[(0 < diff) & (diff < 60)] = 1
    return np.sum(diff == model.trans_eigenvalues_) / sum((diff != 0)), sum((diff != 0)) / len(diff)


if __name__ == '__main__':
    # num_samples = NUM_SAMPLES
    space_dims = [
                4, 10, 16, 22,
                28]
    methods = [
        'mean', 'median', 'mm_mix', 'sign',
        'loc_corr_adj']
    datasets = ['mnist', 'emnist']
    # num_samples_list = [2_000, 10_000, 50_000, 250_000]

    # print(df_error)
    df_cov_acc = pd.DataFrame(columns=space_dims, index=datasets, dtype=float)
    for dataset in datasets:
        df_acc = pd.DataFrame(columns=methods, index=space_dims, dtype=float)
        save_file = DATA_PATH / f"{dataset}_acc.csv"
        for space_dim in space_dims:
            torch.manual_seed(42)
            rng = default_rng(42)

            trans_mat = np.eye(space_dim * space_dim).reshape(space_dim, space_dim, space_dim, space_dim)
            trans_mat = trans_mat[:, :, :, ::-1]
            trans_mat = trans_mat.reshape(space_dim * space_dim, space_dim * space_dim)

            dim = (space_dim, space_dim)
            strt_msg = f"Starting Experiment: methods = {', '.join(methods)}, space_dim = {space_dim}"
            logging.info(strt_msg)
            if dataset == 'emnist':
                if space_dims == 28:
                    transform = transforms.Compose(
                        [
                            lambda x: funct.rotate(x, angle=-90),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            lambda x: x.view(-1),
                            lambda x: x.numpy()])
                else:
                    transform = transforms.Compose(
                        [
                            lambda x: funct.rotate(x, angle=-90),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(dim),
                            transforms.ToTensor(),
                            lambda x: x.view(-1),
                            lambda x: x.numpy()])
            else:
                if space_dims == 28:
                    transform = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            lambda x: x.view(-1),
                            lambda x: x.numpy()])
                else:
                    transform = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(dim),
                            transforms.ToTensor(),
                            lambda x: x.view(-1),
                            lambda x: x.numpy()])
            if dataset == 'emnist':
                trainset = torchvision.datasets.EMNIST(root=DATA_PATH, train=True,
                                                       download=True, transform=transform,
                                                       split="digits"
                                                       )
            else:
                trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True,
                                                      download=True, transform=transform,
                                                      )



            data = np.array([x for i, (x, label) in enumerate(trainset)])
            data = data / np.std(data)
            rng.shuffle(data)

            cov = np.cov(data, rowvar=False)
            eigen = np.linalg.eig(cov)
            cov_eigenvalues = np.real(eigen[0])
            eigenvectors = np.real(eigen[1])
            mu = np.mean(data, axis=0)
            sol = np.linalg.solve(eigenvectors, mu)
            num_trivial_vectors = np.sum((sol < SymmetryFinder.ignore_threshold) & \
                                         (cov_eigenvalues < SymmetryFinder.ignore_threshold))
            print(space_dim, dataset, num_trivial_vectors)
            swaps = int((dim[0] * dim[1] - num_trivial_vectors) // 2)
            # print(type(swaps))
            for i, method in enumerate(methods):
                sym = SymmetryFinder(fit_method=method, select_method=swaps)
                sym.fit(data, cov=cov)
                accs = horiz_flip_gt_accuracy(sym, dim=dim)
                msg = f"method = {method}, accs = {accs}"
                logging.info(msg)
                print(msg)
                df_acc.loc[space_dim, method] = accs[0]
                df_cov_acc.loc[dataset, space_dim] = accs[1]
        df_acc.to_csv(save_file)
    df_cov_acc.to_csv(DATA_PATH / "cov_acc.csv")
