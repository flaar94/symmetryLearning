"""
In this experiment we test how robust the label-based approaches are when the symmetry isn't "equal". By this I mean,
the transformation isn't quite a symmetry of the distribution, but it will take datapoints into allowable datapoints.
For example, if we take a dataset and include transformed datapoints, but at a lower rate than untransformed datapoints
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from core import SymmetryFinder, SymmetryFinderLabel
from numpy.random import default_rng
import torchvision
import logging
import torchvision.transforms.functional as funct
import torchvision.transforms as transforms
from sklearn.utils import shuffle
import itertools
import pandas as pd
import random
import dill
from pathlib import Path

DATA_PATH = Path("..") / "data"
RESULT_DATA_PATH = Path("..") / "result_data"
logging.basicConfig(filename="../logs/robustness_under_unequal_symmetry.log", level=logging.INFO)


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
    space_dims = [10, 28]
    datasets = ['mnist', 'emnist']
    p_diffs = [0., 0.01, 0.05, 0.1, 0.2]

    # num_samples_list = [2_000, 10_000, 50_000, 250_000]

    # print(df_error)
    acc_records = []
    for dataset, space_dim, p_diff in itertools.product(datasets, space_dims, p_diffs):
        p = 0.5 - p_diff
        torch.manual_seed(42)
        rng = default_rng(42)

        dim = (space_dim, space_dim)
        strt_msg = f"Starting labeled refinement Experiment, dataset = {dataset}, space_dim = {space_dim}, p_diff={p_diff}"
        logging.info(strt_msg)
        print(strt_msg)
        if dataset == 'emnist':
            if space_dims == 28:
                transform = transforms.Compose(
                    [
                        lambda x: funct.rotate(x, angle=-90),
                        transforms.RandomHorizontalFlip(p),
                        transforms.ToTensor(),
                        lambda x: x.view(-1),
                        lambda x: x.numpy()])
            else:
                transform = transforms.Compose(
                    [
                        lambda x: funct.rotate(x, angle=-90),
                        transforms.RandomHorizontalFlip(p),
                        transforms.Resize(dim),
                        transforms.ToTensor(),
                        lambda x: x.view(-1),
                        lambda x: x.numpy()])
        else:
            if space_dims == 28:
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p),
                        transforms.ToTensor(),
                        lambda x: x.view(-1),
                        lambda x: x.numpy()])
            else:
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p),
                        transforms.Resize(dim),
                        transforms.ToTensor(),
                        lambda x: x.view(-1),
                        lambda x: x.numpy()])
        if dataset == 'emnist':
            trainset = torchvision.datasets.EMNIST(root=str(DATA_PATH), train=True,
                                                   download=True, transform=transform,
                                                   split="digits"
                                                   )
        else:
            trainset = torchvision.datasets.MNIST(root=str(DATA_PATH), train=True,
                                                  download=True, transform=transform,
                                                  )

        data = np.array([x for i, (x, label) in enumerate(trainset)])
        labels = np.array([label for i, (x, label) in enumerate(trainset)])
        data = data / np.std(data)
        data, labels = shuffle(data, labels)

        cov = np.cov(data, rowvar=False)
        eigen = np.linalg.eigh(cov)
        cov_eigenvalues = np.real(eigen[0])
        eigenvectors = np.real(eigen[1])
        mu = np.mean(data, axis=0)
        sol = np.linalg.solve(eigenvectors, mu)
        num_trivial_vectors = np.sum((sol < SymmetryFinder.ignore_threshold) & \
                                     (cov_eigenvalues < SymmetryFinder.ignore_threshold))
        print(space_dim, dataset, num_trivial_vectors)
        swaps = int((dim[0] * dim[1] - num_trivial_vectors) // 2)
        # print(type(swaps))
        # Baseline unidirectional symmetry finding
        for bidirectional, heal_eigenvectors in itertools.product([False, True], [False, True]):
            torch.cuda.empty_cache()
            sym_lbl = SymmetryFinderLabel(select_method=swaps,
                                          bidirectional=bidirectional,
                                          heal_eigenvectors=heal_eigenvectors
                                          )
            sym_lbl.fit(data, labels)
            accs = horiz_flip_gt_accuracy(sym_lbl, dim=dim)
            torch.cuda.empty_cache()
            ground_truth_error = sym_lbl.gt_swap_score(data)
            msg = f"label_based {bidirectional=}, {heal_eigenvectors=}: " \
                  f"vector_accuracy={accs[1] * 100:.2f}%, selection_accuracy={accs[0] * 100:.2f}% " \
                  f"ground truth MSE on data: {ground_truth_error:.4f}"
            acc_records.append(
                {"dataset": dataset, "space_dim": space_dim, "method": "label_based", "p_diff": p_diff,
                "bidirectional": bidirectional, "heal_eigenvectors": heal_eigenvectors,
                "vector_accuracy": accs[1],
                "selection_accuracy": accs[0],
                "gt_data_mse": ground_truth_error.cpu().item()
                 }
            )
            logging.info(msg)
            print(msg)
        # df_acc.loc[space_dim, method] = accs[0]
        # df_cov_acc.loc[dataset, space_dim] = accs[1]
    pd.DataFrame(acc_records).to_csv(RESULT_DATA_PATH / "robustness_under_unequal_symmetry.csv", index=False, header=True)
