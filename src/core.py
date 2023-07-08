import itertools
import torch
import dill
import os.path as path
import time
import logging
import numpy as np
from numpy.random import default_rng
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata, ortho_group, skewtest, kurtosis
from metrics import DebiasedMMDLoss, se_kernel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


UTIL_DATA = '../data/util_data.pkl'

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.orthogonal_(m.weight)

class MyInit:
    def __init__(self, fc1, trans):
        self.fc1 = fc1
        self.trans = trans

    def __call__(self, model):
        with torch.no_grad():
            model.fc1.weight.copy_(self.fc1)
            model.trans.copy_(self.trans)


def train_unsup(trainloader, Model, device, Optimizer, criterion, weight_criterion, init=None,
                use_saved=False, epochs=100, save_file='../data/state_dict.pt',
                error_display_stride=1, inter_error_stride=20, optimizer_params=None,
                weight_penalty_adj=3_000):
    model = Model()

    if optimizer_params is None:
        optimizer_params = {}
    optimizer = Optimizer(model.parameters(), **optimizer_params)

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('../logs/train_unsup.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(f"trainloader={trainloader}, Model={model}, device={device}, optimizer={optimizer}, "
                f"criterion={criterion}, use_saved={use_saved}, epochs={epochs}, save_file={save_file}")

    true_epoch = 0
    total_time = 0.
    if use_saved and path.exists(save_file) and path.exists(UTIL_DATA):
        model.load_state_dict(torch.load(save_file))
        with open(UTIL_DATA, 'rb') as f:
            true_epoch, total_time = dill.load(f)
    elif init is not None:
        init(model)

    model.to(device)
    model.trans = model.trans.to(device)
    id_mat = torch.eye(model.fc1.weight.shape[1], requires_grad=False, device=device)
    epoch_loss = 0.0
    true_train_total = 0.0
    try:
        for epoch in range(epochs):
            # loop over the dataset multiple times
            start_time = time.time()
            true_epoch += 1
            running_loss = 0.0
            running_ortho_loss = 0.0
            running_ground_truth_loss = 0.0
            total_train = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs = data
                inputs = inputs.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(inputs, outputs)
                running_loss += loss.detach()
                epoch_loss += loss.detach()

                orth_loss = weight_criterion(model.fc1.weight.t() @ model.fc1.weight, id_mat) * weight_penalty_adj
                running_ortho_loss += orth_loss.detach()

                loss += orth_loss

                total_train += 1
                true_train_total += 1

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                if i % inter_error_stride == (inter_error_stride - 1) and (true_epoch % error_display_stride) == 1:
                    with torch.no_grad():
                        ground_truth = torch.block_diag(
                            *[torch.tensor([[0, 1], [1, 0]], dtype=torch.float, device=device) for _ in
                              range(data.shape[1] // 2)],
                        )
                        mat = model.fc1.weight.t() @ model.trans @ model.fc1.weight
                        ground_truth_loss = torch.sqrt(weight_criterion(mat, ground_truth))
                        determinant = model.fc1.weight.det()
                        ground_truth_loss = ground_truth_loss.detach()
                    # print every n mini-batches
                    partial_err_msg = f'[{true_epoch}, {i + 1}] ' \
                                      f'loss: {running_loss / total_train:.4f}, ' \
                                      f'ortho_loss: {running_ortho_loss / total_train:.4f}, ' \
                                      f'ground_truth_loss: {ground_truth_loss:.4f}, ' \
                                      f'base change det: {determinant:.2f}'
                    print(partial_err_msg)
                    logger.info(partial_err_msg)

                    running_loss = 0.0
                    running_ortho_loss = 0.0
                    total_train = 0.0
            total_time += time.time() - start_time
            if true_epoch % error_display_stride == 1:
                tot_err_msg = f'total error = {epoch_loss / true_train_total:.4f}'
                print(tot_err_msg)
                logger.info(tot_err_msg)
                time_msg = f'Finished epoch, cumulative time: {total_time}s'
                print(time_msg)
                logger.info(time_msg)
                epoch_loss = 0.0
                true_train_total = 0.0

                torch.save(model.state_dict(), save_file)
                with open(UTIL_DATA, 'wb') as f:
                    dill.dump((true_epoch, total_time), f)
    finally:
        torch.save(model.state_dict(), save_file)
        with open(UTIL_DATA, 'wb') as f:
            dill.dump((true_epoch, total_time), f)

    finish_msg = "Finished training!"
    print(finish_msg)
    logger.info(finish_msg)
    return epoch_loss / true_train_total if true_train_total > 0 else float('inf')


class SymmetryFinder(BaseEstimator, RegressorMixin):
    default_bootstraps = 100
    default_alpha = 0.05
    ranking_gamma = 10 ** (-10)
    ignore_threshold = 10 ** (-5)
    default_mean_wt = 0.5
    default_repeats = 5
    device = 'cuda:0'

    def __init__(self, select_method='bs_mean', fit_method='mean', scoring_sig=3, bayes_correct=False,
                 cov_adj_score=False, la=10 ** (-5), tv_ratio=1, seed=42, **kwargs):
        super().__init__()
        if fit_method == 'median':
            fit_method = 'mm_mix'
            kwargs["mean_wt"] = 0

        if select_method == 'bs_mean' or select_method == 'bs_mean_split' or select_method == 'mean_rotate_bs':
            self.bootstraps = kwargs['bootstraps'] if 'bootstraps' in kwargs else self.default_bootstraps
            self.alpha = kwargs['alpha'] if 'alpha' in kwargs else self.default_alpha
        elif isinstance(select_method, int) or (select_method == ''):
            pass
        elif select_method == 'mmd':
            self.repeats = kwargs['repeats'] if 'repeats' in kwargs else self.default_repeats
        else:
            raise ValueError(f"init error: select_method = {select_method} is an invalid parameter")

        if fit_method == 'mm_mix':
            self.mean_wt = kwargs['mean_wt'] if 'mean_wt' in kwargs else self.default_mean_wt

        self.fit_method = fit_method
        self.__select_method = select_method
        self.scoring_sig = scoring_sig
        self.cov_adj_score = cov_adj_score
        self.la = la
        self.scoring_fn = DebiasedMMDLoss(kernel=se_kernel, sig2=self.scoring_sig, la=self.la)
        self.fine_tuned = False
        self.bayes_correct = bayes_correct
        self.random_state = np.random.RandomState(seed)
        self.tv_ratio = tv_ratio

        # Used for partial fitting
        self.scatter_ = torch.zeros(1)
        self.sum_ = torch.zeros(1)
        self.n_ = torch.zeros(1)

    def _find_neg_eigens_bootstrap(self, X):
        pos_bs = []
        neg_bs = []
        for _ in range(self.bootstraps):
            data_bs = resample(X)
            cov_bs = np.cov(data_bs, rowvar=False)

            eigenvalues_bs = np.linalg.eig(cov_bs)[0]
            eigenvectors_bs = np.linalg.eig(cov_bs)[1]
            mu_bs = np.mean(data_bs, axis=0)
            sol_bs = np.linalg.solve(eigenvectors_bs, mu_bs)
            triv_dims_bs = (sol_bs < self.ignore_threshold) & (eigenvalues_bs < self.ignore_threshold)
            filtered_sol_bs = sol_bs * (~triv_dims_bs)
            pos_bs.append((filtered_sol_bs > 0).sum())
            neg_bs.append((filtered_sol_bs < 0).sum())
        # print(pos_bs)
        # print(neg_bs)
        pos_bs.sort()
        neg_bs.sort()
        position = int(self.bootstraps * self.alpha // 2)
        return pos_bs[position], neg_bs[position]

    def _mean_rotate_bs(self, X):
        """
        Randomly rotate the mean, and then shift all data points to move the mean there. The symmetries should be broken.
        Then bound the false discovery rate by alpha.

        :param X:
        :return:
        """
        position = int(self.bootstraps * X.shape[1] * self.alpha)
        if self.fit_method == 'sign':
            diffs = []
            pos_data = X @ self.eigenvectors_
            for _ in range(self.bootstraps):
                g = ortho_group.rvs(dim=X.shape[1])
                shift = self.mu_ @ g - self.mu_
                pos_data_frac_bs = ((pos_data + (shift @ self.eigenvectors_)) > 0).sum(axis=0) / X.shape[0]
                new_diffs = np.abs(pos_data_frac_bs - 0.5)
                diffs.extend(new_diffs)
                # distr = list()
                # distr.sort()
        else:
            raise (Exception(f'{self.fit_method} not implemented for mean rotate bs'))
        diffs.sort()
        return diffs[position]
        # if self.fit_method == 'mean':
        #     for _ in range(self.bootstraps):
        #         g = ortho_group.rvs(dim=X.shape[1])
        #         sol_bs = np.linalg.solve(self.eigenvectors_, self.mu_ @ g)

    def partial_fit(self, X: np.array, y: np.array = None):
        with torch.no_grad():
            self.scatter_ += torch.sum(X @ X.t(), dim=0)
            self.sum_ += torch.sum(X, dim=0)
            self.n_ += X.shape[0]

    def fit(self, X: np.array, y: np.array = None, cov=None):
        if cov is None:
            if X is None:
                cov = (self.scatter_ / (self.n_ - 1) - (self.sum_ / self.n_) ** 2).cpu().numpy()
            else:
                cov = np.cov(X, rowvar=False)

        if self.tv_ratio != 1:
            X_train, X_val = train_test_split(X, train_size=int(X.shape[0] * self.tv_ratio),
                                              random_state=self.random_state)
        else:
            X_train, X_val = X, X

        self.compute_base_stats(X, cov)

        self.make_ranking(X_train)

        if self.cov_adj_score:
            self.scoring_fn.add_cov(cov)

        self.model_selection(X_val)

        return self

    def compute_base_stats(self, X, cov):
        self.dim_ = X.shape[1]
        self.cov_eigenvalues_ = np.maximum(np.real(np.linalg.eig(cov)[0]), 0)
        self.eigenvectors_ = np.real(np.linalg.eig(cov)[1])

        cov_eigenval_ranking = rankdata(self.cov_eigenvalues_, 'ordinal').astype(np.int64) - 1
        self.cov_eigenvalues_[cov_eigenval_ranking] = np.array(self.cov_eigenvalues_)
        self.eigenvectors_[:, cov_eigenval_ranking] = np.array(self.eigenvectors_)

        self.mu_ = np.mean(X, axis=0)

        self.sol_ = np.linalg.solve(self.eigenvectors_, self.mu_)
        self.trivial_vectors_ = (self.sol_ < self.ignore_threshold) & (self.cov_eigenvalues_ < self.ignore_threshold)
        self.sol_ *= ~self.trivial_vectors_

    def make_ranking(self, X: np.array):

        # if self.cov_adj_score:
        #     self.cov_ = cov

        if self.fit_method == 'mean':
            self.stats_ = np.abs(self.sol_ / (np.sqrt(self.cov_eigenvalues_) + self.ranking_gamma))
        elif self.fit_method == 'mm_mix':
            self.stats_ = (1 - self.mean_wt) * np.abs(np.median(X @ self.eigenvectors_, axis=0)) + \
                          self.mean_wt * np.abs(self.sol_) \
                          / (2 * np.sqrt(self.cov_eigenvalues_) + self.ranking_gamma)

        elif self.fit_method == 'skew':
            self.stats_ = -skewtest(X @ self.eigenvectors_)[1]
        elif self.fit_method == 'ms_mix':
            self.stats_ = (skewtest(X @ self.eigenvectors_)[0]
                           + np.abs(self.sol_ / (np.sqrt(self.cov_eigenvalues_) + self.ranking_gamma))) / 2
        elif self.fit_method == 'cov_adj':
            left_shifted = np.abs(self.cov_eigenvalues_ - np.roll(self.cov_eigenvalues_, 1))
            right_shifted = np.abs(self.cov_eigenvalues_ - np.roll(self.cov_eigenvalues_, -1))
            delta_k = np.minimum(left_shifted, right_shifted)
            cov_adj = (1 + np.linalg.norm(self.mu_)) * kurtosis(X @ self.eigenvectors_, fisher=False) / delta_k
            self.stats_ = np.abs(self.sol_ / (np.sqrt(self.cov_eigenvalues_) + cov_adj + self.ranking_gamma))
        elif self.fit_method == 'loc_cov_adj':
            mu_i = self.mu_ @ self.eigenvectors_
            mu_diff = np.maximum((np.abs(mu_i)).reshape(-1, 1) - (np.abs(mu_i)).reshape(1, -1), 0)
            la_diff = np.abs(self.cov_eigenvalues_.reshape(-1, 1) - self.cov_eigenvalues_.reshape(1, -1)) + np.eye(
                X.shape[1])
            X2_cent = (X @ self.eigenvectors_ - mu_i.reshape(1, -1)) ** 2
            expected_cov = np.sqrt(X2_cent.T @ X2_cent / X.shape[0])
            comb = mu_diff * expected_cov / (la_diff + self.ranking_gamma)
            comb = np.nan_to_num(comb, nan=0.0, posinf=0.0, neginf=0.0)
            tot = np.sum(comb, axis=0)
            self.stats_ = np.abs(self.sol_ / (np.sqrt(self.cov_eigenvalues_) + tot + self.ranking_gamma))
        elif self.fit_method == 'loc_corr_adj':
            mu_i = self.mu_ @ self.eigenvectors_
            mu_diff = np.maximum((np.abs(mu_i)).reshape(-1, 1) - (np.abs(mu_i)).reshape(1, -1), 0)
            la_diff = np.abs(self.cov_eigenvalues_.reshape(-1, 1) - self.cov_eigenvalues_.reshape(1, -1)) + np.eye(
                X.shape[1])
            X_i = X @ self.eigenvectors_
            X2_cent = (X_i - mu_i.reshape(1, -1)) ** 2
            expected_cov = X2_cent.T @ X2_cent / (X.shape[0] *
                                                  np.std(X_i, axis=0).reshape(-1, 1) *
                                                  np.std(X_i, axis=0).reshape(1, -1) + self.ranking_gamma)
            # Prevent trivial dimensions from creating NaN problems
            expected_cov = np.nan_to_num(expected_cov, nan=0.0)
            comb = mu_diff * expected_cov / (la_diff + self.ranking_gamma)
            tot = np.sum(comb, axis=0)
            self.stats_ = np.abs(self.sol_) / (np.sqrt(self.cov_eigenvalues_) + tot + self.ranking_gamma)
        elif self.fit_method == 'corr_adj_eb':
            mu_i = self.mu_ @ self.eigenvectors_
            mu_diff = np.maximum((np.abs(mu_i)).reshape(-1, 1) - (np.abs(mu_i)).reshape(1, -1), 0)
            la_diff = np.abs(self.cov_eigenvalues_.reshape(-1, 1) - self.cov_eigenvalues_.reshape(1, -1)) + np.eye(
                X.shape[1])
            X_i = X @ self.eigenvectors_
            X2_cent = (X_i - mu_i.reshape(1, -1)) ** 2
            expected_cov = X2_cent.T @ X2_cent / (X.shape[0] *
                                                  np.std(X_i, axis=0).reshape(-1, 1) *
                                                  np.std(X_i, axis=0).reshape(1, -1))
            # Prevent trivial dimensions from creating NaN problems
            expected_cov = np.nan_to_num(expected_cov, nan=0.0)
            comb = mu_diff * expected_cov / la_diff
            tot = np.sum(comb, axis=0)
            psi = (np.sqrt(self.cov_eigenvalues_) + self.ranking_gamma) ** 2
            phi = 2 * np.exp(-2.38) * self.cov_eigenvalues_ ** 0.8 + psi / X.shape[0]
            # print((X.shape[0]/psi)[-50:])
            # print((1/phi)[-50:])
            print(np.log(phi / psi)[-50:])
            print(np.log(X.shape[0]))
            self.stats_ = np.abs(self.sol_) * np.sqrt(X.shape[0] / psi - 1 / phi) / np.sqrt(
                (np.log(phi / psi) + np.log(X.shape[0])))
        elif self.fit_method == 'sign':
            pos_data_frac = ((X @ self.eigenvectors_) > 0).sum(axis=0) / X.shape[0]
            self.stats_ = np.abs(pos_data_frac - 0.5)
            if self.select_method == 'bs_mean' or isinstance(self.select_method, int):
                pass
        if self.bayes_correct:
            self.stats_ /= (np.sqrt(self.cov_eigenvalues_) + self.ranking_gamma)

        self.stats_[self.trivial_vectors_] = np.inf
        self.ranking_ = rankdata(self.stats_)

    def model_selection(self, X=None):
        if (self.select_method == 'bs_mean' or self.select_method == 'bs_mean_split'):
            num_pos, num_neg = self._find_neg_eigens_bootstrap(X)
            num_static = num_pos + num_neg
            # print(X.shape[1] - num_static)
        elif isinstance(self.select_method, int):
            num_static = self.dim_ - self.select_method
            # print(num_static)
        elif (self.select_method == 'mmd') or (self.select_method is None):
            pass
        else:
            raise ValueError(f"fitting error: select_method = {self.select_method} is an invalid parameter")

        if self.select_method == 'bs_mean' or isinstance(self.select_method, int):
            static_vectors = self.ranking_ >= self.dim_ + 1 - num_static
        elif self.select_method == 'bs_mean_split':
            static_vectors = ((self.ranking_ <= num_neg) | (self.ranking_ >= self.dim_ + 1 - num_pos))
        elif self.select_method == 'mmd':
            if X is None:
                raise ValueError("Cannot change select method to 'mmd' after fitting")
            num_trivial = np.sum(self.trivial_vectors_)
            self.mmd_errors_ = [0. for _ in range(num_trivial + 1, self.dim_ + 1)]
            for num_swaps in range(num_trivial + 1, self.dim_ + 1):

                static_vectors = self.ranking_ >= num_swaps + 1
                static_vectors |= self.trivial_vectors_
                self.trans_eigenvalues_ = static_vectors * 2. - 1
                # print(self.trans_eigenvalues_)
                self.trans_ = self.eigenvectors_ @ np.diag(self.trans_eigenvalues_) @ self.eigenvectors_.T

                for _ in range(self.repeats):
                    self.mmd_errors_[num_swaps - (num_trivial + 1)] += self.score(X)
                self.mmd_errors_[num_swaps - (num_trivial + 1)] /= self.repeats

                print(num_swaps, self.mmd_errors_[num_swaps - (num_trivial + 1)])

            # clear temporary values in order to be consistent with other methods
            del self.trans_eigenvalues_
            del self.trans_
            print(self.mmd_errors_)
            best_num_swaps = np.argmax(self.mmd_errors_) + num_trivial + 2
            static_vectors = self.ranking_ >= best_num_swaps
        else:
            raise ValueError(f'Invalid method {self.select_method}')

        static_vectors |= self.trivial_vectors_
        self.trans_eigenvalues_ = static_vectors * 2. - 1
        self.trans_ = self.eigenvectors_ @ np.diag(self.trans_eigenvalues_) @ self.eigenvectors_.T

    @property
    def select_method(self):
        return self.__select_method

    @select_method.setter
    def select_method(self, value):
        self.__select_method = value
        self.model_selection()

    def predict(self, X, use_fine_tuned=True):
        if use_fine_tuned and self.fine_tuned:
            return X @ self.ft_trans_.T
        else:
            return X @ self.trans_.T

    def fine_tune(self, X, lr=0.01, epochs=100, weight_penalty_adj=3_000, bandwidth=10):
        BATCH_SIZE = 1_024
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float, device=self.device))
        # tensor_trans = torch.tensor(self.trans_.T, dtype=torch.double, device=self.device)
        d = X.shape[1]

        class LinearNN(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim, bias=False)
                self.trans = torch.block_diag(
                    *[torch.tensor([[0, 1], [1, 0]], dtype=torch.float) for _ in range(dim // 2)])

            def forward(self, x):
                x = self.fc1(x)
                x = F.linear(x, self.trans)
                x = F.linear(x, self.fc1.weight.t())
                return x

        linear_nn = LinearNN(d)
        linear_nn.to(self.device)
        linear_nn.trans = linear_nn.trans.to(self.device)
        my_init = MyInit(torch.tensor(self.eigenvectors_.T, device=self.device),
                         torch.tensor(np.diag(self.trans_eigenvalues_), device=self.device))
        my_init(linear_nn)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # criterion = DebiasedMMDLoss(kernel = poly_kernel, r=1, m=2, gamma=0.3)
        criterion = DebiasedMMDLoss(kernel=se_kernel, sig2=bandwidth)
        optimizer = optim.SGD(linear_nn.parameters(), lr=lr, momentum=0.5)
        weight_criterion = nn.MSELoss()
        true_epoch = 0
        id_mat = torch.eye(linear_nn.fc1.weight.shape[1], requires_grad=False, device=self.device)
        for epoch in range(epochs):
            # loop over the dataset multiple times
            true_epoch += 1
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, = data
                inputs = inputs.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = linear_nn(inputs)
                loss = criterion(inputs, outputs)

                orth_loss = weight_criterion(linear_nn.fc1.weight.t() @ linear_nn.fc1.weight,
                                             id_mat) * weight_penalty_adj

                loss += orth_loss
                loss.backward()
                optimizer.step()
                if torch.isnan(linear_nn.fc1.weight).any():
                    raise ValueError("Fine Tune failure. NaNs in the model")

        self.ft_trans_ = (linear_nn.fc1.weight.t() @ linear_nn.trans @ linear_nn.fc1.weight).detach().cpu().numpy()
        self.fine_tuned = True

    def score(self, X, y=None, sample_weight=None):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.double, device=self.device))
        dataloader = DataLoader(dataset, batch_size=1_024, shuffle=True)
        tensor_trans = torch.tensor(self.trans_.T, dtype=torch.double, device=self.device)
        with torch.no_grad():
            running_loss = torch.tensor(0., dtype=torch.double, device=self.device, requires_grad=False)
            for batch in dataloader:
                batch_data = batch[0]
                running_loss += self.scoring_fn(batch_data @ tensor_trans, batch_data).detach()

        return -running_loss.cpu().numpy()


class SymmetryFinderLabel(SymmetryFinder):
    def __init__(self, select_method='', fit_method='', **kwargs):
        kwargs['select_method'] = select_method
        kwargs['fit_method'] = fit_method
        super().__init__(**kwargs)

    def fit(self, X: np.array, y: np.array = None, overall_cov=None):

        if self.tv_ratio != 1:
            X_train, X_val = train_test_split(X, train_size=int(X.shape[0] * self.tv_ratio),
                                              random_state=self.random_state)
        else:
            X_train, X_val = X, X

        if overall_cov is None:
            if X is None:
                overall_cov = (self.scatter_ / (self.n_ - 1) - (self.sum_ / self.n_) ** 2).cpu().numpy()
            else:
                overall_cov = np.cov(X_train, rowvar=False)

        self.compute_base_stats(X_train, overall_cov)

        self.num_labels_ = len(np.unique(y))

        theta_tilde = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
        for label in np.unique(y):
            X_label = X_train[y == label]
            mu_label = self.eigenvectors_.T @ np.mean(X_label, axis=0)
            cov_label = np.abs(self.eigenvectors_.T @ np.cov(X_label, rowvar=False) @ self.eigenvectors_)
            corr_label = cov_label / (np.sqrt(cov_label.diagonal() @ cov_label.diagonal().T) + self.ranking_gamma)
            se_label = mu_label ** 2 / (cov_label.diagonal() + self.ranking_gamma)

            theta_tilde[:-1, :-1] += ((np.log(1 + corr_label + self.ranking_gamma) -
                                      np.log(1 - corr_label + self.ranking_gamma)) / 2) ** 2
            theta_tilde[-1, :-1] += se_label
            theta_tilde[:-1, -1] += se_label
        theta_tilde /= self.num_labels_
        theta_tilde -= np.diag(theta_tilde.diagonal())

        self._dissimilarity_analysis(theta_tilde)
        print(theta_tilde[-4:, -4:])
        # reindexer = np.array([pos - 1 if x == 1 else X.shape[1] - neg for x, pos, neg in
        #        zip(self.trans_eigenvalues_,
        #            itertools.accumulate((self.trans_eigenvalues_ == 1).astype(int)),
        #            itertools.accumulate((self.trans_eigenvalues_ == -1).astype(int)))])
        # print(theta_tilde[:-1, :-1][reindexer][:, reindexer])
        # plt.imshow(theta_tilde[:-1, :-1][reindexer][:, reindexer])
        # plt.vlines(sum(self.trans_eigenvalues_ == 1), ymin=0, ymax=X.shape[1] - 1)
        # plt.hlines(sum(self.trans_eigenvalues_ == 1), xmin=0, xmax=X.shape[1] - 1)
        # plt.show()
        return self

    def _dissimilarity_analysis(self, theta):
        # Start the fixed point cluster with the auxiliary node
        static_vectors = np.array([False if i < theta.shape[0] - 1 else True for i in range(theta.shape[0])])
        for i in range(theta.shape[0]):
            # Create matrix of distances between and within clusters, then compute average distance
            between_sim = np.mean(theta[:, static_vectors], axis=1)
            within_sim = np.mean(theta[:, ~static_vectors], axis=1)

            diff = (between_sim - within_sim) * ~static_vectors
            if isinstance(self.select_method, int):
                static_vectors[np.argmax(diff)] = True
                if self.select_method == theta.shape[0] - 2 - i:
                    break
            else:
                if max(diff) > 0:
                    print(between_sim[np.argmax(diff)], within_sim[np.argmax(diff)])
                    static_vectors[np.argmax(diff)] = True
                else:
                    break
        else:
            raise TimeoutError("Error, clustering algorithm failed to terminate")
        # Remove auxiliary node
        static_vectors = static_vectors[:-1]

        static_vectors |= self.trivial_vectors_
        self.trans_eigenvalues_ = static_vectors * 2. - 1
        # print(self.trans_eigenvalues_)
        self.trans_ = self.eigenvectors_ @ np.diag(self.trans_eigenvalues_) @ self.eigenvectors_.T


class SymmetryFinder2(BaseEstimator, RegressorMixin):
    default_bootstraps = 200
    default_alpha = 0.05
    ranking_gamma = 0
    ignore_threshold = 10 ** (-5)

    def __init__(self, fit_method='mean', **kwargs):
        super().__init__()
        self.bootstraps = kwargs['bootstraps'] if 'bootstraps' in kwargs else self.default_bootstraps
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else self.default_alpha
        self.fit_method = fit_method

    def _bootstrap(self, X):
        eigen_dists = []
        univ_stats = [[] for _ in range(X.shape[1])]
        nearest_neighbor = NearestNeighbors(n_neighbors=1)
        nearest_neighbor.fit(self.eigenvectors_.T)
        # for _ in range(self.bootstraps):
        for _ in range(self.bootstraps):
            data_bs = resample(X)
            cov_bs = np.cov(data_bs, rowvar=False)

            # eigenvalues_bs = np.linalg.eig(cov_bs)[0]
            eigenvectors_bs = np.linalg.eig(cov_bs)[1]
            if self.fit_method == 'mean':
                mu_bs = np.mean(data_bs, axis=0)
                sol_bs = np.linalg.solve(eigenvectors_bs, mu_bs)

                dists, eigen_corresp = nearest_neighbor.kneighbors(eigenvectors_bs.T)
                # We reorder dists and sol_bs to match the indices of self.eigenvectors_ instead of eigenvectors_bs
                eigen_dists.append(dists[eigen_corresp].reshape(-1))
                sols = sol_bs[eigen_corresp].reshape(-1)
                for i, dist in enumerate(eigen_dists[-1]):
                    # Add in only the statistics when there is a corresponding eigenvector
                    if dist < 0.5:
                        univ_stats[i].append(sols[i])
        for univ_stat in univ_stats:
            univ_stat.sort()
        eigen_dists = np.sort(np.stack(eigen_dists), axis=0)
        return eigen_dists, univ_stats

    def fit(self, X: np.array, y: np.array = None):
        cov = np.cov(X, rowvar=False)
        self.cov_eigenvalues_ = np.real(np.linalg.eig(cov)[0])
        self.eigenvectors_ = np.real(np.linalg.eig(cov)[1])
        self.mu_ = np.mean(X, axis=0)

        self.sol_ = np.linalg.solve(self.eigenvectors_, self.mu_)
        self.trivial_vectors_ = (self.sol_ < self.ignore_threshold) & (self.cov_eigenvalues_ < self.ignore_threshold)
        self.sol_ *= ~self.trivial_vectors_

        if self.fit_method == 'mean':
            eigen_dists, univ_stats = self._bootstrap(X)
            self.indistinguishable_ = (np.median(eigen_dists, axis=0) > 0.5).reshape(-1)
            self.univ_stats_ = univ_stats
            self.eigen_dists_ = eigen_dists
            static_vectors = []
            for i, univ_stat in enumerate(univ_stats):
                position = int(len(univ_stat) * self.alpha / 2)
                if (univ_stat[position] * univ_stat[-position - 1]) > 0:
                    static_vectors.append(True)
                else:
                    static_vectors.append(False)
            static_vectors = np.array(static_vectors, dtype=np.bool)
        else:
            raise ValueError(f"fit_method = {self.fit_method} invalid")
        static_vectors |= self.trivial_vectors_
        self.trans_eigenvalues_ = static_vectors * 2. - 1
        self.trans_ = self.eigenvectors_ @ np.diag(self.trans_eigenvalues_) @ self.eigenvectors_.T

        return self

    def predict(self, X):
        return X @ self.trans_.T


class SymmetryFinderMeanBS(BaseEstimator, RegressorMixin):
    default_bootstraps = 100
    default_alpha = 0.05
    ranking_gamma = 0
    ignore_threshold = 10 ** (-5)

    def __init__(self, fit_method='mean', **kwargs):
        super().__init__()
        self.bootstraps = kwargs['bootstraps'] if 'bootstraps' in kwargs else self.default_bootstraps
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else self.default_alpha
        self.fit_method = fit_method

    def _bootstrap(self, X):
        eigen_dists = []
        univ_stats = []
        # for _ in range(self.bootstraps):
        for _ in range(self.bootstraps):
            data_bs = resample(X)
            # cov_bs = np.cov(data_bs, rowvar=False)
            #
            # eigenvalues_bs = np.linalg.eig(cov_bs)[0]
            # eigenvectors_bs = np.linalg.eig(cov_bs)[1]
            if self.fit_method == 'mean':
                mu_bs = np.mean(data_bs, axis=0)
                sol_bs = np.linalg.solve(self.eigenvectors_, mu_bs)
                univ_stats.append(sol_bs.reshape(-1))
        univ_stats = np.sort(np.stack(univ_stats), axis=0)
        return univ_stats

    def fit(self, X: np.array, y: np.array = None):
        cov = np.cov(X, rowvar=False)
        self.cov_eigenvalues_ = np.real(np.linalg.eig(cov)[0])
        self.eigenvectors_ = np.real(np.linalg.eig(cov)[1])
        self.mu_ = np.mean(X, axis=0)

        self.sol_ = np.linalg.solve(self.eigenvectors_, self.mu_)
        self.trivial_vectors_ = (self.sol_ < self.ignore_threshold) & (self.cov_eigenvalues_ < self.ignore_threshold)
        self.sol_ *= ~self.trivial_vectors_

        if self.fit_method == 'mean':
            univ_stats = self._bootstrap(X)
            self.univ_stats_ = univ_stats
            position = int(self.bootstraps * self.alpha / 2)
            static_vectors = (univ_stats[position] * univ_stats[-position - 1]) > 0
        else:
            raise ValueError(f"fit_method = {self.fit_method} invalid")
        static_vectors |= self.trivial_vectors_
        self.trans_eigenvalues_ = static_vectors * 2. - 1
        self.trans_ = self.eigenvectors_ @ np.diag(self.trans_eigenvalues_) @ self.eigenvectors_.T
        return self

    def predict(self, X):
        return X @ self.trans_.T
