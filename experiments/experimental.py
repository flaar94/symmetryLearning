import torch
import numpy as np
import matplotlib.pyplot as plt
from core import SymmetryFinder, SymmetryFinder2, SymmetryFinderMeanBS, train_unsup
from synthetic_dataset_builder import create_weak_gumbel_dataset
from scipy.linalg import block_diag
from numpy.random import default_rng
import itertools
import logging
import pandas as pd
import time

logging.basicConfig(filename="../logs/experimental.log", level=logging.INFO)

rng = default_rng(42)

DATA_PATH = '../data/weak_dataset.npy'
# NUM_SAMPLES = 50_000
# SPACE_DIM = 30
TESTS = 10


def ground_truth_loss(mat1, mat2):
    return np.sqrt(np.mean((mat1 - mat2) ** 2))


if __name__ == '__main__':
    # num_samples = NUM_SAMPLES
    tests = TESTS
    space_dims = [100, 200, 400]
    num_samples_list = [10_000, 50_000]
    df_error = pd.DataFrame(columns=space_dims, index=num_samples_list, dtype=str)
    print(df_error)
    for space_dim, num_samples in itertools.product(space_dims, num_samples_list):
        sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(space_dim // 2)])
        errors = [[] for _ in range(3)]
        # error_diffs = []
        # scores = [[] for _ in range(1)]
        # swaps_list = []
        strt_msg = f"Starting Experiment: num_samples = {num_samples}, space_dim = {space_dim}, tests = {tests}"
        print(strt_msg)
        logging.info(strt_msg)
        start_time = time.perf_counter()
        for i in range(tests):
            sample_arr = create_weak_gumbel_dataset(num_samples=num_samples, space_dim=space_dim)
            np.save(DATA_PATH, sample_arr)

            class TrainSet(torch.utils.data.Dataset):
                def __init__(self, path, transform=None):
                    super().__init__()
                    self.path = path
                    self.data = np.load(path)
                    self.rows = self.data.shape[0]
                    self.cols = self.data.shape[1]
                    self.transform = transform

                def __len__(self):
                    return self.rows

                def __getitem__(self, idx):
                    sample = torch.tensor(self.data[idx], dtype=torch.float)
                    if self.transform:
                        sample = self.transform(sample)

                    return sample


            trainset = TrainSet(DATA_PATH)
            trainset.data = trainset.data / np.std(trainset.data)
            data = trainset[:].numpy()
            rng.shuffle(data)
            swaps = space_dim // 2
            sym0 = SymmetryFinder(alpha=0.05, fit_method='loc_corr_adj', select_method=swaps, bootstraps=100,
                                  scoring_sig=3)
            if not i:
                logging.info(f"sym0 = {str(sym0)}")
            # cv = GridSearchCV(sym0, param_grid={'select_method': range(1, SPACE_DIM + 1)},
            #                   cv=RepeatedKFold(n_splits=5, n_repeats=5))
            # cv.fit(data)
            # sym0 = cv.best_estimator_

            sym0.fit(data)

            # lr = 0.1
            # for _ in range(10):
            #     try:
            #         sym0.fine_tune(data, lr=lr, epochs=100, bandwidth=10, weight_penalty_adj=100)
            #         break
            #     except ValueError:
            #         lr *= 0.3
            # else:
            #     Exception("Model Inherently Unstable")

            # print('MMD error:', np.round(cv.cv_results_['mean_test_score'], 2))
            # swaps = int((cv.best_estimator_.trans_eigenvalues_ == -1).sum())
            # swaps = int((sym0.trans_eigenvalues_ == -1).sum())
            # print(swaps)
            # print('num swaps:', swaps)
            # swaps_list.append(swaps)
            #
            # swaps = space_dim // 2
            sym1 = SymmetryFinder(alpha=0.05, fit_method='sign', select_method=swaps, bootstraps=100, scoring_sig=3)
            sym1.fit(data)

            sym2 = SymmetryFinder(alpha=0.05, fit_method='mm_mix', select_method=swaps, bootstraps=100, scoring_sig=3)
            sym2.fit(data)
            #
            # sym3 = SymmetryFinder(alpha=0.05, fit_method='median', select_method=swaps, bootstraps=100, scoring_sig=3)
            # sym3.fit(data)
            #
            # sym4 = SymmetryFinder(alpha=0.05, fit_method='skew', select_method=swaps, bootstraps=100, scoring_sig=3)
            # sym4.fit(data)
            #
            errors[0].append(ground_truth_loss(sym0.trans_, sym_mat))
            # errors[1].append(ground_truth_loss(sym1.trans_, sym_mat))
            # print(errors[0][-1])
            # error_diffs.append(ground_truth_loss(sym0.trans_, sym_mat) - ground_truth_loss(sym0.ft_trans_, sym_mat))
            errors[1].append(ground_truth_loss(sym1.trans_, sym_mat))
            errors[2].append(ground_truth_loss(sym2.trans_, sym_mat))
            # errors[3].append(ground_truth_loss(sym3.trans_, sym_mat))
            # errors[4].append(ground_truth_loss(sym4.trans_, sym_mat))
            # print('ground truth error for variable:', errors[0][-1])
            # print(f'ground truth error for {SPACE_DIM // 2}: {errors[1][-1]}')

            # scores[0].append(sym0.score(data))
            # scores[1].append(sym1.score(data))
            # scores[2].append(sym2.score(data))
            # print(scores[0][-1], errors[0][-1])
        print(time.perf_counter() - start_time)
        logging.info(str(errors))

        accs = [str(len([0 for x in error if x < 0.20 * (10 / space_dim)]) / tests) for error in errors]
        acc_msg = f"Accuracies: {', '.join(accs)}"
        print(acc_msg)
        logging.info(acc_msg)

        char_errors = [f'{np.round(np.mean(error), 3)} +/- {np.round(np.std(error) / np.sqrt(tests), 3)}' for error in errors]
        err_msg = f"Errors: {', '.join(char_errors)}"
        print(err_msg)
        logging.info(err_msg)
        df_error.loc[
            num_samples, space_dim] = f'{np.round(np.mean(errors[0]), 3)} $\\pm$ {np.round(np.std(errors[0]) / np.sqrt(tests), 3)}'
        # print(f'error diffs: '
        #       f'{np.round(np.mean(errors[0]), 3)} +/- {np.round(np.std(errors[0]) / np.sqrt(tests), 3)} '
        #       f'{np.round(np.mean(errors[1]), 3)} +/- {np.round(np.std(errors[1]) / np.sqrt(tests), 3)} '
              # f'{np.round(np.mean(errors[2]), 3)} +/- {np.round(np.std(errors[2])/np.sqrt(tests), 3)} '
        #       # f'{np.round(np.mean(errors[3]), 3)} +/- {np.round(np.std(errors[3]) / np.sqrt(tests), 3)} '
        #       # f'{np.round(np.mean(errors[4]), 3)} +/- {np.round(np.std(errors[4]) / np.sqrt(tests), 3)} '
        #       )

        # print('scores', np.round(np.mean(scores[0]), 2),
        #       np.round(np.mean(scores[1]), 2),
        #       np.round(np.mean(scores[2]), 2)
        #       )
        # fig, ax = plt.subplots(1, 5)
        # fig.suptitle("Histogram of Errors", fontsize=14)
        # bins = 30
        # ax[0].hist(errors[0], bins=bins)
        # ax[0].set_title("Mean")
        # ax[0].set_ylabel("Number of Tests")
        #
        #
        # ax[1].hist(errors[1], bins=bins)
        # ax[1].set_title("Median")
        # ax[1].set_xlabel("MMD error (SE Kernel $\sigma=3$)")
        # # ax[1].title('median')
        # ax[2].hist(errors[2], bins=bins)
        # ax[2].set_title("MM Mix")
        # # ax[2].title('mean_median mix')
        # ax[3].hist(errors[3], bins=bins)
        # ax[3].set_title("Sign")
        #
        # ax[4].hist(errors[4], bins=bins)
        # ax[4].set_title("Skew")
        # fig.tight_layout()
        # plt.show()
        # fig.savefig("../figures/histogram_of_errors.png")

        # fig, ax = plt.subplots(1, 3)
        # ax[0].hist(scores[0], bins=bins)
        # ax[1].hist(scores[1], bins=bins)
        # ax[2].hist(scores[2], bins=bins)
        # plt.show()
        # plt.hist(swaps_list, bins=bins)
        # plt.show()
        # fig, ax = plt.subplots(1, 5)
        # plt.suptitle("Histogram of Errors", fontsize=14)
        print(df_error)

        bins = 30
        plt.hist(errors[0], bins=bins)
        plt.title("Histogram of Errors")
        plt.ylabel("Number of Tests")
        plt.xlabel("Ground Truth Error")
        plt.show()

        # bins = 30
        # plt.hist(errors[1], bins=bins)
        # plt.title("Histogram of Errors (Model Known)")
        # plt.ylabel("Number of Tests")
        # plt.xlabel("Ground Truth Error")
        # plt.show()
        #
        # plt.hist(swaps_list, bins=10)
        # plt.title("Histogram of Swaps")
        # plt.ylabel("Number of Tests")
        # plt.xlabel("Swaps")
        # plt.show()

        # swap_counts = Counter(swaps_list)
        # print(swap_counts)

        # bins = 30
        # plt.hist(error_diffs, bins=bins)
        # plt.title("Histogram of Fine Tuning Error Improvements")
        # plt.ylabel("Number of Tests")
        # plt.xlabel("Ground Truth Error")
        # plt.show()

        # with open('../data/fine_tune_data.pkl', 'bw') as f:
        #     dill.dump(errors, f)