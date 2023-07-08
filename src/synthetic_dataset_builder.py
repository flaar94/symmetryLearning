from sklearn.datasets import make_classification
from scipy.linalg import block_diag
from numpy.random import default_rng
from scipy.stats import invwishart, multivariate_normal, ortho_group, gumbel_r, truncnorm
import itertools
import numpy as np
import torch
from pathlib import Path

rng = default_rng()

SPACE_DIM = 10
SAMPLES = 250_000
CLUSTERS = 2
DATA_DIM_DIV = 1  # if i % DATA_DIM_DIV == 0 then the dimension has larger variance
DENSE = False
GUMBEL_DIST = True
STRONG_SYM = False

DATA_PATH = Path("..") / "data"

GUMBEL_PATH = DATA_PATH / 'gumbel_dataset'
STRICT_PATH = DATA_PATH / 'strict_dataset'
WEAK_GUMBEL_PATH = DATA_PATH / 'weak_dataset'


def random_gl(space_dim=SPACE_DIM):
    base_mat = truncnorm.rvs(-2, 2, size=(space_dim, space_dim))
    upper_mat = np.triu(base_mat)
    lower_mat = np.tril(base_mat)
    diag_spaces = [truncnorm.rvs(0.4, 2) / 4 for _ in range(space_dim)]
    diag_entries = list(itertools.accumulate(diag_spaces))
    diag_entries = rng.permutation(diag_entries)
    for i, entry in enumerate(diag_entries):
        upper_mat[i, i] = entry
        lower_mat[i, i] = 1
    # print(diag_entries)
    return lower_mat @ upper_mat


def create_weak_gumbel_dataset(num_samples=SAMPLES, space_dim=SPACE_DIM):
    samples = []
    sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(space_dim // 2)])
    for x in range(CLUSTERS):
        # distr_trans = ortho_group.rvs(space_dim)
        mu = truncnorm.rvs(-2, 2, size=space_dim) * 10
        distr_trans = random_gl(space_dim=space_dim)
        # print(mu)
        data = gumbel_r.rvs(size=(num_samples // (2 * CLUSTERS), space_dim))
        data = data @ distr_trans + mu
        samples.append(data)
        new_data = gumbel_r.rvs(size=(num_samples // (2 * CLUSTERS), space_dim))
        new_data = new_data @ distr_trans + mu
        transformed_samples = new_data @ sym_mat
        samples.append(transformed_samples)
    sampl_arr = np.concatenate(samples)
    # cov_arr = np.cov(sampl_arr, rowvar=False)
    # print(np.linalg.eig(cov_arr)[0])
    return sampl_arr


if __name__ == '__main__':
    samples = []
    sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(SPACE_DIM // 2)])
    if STRONG_SYM:
        if not GUMBEL_DIST:
            for x in range(CLUSTERS):
                mu = rng.normal(0, 1, size=SPACE_DIM)
                if DENSE:
                    iwh = invwishart(df=SPACE_DIM + 1, scale=np.eye(SPACE_DIM) / 3.)
                else:
                    iwh = invwishart(df=SPACE_DIM + 1,
                                     scale=np.diag([1 if not i % DATA_DIM_DIV else 0.001 for i in range(SPACE_DIM)]))
                cov = iwh.rvs()
                cluster_sampler = multivariate_normal(mean=mu, cov=cov)
                print(mu, cov.diagonal())
                # print(cov)
                # print(cluster_sampler.rvs(size=SAMPLES // (2 * SPACE_DIM)))
                cluster_samples = cluster_sampler.rvs(size=SAMPLES // (2 * CLUSTERS))
                samples.append(cluster_samples)
                transformed_samples = cluster_samples @ sym_mat
                samples.append(transformed_samples)
            # print(np.concatenate(samples).shape)
            np.save(STRICT_PATH, np.concatenate(samples))
        else:
            for x in range(CLUSTERS):
                # distr_trans = ortho_group.rvs(SPACE_DIM)
                mu = truncnorm.rvs(-2, 2, size=SPACE_DIM)
                distr_trans = random_gl()
                print(mu)
                data = gumbel_r.rvs(size=(SAMPLES // (2 * CLUSTERS), SPACE_DIM))
                data = data @ distr_trans + mu
                samples.append(data)
                transformed_samples = data @ sym_mat
                samples.append(transformed_samples)
            # print(np.concatenate(samples).shape)
            np.save(GUMBEL_PATH, np.concatenate(samples))
    else:
        sample_arr = create_weak_gumbel_dataset()
        np.save(WEAK_GUMBEL_PATH, sample_arr)

# strict_dataset, _ = make_classification(n_samples=SAMPLES // 2,
#                                          n_features=SPACE_DIM,
#                                          n_informative=10,
#                                          n_redundant=2,
#                                          n_classes=3,
#                                          n_clusters_per_class=1)
#
# scramble_matrix = ortho_group.rvs(SPACE_DIM)
# strict_dataset = strict_dataset @ scramble_matrix
# sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(SPACE_DIM // 2)])
# strict_dataset = np.concatenate((strict_dataset, strict_dataset @ sym_mat))
# np.save(STRICT_PATH, strict_dataset)

# Need to use data from same make_classification call
# MODEL_PATH = 'model_dataset'
# model_dataset, _ = make_classification(n_samples=SAMPLES // 2,
#                                          n_features=SPACE_DIM,
#                                          n_informative=10,
#                                          n_redundant=2,
#                                          n_classes=3,
#                                          n_clusters_per_class=1)
#
# scramble_matrix = ortho_group.rvs(SPACE_DIM)
# model_dataset = model_dataset @ scramble_matrix
# model_dataset2, _ = make_classification(n_samples=SAMPLES // 2,
#                                          n_features=SPACE_DIM,
#                                          n_informative=10,
#                                          n_redundant=2,
#                                          n_classes=3,
#                                          n_clusters_per_class=1)
# model_dataset2 = model_dataset2 @ scramble_matrix
#
# sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(SPACE_DIM // 2)])
# model_dataset = np.concatenate((model_dataset, model_dataset2 @ sym_mat))
# np.save(MODEL_PATH, model_dataset)
#
# RARE_PATH = 'rare_dataset'
# rare_dataset, _ = make_classification(n_samples=SAMPLES * 9 // 10,
#                                          n_features=SPACE_DIM,
#                                          n_informative=10,
#                                          n_redundant=2,
#                                          n_classes=3,
#                                          n_clusters_per_class=1)
#
# scramble_matrix = ortho_group.rvs(SPACE_DIM)
# rare_dataset = rare_dataset @ scramble_matrix
# rare_dataset2, _ = make_classification(n_samples=SAMPLES // 10,
#                                          n_features=SPACE_DIM,
#                                          n_informative=10,
#                                          n_redundant=2,
#                                          n_classes=3,
#                                          n_clusters_per_class=1)
# rare_dataset2 = rare_dataset2 @ scramble_matrix
#
# sym_mat = block_diag(*[np.array([[0, 1], [1, 0]]) for _ in range(SPACE_DIM // 2)])
# rare_dataset = np.concatenate((rare_dataset, rare_dataset2 @ sym_mat))
# np.save(RARE_PATH, rare_dataset)
