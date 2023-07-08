from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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


def split(X):
    return X[:X.shape[0] // 2], X[X.shape[0] // 2:]


def se_kernel(X, Y, sig2=1, prec=None):

    if prec is None:
        X_norms = torch.mean(X ** 2, dim=1)
        Y_norms = torch.mean(Y ** 2, dim=1)
    else:
        X_norms = torch.mean(X @ prec @ X.t(), dim=1)
        Y_norms = torch.mean(Y @ prec @ Y.t(), dim=1)
    if prec is None:
        prec = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    # print(torch.exp(X_norms.unsqueeze(1) / (2 * sig2)).shape, torch.exp(Y_norms.unsqueeze(0) / (2 * sig2)).shape)
    # print(X.shape, Y.shape)
    if prec is None:
        return torch.exp(X @ Y.t() / (2 * sig2)) / (torch.exp(X_norms.unsqueeze(1) / (2 * sig2)) @
                                                    torch.exp(Y_norms.unsqueeze(0) / (2 * sig2))) * sig2
    else:
        return torch.exp(X @ prec @ Y.t() / (2 * sig2)) / (torch.exp(X_norms.unsqueeze(1) / (2 * sig2)) @
                                                           torch.exp(Y_norms.unsqueeze(0) / (2 * sig2))) * sig2

def poly_kernel(X, Y, r=1, m=2, gamma=0.01, prec=None):
    return (r + gamma * X @ Y.t()) ** m

class MMDLoss(nn.Module):
    def __init__(self, kernel=se_kernel, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.kwargs = kwargs

    def forward(self, X, Y):
        kernel_dists = self.kernel(X, X, **self.kwargs) + self.kernel(Y, Y, **self.kwargs) - 2 * self.kernel(X, Y,
                                                                                                             **self.kwargs)
        loss = torch.mean(kernel_dists)
        return loss


class SplitMMDLoss(nn.Module):
    def __init__(self, kernel=se_kernel, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.kwargs = kwargs

    def forward(self, X, Y):
        X1, X2 = split(X)
        Y1, Y2 = split(Y)
        kernel_dists = self.kernel(X1, X2, **self.kwargs) + self.kernel(Y1, Y2, **self.kwargs) - \
                       self.kernel(X1, Y2, **self.kwargs) - self.kernel(X2, Y1, **self.kwargs)
        loss = torch.mean(kernel_dists)
        return loss


class DebiasedMMDLoss(nn.Module):
    def __init__(self, kernel=se_kernel, cov=None, la=10 ** -5, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.kwargs = kwargs
        self.la = la
        with torch.no_grad():
            self.prec = np.linalg.inv((1 - self.la) * torch.tensor(cov, requires_grad=False) + self.la * torch.eye(cov.shape[0])) if cov is not None else None
        # print(f"before = {self.prec}")

    def add_cov(self, cov):
        with torch.no_grad():
            self.prec = np.linalg.inv((1 - self.la) * torch.tensor(cov, requires_grad=False) + self.la * torch.eye(cov.shape[0])) if cov is not None else None

    def forward(self, X, Y):
        if isinstance(X, np.ndarray) or isinstance(Y, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
            Y = torch.tensor(Y, dtype=torch.float)
        if isinstance(self.prec, np.ndarray):
            self.prec = torch.tensor(self.prec, dtype=X.dtype, device=X.device)

        # print(self.prec)

        kernel_dists = self.kernel(X, X, prec=self.prec, **self.kwargs) + \
                       self.kernel(Y, Y, prec=self.prec, **self.kwargs) - \
                       2 * self.kernel(X, Y, prec=self.prec, **self.kwargs)
        mask = torch.eye(*kernel_dists.shape, device=kernel_dists.device).bool()
        kernel_dists.masked_fill_(mask, 0)
        loss = torch.mean(kernel_dists)
        return loss
