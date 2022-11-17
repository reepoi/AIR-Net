from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


rng = np.random.RandomState(seed=20210909)
# rng = np.random.RandomState(seed=1935912)

Shape = namedtuple('Shape', ['rows', 'cols'])


class DeepMatrixFactorization(nn.Module):
    def __init__(self, matrix_factor_dimensions: Shape):
        super().__init__()
        self.matrix_factors = self.build_matrix_factorization(matrix_factor_dimensions)

    def build_matrix_factorization(self, matrix_factor_dimensions: Shape):
        seq = nn.Sequential()
        init_scale = 1e-3
        depth = len(matrix_factor_dimensions) - 1
        n = matrix_factor_dimensions[0].rows
        scale = init_scale**(1. / depth) * n**(-0.5)
        for d_row, d_col in matrix_factor_dimensions:
            lin = nn.Linear(d_row, d_col, bias=False)
            nn.init.normal_(lin.weight, mean=0, std=scale)
            seq.add_module(str(len(seq)), lin)
        return seq

    def forward(self, _):
        matrix_factors = list(self.matrix_factors.children())
        weight = matrix_factors[0].weight.T
        for c in matrix_factors[1:]:
            weight = c(weight)
        return weight


def mean_sqrd_error(a, b, lib=torch):
    """
    Calculates the mean squared error between two ndarrays.

    Parameters
    ----------
    a: numeric
        An ndarray.
    
    b: numeric
        An ndarray.
    
    lib: module
        The ndarray library to use, i.e. torch or numpy.
    
    Returns
    -------
        float
    """
    return lib.mean((a - b)**2)


def l2_sqrd_error(a, b, lib=torch):
    """
    Calculates the square of the l2 vector norm of the difference
    of two ndarrays of compatible shape.

    Parameters
    ----------
    a: numeric
        An ndarray.
    
    b: numeric
        An ndarray.
    
    lib: module
        The ndarray library to use, i.e. torch or numpy.
    
    Returns
    -------
        float
    """
    return lib.sum((a - b)**2)


def norm_mean_abs_error(yhat, y, lib=torch):
    """
    Calculates the normalized mean absolute error between
    two ndarrays.

    Parameters
    ----------
    yhat: numeric
        The ndarray that is "predicted".
    
    y: numeric
        The groud-truth ndarray.

    lib: module
        The ndarray library to use, i.e. torch or numpy.
    
    Returns
    -------
        float
    """
    return lib.mean(lib.abs(yhat - y)) / lib.mean(lib.abs(y))


def get_bit_mask(shape, rate):
    """
    Creates a matrix of zeros and ones.

    Parameters
    ----------
    shape: iterable
        The size of the matrix to create.
    
    rate: scalar
        The expected percentage of zeros in the matrix.
    
    Returns
    -------
        An ndarray.
    """
    return torch.tensor(rng.random(shape) > rate).int().to(device)


def train(max_epochs, matrix_factor_dimensions, masked_matrix, mask,
             meets_stop_criteria=lambda epoch, loss: False,
             report_frequency=100,
             report=lambda reconstructed_matrix, epoch, loss, last_report: None):
    """
    Run a training loop for the DeepMatrixFactorization model.

    Parameters
    ----------
    max_epochs: int
        The maximum number of iterations of the training loop.
    
    matrix_factor_dimensions: list of Shape
        The dimensions of the matrix factors in the DeepMatrixFactorization model.
    
    masked_matrix: numeric
        The masked ground-truth matrix to be reconstructed.
    
    mask: numeric
        The bit-mask matrix that was used to create ``masked_matrix``.

    meets_stop_criteria: function
        A function that returns a boolean whether to terminate the training loop
        early. This function will be passed the current epoch and loss.

    report_frequency: int
        The number of epochs that should pass before calling the ``report`` function.
    
    report: function
        A function for the caller of ``train`` to view the current state of the
        DeepMatrixFactorization model, and of the training loop. This function will
        be passed the current reconstructed matrix, epoch, loss, and a boolean whether
        it is the last time ``report`` will be called.
    
    Returns
    -------
        The reconstructed matrix produced by the DeepMatrixFactorization model.
    """
    to_report = lambda x: x.detach().cpu().numpy()
    mask.requires_grad_(False)
    masked_matrix.requires_grad_(False)
    mask_nonzeros = mask.nonzero()
    nonzero_mask = lambda x: x[mask_nonzeros]

    model = DeepMatrixFactorization(matrix_factor_dimensions).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for e in range(max_epochs):

        # Compute prediction error
        reconstructed_matrix = model(masked_matrix)
        # loss = 0.5 * l2_sqrd_error(reconstructed_matrix * mask, masked_matrix)
        loss = norm_mean_abs_error(nonzero_mask(reconstructed_matrix), nonzero_mask(masked_matrix))

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        # Gradient step
        optimizer.step()

        if e % report_frequency == 0:
            report(to_report(reconstructed_matrix), e, to_report(loss), False)

        if meets_stop_criteria(e, to_report(loss)):
            break
        
    report(to_report(reconstructed_matrix), e, to_report(loss), True)

    return reconstructed_matrix


def iterated_soft_thresholding(matrix, mask, err=1e-6, normfac=1, insweep=200, tol=1e-4, decfac=0.9,
                               report_frequency=100,
                               report=lambda reconstructed_matrix, epoch, loss, last_report: None):
    reconstructed_matrix = np.zeros(matrix.shape)
    alpha = 1.1 * normfac
    # lam = lambda
    masked = matrix * mask # y
    lam_init = decfac * np.max(np.abs(masked)) # interpretation of matlab code
    lam = np.copy(lam_init)

    loss_func = lambda RCm: np.linalg.norm(masked - mask * RCm, ord=2) + lam * np.linalg.norm(RCm.ravel(), ord=1)

    loss = loss_func(reconstructed_matrix)

    e = 1
    while lam > lam_init * tol:
        for _ in range(insweep):
            loss_prev = loss
            reconstructed_matrix += (masked - mask * reconstructed_matrix) / alpha

            U, S, Vh = np.linalg.svd(reconstructed_matrix, full_matrices=False)
            S = soft_threshold(S, lam / (2 * alpha))
            reconstructed_matrix = np.matmul(U * S, Vh)

            loss = loss_func(reconstructed_matrix)

            if e % report_frequency == 0:
                report(reconstructed_matrix, e, loss, False)

            if np.abs(loss - loss_prev) / np.abs(loss + loss_prev) < tol:
                break

            e += 1
            
        if np.linalg.norm(masked - mask * reconstructed_matrix, ord=2) < err:
            break

        lam *= decfac
        print(lam, lam_init * tol)

    report(reconstructed_matrix, e, loss, True)

    return reconstructed_matrix


def soft_threshold(matrix, threshold):
    return np.sign(matrix) * np.maximum(0, np.abs(matrix) - threshold)