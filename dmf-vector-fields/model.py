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
    return np.array(rng.random(shape) > rate, dtype=int)


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
        loss = 0.5 * l2_sqrd_error(reconstructed_matrix * mask, masked_matrix)
        # loss = norm_mean_abs_error(nonzero_mask(reconstructed_matrix), nonzero_mask(masked_matrix))

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


def iterated_soft_thresholding(masked_matrix, mask, err=1e-6, normfac=1, insweep=200, tol=1e-4, decfac=0.9,
                               report_frequency=100,
                               report=lambda reconstructed_matrix, epoch, loss, last_report: None):
    """
    Matrix Completion via Iterated Soft Thresholding

    .. math::

       \min_X ||X||_* \text{ subject to } \frac{1}{2}||(X - \tilde{X}) \odot M||_F^2 < \varepsilon

    min nuclear-norm(X) subject to ||y - M(X)||_2<err

    Parameters
    ----------
    masked_matrix: numeric
        The masked ground-truth matrix to be reconstructed.
    
    mask: numeric
        The bit-mask matrix that was used to create ``masked_matrix``.
    
    err: positive scalar, default 1e-6
        :math:`\varepsilon` in the constraint of the minization problem.
    
    normfac: scalar, default 1
        The largest eigenvalue of the matrix ``np.matmul(mask.T, mask)``.
        Since ``mask`` is a bit-mask matrix, it should be that ``normfac = 1``.

    insweep: int, default 200
        The maximum number of internal sweeps for solving :math:`||(X - \tilde{X}) \odot M||_F^2 + \lambda||X||_*`
    
    tol: scalar, default 1e-4
        The tolerance for (???).
    
    decfac: scalar, default 0.9
        The decrease factor for cooling :math:`\lambda` (???).

    report_frequency: int, default 100
        The number of epochs that should pass before calling the ``report`` function.
    
    report: function
        A function for the caller of ``iterated_soft_thresholding``
        to view the current state of the solver. This function will
        be passed the current reconstructed matrix, epoch, loss, and
        a boolean whether it is the last time ``report`` will be called.

    Returns
    -------
        The reconstructed matrix produced by the algorithm.
    
    References
    ----------
    .. [1] Majumdar, A.: Singular Value Shrinkage. In: Compressed Sensing for Engineers.
       pp. 110-119. CRC Press/Taylor &amp; Francis, Boca Raton, FL (2019). 
    """
    to_report = lambda x: x.detach().cpu().numpy()

    shape = masked_matrix.shape

    # Vectorize the matrices
    mask = mask.ravel()
    masked_matrix = masked_matrix.ravel()
    reconstructed_matrix = torch.zeros_like(masked_matrix).to(device)

    alpha = 1.1 * normfac
    # lam = lambda
    lam_init = decfac * torch.max(torch.abs(mask * masked_matrix)).item()
    lam = lam_init

    l2 = lambda v: torch.linalg.norm(v)
    l1 = lambda v: torch.linalg.norm(v, ord=1)
    constraint = lambda RCm: l2(masked_matrix - torch.matmul(mask, RCm))
    loss_func = lambda RCm, lam: constraint(RCm) + lam * l1(RCm)

    loss = loss_func(reconstructed_matrix, lam)

    e = 0
    while lam > lam_init * tol:
        for _ in range(insweep):
            loss_prev = loss
            reconstructed_matrix += mask * (masked_matrix - mask * reconstructed_matrix) / alpha

            U, S, Vh = torch.linalg.svd(reconstructed_matrix.reshape(shape), full_matrices=False)

            S = soft_threshold(S, lam / (2 * alpha))
            reconstructed_matrix = torch.matmul(U * S, Vh).ravel()

            loss = loss_func(reconstructed_matrix, lam)

            if e % report_frequency == 0:
                report(to_report(reconstructed_matrix.reshape(shape)), e, loss, False)

            e += 1

            if torch.abs(loss - loss_prev) / torch.abs(loss + loss_prev) < tol:
                break
            
        if constraint(reconstructed_matrix) / 2 < err:
            break

        lam *= decfac

    report(to_report(reconstructed_matrix.reshape(shape)), e, loss, True)

    return reconstructed_matrix.reshape(shape)


def soft_threshold(matrix, threshold):
    return torch.sign(matrix) * torch.maximum(torch.zeros_like(matrix).to(device), torch.abs(matrix) - threshold)
