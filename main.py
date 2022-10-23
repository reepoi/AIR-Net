# Image type matrix completion
# Loss fixed points
import csv
from collections import namedtuple
import enum
from MinPy import demo, loss as lossm, net, reg
import torch
from MinPy.toolbox import dataloader, plot, pprint
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


rng = np.random.RandomState(seed=20210909)

Shape = namedtuple('Shape', ['rows', 'cols'])
Regularizer = namedtuple('RegularizerSet', ['weight_decay', 'regularizer', 'optimizer'])


def csv_to_tensor(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        rows = []
        for row in reader:
            rows.append([float(f) for f in row])
    return torch.tensor(rows)


def write_csv(matrix, filename):
    np.savetxt(f'{filename}.csv', matrix, delimiter=',')

    
def get_bit_mask(matrix, rate):
    return torch.tensor(rng.random(matrix.shape) > rate).int().to(device)


def total_variation_regularization(weight_decay, dimension, similarity_type):
    regularizer = reg.TotalVariationRegularization(dimension, similarity_type).to(device)
    optimizer = None
    # optimizer = torch.optim.Adam(regularizer.parameters())
    return Regularizer(weight_decay=weight_decay, regularizer=regularizer, optimizer=optimizer)
    
    
def dirichlet_energy_regularization(weight_decay, dimension, similarity_type):
    regularizer = reg.DirichletEnergyRegularization(dimension, similarity_type).to(device)
    optimizer = torch.optim.Adam(regularizer.parameters())
    return Regularizer(weight_decay=weight_decay, regularizer=regularizer, optimizer=optimizer)


def distance_regularization(weight_decay, points, sigma):
    regularizer = reg.DistanceRegularization(points, sigma)
    return Regularizer(weight_decay=weight_decay, regularizer=regularizer, optimizer=None)


def paper_regularization(weight_decay, dimension, similarity_type):
    regularizer = reg.auto_reg(dimension, similarity_type)
    return Regularizer(weight_decay=weight_decay, regularizer=regularizer, optimizer=None)


def run_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=None):
    if not regularizers:
        regularizers = []

    model = net.MyDeepMatrixFactorization(matrix_factor_dimensions).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    height, width = matrix.shape

    nmae_losses = []
    for e in range(epochs):

        # Compute prediction error
        reconstructed_matrix = model(matrix * mask)
        loss = (
            lossm.mse(reconstructed_matrix, matrix, mask)
            + sum((r.weight_decay * r.regularizer(reconstructed_matrix) for r in regularizers), start=0)
        )

        # Backpropagation
        optimizer.zero_grad()
        regularizer_optimizers = [r.optimizer for r in regularizers if r.optimizer]
        for o in regularizer_optimizers:
            o.zero_grad()

        loss.backward()

        optimizer.step()
        for o in regularizer_optimizers:
            o.step()

        nmae_losses.append(lossm.nmae(reconstructed_matrix, matrix, mask).detach().cpu().numpy())

        if e % 100 == 0:
            pprint.my_progress_bar(e, epochs, nmae_losses[-1])
        # if e % 5000 == 0:
        #     plot.gray_im(reconstructed_matrix.cpu().detach().numpy())

    return reconstructed_matrix, nmae_losses


def run_paper_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=None):
    if not regularizers:
        regularizers = []

    dmf = demo.BasicDeepMatrixFactorization(matrix_factor_dimensions, [r.regularizer for r in regularizers]) # Define model

    eta = [r.weight_decay for r in regularizers]

    #Training model
    for ite in range(epochs):
        dmf.train(matrix, mu=1, eta=eta, mask_in=mask)

        if ite % 100 == 0:
#             pprint.my_progress_bar(e, epochs, nmae_losses[-1])
            pprint.progress_bar(ite, epochs, dmf.loss_dict) # Format the loss of the output training and print out the training progress bar

        # if ite % 5000 == 0:
        #     plot.gray_im(dmf.net.data.cpu().detach().numpy()) # Display the training image, you can set parameters to save the image

    return dmf.net.data, dmf.loss_dict['nmae_test']


rng = np.random.RandomState(seed=20210909)


if __name__ == '__main__':
    stacked_velocity_data = csv_to_tensor('./amir/vel_2Daneu_crop.csv').to(device)
    t0_data = csv_to_tensor('./amir/vel_2Daneu_crop.0.no_header.csv').to(device)
    point_coordinates = t0_data[:, [3, 4]]
    velocity_data_possible_indices = np.arange(len(stacked_velocity_data))[0::2]
    point_selectors = rng.permutation(100)
    row_indices = velocity_data_possible_indices[point_selectors]
    # print(row_indices)
    matrix = stacked_velocity_data[row_indices]
    matrix_points = point_coordinates[point_selectors]
    for i, p in enumerate(t0_data[point_selectors]):
        assert torch.abs(p[0] - matrix[i,0]) < 1e-3, f'{t0_data[point_selectors][i,0]}, {matrix[i,0]}'
    # matrix = stacked_velocity_data
    matrix = dataloader.get_data(height=240,width=240,pic_name='./train_pics/Barbara.jpg').to(device)
    plot.gray_im(matrix.cpu())
    mask_rate = 0.3
    mask = get_bit_mask(matrix, rate=mask_rate)
    plot.gray_im((matrix * mask).cpu())

    rows, cols = matrix.shape
    matrix_factor_dimensions = [
        Shape(rows=rows, cols=rows),
        Shape(rows=rows, cols=rows),
        Shape(rows=rows, cols=cols)
    ]
    epochs = 10_001
    RCMatrix_PaperDMFAIR, PaperDMFAIR_losses = run_paper_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=[
        paper_regularization(1e-4, rows, 'row'),
        paper_regularization(1e-4, cols, 'col')
    ])
    # RCMatrix_MyDMF, MyDMF_losses = run_test(epochs, matrix_factor_dimensions, matrix, mask)
    # RCMatrix_MyDMFAIR, MyDMFAIR_losses = run_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=[
    #     # distance_regularization(1e-10, matrix_points, 10),
    #     total_variation_regularization(1e-5, rows, reg.TotalVariationRegularization.ROW_SIMILARITY),
    #     # dirichlet_energy_regularization(1e-5, rows, reg.DirichletEnergyRegularizationMode.ROW_SIMILARITY),
    #     # dirichlet_energy_regularization(1e-5, cols, reg.DirichletEnergyRegularizationMode.COL_SIMILARITY)
    # ])