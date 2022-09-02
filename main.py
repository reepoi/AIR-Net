# Image type matrix completion
# Loss fixed points
import csv
import enum
from MinPy import demo, loss as lossm, net, reg
import torch
from MinPy.toolbox import dataloader, plot, pprint
import numpy as np

class MissMode(enum.Enum):
    RANDOM = enum.auto() # Random pixels are set to zero
    PATCH = enum.auto() # Entire pixel blocks are set to zero
    FIXED = enum.auto() # A mask from a file is used to determine with pixels are set to zero

class Algorithm(enum.Enum):
    DMF = enum.auto()
    DMF_AIR = enum.auto()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
HEIGHT = 240
WIDTH = 49
EPOCHS = 30_001
np.random.seed(20210909)

def csv_to_tensor(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        rows = []
        for row in reader:
            rows.append([float(f) for f in row])
    return torch.tensor(rows)


def train_my_dmf(model, loss_fn, optimizer, matrix, mask):
    nmae_losses = []
    model.train()
    for e in range(EPOCHS):

        # Compute prediction error
        reconstructed_matrix = model(matrix * mask)
        loss = loss_fn(reconstructed_matrix, matrix, mask)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nmae_losses.append(lossm.nmae(reconstructed_matrix, matrix, mask).detach().cpu().numpy())

        if e % 100 == 0:
            pprint.my_progress_bar(e, EPOCHS, nmae_losses[-1])
        if e % 5000 == 0:
            plot.gray_im(reconstructed_matrix.cpu().detach().numpy())

    return reconstructed_matrix, nmae_losses


def train_my_dmf_air(model, loss_fn, optimizer, matrix, mask):
    nmae_losses = []
    model.train()
    regularizer_row = reg.auto_reg(HEIGHT, 'row')
    regularizer_col = reg.auto_reg(WIDTH, 'col')

    for e in range(EPOCHS):

        # Compute prediction error
        reconstructed_matrix = model(matrix * mask)
        loss = (
            loss_fn(reconstructed_matrix, matrix, mask)
            + 1e-4 * regularizer_row.init_data(reconstructed_matrix)
            + 1e-4 * regularizer_col.init_data(reconstructed_matrix)
        )

        # Backpropagation
        regularizer_row.opt.zero_grad()
        regularizer_col.opt.zero_grad()
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        regularizer_row.update(reconstructed_matrix)
        regularizer_col.update(reconstructed_matrix)

        nmae_losses.append(lossm.nmae(reconstructed_matrix, matrix, mask).detach().cpu().numpy())

        if e % 100 == 0:
            pprint.my_progress_bar(e, EPOCHS, nmae_losses[-1])
        if e % 5000 == 0:
            plot.gray_im(reconstructed_matrix.cpu().detach().numpy())

    return reconstructed_matrix, nmae_losses


def train_dmf_air(matrix_dimensions, matrix, mask):
    reg_row = reg.auto_reg(HEIGHT, 'row')
    reg_col = reg.auto_reg(WIDTH, 'col')
    regularizers = [reg_row, reg_col]
    dmf = demo.BasicDeepMatrixFactorization(matrix_dimensions, regularizers) # Define model

    eta = [1e-4, 1e-4]

    #Training model
    for ite in range(EPOCHS):
        dmf.train(matrix, mu=1, eta=eta, mask_in=mask)

        if ite % 100 == 0:
            pprint.progress_bar(ite, EPOCHS, dmf.loss_dict) # Format the loss of the output training and print out the training progress bar

        if ite % 5000 == 0:
            plot.gray_im(dmf.net.data.cpu().detach().numpy()) # Display the training image, you can set parameters to save the image

    return dmf.net.data, dmf.loss_dict['nmae_test']


def drive(miss_mode, image_path, mask_path):
    pic = csv_to_tensor(image_path)[:HEIGHT].cuda()
    # pic = dataloader.get_data(height=HEIGHT,width=WIDTH,pic_name=image_path).cuda() # Read grayscale image

    plot.gray_im(pic.cpu()) # Display grayscale image

    transformer = dataloader.data_transform(z=pic,return_type='tensor')

    if miss_mode is MissMode.RANDOM:
        mask_in = transformer.get_drop_mask(rate=0.6) # 'rate' is the loss rate
        mask_in[mask_in < 1] = 0
    elif miss_mode is MissMode.PATCH:
        mask_in = torch.ones(HEIGHT, WIDTH).cuda()
        mask_in[70:100, 150:190] = 0
        mask_in[200:230, 200:230] = 0
    elif miss_mode is MissMode.FIXED:
        mask_in = dataloader.get_data(height=HEIGHT, width=WIDTH, pic_name=mask_path)
        mask_in[mask_in < 1] = 0
    else:
        raise ValueError(f'Invalid MissMode: {miss_mode.name}')

    plot.gray_im(pic.cpu() * mask_in.cpu())

    matrix_dimensions = [
        (HEIGHT, HEIGHT),
        (HEIGHT, HEIGHT),
        (HEIGHT, WIDTH)
    ]
    line_dict = {'x_plot': np.arange(0, EPOCHS, 1)}
    mydmf = net.MyDeepMatrixFactorization(matrix_dimensions).to(device)
    _, line_dict['mydmf'] = train_my_dmf(
        mydmf,
        lossm.mse,
        torch.optim.Adam(mydmf.parameters()),
        pic,
        mask_in.cuda()
    )
    mydmfair = net.MyDeepMatrixFactorization(matrix_dimensions).to(device)
    _, line_dict['mydmfair'] = train_my_dmf_air(
        mydmfair,
        lossm.mse,
        torch.optim.Adam(mydmfair.parameters()),
        pic,
        mask_in.cuda()
    )
    # _, line_dict['DMF+AIR'] = train_dmf_air(matrix_dimensions, pic, mask_in.cuda())

    plot.lines(line_dict, save_if=False, black_if=True, ylabel_name='NMAE')
