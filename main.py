# Image type matrix completion
# Loss fixed pionts
import enum
import MinPy
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

HEIGHT = 240
WIDTH = 240
EPOCHS = 10_001
np.random.seed(20210909)

def drive(miss_mode, image_path, mask_path):
    pic = dataloader.get_data(height=HEIGHT,width=WIDTH,pic_name=image_path).cuda() # Read grayscale image

    plot.gray_im(pic.cpu()) # Display grayscale image

    transformer = dataloader.data_transform(z=pic,return_type='tensor')

    if miss_mode is MissMode.RANDOM:
        mask_in = transformer.get_drop_mask(rate=0.3) # 'rate' is the loss rate
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

    line_dict={'x_plot': np.arange(0, EPOCHS, 1)}
    for algo in [Algorithm.DMF, Algorithm.DMF_AIR]:
        reg_hc = MinPy.reg.hc_reg(name='lap')
        reg_row = MinPy.reg.auto_reg(HEIGHT, 'row')
        reg_col = MinPy.reg.auto_reg(WIDTH, 'col')
        reg_cnn = MinPy.reg.cnn_reg()
        dmf = MinPy.demo.basic_dmf([HEIGHT, HEIGHT, HEIGHT, WIDTH], [reg_hc, reg_row, reg_col, reg_cnn]) # Define model

        eta = [None] * 4 if algo is Algorithm.DMF else [None, 1e-4, 1e-4, None]

        #Training model
        for ite in range(EPOCHS):
            dmf.train(pic, mu=1, eta=eta, mask_in=mask_in)

            if ite % 100 == 0:
                pprint.progress_bar(ite, EPOCHS, dmf.loss_dict) # Format the loss of the output training and print out the training progress bar

            if ite % 5000 == 0:
                plot.gray_im(dmf.net.data.cpu().detach().numpy()) # Display the training image, you can set parameters to save the image
                print('RMSE:', torch.sqrt(torch.mean((pic-dmf.net.data)**2)).detach().cpu().numpy())
                #plot.gray_im(dmf.net.show_img().cpu().detach().numpy()) # Display the training image, you can set parameters to save the image

        # save loss values for plotting
        plot_keys_ignore = {'loss_all', 'loss_fid', 'loss_auto_reg_col', 'loss_auto_reg_row'}
        for k, v in dmf.loss_dict.items():
            if k not in plot_keys_ignore and v:
                line_dict[algo.name] = np.array(v)


    plot.lines(line_dict, save_if=False, black_if=True, ylabel_name='NMAE')
