import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import misc
def im(img,path='./test.jpg',save_if=False):
    fig, ax = plt.subplots()
    show_pic = np.clip(img,0,1)
    ax.imshow(show_pic,vmin=0,vmax=1)
    ax.grid(0)
    ax.axis('off')
    if save_if:
        fig.savefig(path)
        print('Saved', path)
    return fig, ax


def gray_im(img,path='./test.jpg',save_if=False):
    fig, ax = plt.subplots()
    show_pic = np.clip(img,0,1)
    ax.imshow(show_pic,'gray',vmin=0,vmax=1)
    ax.grid(0)
    ax.axis('off')
    if save_if:
        fig.savefig(path)
        print('Saved', path)
    return fig, ax



def lines(line_dict,xlabel_name='epoch',ylabel_name='MSE',ylog_if=False,save_if=False,path='./lines.jpg',black_if=False):
    fig, ax = plt.subplots()
    if black_if:
        sns.set()
    else:
        sns.set_style("whitegrid")  
    for name in line_dict.keys():
        if name != 'x_plot':
            plt.plot(line_dict['x_plot'],line_dict[name],label=name)
    ax.legend()
    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)
    if ylog_if:
        plt.yscale('log')
    if save_if:
        fig.savefig(path)
    return fig, ax


