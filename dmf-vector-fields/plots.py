import io

import PIL
import matplotlib.pyplot as plt


def matplotlib_to_PIL_Image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def quiver(X, Y, U, V, scale=None, save_path=None):
    fig, ax = plt.subplots()

    if scale:
        ax.quiver(X, Y, U, V, scale=scale)
    else:
        ax.quiver(X, Y, U, V)

    if save_path:
        fig.savefig(save_path)

    return fig, ax
