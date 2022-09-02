import sys


def my_progress_bar(epoch_now, epoch_all, loss):
    print("\r", end="")
    print("loss: {:.2}. Trainning progress: {:.3}%: ".format(loss, epoch_now/epoch_all*100),  "▋" * (int(epoch_now/epoch_all*100 // 2)), end='')
    sys.stdout.flush()

def progress_bar(epoch_now,epoch_all,loss_dict):
    print("\r", end="")
    print("loss_fid:{:.2},nmae_test:{:.2}, loss_all:{:.2}. Trainning progress: {:.3} %: ".format(loss_dict['loss_fid'][-1], loss_dict['nmae_test'][-1],loss_dict['loss_all'][-1],epoch_now/epoch_all*100),  "▋" * (int(epoch_now/epoch_all*100 // 2)), end='')
    sys.stdout.flush()
