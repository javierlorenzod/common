import numpy as np
import os
import random
import torch
from functools import reduce
from loguru import logger

def resize_imgs_keeping_aspect_ratio(img_list, new_size=512, background_value=0, device="cpu"):
    """
    Code mainly based on
     https://discuss.pytorch.org/t/resize-images-torch-tensor-already-loaded-on-gpu-keeping-their-aspect-ratio/83172/9
    :param img_list: list of images, previously in FloatTensor format, each with size [C, H, W]
    :param new_size: input size of the model. It is assumed that it is square
    :param background_value: value of the padding pixels. In the example is 126 (medium gray)
    :param device: device in which all the processing is done
    :return:
    """
    tensor_imgs = torch.stack(img_list, dim=0).to(device)  # Size: [B, C, H, W] where N is the number of images or batch size
    tensor_height, tensor_width = tensor_imgs.shape[2:4]
    batch_size = tensor_imgs.shape[0]
    tall = True if tensor_height >= tensor_width else False
    coef = new_size / float(tensor_height) if tall else new_size / float(tensor_width)
    new_dimension = (new_size, int(tensor_width * coef)) if tall else (int(tensor_height * coef), new_size)
    y = torch.nn.functional.interpolate(tensor_imgs, new_dimension)
    res = torch.ones(batch_size, 3, new_size, new_size) * background_value
    margin = (new_size - new_dimension[1]) // 2 if tall else (new_size - new_dimension[0]) // 2
    if tall:
        res[:, :, :, margin:new_size - margin] = y
    else:
        res[:, :, margin:new_size - margin, :, ] = y
    return res


def pytorch_count_params(model):
    """
    count number trainable parameters in a pytorch model
    """
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params


def activate_reproducibility(seed=0):
    """
    Function called at the beginning of training. Needed to have the same random seeds for all trainings and because
    of that, adquire the same results for the same parameters.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assure_torch_reproducibility(seed: int, use_cuda: bool):
    # https://stackoverflow.com/questions/30483246/how-to-check-if-a-python-module-has-been-imported
    ## Pytorch
    torch.manual_seed(seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assure_numpy_reproducibility(seed: int):
    np.random.seed(seed)


def set_device(disable_cuda=False):
    """
    Returns the device chosen (if disable_cuda is False, CUDA will be chosen if available)
    Source: https://pytorch.org/docs/stable/notes/cuda.html
    """
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


# From: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path='./results', checkpoint_filename='best_model_checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.cp_path = checkpoint_path
        self.cp_filename = checkpoint_filename
        self.cp_path = os.path.join(self.cp_path, self.cp_filename)
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.cp_path)
        self.val_loss_min = val_loss