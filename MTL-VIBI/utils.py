import os
import sys
import re
import time
import math
import copy
import numpy as np
import _pickle as pickle
import argparse
import torch
from PIL import Image
from PIL import ImageDraw
from torch import nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import label_binarize

class UnknownDatasetError(Exception):
    def __str__(self):
        return "ERROR: unknown datasets error"

class UnknownModelError(Exception):
    def __str__(self):
        return "ERROR: unknown model type"

def check_model_name(model_name, file_path = 'models'):
    """
    Check whether model name is overlapped or not 
    """

    if model_name in os.listdir(file_path):
        
        valid = {"yes": True, "y": True, "ye": True, 'true': True,
                 't': True, '1': True,
                 "no": False, "n": False, 'false': False,
                 'f': False, '0': False}
        sys.stdout.write("The file {} already exists. Do you want to overwrite it? [yes/no]".\
                         format(model_name))
        choice = input().lower()
    
        if choice in valid:
            if not valid[choice]:
                sys.stdout.write("Please assign another name. (ex. 'original_2.ckpt')")
                model_name = input().lower()
                check_model_name(model_name = model_name, file_path = file_path)
                
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            check_model_name(model_name = model_name, file_path = file_path)
            
    return model_name

def compare_with_previous(correct, correct_pre, compare = True):
    
    if compare and correct < correct_pre:

        valid = {"yes": True, "y": True, "ye": True, 'true': True,
                 't': True, '1': True,
                 "no": False, "n": False,
                 'false': False, 'f': False, '0': False}
        sys.stdout.write('the accuracy is not improved from the last model. Do you want to proceed it? [yes/no]')
        choice = input().lower()
        
        if choice in valid:
            if not valid[choice]: sys.exit(0)
        
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            compare_with_previous(correct = correct,
                                  correct_pre = correct_pre,
                                  compare = compare)
    
def get_selected_words(x_single, score, id_to_word, k): 
    
    selected = np.argsort(score)[-k:] 
    selected_k_hot = np.zeros(400)
    selected_k_hot[selected] = 1.0

    x_selected = (x_single * selected_k_hot).astype(int)
    
    return x_selected 

def create_dataset_from_score(x, scores, k):
    
    with open('data/id_to_word.pkl','rb') as f:
        id_to_word = pickle.load(f)

    new_data = []
    for i, x_single in enumerate(x):
        x_selected = get_selected_words(x_single,
                                        scores[i],
                                        id_to_word, k)
        new_data.append(x_selected) 

    np.save('data/x_val-L2X.npy', np.array(new_data))

def calculate_acc(pred, y):

    return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))

def label2binary(label, classes):

    classes = list(classes)
    if len(classes) == 2:

        classes.append(-1)
        res = label_binarize(label, classes = classes)

        return res[:, :-1]
    
    else:

        return label_binarize(label, classes = classes)

def idxtobool(idx, size, is_cuda):
    
    V = cuda(torch.zeros(size, dtype = torch.long), is_cuda)
    if len(size) > 2:
        
        for i in range(size[0]):
            for j in range(size[1]):
                subidx = idx[i, j, :]
                V[i, j, subidx.type(torch.long)] = float(1)
                
    elif len(size) is 2:
        
        for i in range(size[0]):
            subidx = idx[i, :]
            V[i, subidx.type(torch.long)] = float(1)
            
    else:
        
        raise argparse.ArgumentTypeError('len(size) should be larger than 1')
            
    return V

def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y',
                     '1', 'True', 'Y', 'Yes',
                     'YES', 'YEs', 'ye'):
        return True
    
    elif v.lower() in ('no', 'false', 'f', 'n',
                       '0', 'False', 'N', 'NO',
                       'No'):
        return False
    
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def cuda(tensor, is_cuda):
    '''
    Send the tensor to cuda
    
    Args:
        is_cuda: logical. True or False 
    
    Credit: https://github.com/1Konny/VIB-pytorch
    '''

    if is_cuda:
        return tensor.cuda()
    
    else:
        return tensor

def timeSince(since):
    """
    Credit: https://github.com/1Konny/VIB-pytorch
    """

    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    
    return '%dm %ds' % (m, s)

def query_yes_no(question, default = "yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credit: fmark and Nux
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """

    valid = {"yes": True, "y": True, "ye": True,
             'true': True, 't': True, '1': True,
             "no": False, "n": False, 'false': False,
             'f': False, '0': False}
    
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")

class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):

        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay) * state_dict[key] + (1 - self.decay) *new_state_dict[key]

        self.model.load_state_dict(state_dict)

class save_batch(object):

    def __init__(self, dataset, batch, label, label_pred,
                 label_approx, index,
                 filename, is_cuda, word_idx=None):
        
        self.batch = batch
        self.label = label
        self.label_pred = label_pred
        self.label_approx = label_approx
        self.index = index
        self.filename = filename
        self.is_cuda = is_cuda
        self.word_idx = word_idx
        func = getattr(self, dataset)
        self.output = func()
        
        
    def mnist(self):
        """
        draw and save MNIST images and selected chunks
        """
        
        img = copy.deepcopy(self.batch)    
        n_img = img.size(0)

        for i in range(n_img):
            
            dpi = 100
            fig = plt.figure(frameon=False)
            fig.set_size_inches(img.shape[2] / dpi, img.shape[3] / dpi)
                        
            plt.axis('off')

            img0 = img[i].squeeze(0) 
            cmap = plt.get_cmap('Reds')
            plt.imshow(img0.cpu(), alpha=1, cmap=cmap) # alpha设置透明度, cmap可以选择颜色
            print(img0)
            
            # chunk selected
            img2 = img[i].view(-1)  # .numpy()
            
#             img = np.ones(img2.shape, dtype=float)
#             img2 = cuda(torch.tensor(img2), self.is_cuda) 
            img2[:] = float(0.0)
            # 将索引位置置为NAN
            img2[self.index[i].type(torch.long)] = cuda(torch.tensor(float('nan')), self.is_cuda)  # self.index: [100, 1, 20]
            img2 = img2.view(img0.size())  # .numpy()
            # plt.title('BB {}, Apx {}'.format(self.label_pred[i], self.label_approx[i]))
#             plt.imshow(img2.cpu(), cmap='gray')
            plt.imshow(img2.cpu(), cmap='binary')
            
            img_out_path = str(self.filename).split('.')[0] + '_' + str(i) + '.png'       
            fig.savefig(img_out_path, dpi=dpi, bbox_inches='tight', pad_inches=-0.1)

                        
class save_batch2(object):

    def __init__(self, dataset, batch, label, label_pred,
                 label_approx, index,
                 filename, is_cuda, word_idx=None):
        
        self.batch = batch
        self.label = label
        self.label_pred = label_pred
        self.label_approx = label_approx
        self.index = index
        self.filename = filename
        self.is_cuda = is_cuda
        self.word_idx = word_idx
        func = getattr(self, dataset)
        self.output = func()
        
        
    def mnist(self):
        """
        draw and save MNIST images and selected chunks
        """
        
        img = copy.deepcopy(self.batch)    
        n_img = img.size(0)
        n_col = 4
        n_row = n_img // n_col + 1

        fig = plt.figure(figsize=(n_col * 4, n_row * 4)) 

        for i in range(n_img):

            plt.subplot(n_row, n_col, 1 + i)
            plt.axis('off')
            # original image
            img0 = img[i].squeeze(0)  # .numpy()
            print("img0 max:", torch.max(img0))
            
            n = 40
            half_jet_rgba = plt.cm.seismic(np.linspace(0, 1, n))
            half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
            cmap = mpl.colors.ListedColormap(half_jet_rgba)
                        
            color=plt.get_cmap('Reds')(np.linspace(0, 1, n))
            
            
            plt.imshow(img0.cpu(), cmap=color)
            # chunk selected
            img2 = img[i].view(-1)  # .numpy()
            img2[self.index[i].type(torch.long)] = cuda(torch.tensor(float('nan')), self.is_cuda)  # self.index: [100, 1, 20]
            img2 = img2.view(img0.size())  # .numpy()
            plt.title('BB {}, Apx {}'.format(self.label_pred[i], self.label_approx[i]))
            plt.imshow(img2, cmap='gray')

        fig.subplots_adjust(wspace=0.05, hspace=0.35)
        fig.savefig(str(self.filename))
    
    
        
        
def idxtoreview(review, word_idx, index = None):
    
    review = np.array(word_idx)[review.tolist()]
    review = [re.sub(r"<pad>", "", review_sub) for review_sub in review]
    review = [re.sub(' +', ' ', review_sub) for review_sub in review]
    review = [review_sub.strip() for review_sub in review]
    
    if index is not None:    

        review_selected = [len(review_sub) * "_" for review_sub in review]
        for index_sub in index:
            review_selected[index_sub] = review[index_sub]
        review = review_selected
    
    review = " ".join(review)
    review = re.sub(' +', ' ', review)
    review = review.strip()
    
    return review

class index_transfer(object):

    def __init__(self, dataset, idx, filter_size, original_ncol, original_nrow, is_cuda=False):
        
        self.dataset = dataset
        self.idx = idx
        if type(filter_size) is int:
            filter_size = (filter_size, filter_size)
        self.filter_size_row = filter_size[0]  # 2
        self.filter_size_col = filter_size[1]  # 2
        self.original_ncol = original_ncol  # 28
        self.original_nrow = original_nrow  # 28
        self.is_cuda = is_cuda
        func = getattr(self, dataset)
        self.output = func()
    
    def default(self): 

        assert self.original_nrow % self.filter_size_row < 1
        assert self.original_ncol % self.filter_size_col < 1
        bat_size = self.idx.size(0)
        ncol = cuda(torch.LongTensor([self.original_ncol // self.filter_size_col]), self.is_cuda)  # 14

        # 根据索引确定在原图的坐标位置（行，列） 注意:div外加了向下取整符号floor
        idx_2d_unpool0 = torch.add(torch.mul(torch.div(self.idx, ncol), self.filter_size_row).view(-1, 1), cuda(torch.arange(self.filter_size_row),                             self.is_cuda)).view(-1, self.filter_size_row)  # [30, 2]

        idx_2d_unpool1 = torch.add(torch.mul(torch.remainder(self.idx, ncol), self.filter_size_col).view(-1, 1),                                                     cuda(torch.arange(self.filter_size_col), self.is_cuda)).view(-1, self.filter_size_col)  # [30, 2]

        # a = idx_2d_unpool0.view(-1, 1)
        # b = a.expand(-1, self.filter_size_col)
        # c = b.contiguous().view(bat_size, -1)
        # print(c)
        idx_2d_unpool0 = idx_2d_unpool0.view(-1, 1).expand(-1, self.filter_size_col).contiguous().view(bat_size, -1)  # [30, 2, 2]
        idx_2d_unpool1 = idx_2d_unpool1.view(-1, 1).expand(-1, self.filter_size_row).contiguous().view(bat_size, -1)  # [1, 120]

        idx_2d_unpool0 = torch.mul(idx_2d_unpool0, cuda(torch.LongTensor([self.original_ncol]), self.is_cuda))  
        idx_2d_unpool1 = torch.transpose(idx_2d_unpool1.view(-1, self.filter_size_row, self.filter_size_col), 1, 2).contiguous().view(bat_size, -1) 

        idx_unpool = torch.add(idx_2d_unpool0, idx_2d_unpool1)  # 为什么要进行相加
        
        return idx_unpool
        
    def mnist(self):    
        
        return self.default()
    
class Reshape(nn.Module):
    
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)  
    
class Flatten(nn.Module):
    
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def Concatenate(input_global, input_local):
    
    input_global = input_global.unsqueeze(-2)
    input_global = input_global.expand(-1, input_local.size(-2), -1)
            
    return torch.cat((input_global, input_local), -1)

class TimeDistributed(nn.Module):
    
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        assert (len(x.size()) <= 3)

        if len(x.size()) <= 2:
            return self.module(x)

        x = x.permute(0, 2, 1) # reshape x
        y = self.module(x)
        
        if len(y.size()) == 3:
            y = y.permute(0, 2, 1) # reshape y

        return y
