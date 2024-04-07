import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist.data_loader import MNIST_modified
from mnist.Dataloader import clas_seg_Dataset
from torchtext import data
from torchtext.vocab import GloVe

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):

    name = args.dataset
    root = args.root
    batch_size = args.batch_size
    data_loader = dict()
    device = 0 if args.cuda else -1

#     # # 已放在Dataloader中
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,)), ])

    train_data = clas_seg_Dataset(root)  # 训练数据路径
    valid_data = clas_seg_Dataset(root.replace('train', 'val'))    # 测试数据路径
    test_data = clas_seg_Dataset(root.replace('train', 'test'))    # 验证数据路径

    # data loader
    num_workers = 0
    
#     norm_mean = [0.485, 0.456, 0.406]
#     norm_std = [0.229, 0.224, 0.225]

#     train_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(norm_mean, norm_std),
#     ])

#     val_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(norm_mean, norm_std),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
#                               transform=train_transform,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True,
                              pin_memory=True)

    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size,
#                               transform=val_transform,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=False,
                              pin_memory=True)
        
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
#                              transform=test_transform,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=False,
                             pin_memory=True)
    
    data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    data_loader['yc_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    data_loader['ys_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor

    
    data_loader['train'] = train_loader
    data_loader['valid'] = valid_loader
    data_loader['test'] = test_loader

    return data_loader
