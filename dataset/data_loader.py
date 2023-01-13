import os
import torch
from torchvision import transforms


from dataset.dataset import MyDataSet
from utils.utils import read_split_data
from config import config


def data_transforms(str_data='train'):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(config.img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(config.img_size),
                                   transforms.CenterCrop(config.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    if str_data == 'train':
        return data_transform["train"]
    else:
        return data_transform["val"]


def data_loader(str_data='train'):
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(config.data_path)

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transforms('train'))

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transforms('val'))

    # 线程数
    nw = min([os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])

    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    if str_data == 'train':
        return train_loader
    else:
        return val_loader
