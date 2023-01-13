import os
import torch

from config import config

dir_path = os.path.realpath(__file__)


def weight_loading(model):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    weights_path = os.path.dirname(dir_path) + '/pre_weights/' + config.weights
    # 如果存在预训练权重则载入
    if config.weights != "":
        if os.path.exists(weights_path):
            weights_dict = torch.load(weights_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weights_path))

    # 是否冻结权重
    if config.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    return model
