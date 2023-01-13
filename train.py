import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from dataset.data_loader import data_loader
from utils.utils import train_one_epoch, evaluate
from utils.weight_loading import weight_loading
from config.model_config import model_config
from config import config


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    create_model = model_config()

    tb_writer = SummaryWriter()

    # 训练后权重保存位置
    if os.path.exists("./weights/"+str(config.model)) is False:
        os.makedirs("./weights/"+str(config.model))

    train_loader = data_loader(str_data='train')
    val_loader = data_loader(str_data='val')

    # 加载模型
    model = create_model(num_classes=args.num_classes).to(device)
    # 如果存在预训练权重则载入及是否冻结权重
    model = weight_loading(model)

    # 优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    # cosine 余弦退火学习率
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(1, args.epochs+1):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), "./weights/"+str(config.model)+"/model_{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=config.num_classes)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--save_epoch', type=int, default=config.save_epoch)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--lrf', type=float, default=config.lrf)

    # 数据集所在根目录
    parser.add_argument('--data_path', type=str, default=config.data_path)

    # 运行环境
    parser.add_argument('--device', default=config.device, help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
