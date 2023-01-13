import os
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from dataset.data_loader import data_transforms
from config.model_config import model_config
from config import config


def main():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    data_transform = data_transforms('val')

    # load image
    img_path = config.test_img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    create_model = model_config()
    model = create_model(num_classes=config.num_classes).to(device)
    # load model weights
    model_weight_path = "./weights/"+config.model+"/model_"+str(config.model_pre_num)+".pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        print(img.shape)
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
