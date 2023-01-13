from torch2trt import TRTModule
import time
import torch

from config import config

# 运行环境
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

model_trt = TRTModule()

x = torch.ones((1, 3, config.img_size, config.img_size)).to(device)
model_trt.load_state_dict(torch.load(config.model+'_trt.pth'))
tim1 = time.time()
out = model_trt(x)
tim2 = time.time()
print('trt模型推理时间：',tim2-tim1)
print(out)