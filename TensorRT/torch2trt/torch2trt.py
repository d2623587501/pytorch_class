import torch
from torch2trt import torch2trt

from config.model_config import model_config
from config import config

# 运行环境
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# 模型参数加载
model_weight_path = "../../weights/"+config.model+"/model_"+str(config.model_pre_num)+".pth"
torch_model = torch.load(model_weight_path, map_location=device) # pytorch模型权重加载

# 模型实例化
model = model_config()
model = model(num_classes=config.num_classes).to(device)
model.load_state_dict(torch_model)

# 测试数据并导入模型
x = torch.ones((1, 3, config.img_size, config.img_size)).to(device)
model_trt = torch2trt(model, [x])

# 对比转换前后的区别
y = model(x)  # torch模型
y_trt = model_trt(x)  # trt模型
print(torch.max(torch.abs(y - y_trt)))

# trt模型保存
torch.save(model_trt.state_dict(), config.model+'_trt.pth')