import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import torch
import time

from config.model_config import model_config
from config import config

# 运行环境
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

# torch模型参数加载
model_weight_path = "../../weights/"+config.model+"/model_"+str(config.model_pre_num)+".pth"
torch_model = torch.load(model_weight_path, map_location=device) # pytorch模型权重加载

# onnx模型加载
onnx_model_path= config.model+".onnx"
net_session = onnxruntime.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

# torch模型实例化
model = model_config()
model = model(num_classes=config.num_classes).to(device)
model.load_state_dict(torch_model)

# torch模型推理
batch_size = 1  #批处理大小
input_shape = (3, config.img_size, config.img_size)   #输入数据,改成自己的输入shape
model = model.eval()
torch_input = torch.randn(batch_size, *input_shape).to(device)  # 生成张量
torch_time_star = time.time()
torch_out = model(torch_input)
torch_time_end = time.time()
print('torch模型推理时间：', torch_time_end-torch_time_star)

# onnx模型推理
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
onnx_input = torch_input
inputs = {net_session.get_inputs()[0].name: to_numpy(onnx_input)}
onnx_time_star = time.time()
onnx_out = net_session.run(None, inputs)
onnx_time_end = time.time()
print('onnx模型推理时间：', onnx_time_end-onnx_time_star)