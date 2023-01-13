import torch
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

batch_size = 1  #批处理大小
input_shape = (3, config.img_size, config.img_size)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape).to(device)  # 生成张量
export_onnx_file = config.model+".onnx"	 # 目的ONNX文件名

# torch转onnx
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})

import onnx

# 使用 ONNX 的 API 检查 ONNX 模型
onnx_model = onnx.load(config.model+".onnx")
onnx.checker.check_model(onnx_model)
print('ONNX 的 API 检查通过')

# 优化前后对比&验证
# 优化前
model.eval()
with torch.no_grad():
    output_torch = model(x)
print('优化前')

# 优化后
import onnxruntime

session =  onnxruntime.InferenceSession(export_onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
session.get_modelmeta()
output_onnx = session.run(['output'], {"input": x.cpu().numpy()})
print('优化后')

# 优化前后参数区别对比
print('优化前后参数区别对比')
print("{}vs{}".format(output_torch.mean(), output_onnx[0].mean()))