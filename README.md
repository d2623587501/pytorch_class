# pytorch_class

### 1 项目介绍

本项目基于Pytorch框架搭建的图像分类套件，可以通过config文件下的配置进行模型的选取（第2节已支持模型）。接下来的章节有对项目的详细介绍。同时模型支持了TensorRT的推理，具体细节查看（）

### 2 已支持模型

目前本项目已支持的模型如下：（如有需要其它模型，提issues）

```
===========================
## alexNet
[alexnet]
===========================
## vggNet
[vggnet]
===========================
## resNet
[resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d]
===========================
## regNet
[regnet]
===========================
## mobileNet
[mobilenet_v2]
===========================
## shuffleNet
[shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x1_0]
===========================
## denseNet
[densenet121, densenet161, densenet169, densenet201]
===========================
## efficientNet、efficientNetv2
[efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7]
[efficientnetv2_s, efficientnetv2_l, efficientnetv2_m]
===========================
```

### 3 项目文件树及介绍

```
│  predict.py  # 预测代码
│  README.md  # readme
│  train.py  # 训练代码
│
├─config
│  │  config.py  # 模型配置文件，包含模型选取，学习率，batch_size等
│  │  model_config.py  # 模型实例化文件
│  │  model_config.txt  # 目前项目支持的模型类别
│
├─data  # 数据存放位置（可自己给定路径）
├─dataset  
│  │  dataset.py  # 数据增强，归一化等前处理文件
│  │  data_loader.py  # 数据加载文件
│
├─model_zoo  # 目前项目所包含模型搭建文件
│  ├─alexNet
│  │  │  alexNet.py
│  │
│  ├─denseNet
│  │      denseNet.py
│  │
│  ├─efficientNet
│  │      efficientNet.py
│  │      efficientNet_v2.py
│  │
│  ├─googleNet
│  │      googleNet.py
│  │
│  ├─mobileNet
│  │      mobileNet_v2.py
│  │
│  ├─regNet
│  │      regNet.py
│  │
│  ├─resNet
│  │      resNet.py
│  │
│  ├─shuffleNet
│  │      shuffleNet.py
│  │
│  └─vggNet
│          vggNet.py
│
├─TensorRT  # 2种TensorRT推理方式
│  ├─torch2onnx2trt
│  │      onnx2trt.py  # onnx模型转tensorrt模型
│  │      torch2onnx.py  # torch模型转onnx模型
│  │      torch_onnx_predict.py  # torch模型与onnx模型时间推理对比
│  │      trt_predict.py  # tensorrt模型推理及时间
│  │
│  └─torch2trt
│          torch2trt.py  # 依赖NVDIA官方的torch2trt库，torch模型转tensorrt模型
│          trt_predict.py  # tensorrt模型推理及时间
│
├─utils
│  │  utils.py  # 数据集读取处理文件
│  │  weight_loading.py  # 预训练权重加载文件
│  │
│  ├─pre_weights  # 预训练权重存放文件（这部分后续添加自动下载）
│
└─weights  # 训练权重保存文件
    └─alexnet
            model_1.pth
```

### 4 Anaconda环境搭建

```
# 创建python3.7的虚拟环境
conda create -n cv python==3.7
# 进入环境
conda activate cv
# 安装相应的包环境
pip install -r requirements.txt
```

### 5 数据集处理

需要每一个类别一个单独文件夹，如下：

```
├─data
│  │  类别1文件夹
│  │  类别2文件夹
│  │  类别3文件夹
│  │  ......
```



### 6 训练与预测

打开config/config.py，选取训练的模型，训练轮次，保存轮次，指定数据集地址，测试图片等等

```
# 训练
python train.py

# 预测
python predict.py
```

### 7 TensorRT推理

本项目支持两种方式的TensorRT推理

```
# 第一种torch->onnx->tensorrt
# torch模型转onnx模型
python torch2onnx.py
# torch模型与onnx模型时间推理对比
python torch_onnx_predict.py  
# onnx模型转tensorrt模型
python onnx2trt.py  
 # tensorrt模型推理及时间
python trt_predict.py 
```

```
# 第二种依赖NVDIA官方的torch2trt库，torch->tensorrt
# torch模型转tensorrt模型
torch2trt.py  
# tensorrt模型推理及时间
trt_predict.py  
```

