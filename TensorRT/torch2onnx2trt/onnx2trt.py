import os
import tensorrt as trt

from config import config

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, fp16_mode=False, save_engine=True):
    """
    Args:
      max_batch_size: 预先指定大小好分配显存
      fp16_mode:      是否采用FP16
      save_engine:    是否保存引擎
    return:
      ICudaEngine
    """

    # 如果是动态输入，需要显式指定EXPLICIT_BATCH
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # builder创建计算图 INetworkDefinition
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser: # 使用onnx的解析器绑定计算图
        builder.max_workspace_size = 1 << 60           # ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = max_batch_size        # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode
        config.max_workspace_size = 1 << 30            # 1G

        # 动态输入profile优化
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (8, 3, 224, 224))
        config.add_optimization_profile(profile)

        # 解析onnx文件，填充计算图
        if not os.path.exists(onnx_file_path):
            quit("ONNX file {} not found!".format(onnx_file_path))
        print('loading onnx file from path {} ...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print("Begining onnx file parsing")
            if not parser.parse(model.read()):         # 解析onnx文件
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))     # 打印解析错误日志
                return None

        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        print("Completed parsing of onnx file")

        # 使用builder创建CudaEngine
        print("Building an engine from file{}' this may take a while...".format(onnx_file_path))
        #engine=builder.build_cuda_engine(network)    # 非动态输入使用
        engine=builder.build_engine(network, config)  # 动态输入使用
        print("Completed creating Engine")
        if save_engine:
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())

if __name__ == '__main__':
    onnx_file_path = config.model+".onnx"
    fp16_mode = False
    max_batch_size = 1
    trt_engine_path = config.model+".trt"
    build_engine(onnx_file_path, trt_engine_path, max_batch_size, fp16_mode)