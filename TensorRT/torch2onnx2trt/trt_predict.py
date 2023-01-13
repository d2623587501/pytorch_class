import os
import time
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from config import config

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size # 非动态输入
        # size = trt.volume(engine.get_binding_shape(binding))                       # 动态输入
        size = abs(size)                                # 上面得到的size(0)可能为负数，会导致OOM
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)   # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)    # cuda分配空间
        bindings.append(int(device_mem))                # binding在计算图中的缓冲地址
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 如果创建network时显式指定了batchsize，使用execute_async_v2, 否则使用execute_async
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == '__main__':
    engine_file_path = config.model+".trt"
    max_batch_size = 1
    class_num = config.num_classes  # 分类类别数

    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())  # 反序列化
    # 2.将引擎应用到不同的GPU上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 3.推理
    output_shape = (max_batch_size, class_num)
    dummy_input = np.ones([1, 3, 224, 224], dtype=np.float32)
    inputs[0].host = dummy_input.reshape(-1)

    # 如果是动态输入，需以下设置
    context.set_binding_shape(0, dummy_input.shape)

    t1 = time.time()
    trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t2 = time.time()
    # 由于tensorrt输出为一维向量，需要reshape到指定尺寸
    feat = postprocess_the_outputs(trt_outputs[0], output_shape)
    print("TensorRT engine time cost: {}".format(t2 - t1))