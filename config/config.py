# 训练
model = 'alexnet'
img_size = 224
num_classes = 6
save_epoch = 1
epochs = 1
batch_size = 1
lr = 0.01
lrf = 0.01
data_path = '/home/zxce/AI_platform/dataset/shoeTypeClassifierDataset/training'
weights = ''
freeze_layers = False
device = 'cuda:1'
# 预测
test_img_path = 'image2.jpg'
model_pre_num = 1
