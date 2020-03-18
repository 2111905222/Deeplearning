img_path = 'E:/Study/phython-try/DeepLearning/cats_and_dogs_small/train/cats/cat.150.jpg'
from keras.preprocessing import image
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)  # 在0轴增加，代表samples，数量
img_tensor /= 255.  # 图像的数据预处理都是这样，拿到image的张量
print(img_tensor.shape)  # shape：(1, 150, 150, 3)
print(type(img_tensor))  # <class 'numpy.ndarray'>

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])

plt.show()

from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
model.summary()
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]  # 需要的输出是提取模型的前8层的输出，有8个输出，8个numpy数组（每个层对应一个数组）
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # 创建一个模型，给定模型输入，可以返回的输出是上面定义的

activations = activation_model.predict(img_tensor)  # 用模型处理张量 返回的8个Numpy数组组成的列表，每个层激活对应一个Numpy数组
first_layer_activation = activations[0]  # 提取第一个数组，即第一个卷积层的激活
print('first_layer_activation.shape: ', first_layer_activation.shape)  # 输出第一层的输出：[1, 148, 148, 32]，可以看到3*3的核函数，有32个特征图
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')  # 将第一层的第四个通道（特征图）可视化
plt.legend()  # 在Matplotlib中灵活生成图例。多个框框
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')  # 将第一层的第七个通道（特征图）可视化
plt.legend()
plt.show()
import keras

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:  # 取前8层的分别的名称
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):  # activation是处理了图片张量后的模型
    # ﻿﻿zip()函数接受一系列可迭代对象作为参数，将不同对象中相对应的元素打包成一个元组（tuple），x = [1,2,3,4,5] y = ['a','b','c','d']
    # 返回由这些元组组成的list列表，如果传入的参数的长度不等，则返回的list列表的长度和传入参数中最短对象的长度相同。，
    # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
    # This is the number of features in the feature map
    print('layer_activation.shape = ', layer_activation.shape)  # 其实就是activation的每个元素的shape，有第一层、第二层...
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]  # 当前特征图的图片大小size

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row  # 共n_feature这么多张特种图
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):  # 列 1
        for row in range(images_per_row):  # 行
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]  # 找到是特征图中的第几张图片，从左到右, 从上到下输出，1*16+2 1*16+3 ...1*16+15
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()  # 归一化，如果图片偏黑容易出现 RuntimeWarning: invalid value encountered in true_divide
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')  # 将超出0或者255的都换成0和255
            # clip(Array,Array_min,Array_max)
            # 把原始矩阵Array中，比Array_min小的元素都替换成Array_min，比Array_max大的元素都替换成Array_max
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image  # 对应某一张图片放进某一个格子里

    # Display the grid
    scale = 1. / size  # 相对应地缩小图片
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.legend()
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()