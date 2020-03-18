from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# 可视化每一层的前64过滤器，只查看每个卷积块的第一层（'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'）
# 输出放在一个 8 * 8的网格，每个网格都是一个64像素*64像素，两个过滤器之间留有一些黑边
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1,标准化，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1] 将x裁切到[0, 1]区间
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array，再转为RGB
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


model = VGG16(weights='imagenet',
              include_top=False)
# # # 步骤：利用梯度上升？找到令这个过滤器最大响应化的输入图像
# layer_name = 'block3_conv1'  # 可视化block3_conv1的第0个过滤器
# filter_index = 0
#
# layer_output = model.get_layer(layer_name).output  # 拿到这个层的输出
# loss = K.mean(layer_output[:, :, :, filter_index])  # 将这个层的输出的特征图的第0个通道（过滤器）定义为loss
# grads = K.gradients(loss, model.input)[0]  # 用于求loss关于model.input 的导数（梯度）返回的是一个张量列表，列表长度是张量列表model.input的长度
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # 梯度标准化，+1e-5是为了防止除以0
#
# iterate = K.function([model.input], [loss, grads])  # 组成一个function，规定了input是model.input，output是loss和grads
#
# # Let's test it:
#
# loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])  # 规定了输入的格式
# input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.  # 初始化输入值
#
# # Run gradient ascent for 40 steps
# step = 1.  # this is the magnitude of each gradient update
# for i in range(40):  # 迭代40次
#     # Compute the loss value and gradient value
#     loss_value, grads_value = iterate([input_img_data])
#     # Here we adjust the input image in the direction that maximizes the loss
#     input_img_data += grads_value * step  # 梯度上升
# print(input_img_data)
# print(input_img_data[0])
# print(input_img_data.shape)
# img = input_img_data[0]
#
# plt.imshow(deprocess_image(img))
# plt.show()

def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

# plt.imshow(generate_pattern('block3_conv1', 0))
#
# plt.show()


for layer_name in ['block1_conv1', 'block2_conv1']:
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)).astype('uint8')

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.title(layer_name)
plt.show()

