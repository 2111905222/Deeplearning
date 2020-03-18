from keras.applications import VGG16
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_data(**data_monitor):
    title = data_monitor.pop('title')
    data_style = ['b+', 'b-', 'r+', 'r-', 'y*', 'y-']
    # python只支持对于key的遍历，所以不能使用for k,v这种形式，这个时候会提示ValueError: too many values to unpack， 如果想对key，value，则可以使用items方法。
    i = 0

    for label, data in data_monitor.items():
        print('label name&content', label, data)
        epochs = range(1, len(data) + 1)
        print('epochs', epochs)
        plt.plot(epochs, data, data_style[i], label=label)
        i = i + 1
    plt.legend()
    plt.title(title)
    plt.show()


def data_Statistics(history_datapower, title):
    history_dict = history_datapower.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plot_data(acc=acc, val_acc=val_acc, loss_values=loss_values, val_loss_values=val_loss_values, title=title)
    return loss_values, val_loss_values, acc, val_acc, epochs

def smooth_curve(points, factor=0.8):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous * factor + point * (1 - factor))
        else:
            smooth_points.append(point)
    return smooth_points


conv_base = VGG16(
    weights='imagenet',  # 指定模型初始化的权重检查点
    include_top=False,  # 指定模型是否包含密集连接分类器
    input_shape=(150, 150, 3)  # 若不指定，则任意形状
)
print('base skelen', conv_base.summary())  # 最后的特征图形状为（4, 4, 512）
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'E:/Study/phython-try/DeepLearning/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'train')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_feature(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # 确保shape为(samples, 4, 4, 512)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,  # 生成器一轮提取20
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)  # 卷积基一轮处理20张图片
        features[i * batch_size: (i + 1) * batch_size] = features_batch  # 20张图片分批构建入features中
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# 利用冻结的卷积基端到端地训练模型
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255  # 测试集不用数据增强
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# train_features, train_labels = extract_feature(train_dir, 2000)
# validation_features, validation_labels = extract_feature(validation_dir, 1000)
# test_features, test_labels = extract_feature(test_dir, 1000)
# conv_base.trainable = False  # 冻结网络（使得其权重保持不变）只用于特征提取
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
loss_values, val_loss_values, acc, val_acc, epochs = data_Statistics(history, '对VGG的最后几层进行解冻微调（未平滑处理）')

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('解冻后微调并平滑处理的Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, smooth_curve(loss_values), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss_values), 'b', label='Smoothed validation loss')
plt.title('解冻后微调并平滑处理的Training and validation loss')
plt.legend()
plt.show()