import os

base_dir = 'E:/Study/phython-try/DeepLearning/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

from keras.preprocessing.image import ImageDataGenerator


def data_preprocess():  # 从目标文件夹获取数据
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
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
    for data_batch, labels_batch in train_generator:
        print('data batch size:', data_batch.shape)
        print('labels batch size:', labels_batch.shape)
        break
    return train_generator, validation_generator


import matplotlib.pyplot as plt


def plot_data(**data_monitor):
    label_name = data_monitor.pop('label_name')
    data_plot_value = list(data_monitor.values())
    data_plot_key = list(data_monitor.keys())
    data_style = ['b+', 'b-', 'r+', 'r-', 'y*', 'y-']

    i = 0
    for data in data_plot_value:
        epochs = range(1, len(data) + 1)  # 这里需要加1！！！！！！因为range不包括后一个数字本身
        plt.plot(epochs, data, data_style[i], label=label_name[data_plot_key[i]])  # x轴和y轴一定要一样维度和大小！！！（epochs和data）
        i = i + 1
    plt.legend()
    plt.show()
    plt.clf()
    i = 0
    # python只支持对于key的遍历，所以不能使用for k,v这种形式，这个时候会提示ValueError: too many values to unpack， 如果想对key，value，则可以使用items方法。
    for label, data in data_monitor.items():
        plt.plot(epochs, data, data_style[i], label=label)
        i = i + 1
    plt.legend()
    plt.show()


def data_preprocess_power():
    train_datagen = ImageDataGenerator(  # 数据增强
        rescale=1. / 255,
        rotation_range=40,  # 角度值
        width_shift_range=0.2,  # 图像水平和垂直方向上平移的范围（相对于总宽度和总高度的比例）
        height_shift_range=0.2,
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 图像随机缩放的范围
        horizontal_flip=True,  # 随机将一半图像水平翻转，
        fill_mode='nearest'  # 填充新创建像素的方法
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)  # 注意不能增强验证数据
    train_generator = train_datagen.flow_from_directory(  # 以train_datagen模式从目标文件夹获取数据
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(  # 以validation_datagen模式从目标文件夹获取数据
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    return train_generator, validation_generator


def data_Statistics(history_datapower):
    history_dict = history_datapower.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    label_name = {'acc': 'Traing acc', 'val_acc': 'Validation acc', 'loss_values': 'Traing loss',
                  'val_loss_values': 'Validation loss'}
    plot_data(acc=acc, val_acc=val_acc, loss_values=loss_values, val_loss_values=val_loss_values, label_name=label_name)


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # 150是随意定下来的
model.add(layers.MaxPooling2D((2, 2)))  # maxpooling2D后面有()，(2, 2)是元组
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))  # 计算下来剩下28 * 28 所以需要512个神经元
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train_data, validation_data = data_preprocess()
# history = model.fit_generator(
#     train_data,
#     steps_per_epoch=100,
#     epochs=20,
#     validation_data=validation_data,
#     validation_steps=50
# )  # 100(steps_per_epch)*20=2000(总体的训练数据的大小)，及高速fit需要运行多少次梯度下降
# model.save('cats_and_dogs_small_1.h5')

train_generator, validation_generator = data_preprocess_power()  # 数据增强

history_datapower = model.fit_generator(  # 使用数据增强来训练模型
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_2.h5')  # 保存
data_Statistics(history_datapower)
