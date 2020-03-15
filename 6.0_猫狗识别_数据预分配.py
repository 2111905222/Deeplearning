import os, shutil

def data_dis():
    global train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, test_cats_dir, test_dogs_dir
    try:
        original_dataset_dir = 'E:/Study/phython-try/DeepLearning/dogs-vs-cats'
        # The path to the directory where the original
        # dataset was uncompressed
        # The directory where we will
        # store our smaller dataset
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
        os.mkdir(base_dir)

        # 建立三个文件夹train test validation

        os.mkdir(train_dir)

        os.mkdir(validation_dir)

        os.mkdir(test_dir)

        # 在train文件夹下cats

        os.mkdir(train_cats_dir)

        # 在train文件夹下dogs

        os.mkdir(train_dogs_dir)

        # 在validation文件夹下建立cats

        os.mkdir(validation_cats_dir)

        # 在validation文件夹下建立dogs

        os.mkdir(validation_dogs_dir)

        # 在test文件夹中建立cats

        os.mkdir(test_cats_dir)

        # 在test文件夹中建立dogs

        os.mkdir(test_dogs_dir)

        # 往train文件夹复制前1000个猫的图片
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)

        # 将接下来的500个猫图片复制到验证文件夹
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)

        # 将接下来的500个猫图片复制到测试文件夹
        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy first 1000 dog images to train_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 dog images to validation_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 dog images to test_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)
    except FileExistsError:
        print('total training cat images:', len(os.listdir(train_cats_dir)))
        print('total training dog images:', len(os.listdir(train_dogs_dir)))
        print('total validation cat images:', len(os.listdir(validation_cats_dir)))
        print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
        print('total test cat images:', len(os.listdir(test_cats_dir)))
        print('total test dog images:', len(os.listdir(test_dogs_dir)))
        return 1
    finally:
        return 1


data_dis()
