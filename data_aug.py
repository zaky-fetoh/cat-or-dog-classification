import keras.preprocessing.image as image


def getGens(trainPath: str, valPath: str, testPath: str, shape: tuple):
    train_generator = image.ImageDataGenerator(rescale=1. / 255).flow_from_directory(trainPath,
                                                                                     target_size=shape,
                                                                                     class_mode='binary', batch_size=20)
    val_generator = image.ImageDataGenerator(rescale=1. / 255).flow_from_directory(valPath,
                                                                                   target_size=shape,
                                                                                   class_mode='binary', batch_size=20)
    test_generator = image.ImageDataGenerator(rescale=1. / 255).flow_from_directory(testPath,
                                                                                    target_size=shape,
                                                                                    class_mode='binary')
    return train_generator, val_generator, test_generator


def get_generators(train_path, val_path):
    train_generator = image.ImageDataGenerator(rotation_range=90,
                                               width_shift_range=.2,
                                               height_shift_range=.2,
                                               shear_range=.2,
                                               zoom_range=.2,
                                               horizontal_flip=True,
                                               vertical_flip=True,
                                               rescale=1. / 255
                                               ).flow_from_directory(train_path,
                                                                     target_size=(150, 150),
                                                                     class_mode='binary',
                                                                     batch_size=20)

    val_generator = image.ImageDataGenerator(rescale=1. / 255
                                             ).flow_from_directory(val_path,
                                                                   target_size=(150, 150),
                                                                   class_mode='binary',
                                                                   batch_size=20)

    return train_generator, val_generator
