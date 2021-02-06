import keras.layers as layers
import keras.models as models
import keras.optimizers as opt
import keras.losses as lss


def getModel(input_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='B1Conv'
                            , input_shape=input_shape))
    model.add(layers.MaxPool2D((2, 2), name='B1MAx'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='B2Conv'))
    model.add(layers.MaxPool2D((2, 2), name='B2MAx'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='B3Conv'))
    model.add(layers.MaxPool2D((2, 2), name='B3MAx'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', name='B4Conv'))
    model.add(layers.MaxPool2D((2, 2), name='B4MAx'))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=opt.RMSprop(),
                  loss=lss.binary_crossentropy,
                  metrics=['acc'])
    return model


def getModelReg(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            name='B0_conv', input_shape=input_shape))
    model.add(layers.MaxPool2D((2, 2), name='B0_MaxPoo'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='B1_conv'))
    model.add(layers.MaxPool2D((2, 2), name='B1_MaxPoo'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', name='B2_conv'))
    model.add(layers.MaxPool2D((2, 2), name='B2_MaxPoo'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', name='B3_conv'))
    model.add(layers.MaxPool2D((2, 2), name='B3_MaxPoo'))

    model.add(layers.Flatten(name='EndoFE'))

    model.add(layers.Dropout(.2, name='Drop0Ut'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=opt.RMSprop(),
                  loss=lss.binary_crossentropy,
                  metrics=['acc'])

    return model
