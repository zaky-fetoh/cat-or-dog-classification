import numpy as np
import pandas as pd
import keras.models as models
import keras.applications as app
import keras.layers as layers
import OrganizingData as od
import data_aug as dtaug
import keras.optimizers as opt
import keras.losses as lss


def get_vgg(inp=(150, 150, 3)):
    convbase = app.VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=inp)

    modl = models.Sequential()
    modl.add(convbase)
    modl.add(layers.Flatten())
    return modl


def get_generators():
    train_path, val_path, test_path = od.getPaths()
    tr_gen, val_gen = dtaug.get_generators(train_path, val_path)
    return tr_gen, val_gen


def getFeatureTr():
    model = get_vgg()
    featurelist, labellist = list(), list()
    tr_gen, val_gen = get_generators()
    for _ in range(1500 // 20):
        batch, label = next(tr_gen)
        featurelist.append(model.predict(batch))
        labellist.append(label)

    return np.concatenate(featurelist, axis=0), np.concatenate(labellist, axis=0)


def get_densely_NN(inp=(8192,)):
    model = models.Sequential()
    model.add(layers.Dense(1000, activation='relu', input_shape=inp))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=opt.RMSprop(),
                  loss=lss.binary_crossentropy,
                  metrics=['acc'])
    return model


def trainFro():
    Ft, lbl = getFeatureTr()
    modl = get_densely_NN()
    hist = modl.fit(Ft, lbl, batch_size=20, epochs=5).history
    return modl, hist


def Fx1():
    convbase = get_vgg()
    convbase.trainable = False
    dns = get_densely_NN()
    modl = models.Sequential()
    modl.add(convbase)
    modl.add(dns)
    modl.compile(optimizer=opt.RMSprop(),
                 loss=lss.binary_crossentropy,
                 metrics=['acc'])
    return modl


if __name__ == '__main__':
    pass
