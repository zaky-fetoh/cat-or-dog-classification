import model
import data_aug
import OrganizingData as pth


def old1():
    modl = model.getModel((150, 150, 3))
    train_path, val_path, TestPath = pth.getPaths()
    train_generator, val_generator, test_generator = data_aug.getGens(train_path, val_path,
                                                                      TestPath, (150, 150))
    hist = modl.fit_generator(train_generator, steps_per_epoch=1500 // 20, epochs=5
                              ,validation_data=val_generator, validation_steps=500 // 20).history


# $ T0D0 augmentedData and regul-izedM0Del

def reg_model_train():
    modl = model.getModelReg((150, 150, 3))
    train_path, val_path,test_path = pth.getPaths()

    train_generator, val_generator = data_aug.get_generators(train_path, val_path)
    hist = modl.fit_generator(train_generator,
                              steps_per_epoch=1500 // 20,
                              epochs=5,
                              validation_data=val_generator,
                              validation_steps=500 // 20).history

    return modl, hist


modl,plts = reg_model_train()

#T0Do featureEx Truction using a pretainedModel andfine Tuning





























