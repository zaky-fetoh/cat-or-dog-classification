import os
from glob import glob

cats = glob('.\dataset\cat*')
dogs = glob('.\dataset\dog*')

# preparing PAths
basePath = os.getcwd()
trainPath = os.path.join(basePath, 'train')
valPath = os.path.join(basePath, 'validation')
testPath = os.path.join(basePath, 'test')

def getPaths() :
    return trainPath, valPath, testPath



if __name__ == '__main__' :

    for dir in [trainPath, valPath, testPath]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # trraining data prepration
    dogPath = os.path.join(trainPath, 'dog')
    catPath = os.path.join(trainPath, 'cat')
    for dir in [catPath, dogPath]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # $catsFirst
    for pth in cats[:1500]:
        dist = os.path.join(catPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)
    for pth in dogs[:1500]:
        dist = os.path.join(dogPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)

    # $validation (dt) preparation
    vdogPath = os.path.join(valPath, 'dog')
    vcatPath = os.path.join(valPath, 'cat')
    for dir in [vdogPath, vcatPath]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # $Coping Data
    for pth in cats[1500:2000]:
        dist = os.path.join(vcatPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)
    for pth in dogs[1500:2000]:
        dist = os.path.join(vdogPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)

    # $validation (dt) preparation
    tsdogPath = os.path.join(testPath, 'dog')
    tscatPath = os.path.join(testPath, 'cat')
    for dir in [tsdogPath, tscatPath]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    # $Coping Data
    for pth in cats[2000:2500]:
        dist = os.path.join(tscatPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)
    for pth in dogs[2000:2500]:
        dist = os.path.join(tsdogPath, pth.split('\\')[-1])
        os.system('copy ' + pth + ' ' + dist)
