from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision import *

fastprogress.MAX_COLS = 80


def get_data(size, woof, bs, workers=None):
    if size <= 128:
        path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size <= 224:
        path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else:
        path = URLs.IMAGEWOOF if woof else URLs.IMAGENETTE
    path = untar_data(path)

    tfms = get_transforms()
    data = ImageList.from_folder(path).split_by_folder(valid='val').label_from_folder().transform(tfms, size=32)
    data=data.databunch().normalize()

    return data


@call_parse
def main(
):
    "Distributed training of Imagenette."
    gpu = None
    woof = 1
    lr = 1e-3
    size = 128
    alpha = 0.99
    mom= 0.9
    eps = 1e-6
    epochs = 5
    bs= 256
    mixup= 0.
    opt= 'adam'
    arch= 'resnet18'
    dump = 0
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if opt == 'adam':
        opt_func = partial(optim.Adam, betas=(mom, alpha), eps=eps)
    elif opt == 'rms':
        opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt == 'sgd':
        opt_func = partial(optim.SGD, momentum=mom)

    data = get_data(size, woof, bs)

    if not gpu: print(f'lr: {lr};  size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')

    #m = globals()[arch]
    learn = cnn_learner(data,models.resnet18, metrics=[accuracy],pretrained=False,opt_func=opt_func)

    learn.fit_one_cycle(epochs, lr)

