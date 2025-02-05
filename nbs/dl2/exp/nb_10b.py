
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/10b_mixup_label_smoothing.ipynb

from exp.nb_10 import *

class NoneReduce():
    def __init__(self, loss_func):
        self.loss_func,self.old_red = loss_func,None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)

from torch.distributions.beta import Beta

def unsqueeze(input, dims):
    for dim in listify(dims): input = torch.unsqueeze(input, dim)
    return input

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class MixUp(Callback):
    _order = 90 #Runs after normalization and cuda
    def __init__(self, alpha:float=0.4): self.distrib = Beta(tensor([alpha]), tensor([alpha]))

    def begin_fit(self): self.old_loss_func,self.run.loss_func = self.run.loss_func,self.loss_func

    def begin_batch(self):
        if not self.in_train: return #Only mixup things during training
        lambd = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        lambd = torch.stack([lambd, 1-lambd], 1)
        self.lambd = unsqueeze(lambd.max(1)[0], (1,2,3))
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1,self.yb1 = self.xb[shuffle],self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.lambd)

    def after_fit(self): self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train: return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        loss = lin_comb(loss1, loss2, self.lambd)
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.eps)