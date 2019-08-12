import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import datasets
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--hyper_dim', type=int, default=8)
parser.add_argument('--hyper_hidden', type=int, default=8)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

#python3 odenet_boston.py --hyper_dim 256 --hyper_hidden 2 --lr 0.01


class ODEfunc(nn.Module):

    def __init__(self, dim, hypernet_dim, hypernet_hidden_layers, activation=nn.Tanh):
        super(ODEfunc, self).__init__()
        self.dim = dim
        self.params_dim = self.dim**2 + self.dim
        self.activation = activation

        print('Number of outputs in hypernet:', self.params_dim)

        layers = []
        dims = [1] + [hypernet_dim] * \
            hypernet_hidden_layers + [self.params_dim]

        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(self.activation())

        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim].view(self.dim)
        w = params[self.dim:].view(self.dim, self.dim)

        out = 0.5*(nn.functional.linear(x, w, b) + nn.functional.linear(x, -w.t(), b))
        # out = nn.functional.linear(x, w, b)

        return nn.functional.tanh(out)



class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class BostonHousingTrain:
    def __init__(self, n_examples):
        self.n_examples = n_examples

        X, y = datasets.load_boston(return_X_y=True)

        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)

        self.X = X[:self.n_examples, :]
        self.y = y[:self.n_examples]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class BostonHousingTest:
    def __init__(self, n_examples):
        self.n_examples = n_examples

        X, y = datasets.load_boston(return_X_y=True)

        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)

        self.X = X[self.n_examples:, :]
        self.y = y[self.n_examples:]

    def __len__(self):
        return 506 - self.n_examples

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


def get_boston_housing_loaders(batch_size=128, test_train_split=16):

    train_loader = DataLoader(BostonHousingTrain(test_train_split),
                              batch_size=batch_size, shuffle=True,
                              num_workers=1, drop_last=False)

    test_loader = DataLoader(BostonHousingTest(test_train_split),
                             batch_size=batch_size, shuffle=True,
                             num_workers=1, drop_last=False)

    return train_loader, test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def compute_loss(loss_func, model, dataset_loader):
	for x, y in dataset_loader:
		x = x.float().to(device)
		y = y.float().to(device)

	return loss_func(torch.flatten(model(x)), y)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.normal_(m.bias, 0, 0.01)


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(
        args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    feature_layers = [ODEBlock(ODEfunc(13, args.hyper_dim, args.hyper_hidden))]
    fc_layers = [nn.Linear(13, 1)]

    model = nn.Sequential(*feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.MSELoss().to(device)

    train_loader, test_loader = get_boston_housing_loaders(args.batch_size)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    # lr_fn = learning_rate_with_decay(
    #     args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
    #     decay_rates=[1, 0.1, 0.01, 0.001]
    # )

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()

        x = x.float().to(device)
        y = y.float().to(device)
        loss = criterion(torch.flatten(model(x)), y)

        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_loss = compute_loss(criterion, model, train_loader)
                # test_loss = compute_loss(criterion, model, test_loader)
                # logger.info(
                #     "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                #     "Train Loss {:.4f} | Test Loss {:.4f}".format(
                #         itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                #         b_nfe_meter.avg, train_loss, test_loss
                #     )
                # )

                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Loss {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_loss
                    )
                )
