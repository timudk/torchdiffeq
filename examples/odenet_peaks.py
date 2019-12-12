import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str,
                    choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str,
                    default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval,
                    default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='NONE')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--hypernet_dim', type=int, required=True)
parser.add_argument('--hypernet_n_layers', type=int, required=True)
parser.add_argument('--end_time', type=float, default=1.0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

N_CLASSES = 2
N_LIFTING = 0
# HYPERNET_DIM = 7
# HYPERNET_HIDDEN_LAYERS = 2
BATCHSIZE = 128
data_file = '../peaks_data/classes_2/length_6_points_40.pkl'

# python3 odenet_peaks.py --batch_size 51 --lr 0.1 --nepochs 400 --tol 1e-4 --adjoint True N_LIFTING = 2 HYPERNET_DIM = 64 HYPERNET_HIDDEN_LAYERS = 8


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0, 0.1)
        nn.init.normal_(m.bias, 0, 0.1)


class PeaksTrainingSet(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
            # self.data[0] = self.data[0][:256]

        self.x = np.zeros((len(self.data[0]), 2 + N_LIFTING))
        self.y = np.zeros((len(self.data[0]),), dtype=np.int32)

        for i in range(len(self.data[0])):
            self.x[i, 0] = self.data[0][i][0]
            self.x[i, 1] = self.data[0][i][1]
            # if N_LIFTING % 2 == 0:
                # for j in range(1 + int(N_LIFTING / 2)):
                #     self.x[i, 2 * j] = self.data[0][i][0]
                #     self.x[i, 2 * j + 1] = self.data[0][i][1]
            self.y[i] = self.data[0][i][2]

        # plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y[:])
        # plt.show()

        print('Number of training points:', self.__len__())

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])

        return sample


class PeaksTestSet(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.x = np.zeros((len(self.data[1]), 2 + N_LIFTING))
        self.y = np.zeros((len(self.data[1]),), dtype=np.int32)

        for i in range(len(self.data[1])):
            self.x[i, 0] = self.data[1][i][0]
            self.x[i, 1] = self.data[1][i][1]
            # if N_LIFTING % 2 == 0:
            #     for j in range(1 + int(N_LIFTING / 2)):
            #         self.x[i, 2 * j] = self.data[1][i][0]
            #         self.x[i, 2 * j + 1] = self.data[1][i][1]
            self.y[i] = self.data[1][i][2]

        print('Number of test points:', self.__len__())

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])

        return sample


class ODEfunc(nn.Module):

    def __init__(self, dim, hypernet_dim, hypernet_hidden_layers, activation=nn.Tanh):
        super(ODEfunc, self).__init__()
        self.dim = dim
        self.params_dim = self.dim**2 + self.dim

        print('Number of parameters in output net:', self.params_dim)

        layers = []
        dims = [1] + [hypernet_dim] * \
            hypernet_hidden_layers + [self.params_dim]

        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

        print('Number of parameters in hypernet:', sum(p.numel()
                                                       for p in self._hypernet.parameters() if p.requires_grad))

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim].view(self.dim)
        w = params[self.dim:].view(self.dim, self.dim)

        # out = nn.functional.tanh(
        #     0.5 * (F.linear(x, w, b) + F.linear(x, -w.t(), b)))
        out = nn.functional.tanh(F.linear(x, w, b))
        # out = F.linear(x, w, b)
        return out

# class ODEfunc(nn.Module):

#     def __init__(self, dim, hypernet_dim, hypernet_hidden_layers, activation=nn.Tanh):
#         super(ODEfunc, self).__init__()
#         self.dim = dim
#         self.params_dim = self.dim**2 + 2*self.dim

#         print('Number of parameters in output net:', self.params_dim)

#         layers = []
#         dims = [1] + [hypernet_dim] * \
#             hypernet_hidden_layers + [self.params_dim]

#         for i in range(1, len(dims)):
#             layers.append(nn.Linear(dims[i - 1], dims[i]))
#             if i < len(dims) - 1:
#                 layers.append(activation())
#         self._hypernet = nn.Sequential(*layers)
#         self._hypernet.apply(weights_init)

#         print('Number of parameters in hypernet:', sum(p.numel()
#                                                        for p in self._hypernet.parameters() if p.requires_grad))

#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1

#         # print(x)
#         # print(x.narrow(1, 0, self.dim))

#         params = self._hypernet(t.view(1, 1)).view(-1)
#         b = params[:(2*self.dim)].view(2*self.dim)
#         w = params[(2*self.dim):].view(self.dim, self.dim)

#         out_1 = F.linear(x.narrow(1, 0, self.dim), w)
#         out_2 = F.linear(x.narrow(1, self.dim, self.dim), -w.t())
#         # print(out_1)
#         # print(out_2)

#         out = torch.cat((out_1, out_2), 1) + b

#         return nn.functional.tanh(out)


class ODEBlock(nn.Module):

    def __init__(self, odefunc, end_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, end_time]).float()

    def forward(self, x):
        # print('ODEBlock forward pass.')

        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=args.tol, atol=args.tol, method='fixed_adams')
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


def get_peaks_loaders(data_aug=False, batch_size=BATCHSIZE, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        PeaksTrainingSet(data_file), batch_size=batch_size,
        shuffle=True, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        PeaksTestSet(data_file), batch_size=test_batch_size,
        shuffle=False, num_workers=1, drop_last=False
    )

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


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        x = x.float()
        y = one_hot(np.array(y.numpy()), N_CLASSES)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


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


if __name__ == '__main__':

    # makedirs(args.save)
    # logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    # logger.info(args)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    feature_layers = [
        ODEBlock(ODEfunc(2 + N_LIFTING, args.hypernet_dim, args.hypernet_n_layers), args.end_time)]
    # feature_layers = [ODEBlock(ODEfunc(2, HYPERNET_DIM, HYPERNET_HIDDEN_LAYERS))]
    fc_layers = [nn.Linear(2 + N_LIFTING, N_CLASSES)]

    # model = nn.Sequential(*feature_layers, *fc_layers).to(device)
    model = nn.Sequential(*feature_layers, *fc_layers).to(device)

    # logger.info(model)
    # logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = get_peaks_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    # lr_fn = learning_rate_with_decay(
    #     args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
    #     decay_rates=[1, 0.1, 0.01, 0.001]
    # )

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    train_accs = []
    val_accs = []

    is_converged = False

    itr = 0
    while not is_converged and itr < args.nepochs * batches_per_epoch:

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()

        x = x.to(device)
        y = y.to(device)

        x = x.float()
        y = y.long()

        logits = model(x)
        # print(logits.shape)
        loss = criterion(logits, y)
        # print('Loss:', loss)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        # pdb.set_trace()
        # print('Loss.')
        loss.backward()
        # print('Optimizer step.')
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if (itr+1) % (10 * batches_per_epoch) == 0 and not is_converged:
            with torch.no_grad():
                # plt.scatter(model[0](x)[:, 0], model[0](x)[:, 1], c=y[:])
                # plt.show()
                # print('Epoch:', itr // batches_per_epoch)
                train_accs.append(accuracy(model, train_loader))
                # print('Training accuracy:', train_acc)
                # val_acc = accuracy(model, test_loader)
                val_accs.append(accuracy(model, test_loader))
                # print('Test accuracy:', val_acc)
                # if val_acc > best_acc:
                #     if args.save != 'NONE':
                #         torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(
                #             args.save, 'model.pth'))
                #     best_acc = val_acc
                # logger.info(
                #     "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                #     "Test Acc {:.4f} | Train Acc {:.4f}".format(
                #         itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                #         b_nfe_meter.avg, val_acc, train_acc
                #     )
                # )
                print("Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                      "Test Acc {:.4f} | Train Acc {:.4f}".format(
                          (itr+1) // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                          b_nfe_meter.avg, val_accs[-1], train_accs[-1]
                      ))

                if(len(val_accs) >= 3):
                    if(np.abs(val_accs[-1] - val_accs[-2]) < 0.001) and (np.abs(val_accs[-1] - val_accs[-3]) < 0.001):
                        is_converged = True
                        print('Converged at epoch: {}'.format((itr+1) // batches_per_epoch))

        itr += 1

    if not is_converged:
        print('Model has not converged.')

    with torch.no_grad():
        x_0 = np.linspace(-6, 6, 101)
        x_1 = np.ones(101)

        x = np.array([x_0, x_1]).T

        # x = x.to(device)
        # y = y.to(device)

        x = torch.from_numpy(x).float().to(device)
        print(x.shape)

        plt.scatter(model[0](x)[:, 0], model[0](x)[:, 1])
        plt.plot(x_0, x_1)
        plt.show()
