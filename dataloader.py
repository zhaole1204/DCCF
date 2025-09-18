import os

import numpy as np
import scipy.io
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import Dataset
# from torch.nn.functional import normalize
from utils import *


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.V1 = data1
        self.V2 = data2
        self.Y = labels

    def __len__(self):
        return self.V1.shape[0]

    def get(self):
        return [self.V1, self.V2]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
            self.V2[idx])], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()


class Scene15():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Scene15.mat')
        X_data = data['X']
        scaler = MinMaxScaler()
        self.Y = np.array(np.squeeze(data['Y'])).astype(np.int32).reshape(4485, )
        self.V1 = scaler.fit_transform(X_data[0, 0].astype(np.float32))
        self.V2 = scaler.fit_transform(X_data[0, 1].astype(np.float32))
        self.V3 = scaler.fit_transform(X_data[0, 2].astype(np.float32))

    def __len__(self):
        return 4485

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.V1 = np.load(path + 'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = np.load(path + 'SIFT.npy').astype(np.float32)
        self.V3 = np.load(path + 'MFCC.npy').astype(np.float32)
        self.Y = np.load(path + 'label.npy')

    def __len__(self):
        return 6773

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
            x2), torch.from_numpy(x3)], torch.from_numpy(self.Y[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def get(self):
        return [self.V1, self.V2]

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class MSRCv1():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MSRCv1.mat')
        X_data = data['X']
        scaler = MinMaxScaler()
        self.Y = np.array(np.squeeze(data['Y'])).astype(np.int32).reshape(210, )
        self.V1 = scaler.fit_transform(X_data[0, 0].astype(np.float32))
        self.V2 = scaler.fit_transform(X_data[0, 1].astype(np.float32))
        self.V3 = scaler.fit_transform(X_data[0, 2].astype(np.float32))
        self.V4 = scaler.fit_transform(X_data[0, 3].astype(np.float32))
        self.V5 = scaler.fit_transform(X_data[0, 4].astype(np.float32))

    def __len__(self):
        return 210

    def get(self):
        return [self.V1, self.V2, self.V3, self.V4, self.V5]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class COIL20():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'COIL20.mat')
        X_data = data['X']
        scaler = MinMaxScaler()
        self.Y = np.array(np.squeeze(data['Y'])).astype(np.int32).reshape(1440, )
        self.V1 = scaler.fit_transform(X_data[0, 0].astype(np.float32))
        self.V2 = scaler.fit_transform(X_data[0, 1].astype(np.float32))
        self.V3 = scaler.fit_transform(X_data[0, 2].astype(np.float32))

    def __len__(self):
        return 1440

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return ([torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],
                self.Y[idx], torch.from_numpy(np.array(idx)).long())


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.Y = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def get(self):
        return [self.V1, self.V2, self.V3, self.V4, self.V5]

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.V1[idx]), torch.from_numpy(self.V2[idx])], torch.from_numpy(
                self.Y[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
                self.V2[idx]), torch.from_numpy(self.V5[idx])], torch.from_numpy(
                self.Y[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), torch.from_numpy(
                self.V5[idx]), torch.from_numpy(self.V4[idx])], torch.from_numpy(
                self.Y[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.V1[idx]), torch.from_numpy(
                self.V2[idx]), torch.from_numpy(self.V5[idx]), torch.from_numpy(
                self.V4[idx]), torch.from_numpy(self.V3[idx])], torch.from_numpy(
                self.Y[idx]), torch.from_numpy(np.array(idx)).long()


class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 50000

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 50000

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 600

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()


class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 551

    def get(self):
        return [self.V1, self.V2, self.V3]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000, )
        scaler = MinMaxScaler()
        # self.V1 = data['data'][0][0].T.astype(np.float32)
        # self.V2 = data['data'][0][1].T.astype(np.float32)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))

    def __len__(self):
        return 10000

    def get(self):
        return [self.V1, self.V2]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Hand():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'handwritten.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X'][0][5].astype(np.float32))

    def __len__(self):
        return 2000

    def get(self):
        return [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class F_MNIST(Dataset):
    def __init__(self, npz_file, view):
        data = np.load(npz_file)
        x1 = np.reshape(data['view_0'], (60000, 28, 28, 1), order='C')
        x2 = np.reshape(data['view_1'], (60000, 28, 28, 1), order='C')
        self.view1 = x1.astype(np.float32)
        self.view2 = x2.astype(np.float32)
        # self.view1 = np.reshape(x1, (60000, 28, 28, 1), order='C')
        # self.view2 = np.reshape(x2, (60000, 28, 28, 1), order='C')
        Y = data['labels'].astype(np.int32)
        self.labels = np.reshape(Y, 60000)
        self.len = len(self.labels)

    def __len__(self):
        # return 60000
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.view1[idx].reshape(784)
        item1 = self.view2[idx].reshape(784)
        # label = self.labels[idx]
        return [torch.from_numpy(item), torch.from_numpy(item1)], self.labels[idx], torch.from_numpy(
            np.array(idx)).long()


class YouTubeFace(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YouTubeFace.mat')
        scaler = MinMaxScaler()
        self.Y = data['Y'].astype(np.int32).reshape(101499, )
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))

    def __len__(self):
        return 101499

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return ([torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),
                 torch.from_numpy(x4), torch.from_numpy(x5)],
                self.Y[idx], torch.from_numpy(np.array(idx)).long())


class LabelMe(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path + 'LabelMe.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path + 'LabelMe.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'LabelMe.mat')['Y']  # .transpose()
        self.y = labels

    def __len__(self):
        return 2688

    def __getitem__(self, idx):
        return ([torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])],
                torch.from_numpy(
            self.y[idx]), torch.from_numpy(np.array(idx)).long())

class Caltech_6V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
        # for i in [0, 3]:
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        # print(self.class_num)
        # for i in range(5000):
        #     print(data['X1'][i][-1])
        # X1 = data['X1'][:, :-1]
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class YoutubeVideo(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])

        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 3304, 6750]
        view = 3
        class_num = 20
        data_size = 1440
    elif dataset == "Scene15":
        dataset = Scene15('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "MSRCv1":
        dataset = MSRCv1('./data/')
        dims = [24, 576, 512, 256, 254]
        view = 5
        data_size = 210
        class_num = 7
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Synthetic3d":
        dataset = synthetic3d('./data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Prokaryotic":
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "Hand":
        dataset = Hand('./data/')
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "LabelMe":
        dataset = LabelMe('./data/')
        dims = [512, 245]
        view = 2
        data_size = 2688
        class_num = 8
    elif dataset == "Caltech":
        dataset = Caltech_6V('data/Caltech.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "YoutubeVideo":
        dataset = YoutubeVideo("./data/Video-3V.mat", view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "Caltech":
        dataset = Caltech_6V('data/Caltech.mat', view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/NUSWIDE.mat', view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA('data/DHA.mat', view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "YoutubeVideo":
        dataset = YoutubeVideo("./data/Video-3V.mat", view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
