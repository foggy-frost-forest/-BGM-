import librosa
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import math
import os
import shutil
import pickle
import math
import scipy

# 文件路径
data_path = "./data/"
"""数据路径"""
log_path = "./logs/"
"""log文件路径"""
train_data_path = data_path + "train/"
"""训练集路径"""
test_data_path = data_path + "test/"
"""测试集路径"""
demo_data_path = data_path + 'demo/'
"""展示mfcc路径"""

# 数据处理参数
cut_len = 6
"""音频剪切的长度(秒)"""
tag_list = ["激昂类", "轻快类", "伤感类", "温和类"]
"""标签列表"""

# GRU超参数
USE_CUDA = torch.cuda.is_available() & True
"""是否使用GPU"""

net_class_num = len(tag_list)
"""音频类型数目"""
net_learn_rate = 0.003
"""学习率"""
net_batch_size = 400
"""batch size"""

random_seed = 233
"""随机种子"""
wav_sample_rate = 22500
"""取样率"""
skip_train = False
"""是否跳过训练阶段"""
tensorboard_mode = False


class mydataset(Dataset):
    def __init__(self, path, mode):
        self.file_list = os.listdir(path)
        self.mfcc = []
        self.target = []
        self.belong = []
        self.mode = mode

        for file in self.file_list:
            file_tag = file.split(".")[0].split("-")[0]
            if mode == 'online':
                self.audio_cut(path, file, -1)
            elif file_tag in tag_list:
                tag = tag_list.index(file_tag)
                self.audio_cut(path, file, tag)
                print(path + file)
            elif file.split(".")[-1] == "wav":
                print(file + "is not in tag_list")

        self.len = len(self.mfcc)
        self.mfcc = torch.Tensor(self.mfcc)
        self.target = torch.Tensor(self.target)
        self.target = self.target.long()

        if USE_CUDA:
            self.mfcc = self.mfcc.cuda()
            self.target = self.target.cuda()

    def __getitem__(self, item):
        return self.mfcc[item], self.target[item]

    def __len__(self):
        return self.len

    def audio_cut(self, path, file, tag):
        global wav_sample_rate, cut_len
        signal, sample_rate = librosa.load(path + file, sr=wav_sample_rate, mono=True)
        """归一化"""
        smax = signal[1 * wav_sample_rate:-1 * wav_sample_rate].max()
        smin = signal[1 * wav_sample_rate:-1 * wav_sample_rate].min()
        norm = max(abs(smax), abs(smin))
        signal = signal / norm
        for i in range(wav_sample_rate):
            if signal[i] > 1:
                signal[i] = 1
            elif signal[i] < -1:
                signal[i] = -1
            if signal[-i] > 1:
                signal[-i] = 1
            elif signal[-i] < -1:
                signal[-i] = -1
        j = 0
        while j < cut_len:
            i = int(math.ceil(j * sample_rate))
            while (i + cut_len * sample_rate) < len(signal):
                self.target.append(tag)

                mfcc = librosa.feature.mfcc(y=signal[i:i + cut_len * sample_rate], sr=sample_rate, n_mfcc=13, dct_type=2, norm='ortho', lifter=40)[1:]
                mfcc_1 = self.mfcc_difference(mfcc, k=2)
                mfcc_2 = self.mfcc_difference(mfcc_1, k=2)
                self.mfcc.append(np.vstack((mfcc, mfcc_1, mfcc_2)).T)

                self.belong.append(self.file_list.index(file))
                i += cut_len * sample_rate
            if self.mode == 'test' or self.mode == 'online':
                break
            j += 0.5

    def mfcc_difference(self, mfcc: np.ndarray, k: int = 1) -> np.ndarray:
        """
        对mfcc取差分
        """
        dif_mfcc = np.zeros(mfcc.shape)
        if k == 1:
            dif_mfcc[1:-1] = (mfcc[2:] - mfcc[:-2]) / 2
            dif_mfcc[0] = mfcc[1] - mfcc[0]
            dif_mfcc[-1] = mfcc[-1] - mfcc[-2]
        elif k == 2:
            dif_mfcc[2:-2] = (2 * (mfcc[4:] - mfcc[:-4]) + mfcc[3:-1] - mfcc[1:-3]) / 10
            dif_mfcc[0] = mfcc[1] - mfcc[0]
            dif_mfcc[1] = mfcc[2] - mfcc[1]
            dif_mfcc[-1] = mfcc[-1] - mfcc[-2]
            dif_mfcc[-2] = mfcc[-2] - mfcc[-3]
        else:
            print("Wrong in mfcc difference")
        return dif_mfcc


class CGNN(nn.Module):
    def __init__(self):
        super(CGNN, self).__init__()
        self.gru = nn.GRU(input_size=36, hidden_size=128, num_layers=3, bias=True, batch_first=True, dropout=0.8, bidirectional=True)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.linear2 = nn.Linear(64, net_class_num)

    def forward(self, mfcc):
        out, hn = self.gru(mfcc, torch.randn(3 * 2, mfcc.shape[0], 128).cuda())
        out = self.linear1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


def train_model(model, train_dataloader, test_dataloader, test_dataset):
    print("<<<<<<  training  model  >>>>>>")
    best_eval: float = 0
    best_model = model
    best_epoch: int = -1
    loss_func = nn.CrossEntropyLoss().cuda()  # 计算损失函数，交叉熵

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs_num = 50
    for epoch in range(epochs_num):
        print('#### Epoch {}/{} ####'.format(epoch + 1, epochs_num))
        total_loss = 0
        model.train()  # 启用batch normalization和drop out
        for mfcc, target in train_dataloader:
            out = model(mfcc)  # 前向传播
            loss = loss_func(out, target)  # 计算损失
            total_loss = total_loss + loss.item()
            optimizer.zero_grad()  # 清空梯度池
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降
        print('lost =', total_loss)
        train_acc, test_acc = judge(model, train_dataloader, test_dataloader)

        if tensorboard_mode:
            writer.add_scalar("loss", total_loss, epoch)
            writer.add_scalar("train accuracy", train_acc, epoch)
            writer.add_scalar("test accuracy", test_acc, epoch)

            demo(model, test_dataset, epoch)

        if best_eval < test_acc:
            best_model = model
            best_eval = test_acc
            best_epoch = epoch + 1

        if epoch % 10 == 9:
            print("****************epoch({}) : {}".format(epoch, best_eval))

    return best_model, best_eval, best_epoch, model


def judge(model, train_dataloader, test_dataloader):
    model.eval()

    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    for mfcc, target in train_dataloader:
        out = model(mfcc)
        pre_lab = torch.argmax(out, 1)
        train_correct += torch.sum(pre_lab == target).item()
        train_total += len(mfcc)

    for mfcc, target in test_dataloader:
        out = model(mfcc)
        pre_lab = torch.argmax(out, 1)
        test_correct += torch.sum(pre_lab == target).item()
        test_total += len(mfcc)

    print('train accuracy = {}/{} = {:.3f}'.format(train_correct, train_total, train_correct / train_total))
    print('test accuracy = {}/{} = {:.3f}'.format(test_correct, test_total, test_correct / test_total))

    return train_correct / train_total, test_correct / test_total


def file_judge(model, dataset: mydataset):
    print("evaluation of test files")
    book = np.zeros((len(dataset.file_list), net_class_num), dtype=np.int32)
    ans = np.zeros(len(dataset.file_list), dtype=np.int32)
    correct = 0
    almost = 0
    total = 0
    model.eval()
    for i in range(dataset.len):
        out = model(dataset.mfcc[i].reshape(1, dataset.mfcc[i].shape[0], dataset.mfcc[i].shape[1]))
        pre_lab = torch.argmax(out, 1)
        book[dataset.belong[i], pre_lab] += 1
        ans[dataset.belong[i]] = dataset.target[i]

    for file in dataset.file_list:
        if file.split('.')[-1] == 'wav':
            pos = dataset.file_list.index(file)
            total += 1
            cnt = 0
            for i in range(net_class_num):
                if i != ans[pos] and book[pos, i] >= book[pos, ans[pos]]:
                    cnt += 1
            flag = "wrong ******\n"
            if cnt <= 1:
                almost += 1
                flag = "almost **\n"
            if cnt == 0:
                correct += 1
                flag = '\n'
            print("{} ---- {} ---- {} ---".format(file, tag_list[ans[pos]], cnt), end=flag)
            for i in range(net_class_num):
                print("{} {:<3d} |  ".format(tag_list[i], book[pos, i]), end='')
            print("\n")
    print("file accuracy : {}/{} {}".format(correct, total, correct / total))
    print("file almost accuracy : {}/{} {}".format(almost, total, almost / total))


def online_judge(model, dataset: mydataset):
    print("evaluation of test files")
    book = np.zeros((net_class_num), dtype=np.int32)
    model.eval()
    for i in range(dataset.len):
        out = model(dataset.mfcc[i].reshape(1, dataset.mfcc[i].shape[0], dataset.mfcc[i].shape[1]))
        pre_lab = torch.argmax(out, 1)
        book[pre_lab] += 1
    return book


def demo(model, dataset: mydataset, epoch):
    book = np.zeros((len(dataset.file_list), net_class_num), dtype=np.int32)
    model.eval()
    for i in range(dataset.len):
        out = model(dataset.mfcc[i].reshape(1, dataset.mfcc[i].shape[0], dataset.mfcc[i].shape[1]))
        pre_lab = torch.argmax(out, 1)
        book[dataset.belong[i], pre_lab] += 1
    tag_map = {}
    for f in range(len(dataset.file_list)):
        if dataset.file_list[f].split('.')[-1] != 'wav':
            continue
        tag_map.clear()
        for t in range(len(tag_list)):
            tag_map[tag_list[t]] = book[f, t]
            writer.add_scalars(dataset.file_list[f], tag_map, epoch)


def init():
    random.seed(random_seed)
    file_list = os.listdir(log_path)
    for file in file_list:
        try:
            os.remove(log_path + file)
        except:
            shutil.rmtree(log_path + file)


if __name__ == '__main__':
    init()
    try:
        f = open(data_path + 'GRUtraindata.pkl', 'rb')
        train_dataset = pickle.load(f)
        f.close()
        f = open(data_path + 'GRUtestdata.pkl', 'rb')
        test_dataset = pickle.load(f)
        f.close()
        print("<<<<<<  loading  data  >>>>>>")
    except:
        print("<<<<<<  creating  data  >>>>>>")
        train_dataset = mydataset(train_data_path, 'train')  # 得到数据集
        test_dataset = mydataset(test_data_path, 'test')
        f = open(data_path + 'GRUtraindata.pkl', 'wb')
        pickle.dump(train_dataset, f)
        f.close()
        f = open(data_path + 'GRUtestdata.pkl', 'wb')
        pickle.dump(test_dataset, f)
        f.close()

    train_dataloader = DataLoader(train_dataset, batch_size=net_batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=net_batch_size, shuffle=False, drop_last=False, num_workers=0)

    if skip_train:
        f = open(data_path + 'GRUmodel.pkl', 'rb')
        best_model = pickle.load(f)
        forward_mode = 2
        f.close()
    else:
        CGRUmodel = CGNN()  # 得到总模型
        if tensorboard_mode:
            writer = SummaryWriter(log_path)
        if USE_CUDA:
            CGRUmodel = CGRUmodel.cuda()
        best_model, best_eval, best_epoch, last_model = train_model(CGRUmodel, train_dataloader, test_dataloader, test_dataset)  # 训练模型
        print("测试集正确率最高为：", best_eval)

    file_judge(best_model, test_dataset)
    best_model.eval()
    if tensorboard_mode:
        writer.add_graph(best_model, torch.randn(100, 264, 36).cuda())

    if not skip_train:
        save_flag = input('save model?[y/n]')
        if save_flag == 'y':
            f = open(data_path + 'GRUmodel.pkl', 'wb')
            pickle.dump(best_model, f)
            f.close()
