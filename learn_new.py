import librosa
import numpy as np
import functools as ft
import scipy.io.wavfile as siw
import scipy
import scipy.signal
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import torch
import torch.nn as nn
import random
import math

# 超参数
vec_num = 12
"""基础mfcc长度"""
class_num = 13
"""音频类型数目"""
cut_len = 5
"""音频剪切的长度(秒)"""
mfcc_len = 36
"""最终mfcc的长度"""
random.seed(2333)
"""随机种子"""
USE_CUDA = True
"""是否使用GPU"""
epochs = 1000
"""训练次数"""
test_rate = 0.2
"""测试集占比"""
write_wrong = True
"""是否输出预测错误的数据"""
write_path = "C:/Users/dark7/Desktop/大创项目/PyCharm/test 1/"


class WavData:
    sample_rate: int = -1
    data: np.ndarray

    def load(self, file_path):
        self.data, self.sample_rate = librosa.load(file_path)

    def cut(self, begin: float, duration: float):
        if self.sample_rate == -1:
            print("the WavData is empty!")
            raise Exception("the WavData is empty!")

        output = WavData()
        output.sample_rate = self.sample_rate
        x = int(begin * self.sample_rate)
        y = x + int(duration * self.sample_rate)
        output.data = self.data[x: y]
        return output

    def write(self, file_path) -> None:
        if self.sample_rate == -1:
            print("the WavData is empty!")
            raise Exception("the WavData is empty!")
        scipy.io.wavfile.write(file_path, self.sample_rate, self.data)


def mfcc_difference(mfcc: np.ndarray, k=1) -> np.ndarray:
    """
    对mfcc取差分
    """
    dif_mfcc = np.zeros(mfcc.shape)
    if k == 1:
        dif_mfcc[1:-1] = (mfcc[2:] - mfcc[:-2]) / 2
        dif_mfcc[0] = mfcc[1] - mfcc[0]
        dif_mfcc[-1] = mfcc[-1] - mfcc[-2]
    else:
        dif_mfcc[2:-2] = (2 * (mfcc[4:] - mfcc[:-4]) + mfcc[3:-1] - mfcc[1:-3]) / 10
        dif_mfcc[0] = mfcc[1] - mfcc[0]
        dif_mfcc[1] = mfcc[2] - mfcc[1]
        dif_mfcc[-1] = mfcc[-1] - mfcc[-2]
        dif_mfcc[-2] = mfcc[-2] - mfcc[-3]
    return dif_mfcc


def my_mfcc(wav_data: WavData) -> np.ndarray:  # 用librosa库获得mfcc
    """
    用librosa库获得mfcc
    """

    # y:采样点序列，sr：采样率，S：滤波后的序列（如果S!=None，那么y没有用），
    # n_mfcc：mfcc维数，dct_type：dct模式，norm：dct是否使用正交基
    mfcc = librosa.feature.mfcc(
        y=wav_data.data, sr=wav_data.sample_rate, n_mfcc=vec_num, dct_type=2, norm='ortho'
    )

    mfcc_1 = mfcc_difference(mfcc, k=1)
    mfcc_2 = mfcc_difference(mfcc_1, k=1)
    mfcc = np.vstack((mfcc, mfcc_1, mfcc_2))

    return mfcc


def get_file_name_by_path(path: str):
    p = 0
    for i in range(len(path)):
        if path[i] == '/' or path[i] == '\\':
            p = i + 1
    return path[p:]


class MfccData:
    name: str    # 文件地址
    begin: float    # 开始时刻
    length: float   # 时长
    tag: int        # 标签
    num: int        # 属于第几段
    data: WavData   # float类型wav数据
    mfcc: np.ndarray  # mfcc处理数据

    def __init__(self, full_data: WavData, begin: float, duration: float, tag: int, path: str = "", cnt: int = 0):
        """
        获取path路径音频[start,start+duration]段，单位为秒

        tag为此音频标签
        """
        self.name = get_file_name_by_path(path)
        self.begin = begin  # 开始时刻
        self.length = duration  # 时长
        self.tag = tag  # 标签
        self.num = cnt  # 属于第几段
        self.data = full_data.cut(begin, duration)
        self.mfcc = my_mfcc(self.data)
        self.mfcc.resize((1, self.mfcc.shape[0], self.mfcc.shape[1]))

    def write(self, file_name: str) -> None:
        self.data.write(file_name)


def audio_cut(data: [MfccData], path: str, total: int, tag: int) -> None:
    """
    音频取前total秒剪切并放入data，共total/cut_len段，每一小段的时长为cut_len
    """
    full_wav = WavData()
    full_wav.load(path)

    for i in range(cut_len, total, cut_len):
        data.append(MfccData(full_wav, i - cut_len, cut_len, tag, path))


def get_data() -> (torch.FloatTensor, torch.IntTensor, torch.FloatTensor, torch.IntTensor, [MfccData], int, int):
    """
    返回训练集/测试集的x与y (train_x, train_y, test_x, test_y)
    返回测试集数量与全体数据集数量
    """
    data: [MfccData] = []
    abs_path = "C:/Users/dark7/Desktop/大创项目/data/"
    audio_cut(data, abs_path + "0/preview_vocals_广告配音2-男声-沉稳.wav", 71, 0)
    audio_cut(data, abs_path + "0/preview_vocals_广告配音10-男声-沉稳.wav", 37, 0)
    audio_cut(data, abs_path + "0/preview_vocals_广告配音11-男声-沉稳.wav", 31, 0)

    audio_cut(data, abs_path + "1/preview_vocals_广告配音1-女声-活泼.wav", 21, 1)
    audio_cut(data, abs_path + "1/preview_vocals_广告配音4-女声-活泼.wav", 71, 1)
    audio_cut(data, abs_path + "1/preview_vocals_叙述4-女声-平稳.wav", 71, 1)

    audio_cut(data, abs_path + "2/preview_vocals_叙述3-女声-平稳.wav", 71, 2)
    audio_cut(data, abs_path + "2/preview_vocals_叙述7-女声-平稳.wav", 71, 2)

    audio_cut(data, abs_path + "3/preview_vocals_叙述1-女声-平稳.wav", 71, 3)
    audio_cut(data, abs_path + "3/preview_vocals_叙述13-男声-沉稳.wav", 71, 3)

    audio_cut(data, abs_path + "4/preview_vocals_朗诵1-男声-深情.wav", 71, 4)
    audio_cut(data, abs_path + "4/preview_vocals_冥想2-女声.wav", 71, 4)
    audio_cut(data, abs_path + "4/preview_vocals_冥想2-女声.wav", 71, 4)

    random.shuffle(data)

    train_x = np.zeros((1, mfcc_len, 216))
    train_y = np.zeros(1)

    test_x = np.zeros((1, mfcc_len, 216))
    test_y = np.zeros(1)

    total_size = len(data)
    test_size = int(total_size * test_rate)

    for i in data[:test_size]:
        test_x = np.append(test_x, i.mfcc, axis=0)
        test_y = np.append(test_y, i.tag)

    for i in data[test_size:]:
        train_x = np.append(train_x, i.mfcc, axis=0)
        train_y = np.append(train_y, i.tag)

    train_x = np.delete(train_x, 0, axis=0)
    train_y = np.delete(train_y, 0)

    test_x = np.delete(test_x, 0, axis=0)
    test_y = np.delete(test_y, 0)

    train_x = train_x.swapaxes(0, 2)
    train_x = train_x.swapaxes(1, 2)
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    train_y = train_y.long()

    test_x = test_x.swapaxes(0, 2)
    test_x = test_x.swapaxes(1, 2)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    test_y = test_y.long()

    if USE_CUDA:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y, data, test_size, total_size


class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None表示h0、c0初始均为0张量
        out = self.linear(r_out[-1, :, :])  # 选最后一次的隐藏层作为输出
        return out


def train_model(model, data, target, criterion, optimizer, epochs_num) -> nn.Module:
    for epoch in range(epochs_num):
        print('#### Epoch {}/{} ####'.format(epoch, epochs_num - 1))
        model.train()  # 启用batch normalization和drop out
        out = model(data)  # 前向传播
        pre_lab = torch.argmax(out, 1)  # 得到预测值
        loss = criterion(out, target)  # 计算损失
        optimizer.zero_grad()  # 清空运算图
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降
        print('lost =', loss.item())
    return model


def judge(model, train_x, train_y, test_x, test_y, mfcc_data: [MfccData], test_size: int, total_size: int) -> None:
    model.eval()

    out = model(train_x)
    pre_lab = torch.argmax(out, 1)
    count_correct = ft.reduce(lambda x, y: x + int(y[0] == y[1]), zip(pre_lab, train_y), 0)
    print('train accuracy = {}/{}={:.3f}'.format(count_correct, len(pre_lab), count_correct / len(pre_lab)))

    out = model(test_x)
    pre_lab = torch.argmax(out, 1)
    flag_array = [int(x[0] == x[1]) for x in zip(pre_lab, test_y)]
    count_correct = sum(flag_array)
    print(' test accuracy = {}/{}={:.3f}'.format(count_correct, len(pre_lab), count_correct / len(pre_lab)))

    if write_wrong:
        for i in range(len(pre_lab)):
            if pre_lab[i] == 1:
                continue
            md: MfccData = mfcc_data[i]
            md.write(write_path + "output/" + "{} fact {} predict {} start {}.wav".format(
                md.name, test_y[i], pre_lab[i], md.begin))

    return


def main():
    train_x, train_y, test_x, test_y, mfcc_data, test_size, total_size = get_data()  # 得到测试集和数据集
    lstm_model = LstmNet(input_size=train_x.shape[2], hidden_size=64, output_size=5)  # 得到总模型
    if USE_CUDA:
        lstm_model = lstm_model.cuda()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)  # 使用Adam梯度优化策略
    loss_func = nn.CrossEntropyLoss()  # 计算损失函数，交叉熵 square(y-y‘).sum
    lstm_model = train_model(lstm_model, train_x, train_y, loss_func, optimizer, epochs_num=epochs)  # 训练模型
    judge(lstm_model, train_x, train_y, test_x, test_y, mfcc_data, test_size, total_size)  # 检验模型正确率


def wav_io_test():
    data: [MfccData] = []
    abs_path = "C:/Users/dark7/Desktop/大创项目/data/"
    audio_cut(data, abs_path + "0/preview_vocals_广告配音2-男声-沉稳.wav", 10, 0)
    data[0].write("write_test.wav")


if __name__ == '__main__':
    print("program begin!")
    # wav_io_test()
    main()
