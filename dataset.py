# encoding=utf-8
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def load_df():
    df = pd.read_excel('1月.xls')
    df = df.append(pd.read_excel('2月.xls'), ignore_index=True)
    df = df.append(pd.read_excel('3月.xls'), ignore_index=True)
    df = df.append(pd.read_excel('4月.xls'), ignore_index=True)
    return df


def load_column(*column_names):
    df = load_df()
    ret = []
    for name in column_names:
        ret.append(np.array(df[name].tolist()))
    return ret


def corrupt_pick(data, ratio=0.01, delta=None):
    data = np.copy(data)
    if delta is None:
        delta = np.max(data) * 0.15
    ind = random.sample(range(len(data)), int(ratio * data.shape[0]))    # corrupt indices
    for num in ind:
        data[num] = random.gauss(data[num] + delta, delta * 0.1)
    return data, ind


def corrupt_breakdown(data, ratio=0.01):
    data = np.copy(data)
    day_len = 1440
    break_len_mean = int(ratio * data.shape[0] / (data.shape[0] / (day_len * 7)))
    # random break 1 dcay per week
    ind = []
    for i in range(data.shape[0] / (day_len * 7)):
        start = i * day_len * 7
        pick_day = random.randint(0, 6)    # pick a day
        start_point = random.randint(480, 900)      # pick time of day
        break_len = random.randint(break_len_mean - 10, break_len_mean + 10)           # pick break len
        break_mean = data[start + start_point] * 0.5
        break_std = data[start + start_point] * 0.05
        for j in range(break_len):
            data[start + start_point + j] = random.gauss(break_mean, break_std)
            ind.append(start + start_point + j)
    return data, ind


def corrupt(data, ratio=0.01):
    # data[7900:8000] = 0
    ind = set()
    ind_list = []
    data_corrupt, ind_list = corrupt_pick(data, ratio)
    # data_corrupt, ind_list = corrupt_breakdown(data, ratio)
    ind |= set(ind_list)
    return data_corrupt, ind


def corrupt_no_response(data, ratio=0.01, ind=None):
    data = np.copy(data)
    delay = 2000000
    stdev = 100000
    if ind is None:
        ind = random.sample(range(len(data)), int(ratio * data.shape[0]))
    for num in ind:
        data[num] = random.gauss(delay, stdev)
    return data, ind


def corrupt_success_rate(data, ratio=0.01, ind=None):
    data = np.copy(data)
    failure = 0.4
    if ind is None:
        ind = random.sample(range(len(data)), int(ratio * data.shape[0]))
    for num in ind:
        data[num] = random.uniform(0, failure)
    return data, ind


def corrupt_zero(data, ratio=0.01, ind=None):
    data = np.copy(data)
    if ind is None:
        ind = random.sample(range(len(data)), int(ratio * data.shape[0]))
    for num in ind:
        data[num] = random.uniform(0, failure)
    return data, ind


def load_dataset(ratio=0.01, amount_crpt=True, res_crpt=True, succ_crpt=True, pick=True, break_down=False, seasonal=False, ret_set=False):
    random.seed(13)
    date, time, res_time, amount, succ = load_column('date', 'time', 'response_time', 'amount', 'success_rate')
    succ = np.array([float(k.rstrip('%')) / 100 for k in succ])
    # extract day and month
    day = date % 100                                    # 通过整除100得到的余数是日
    month = np.array([int(d / 100) for d in date])        # 通过整除100得到的商值

    ind = set()
    X = np.vstack([day, month, time])

    if res_crpt is True:
        corrupt_res, corrupt_ind = corrupt_no_response(res_time, ratio)
        ind |= set(corrupt_ind)
        X = np.vstack((X, corrupt_res))
    else:
        X = np.vstack((X, res_time))

    if amount_crpt is True:
        if break_down is True:
            corrupt_amount, corrupt_ind = corrupt_breakdown(amount, ratio)
            ind |= set(corrupt_ind)
        if pick is True:
            corrupt_amount, corrupt_ind = corrupt_pick(amount, ratio)
            ind |= set(corrupt_ind)
        X = np.vstack((X, corrupt_amount))
    else:
        X = np.vstack((X, amount))

    if succ_crpt is True:
        corrupt_succ, corrupt_ind = corrupt_success_rate(succ, ratio)
        ind |= set(corrupt_ind)
        X = np.vstack((X, corrupt_succ))
    else:
        X = np.vstack((X, succ))


    test_split = 50 * 1440
    train_set = X[:, :test_split]
    test_set = X[:, test_split:]

    ind_list = np.zeros((X.shape[1]))
    for num in ind:
        ind_list[num] = 1

    train_label = np.array(ind_list[:test_split])
    test_label = np.array(ind_list[test_split:])

    if ret_set is True:
        train_label_set = set()
        test_label_set = set()
        for i in range(len(train_label)):
            if train_label[i] == 1:
                train_label_set.add(i)
        for i in range(len(test_label)):
            if test_label[i] == 1:
                test_label_set.add(i)
        return train_set, train_label_set, test_set, test_label_set
    else:
        return train_set, train_label, test_set, test_label


load_dataset()
