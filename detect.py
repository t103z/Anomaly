# encoding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from dataset import load_dataset
from detector import StlDtector, RpcaDetector
from sklearn.ensemble import IsolationForest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score


def isoForests(Training_data,contamination=0.01):
    clf = IsolationForest(contamination=contamination,n_estimators=100)
    label=np.ones([len(Training_data),1])
    anomaly_score=np.ones([len(Training_data),1])
    for i in range(int(len(Training_data)/1440)):
        print(i)
        clf.fit(Training_data[i*1440:(i+1)*1440,:])
        label[i*1440:(i+1)*1440,0] = clf.predict(Training_data[i*1440:(i+1)*1440,:])
        anomaly_score[i*1440:(i+1)*1440,0] = clf.decision_function(Training_data[i*1440:(i+1)*1440,:])  # the lower the score is, the more anomaly the sample is.
    return anomaly_score,label


if __name__ == '__main__':
    random.seed(13)         # random seed for reproductivity
    train = False
    train_set, train_label, test_set, test_label = load_dataset(ratio=0.01, ret_set=False, amount_crpt=True, res_crpt=True, succ_crpt=False, break_down=True, pick=False)

    print len(train_label)

    res_time_train = train_set[3, :]
    amount_train = train_set[4, :]
    succ_train = train_set[5, :]
    res_time_test = test_set[3, :]
    amount_test = test_set[4, :]
    succ_test = test_set[5, :]

    day_len = 1440
    week_len = 7 * day_len

    # anom_score, pred = isoForests(np.transpose(test_set))

    # print 'plotting'
    # plt.scatter(amount, res_time)
    # plt.ylim([0, 500])
    # plt.xlim([0, 500])
    # plot_hist(np.divide(res_time, amount)[:1440], 500)
    # plt.show()

    # data = succ
    # data, corrupt_ind = corrupt(data)
    # plt.plot(data[day_len:10 * day_len])
    # plt.show()

    interval = 14 * day_len

    stl_detector_succ = StlDtector(period=day_len, interval=7 * day_len)
    stl_detector_res = StlDtector(period=day_len, interval=7 * day_len, esd_rate=0.01, nt=3)
    stl_detector_pick = StlDtector(period=day_len, interval=7 * day_len)
    stl_detector_bd = StlDtector(period=day_len, interval=14 * day_len)
    rpca_detector_pick = RpcaDetector(period=day_len, esd_rate=0.01, interval=7 * day_len)
    rpca_detector_bd = RpcaDetector(period=day_len)
    rpca_detector_res = RpcaDetector(period=day_len, interval=14 * day_len, lamb_rate=0.8, esd_rate=0.011)
    rpca_detector_succ = RpcaDetector(period=day_len, interval=4 * day_len)

    # anom_ind = rpca_detector_succ.detect_anom(data)
    # anom_ind = stl_detector_succ.detect_anom(data)
    anom_detect = []
    anom_detect.append(stl_detector_pick.detect_anom(amount_train, ret_set=False))
    anom_detect.append(stl_detector_bd.detect_anom(amount_train, ret_set=False, plot=True))
    anom_detect.append(stl_detector_succ.detect_anom(succ_train, ret_set=False))
    anom_detect.append(stl_detector_res.detect_anom(res_time_train, ret_set=False))
    anom_detect.append(rpca_detector_pick.detect_anom(amount_train, ret_set=False))
    anom_detect.append(rpca_detector_bd.detect_anom(amount_train, ret_set=False))
    anom_detect.append(rpca_detector_succ.detect_anom(succ_train, ret_set=False))
    anom_detect.append(rpca_detector_res.detect_anom(res_time_train, ret_set=False))
    anom_detect_train = np.transpose(np.vstack(anom_detect))

    anom_detect = []
    anom_detect.append(stl_detector_pick.detect_anom(amount_test, ret_set=False))
    anom_detect.append(stl_detector_bd.detect_anom(amount_test, ret_set=False))
    anom_detect.append(stl_detector_succ.detect_anom(succ_test, ret_set=False))
    anom_detect.append(stl_detector_res.detect_anom(res_time_test, ret_set=False))
    anom_detect.append(rpca_detector_pick.detect_anom(amount_test, ret_set=False))
    anom_detect.append(rpca_detector_bd.detect_anom(amount_test, ret_set=False))
    anom_detect.append(rpca_detector_succ.detect_anom(succ_test, ret_set=False))
    anom_detect.append(rpca_detector_res.detect_anom(res_time_test, ret_set=False))
    anom_detect_test = np.transpose(np.vstack(anom_detect))

    lr = LogisticRegression()
    svc = SVC()

    print anom_detect_train.shape
    print train_label.shape

    lr.fit(anom_detect_train, train_label)

    pred = lr.predict(anom_detect_test)

    # pred = []
    # for i in range(anom_detect_test.shape[0]):
    #     pred.append(np.max(anom_detect_test[i, :]))

    precision = precision_score(test_label, pred)
    recall = recall_score(test_label, pred)
    f1 = f1_score(test_label, pred)

    # evaluation
    # if train is True:
    #     correct = train_label & anom_ind
    #     precision = float(len(correct)) / len(anom_ind)
    #     recall = float(len(correct)) / len(train_label)
    # else:
    #     correct = test_label & anom_ind
    #     precision = float(len(correct)) / len(anom_ind)
    #     recall = float(len(correct)) / len(test_label)
    #
    # f1 = stats.hmean([precision, recall])

    print 'Precision: %f' % precision
    print 'Recall: %f' % recall
    print 'F1: %f' % f1
