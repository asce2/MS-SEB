import torch
import torch.nn as nn
import h5py
import numpy as np
import torch.utils.data as Data
import sys
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve, auc
import os
import time
from datetime import datetime
from torch.utils.data import random_split
import math
from tqdm import tqdm
from utils_single import six_scores
# from model import bert4 as lstm_dsmil
from model import SE as lstm_dsmil
from model import quchuBERT as lstm_dsmil
import pandas as pd


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_data(data_dir, shape_dir, bert_dir, ratio, batch_size):
    train_path = '{}train.hdf5'.format(data_dir)
    test_path = '{}test.hdf5'.format(data_dir)
    bert_path = '{}zbert16.hdf5'.format(bert_dir)

    # train_shape_path = '{}train_shape.hdf5'.format(shape_dir)
    # test_shape_path = '{}test_shape.hdf5'.format(shape_dir)

    with h5py.File(train_path, 'r') as fa:
        train_valid_data = np.asarray(fa['data']).astype(float)
        train_valid_label = np.asarray(fa['label']).astype(float)
        fa.close()

    # with h5py.File(train_shape_path, 'r') as fb:
    #     train_valid_shape = np.asarray(fb['shape']).astype(float)
    #     fb.close()

    with h5py.File(test_path, 'r') as fc:
        test_data = np.asarray(fc['data']).astype(float)
        test_label = np.asarray(fc['label']).astype(float)
        fc.close()

    # with h5py.File(test_shape_path, 'r') as fd:
    #     test_shape = np.asarray(fd['shape']).astype(float)
    #     fd.close()

    with h5py.File(bert_path, 'r') as fe:
        train_bert = np.asarray(fe['train_bert']).astype(float)
        test_bert = np.asarray(fe['test_bert']).astype(float)
        fe.close()

    train_valid_data = torch.from_numpy(train_valid_data)
    train_valid_label = torch.from_numpy(train_valid_label)
    print(train_valid_label.shape)
    train_valid_label = torch.squeeze(train_valid_label, 1)
    print(train_valid_label.shape)

    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)
    test_label = torch.squeeze(test_label, 1)

    # -----------------------------------------------------------
    # shape:
    # train_valid_shape = torch.from_numpy(train_valid_shape)
    # test_shape = torch.from_numpy(test_shape)
    # -----------------------------------------------------------

    train_valid_bert = torch.from_numpy(train_bert)
    test_bert = torch.from_numpy(test_bert)

    # train_valid_dataset = Data.TensorDataset(train_valid_data, train_valid_label, train_valid_shape, train_valid_bert)
    train_valid_dataset = Data.TensorDataset(train_valid_data, train_valid_label, train_valid_bert)
    train_dataset, valid_dataset = random_split(dataset=train_valid_dataset,
                                                lengths=[math.ceil(len(train_valid_dataset) * ratio),
                                                         len(train_valid_dataset) - math.ceil(
                                                             len(train_valid_dataset) * ratio)])

    # 在服务器上的时候有generator不行
    # train_dataset, valid_dataset = random_split(dataset=train_valid_dataset,
    #                                             lengths=[math.ceil(len(train_valid_dataset) * ratio),
    #                                                      len(train_valid_dataset) - math.ceil(
    #                                                          len(train_valid_dataset) * ratio)],
    #                                             generator=torch.Generator().manual_seed(0))

    test_dataset = Data.TensorDataset(test_data, test_label, test_bert)

    # 每次加载loader，数据都会被shuffle
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
    )

    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        shuffle=True,
        batch_size=batch_size,
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=batch_size,
    )
    return train_loader, valid_loader, test_loader


def epoch_train_da(data_loader, optimizer, criterion, d_a, epoch):
    bag_labels = []
    bag_predictions = []
    start = time.time()
    epoch_loss = 0
    d_a.train()
    ProgressBar = tqdm(data_loader, file=sys.stdout)
    # for i, (data, label, shape, other) in enumerate(ProgressBar):
    for i, (data, label, other) in enumerate(ProgressBar):
        labels = label.tolist()
        bag_labels.append(labels)

        optimizer.zero_grad()  # 梯度归零
        ProgressBar.set_description('Epoch %d Train ' % (epoch + 1))

        data_tensor = data.type(torch.FloatTensor)
        data_tensor = data_tensor.cuda()

        # shape_tensor = shape.type(torch.FloatTensor)
        # shape_tensor = shape_tensor.cuda()

        other_tensor = other.type(torch.FloatTensor)
        other_tensor = other_tensor.cuda()

        # print(data_tensor.shape)
        label = label.type(torch.FloatTensor)
        label = label.cuda()
        # classes, bag_prediction, _, _, _, = d_a(data_tensor, shape_tensor, other_tensor)
        bag_prediction = d_a(data_tensor, other_tensor)
        loss_bag = criterion(bag_prediction.view(-1), label)
        loss_total = loss_bag

        loss_total = loss_total.mean()

        loss_total.backward()  # 反向传播计算得到每个参数的梯度值
        optimizer.step()  # 通过梯度下降执行一步参数更新

        # bag_predictions.append(torch.sigmoid(bag_prediction).detach().cpu().squeeze().numpy())
        bag_predictions.append(torch.sigmoid(bag_prediction).detach().cpu().squeeze().tolist())

        epoch_loss = epoch_loss + loss_total.item() * data.size(0)
    ProgressBar.close()

    bag_labels = daping(bag_labels)
    bag_predictions = daping(bag_predictions)

    accuracy, precision, recall, fscore, roc_auc, pr_auc = six_scores(bag_labels, bag_predictions)
    finish = time.time()

    print('Epoch {} training consumed: {:.2f}s'.format(epoch + 1, finish - start))

    log_dic = {
        'epoch': epoch + 1,
        'lr': optimizer.param_groups[0]['lr'],
        'train_loss:': epoch_loss / len(data_loader.dataset),
        'train_acc': accuracy,
        'train_rocauc': roc_auc,
        'train_prauc': pr_auc

    }

    return log_dic, epoch_loss / len(data_loader.dataset)


def epoch_valid_da(valid_loader, criterion, d_a, num_epoch, epoch, log_dic, df_file):
    start = time.time()
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    with torch.no_grad():
        ProgressBar = tqdm(valid_loader, file=sys.stdout)
        # for i, (data, label, shape, other) in enumerate(ProgressBar):
        for i, (data, label, other) in enumerate(ProgressBar):
            ProgressBar.set_description('Epoch %d Valid ' % (epoch + 1))

            labels = label.tolist()
            bag_labels.append(labels)

            data_tensor = data.type(torch.FloatTensor)
            data_tensor = data_tensor.cuda()
            label = label.type(torch.FloatTensor)
            label = label.cuda()

            # shape_tensor = shape.type(torch.FloatTensor)
            # shape_tensor = shape_tensor.cuda()

            other_tensor = other.type(torch.FloatTensor)
            other_tensor = other_tensor.cuda()

            # classes, bag_prediction, final_predict, _, _ = d_a(data_tensor, shape_tensor, other_tensor)
            bag_prediction = d_a(data_tensor, other_tensor)
            loss_bag = criterion(bag_prediction.view(-1), label)
            loss_total = loss_bag
            loss_total = loss_total.mean()
            # bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().tolist())
            epoch_loss = epoch_loss + loss_total.item() * data.size(0)

        ProgressBar.close()

        # bag_labels = [float(x) for item in bag_labels for x in item]
        # bag_predictions = [float(x) for item in bag_predictions for x in item]

        bag_labels = daping(bag_labels)
        bag_predictions = daping(bag_predictions)

    epoch_loss = epoch_loss / len(valid_loader.dataset)
    accuracy, precision, recall, fscore, roc_auc, pr_auc = six_scores(bag_labels, bag_predictions)

    finish = time.time()

    print('Epoch {} valid consumed: {:.2f}s'.format(epoch + 1, finish - start))

    # print('bag_labels[0:50]:', bag_labels[0:50])
    # print('bag_predictions[0:50]:', bag_predictions[0:50])

    print('Epoch {}/{} valid loss: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, '
          'fscore: {:.4f}, roc_auc score: {:.4f}, pr_auc score: {:.4f}, '.format(epoch+1, num_epoch,
           epoch_loss, accuracy, precision, recall, fscore, roc_auc, pr_auc))

    log_dic['valid_loss'] = epoch_loss
    log_dic['valid_acc'] = accuracy
    log_dic['valid_precision'] = precision
    log_dic['valid_recall'] = recall
    log_dic['valid_fscore'] = fscore
    log_dic['valid_rocauc'] = roc_auc
    log_dic['valid_prauc'] = pr_auc
    df = pd.read_pickle(df_file)
    df = df.append([log_dic])
    df.reset_index(inplace=True, drop=True)
    df.to_pickle(df_file)

    return accuracy, roc_auc, epoch_loss


def model_test_da(test_loader, criterion, d_a):

    start = time.time()
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    ProgressBar = tqdm(test_loader, file=sys.stdout)
    with torch.no_grad():
        # for i, (data, label, shape, other) in enumerate(ProgressBar):
        for i, (data, label, other) in enumerate(ProgressBar):
            ProgressBar.set_description('\033[33mTest\033[0m ')

            labels = label.tolist()
            bag_labels.append(labels)

            data_tensor = data.type(torch.FloatTensor)
            data_tensor = data_tensor.cuda()
            label = label.type(torch.FloatTensor)
            label = label.cuda()

            # shape_tensor = shape.type(torch.FloatTensor)
            # shape_tensor = shape_tensor.cuda()

            other_tensor = other.type(torch.FloatTensor)
            other_tensor = other_tensor.cuda()

            # classes, bag_prediction, final_predict, _, _ = d_a(data_tensor, shape_tensor, other_tensor)
            bag_prediction = d_a(data_tensor, other_tensor)

            loss_bag = criterion(bag_prediction.view(-1), label)
            loss_total = loss_bag
            loss_total = loss_total.mean()
            # bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().tolist())

            epoch_loss = epoch_loss + loss_total.item() * data.size(0)

        ProgressBar.close()

        # bag_labels = [float(x) for item in bag_labels for x in item]
        # bag_predictions = [float(x) for item in bag_predictions for x in item]
        bag_labels = daping(bag_labels)
        bag_predictions = daping(bag_predictions)

        epoch_loss = epoch_loss / len(test_loader.dataset)

        accuracy, precision, recall, fscore, roc_auc, pr_auc = six_scores(bag_labels, bag_predictions)

        finish = time.time()

        print('\033[33mTesting consumed: {:.2f}s\033[0m'.format(finish - start))
        print('\033[33mbag_labels[0:50]:\033[0m', bag_labels[0:50])
        print('\033[33mbag_predictions[0:50]:\033[0m', bag_predictions[0:50])

        return epoch_loss, accuracy, precision, recall, fscore, roc_auc, pr_auc


def daping(list1):
    list2 = []
    while list1:
        head = list1.pop(0)
        if isinstance(head, list):
            for x in head:
                list2.append(float(x))

        else:
            list2.append(float(head))
    return list2


def main():
    # ---------------------------初始参数设置----------------------------
    after_test = True
    model_name = 'lstm_dsmil'
    ratio = 0.85
    # learning_rate = 0.0002
    learning_rate = 0.001
    num_epoch = 30
    weight_dacay = 1e-5
    batch_size = 64

    # ------------------------------------------------------------------

    # ---------------------------数据加载--------------------------------
    # 换地址的时候就4个
    # data_dir  result_path log_dir checkpoint_path
    # 第二个 002
    # data_dir = './processed/ChIPSeqData/wgEncodeAwgTfbsHaibK562Rad21V0416102UniPk/'  标准格式
    data_dir = '../processed/002/'
    shape_dir = '../processed/002/'
    bert_dir = '../processed/bert/'

    data_name = data_dir.split('/')[-2]
    result_path = './runs/002_result'
    log_dir = 'runs/002/'
    checkpoint_path = './checkpoint/002/'

    train_loader, valid_loader, test_loader = get_data(data_dir, shape_dir, bert_dir, ratio=ratio, batch_size=batch_size)
    # ------------------------------------------------------------------
    best_auc = 0.0
    net = model_name
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    time_now = datetime.now().strftime(DATE_FORMAT)
    print("Dataset: {}".format(data_dir))

    weight_dir = os.path.join(checkpoint_path, net, time_now)
    checkpoint_path = os.path.join(checkpoint_path, net, time_now)

    # create model_weights folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}_{epoch}.pth')
    # -----------------------------------------------------------------------

    # ---------------------单纯测试的时候注释掉-------------------------------
    # record the epoch
    df_path = os.path.join(log_dir, net, time_now)
    create_folder(df_path)
    df_file = os.path.join(df_path, 'df_log.pickle')
    if not os.path.isfile(df_file):
        df_ = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'train_acc',
                                    'train_rocauc', 'train_prauc',
                                    'valid_loss', 'valid_acc', 'valid_precision',
                                    'valid_recall', 'valid_fscore', 'valid_rocauc'
                                    'valid_prauc'])

        df_.to_pickle(df_file)
        print('log DataFrame created!')

    # -----------------------------------------------------------------------
    d_a = lstm_dsmil.Lstm_DSMIL_Att().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(d_a.parameters(), lr=learning_rate, betas=(0.5, 0.9), weight_decay=weight_dacay)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, 0)
    for epoch in range(0, num_epoch):

        e = "{}".format(epoch)
        log_dic, train_loss = epoch_train_da(train_loader, optimizer, criterion, d_a, epoch)

        accuracy, auc_valid, valid_loss = epoch_valid_da(valid_loader, criterion, d_a, num_epoch, epoch, log_dic, df_file)

        # -----------------------------------------------------------------
        # 保存epoch个模型
        # weights_path = checkpoint_path.format(net=model_name, epoch=e)
        # print('保存weights文件至：{}'.format(weights_path))
        # torch.save(d_a.state_dict(), weights_path)
        # -----------------------------------------------------------------

        if (best_auc < auc_valid) & (valid_loss < 0.9):
            weights_path = checkpoint_path.format(net=model_name, epoch=e)
            print('保存weights文件至：{}'.format(weights_path))
            torch.save(d_a.state_dict(), weights_path)
            best_auc = auc_valid
            # save best All
            continue

        # 按照Pytorch的定义是用来更新优化器的学习率的，一般是按照epoch为单位进行更换，
        # 即多少个epoch后更换一次学习率，因而scheduler.step()放在epoch这个大循环下

        schedular.step()

    # ------------------------------测试----------------------------------------------
    # 这个地址训练测试一起的时候注释掉
    # weights_path = './checkpoint/WL/dsmil2/Friday_18_March_2022_16h_03m_53s/dsmil2.pth'
    if after_test:

        d_a.load_state_dict(torch.load(weights_path))
        test_loss, accuracy, precision, recall, fscore, roc_auc, pr_auc = model_test_da(test_loader, criterion, d_a)

        print('\033[33mtest loss: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, '
              'fscore: {:.4f}, roc_auc score: {:.4f}, pr_auc score: {:.4f},\033[0m '.format(test_loss, accuracy, precision,
                                                                                            recall, fscore, roc_auc,
                                                                                            pr_auc))
    with open(result_path, 'a') as fw:
        fw.write(str(data_name))
        fw.write(' ')
        fw.write(' test_loss ')
        fw.write(str(round(test_loss, 4)))
        fw.write(' accuracy ')
        fw.write(str(round(accuracy, 4)))
        fw.write(' precision ')
        fw.write(str(round(precision, 4)))
        fw.write(' recall ')
        fw.write(str(round(recall, 4)))
        fw.write(' fscore ')
        fw.write(str(round(fscore, 4)))
        fw.write(' rocauc ')
        fw.write(str(round(roc_auc, 4)))
        fw.write(' pr_auc ')
        fw.write(str(round(pr_auc, 4)))
        fw.write('\n')
        fw.close()


if __name__ == '__main__':
    main()