# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:15:19 2020

@author: Administrator
"""



import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from random import shuffle
import os
import itertools
import json
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
import cv2

# 图像灰度化，便于降数据集四维降成三维
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
 
    return gray





if __name__=='__main__':

    def create_y_true(index, real_label):
        new_label = real_label.copy()
        for i, label in enumerate(new_label):
            if label != index:
                new_label[i] = 0
            else:
                new_label[i] = 1

        return new_label


    def create_y_scores(index, pred_prob):
        scores = []
        for i, prob in enumerate(pred_prob):
            scores.append(prob[index])

        return scores


    def create_true_score(index, real_test, pred_prob):
        y_true = create_y_true(index, real_test)
        y_scores = create_y_scores(index, pred_prob)

        return y_true, y_scores


    def get_roc(pos_prob, y_true):
        pos = y_true[y_true == 1]
        neg = y_true[y_true == 0]
        threshold = np.sort(pos_prob)[::-1]
        y = y_true[pos_prob.argsort()[::-1]]
        tpr_all = [0];
        fpr_all = [0]
        tpr = 0;
        fpr = 0
        x_step = 1 / float(len(neg))
        y_step = 1 / float(len(pos))
        y_sum = 0
        for i in range(len(threshold)):
            if y[i] == 1:
                tpr += y_step
                tpr_all.append(tpr)
                fpr_all.append(fpr)
            else:
                fpr += x_step
                fpr_all.append(fpr)
                tpr_all.append(tpr)
                y_sum += tpr
        return fpr_all, tpr_all, y_sum * x_step



    class ConfusionMatrix(object):
        """
        注意，如果显示的图像不全，是matplotlib版本问题
        本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
        需要额外安装prettytable库
        """

        def __init__(self, num_classes: int, labels: list):
            self.matrix = np.zeros((num_classes, num_classes))
            self.num_classes = num_classes
            self.labels = labels

        def update(self, preds, labels):
            for p, t in zip(preds, labels):
                self.matrix[p, t] += 1

        def summary(self):
            # calculate accuracy
            sum_TP = 0
            for i in range(self.num_classes):
                sum_TP += self.matrix[i, i]
            acc = sum_TP / np.sum(self.matrix)
            print("the model accuracy is ", acc)

            # precision, recall, specificity
            table = PrettyTable()
            table.field_names = ["", "Precision", "Recall", "Specificity"]
            for i in range(self.num_classes):
                TP = self.matrix[i, i]
                FP = np.sum(self.matrix[i, :]) - TP
                FN = np.sum(self.matrix[:, i]) - TP
                TN = np.sum(self.matrix) - TP - FP - FN
                Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
                Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
                Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
                table.add_row([self.labels[i], Precision, Recall, Specificity])
            print(table)

        def plot(self):
            matrix = self.matrix
            print(matrix)
            plt.imshow(matrix, cmap=plt.cm.Blues)

            # 设置x轴坐标label
            plt.xticks(range(self.num_classes), self.labels, rotation=45)
            # 设置y轴坐标label
            plt.yticks(range(self.num_classes), self.labels)
            # 显示colorbar
            plt.colorbar()
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            plt.title('Confusion matrix')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            # 在图中标注数量/概率信息
            thresh = matrix.max() / 2
            for x in range(self.num_classes):
                for y in range(self.num_classes):
                    # 注意这里的matrix[y, x]不是matrix[x, y]
                    info = int(matrix[y, x])
                    plt.text(x, y, info,
                             verticalalignment='center',
                             horizontalalignment='center',
                             color="white" if info > thresh else "black")
            plt.tight_layout()
            plt.show()

    n_class = 10
    svmscore = []
    knnscore = []
    
    #读取数据集
    train_data_file = 'DataX.npy'
    train_lable_file = 'LableY.npy'
    DataArray = np.load(train_data_file)        
    LabelArray = np.load(train_lable_file)
    print('load finish')
    
    # 对数据集进行打乱    # 
    index = [i for i in range(len(LabelArray))]
    shuffle(index)    
    DataArrayA = DataArray[index, :, :, :]
    LabelArrayA = LabelArray[index, :] 
    
    # 将图像数据从思维降成三维
    DataArrayAA = np.ones((len(LabelArray),227,227))
    # DataArrayAAA = np.ones((len(LabelArray),8,8))
    for i in range(len(LabelArray)):
        DataArrayAA[i] = rgb2gray(DataArrayA[i])
        # DataArrayAAA[i] = cv2.resize(DataArrayAA[i],(8,8),interpolation=cv2.INTER_AREA)
        # DataArrayAAA = np.conconcatenate([DataArrayAAA,DataArrayTmp])
    print('gray change finished') 
    
    # 截取90%数据进行训练，10%数据进行测试
    trainlen = int(len(LabelArray) * 0.9)
    Train_X = DataArrayAA[0:trainlen,:,:]
    Train_Y = LabelArrayA[0:trainlen,:]
    Test_X = DataArrayAA[trainlen:len(LabelArray),:,:]
    Test_Y = LabelArrayA[trainlen:len(LabelArray),:]
    
    # 以下进行降维（根据fit函数的参数要求），将X从三维降成二维，将Y从二维降成一维
    Train_XX = np.reshape(Train_X,[len(Train_X),-1])#X降维
    print('Train_XX reshape finished') 
    Train_YY = np.argmax(Train_Y,axis=1)#Y也需要降维
    Test_XX = np.reshape(Test_X,[len(Test_X),-1])#降维
    Test_YY = np.argmax(Test_Y,axis=1)
    print(Test_YY)
    print('Test_XX reshape finished') 
    
    # 分别采用SVM和KNN算法构建模型进行训练及测试
    svm = SVC()
    knn = KNeighborsClassifier()
    svm.fit(Train_XX,Train_YY)
    svmscore.append(svm.score(Test_XX,Test_YY))
    print('svmAcc is',svmscore)

    
    knn = KNeighborsClassifier()
    knn.fit(Train_XX,Train_YY)
    knnscore.append(knn.score(Test_XX,Test_YY))
    print('knnAcc is',knnscore)
    label_path = './class_indices.json'
    assert os.path.exists(label_path), "cannot find {}".format(label_path)
    json_file = open(label_path, 'r', encoding='UTF-8')
    class_indict = json.load(json_file)

    # print(Test_XX)
    Alex_Res = knn.predict(Test_XX)
    # print(Alex_Res)
    # #bb = np.argmax(Test_YY, axis=1)
    # #print(bb)
    aa = Alex_Res.argmax(axis=1)
    # print(aa)
    ab = Test_Y.argmax(axis=1)
    #Test_XX = np.reshape(Test_XX, [len(Test_XX), -1])
    Test_XX = np.argmax(Test_XX, axis=1)
    print(Test_XX)
    print(Test_YY)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=10, labels=labels)
    confusion.update( Test_XX/1000, Test_YY)
    confusion.plot()
    confusion.summary()


    # print(get_roc(aa,bb))
    # fpr, tpr,auc =get_roc(aa,bb)
    # lw = 2
    # #auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %f)' % auc)
    # plt.plot([0, 0.2], [0, 0.2], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 0.2])
    # plt.ylim([0.0, 0.2])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    def draw(index,real_test,pred_prob):
        y_true, y_scores = create_true_score(index,real_test,pred_prob)
        fpr,tpr,auc =get_roc(np.array(y_scores),np.array(y_true))
        lw = 2
        # auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()



    for i in range(10):
        draw(i,ab,Alex_Res)