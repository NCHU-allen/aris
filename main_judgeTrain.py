import numpy as np
import os
import cv2
import model
from keras.models import load_model

from keras.optimizers import *
# import model_deeplab3plus as DV3
# import model_FCDenseNet as FCDN
import classifiation
import postprocessing
import excel
import estimate
import data

'''
    這個檔案訓練跟測試 JudgeNet

    參數：
        date：存檔的檔案名稱中，日期
        training_num：訓練資料數量
        name_loss：使用的Loss名稱
        name_model：模型使用的名稱
        name：最終存的模型、預測影像、excel資料的名稱，這影響到這次程式運行會有多少個實驗要跑
        input_shape：輸入模型的資料大小
        batch：在訓練或測試的batch size
        train_flag：是否訓練，1 此階段要訓練 / 0 此階段不訓練
        test_flag：是否測試，1 此階段要測試 / 0 此階段不測試
        epochs：訓練epoch數量
        threshold：將資料分成兩類的 IoU門檻值

'''

if __name__ == "__main__":
    date = "20201216"
    training_num = 49461
    name_loss = "CE"
    name_model = ["JudgeAlexNet"]
    # 要先確認門檻
    threshold = np.array([0.5])
    name = [date + "_256_" + str(training_num) + "(th05)_" + name_model[0] + "_" + name_loss]

    excel_train = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_train_iou.xlsx"  # 模型根據訓練資料預測的excel資料
    excel_valid = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_valid_iou.xlsx"  # 模型根據驗證資料預測的excel資料
    excel_test = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_iou.xlsx"         # 模型根據測試資料預測的excel資料


    for i in range(len(name)):
        # print("Train data shape {}\n{}".format(train_x.shape, train_y.shape))
        print("Building model.")
        input_shape = (256, 256, 3)
        model_select = classifiation.Alexnet(input_shape, output_class= 2)
        # model_select.load_weights(".\\result\\model_record\\20201216_256_51984(th09)_JudgeAlexNet_CE.h5")

        epochs = 30
        batch = 10
        train_flag = 0
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])

        print("Model building.")
        model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            total_num = 49461
            train_y = model_build.GT_data_transfer(excel_train, total_num, extract_index= "iou", threshold= threshold[0])   # 生成分類模型的 ground truth
            train_x = np.load(".\\npy\\total-70658_1-1_train_x_49461.npy")

            total_num = 7065
            valid_y = model_build.GT_data_transfer(excel_valid, total_num, extract_index="iou", threshold=threshold[0]) # 生成分類模型的 ground truth
            valid_x = np.load(".\\npy\\total-70658_1-1_valid_x_7065.npy")

            # 開始訓練
            model_build.train(x=train_x, y=train_y, x_valid= valid_x, y_valid= valid_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            total_num = 14132
            test_y = model_build.GT_data_transfer(excel_test, total_num, extract_index="iou", threshold=threshold[0]) # 生成分類模型的 ground truth
            test_x = np.load(".\\npy\\total-70658_1-1_test_x_14132.npy")
            model_build.test(x= test_x, y=test_y, batch_size=batch, data_start= 1)