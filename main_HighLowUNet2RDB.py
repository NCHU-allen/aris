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
    這個檔案訓練跟測試第二分支的兩個模型(兩種不同資料[IoU-High, IoU-Low])

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


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符號
    path = path.rstrip("\\")

    # 判斷路徑是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判斷結果
    if not isExists:
        # 如果不存在則建立目錄
        print("Building the file.")
        # 建立目錄操作函式
        os.makedirs(path)
        return True
    else:
        # 如果目錄存在則不建立，並提示目錄已存在
        print("File is existing.")
        return False

if __name__ == "__main__":
    date = "20201128"
    training_num = 51984
    name_loss = "CE"
    name_model = ["UNet(2RDB8-DtoU-5)"]
                  # "UNet(ELu)",
                  # "SegNet",
                  # "DV3",
                  # "UNet"]
    threshold = 0.5

    name = [date + "_256_" + str(training_num) + "(05high)_" + name_model[0] + "_" + name_loss,
            date + "_256_" + str(training_num) + "(05low)_" + name_model[0] + "_" + name_loss]
            # date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[2] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[3] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[4] + "_" + name_loss]

    excel_train = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_train_iou.xlsx"
    excel_valid = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_valid_iou.xlsx"
    excel_test = ".\\result\\data\\20210118_256_49461_UNet(2-RDB8)_CE_iou.xlsx"

    for i in range(len(name)):

        # print("Train data shape {}\n{}".format(train_x.shape, train_y.shape))
        print("Building model.")
        input_shape = (256, 256, 3)
        model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                        name="unet_2RD-5",
                                        input_size=input_shape,
                                        block_num=2)
        model_select.load_weights(".\\result\\model_record\\20201128_256_51984(05low)_UNet(2RDB8-DtoU-5)_CE_trainNum26596_validNum2674.h5")
        print("Loading data.")

        # 第一個是設定 IoU-High 相關設定
        if i== 0:
            train_flag = 0
            model_select.load_weights(".\\result\\model_record\\20201128_256_51984(05high)_UNet(2RDB8-DtoU-5)_CE_trainNum22865_validNum4391.h5")
            print("EX high data")
            total_num = 49461
            # 提取大於IoU門檻的資料
            # (train_x, train_y) = data.extract_high_result(np.load(".\\npy\\total-70658_1-1_train_x_49461.npy"),
            #                                               np.load(".\\npy\\total-70658_1-1_train_y_49461.npy"),
            #                                               excel_train,
            #                                               total_num,
            #                                               threshold= threshold)
            # total_num = 7065
            # (valid_x, valid_y) = data.extract_high_result(np.load(".\\npy\\total-70658_1-1_valid_x_7065.npy"),
            #                                               np.load(".\\npy\\total-70658_1-1_valid_y_7065.npy"),
            #                                               excel_valid,
            #                                               total_num,
            #                                               threshold= threshold)

        # 第二個是設定 IoU-Low 相關設定
        else:
            # print("EX low data")
            # total_num = 49461
            # 提取小於IoU門檻的資料
            # (train_x, train_y) = data.extract_low_result(np.load(".\\npy\\total-70658_1-1_train_x_49461.npy"),
            #                                               np.load(".\\npy\\total-70658_1-1_train_y_49461.npy"),
            #                                               excel_train,
            #                                               total_num,
            #                                               threshold=threshold)
            # total_num = 7065
            # (valid_x, valid_y) = data.extract_low_result(np.load(".\\npy\\total-70658_1-1_valid_x_7065.npy"),
            #                                               np.load(".\\npy\\total-70658_1-1_valid_y_7065.npy"),
            #                                               excel_valid,
            #                                               total_num,
            #                                               threshold=threshold)
            train_flag = 0

        epochs = 30
        batch = 3

        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            model_build.train(x_train=train_x, y_train=train_y, x_valid= valid_x, y_valid= valid_y, batch_size=batch, epochs= epochs, save_path= model_build.name + "_trainNum" + str(len(train_x)) +"_validNum" + str(len(valid_x)))

        if test_flag:
            print("Start testing.")
            total_num = 14132
            # 提取大於IoU門檻的資料
            if i == 0:
                (test_x, test_y) = data.extract_high_result(np.load(".\\npy\\total-70658_1-1_test_x_14132.npy"),
                                                            np.load(".\\npy\\total-70658_1-1_test_y_14132.npy"),
                                                            excel_test,
                                                            total_num,
                                                            threshold=threshold)
            # 提取小於IoU門檻的資料
            else:
                (test_x, test_y) = data.extract_low_result(np.load(".\\npy\\total-70658_1-1_test_x_14132.npy"),
                                                           np.load(".\\npy\\total-70658_1-1_test_y_14132.npy"),
                                                           excel_test,
                                                           total_num,
                                                           threshold=threshold)

            model_build.test(x_test= test_x, y_test=test_y, batch_size=batch, save_path= "20201128_256_51984(05high)_UNet(2RDB8-DtoU-5)_CE_trainNum22865_validNum4391")