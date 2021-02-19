import numpy as np
import os
import cv2
import model
from keras.models import load_model

from keras.optimizers import *
import model_deeplab3plus as DV3
import model_FCDenseNet as FCDN
import classifiation
import postprocessing
import excel
import estimate
import data
import hrnet
import gc

'''
    這個檔案包含了個別模型的訓練跟測試
    
    使用模型
        "UNet",
        "SegNet",
        "DV3",
        "FCDN",
        "HRNet",
        "UNet(2-RDB8)"
        
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
    date = "20210118"
    training_num = 49461
    name_loss = "CE"
    name_model = ["UNet",
                  "SegNet",
                  "DV3",
                  "FCDN",
                  "HRNet",
                  "UNet(2-RDB8)"]

    name = [date + "_256_" + str(training_num) + "_" + name_model[0] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[2] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[3] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[4] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[5] + "_" + name_loss]

    print("Loading data.")

    for i in range(len(name)):
        print("Building model.")
        input_shape = (256, 256, 3)

        if name_model[i] == "UNet":

            model_select = model.Unet(size= input_shape) # 搭建新的模型
            model_select.load_weights(".\\result\\model_record\\20210118_256_49461_UNet_CE.h5") # 載入現有完成訓練的權重
            batch = 10
            train_flag = 0
        elif name_model[i] == "SegNet":
            continue
            model_select = model.segnet(input_shape= input_shape,
                                        n_labels= 1,
                                        kernel=3,
                                        pool_size=(2, 2),
                                        output_mode="sigmoid")
            model_select.load_weights(".\\result\\model_record\\20210118_256_49461_SegNet_CE.h5")
            batch = 10
            train_flag = 0
        elif name_model[i] == "DV3":
            continue
            model_select = DV3.Deeplabv3(weights=None, input_shape= input_shape, classes=1)
            model_select.load_weights(".\\result\\model_record\\20210118_256_49461_" + name_model[i] + "_CE.h5")
            batch = 3
            train_flag = 1
        elif name_model[i] == "FCDN":
            continue
            model_select = FCDN.Tiramisu(input_shape= input_shape)
            # model_select.load_weights(".\\result\\model_record\\20210118_256_49461_" + name_model[i] + "_CE.h5")
            batch = 3
            train_flag = 1
        elif name_model[i] == "HRNet":
            continue
            model_select = hrnet.seg_hrnet(input_shape[0], input_shape[1], input_shape[2], classes=1)
            # model_select.load_weights(".\\result\\model_record\\20210118_256_49461_" + name_model[i] + "_CE.h5")
            batch = 3
            train_flag = 1
        elif name_model[i] == "UNet(2-RDB8)":
            model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                            name="unet_2RD-5",
                                            input_size=input_shape,
                                            block_num=2)
            model_select.load_weights(".\\result\\model_record\\20210118_256_49461_UNet(2-RDB8)_CE-bestweights-epoch002-loss0.32382-val_loss0.24472.h5")
            batch = 3
            train_flag = 0

        epochs = 30
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("Model building.")
        # model.model()是自訂義的 class，主要是針對訓練跟測試做包裝的類別
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            train_x = np.load(".\\npy\\total-70658_1-1_train_x_49461.npy")
            train_y = np.load(".\\npy\\total-70658_1-1_train_y_49461.npy")
            valid_x = np.load(".\\npy\\total-70658_1-1_valid_x_7065.npy")
            valid_y = np.load(".\\npy\\total-70658_1-1_valid_y_7065.npy")
            model_build.train(x_train=train_x, y_train=train_y, x_valid= valid_x, y_valid= valid_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            test_x = np.load(".\\npy\\total-70658_1-1_test_x_14132.npy")
            test_y = np.load(".\\npy\\total-70658_1-1_test_y_14132.npy")
            model_build.test(x_test= test_x, y_test=test_y, batch_size=batch, save_path= name[i])