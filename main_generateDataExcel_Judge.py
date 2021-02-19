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

'''
    這個檔案主要是根據已經完成訓練的 RDB U-Net針對訓練跟驗證資料生成預測資料

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
    name_model = ["UNet(2-RDB8)"]

    name = [date + "_256_" + str(training_num) + "_" + name_model[0] + "_" + name_loss]

    for i in range(len(name)):
        print("Building model.")
        input_shape = (256, 256, 3)

        model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                        name="unet_2RD-5",
                                        input_size=input_shape,
                                        block_num=2)
        model_select.load_weights(".\\result\\model_record\\" + date + "_256_" + str(training_num) + "_" + name_model[0] + "_" + name_loss + ".h5")

        batch = 3
        epochs = 30
        train_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)
        # model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            train_x = np.load(".\\npy\\total-70658_1-1_train_x_49461.npy")
            train_y = np.load(".\\npy\\total-70658_1-1_train_y_49461.npy")
            model_build.test(x_test= train_x, y_test=train_y, batch_size=batch, save_path= model_build.name + "_train")

            valid_x = np.load(".\\npy\\total-70658_1-1_valid_x_7065.npy")
            valid_y = np.load(".\\npy\\total-70658_1-1_valid_y_7065.npy")
            model_build.test(x_test=valid_x, y_test=valid_y, batch_size=batch, save_path= model_build.name + "_valid")