import cv2
import numpy as np
import os
import excel

# file.rstrip(".tif") """remove .tif"""

def dataset_generator(size = 256):
    count_file = 0
    count_file_read = 0

    for filename in os.listdir(".\\original dataset\\image"):
        count_file_read +=1
        file_id = filename.rstrip(".tif")
        image = cv2.imread(".\\original dataset\\image\\" + file_id + ".tif")
        print("{}. Image name：{}".format(count_file_read, file_id))
        label = cv2.imread(".\\original dataset\\label\\" + file_id + ".tif")
        print("{}. Label name：{}".format(count_file_read, file_id))
        label_vis = cv2.imread(".\\original dataset\\label\\" + file_id + "_vis.tif")
        print("{}. Label_vis name：{}".format(count_file_read, file_id))
        (x, y, channels) = image.shape
        print("image shape")
        print(image.shape)
        print("label shape")
        print(label.shape)

        for dx in range((x // size) + 1):
            for dy in range((y // size) + 1):
                x_low = size * dx
                x_heigh = size * (dx + 1)
                y_low = size * dy
                y_heigh = size * (dy + 1)

                if x_heigh > x or y_heigh > y:
                    save_tif = np.zeros([size,size, channels], dtype= np.uint8)
                    save_label = np.zeros([size,size,channels], dtype= np.uint8)
                    save_label_vis = np.zeros([size, size, channels], dtype = np.uint8)

                    if x_heigh > x and y_heigh > y:
                        save_tif[: (x-x_low), :(y-y_low)] = image[x_low:x, y_low:y]
                        save_label[: (x-x_low), :(y-y_low)] = label[x_low:x, y_low:y]
                        save_label_vis[: (x-x_low), :(y-y_low)] = label_vis[x_low:x, y_low:y]
                    elif x_heigh > x:
                        save_tif[: (x-x_low), :] = image[x_low:x, y_low:y_heigh]
                        save_label[: (x - x_low), :] = label[x_low:x, y_low:y_heigh]
                        save_label_vis[: (x - x_low), :] = label_vis[x_low:x, y_low:y_heigh]
                    elif y_heigh > y:
                        save_tif[:, :(y-y_low)] = image[x_low:x_heigh, y_low:y]
                        save_label[:, :(y - y_low)] = label[x_low:x_heigh, y_low:y]
                        save_label_vis[:, :(y - y_low)] = label_vis[x_low:x_heigh, y_low:y]
                else:
                    save_tif = image[x_low:x_heigh, y_low:y_heigh]
                    save_label = label[x_low:x_heigh, y_low:y_heigh]
                    save_label_vis = label_vis[x_low:x_heigh, y_low:y_heigh]
                if (count_file + 1)>386619:
                    print("{}. image save：".format(count_file_read))
                    print(count_file + 1)
                    cv2.imwrite(".\\self_dataset\\SI\\" + str(count_file + 1) + "_" + file_id + ".tif", save_tif)
                    cv2.imwrite(".\\self_dataset\\label\\" + str(count_file + 1) + "_" + file_id + ".tif", save_label)
                    cv2.imwrite(".\\self_dataset\\label_vis\\" + str(count_file + 1) + "_" + file_id + ".tif", save_label_vis)

                count_file +=1

# self_dataset_stragey1_512 & 1024都是用這個策略
def dataset_generator_self(size = 256):
    count_file = 0
    count_file_read = 0
    positive_data_count = 0

    for filename in os.listdir(".\\original dataset\\image"):
        count_file_read +=1
        file_id = filename.rstrip(".tif")
        image = cv2.imread(".\\original dataset\\image\\" + file_id + ".tif")
        print("{}. Image name：{}".format(count_file_read, file_id))
        label = cv2.imread(".\\original dataset\\label\\" + file_id + ".tif")
        print("{}. Label name：{}".format(count_file_read, file_id))
        label_vis = cv2.imread(".\\original dataset\\label\\" + file_id + "_vis.tif")
        print("{}. Label_vis name：{}".format(count_file_read, file_id))
        (x, y, channels) = image.shape
        print("image shape")
        print(image.shape)
        print("label shape")
        print(label.shape)

        for dx in range((x // size) + 1):
            for dy in range((y//size) + 1):
                x_low = size * dx
                x_heigh = size * (dx + 1)
                y_low = dy * size
                y_heigh = y_low + size

                if x_heigh > x or y_heigh > y:
                    save_tif = np.zeros([size,size, channels], dtype= np.uint8)
                    save_label = np.zeros([size,size,channels], dtype= np.uint8)
                    save_label_vis = np.zeros([size, size, channels], dtype = np.uint8)

                    if x_heigh > x and y_heigh > y:
                        save_tif[: (x-x_low), :(y-y_low)] = image[x_low:x, y_low:y]
                        save_label[: (x-x_low), :(y-y_low)] = label[x_low:x, y_low:y]
                        save_label_vis[: (x-x_low), :(y-y_low)] = label_vis[x_low:x, y_low:y]
                    elif x_heigh > x:
                        save_tif[: (x-x_low), :] = image[x_low:x, y_low:y_heigh]
                        save_label[: (x - x_low), :] = label[x_low:x, y_low:y_heigh]
                        save_label_vis[: (x - x_low), :] = label_vis[x_low:x, y_low:y_heigh]
                    elif y_heigh > y:
                        save_tif[:, :(y-y_low)] = image[x_low:x_heigh, y_low:y]
                        save_label[:, :(y - y_low)] = label[x_low:x_heigh, y_low:y]
                        save_label_vis[:, :(y - y_low)] = label_vis[x_low:x_heigh, y_low:y]
                else:
                    save_tif = image[x_low:x_heigh, y_low:y_heigh]
                    save_label = label[x_low:x_heigh, y_low:y_heigh]
                    save_label_vis = label_vis[x_low:x_heigh, y_low:y_heigh]

                if np.sum(save_tif) == 0:
                    continue

                if np.sum(save_label):
                    positive_data_count +=1
                elif positive_data_count != 0 & np.sum(save_label) == 0:
                    positive_data_count -= 1
                else:
                    continue
                print("{}. image save：".format(count_file_read))
                print(count_file + 1)
                cv2.imwrite(".\\self_dataset_stragey1_1024\\SI\\" + str(count_file + 1) + "_" + file_id + ".tif",
                            save_tif)
                cv2.imwrite(".\\self_dataset_stragey1_1024\\label\\" + str(count_file + 1) + "_" + file_id + ".tif",
                            save_label)
                cv2.imwrite(".\\self_dataset_stragey1_1024\\label_vis\\" + str(count_file + 1) + "_" + file_id + ".tif",
                            save_label_vis)
                count_file +=1
# 讀取 dataset資料轉換成輸入模型大小的資料對
def dataset_read(dataset_path, total = 35922, size= 256):
    x = np.zeros([total, size, size, 3], dtype=np.float32)
    y = np.zeros([total, size, size, 1], dtype=np.uint8)
    count_file_read = 0

    for filename in os.listdir(dataset_path + "\\SI"):
        if count_file_read % 100 == 0:
            print("{}. Read file：{}".format(count_file_read + 1, filename))

        image = cv2.imread(dataset_path + "\\SI\\" + filename)
        label = cv2.imread(dataset_path + "\\label\\" + filename)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_LINEAR)

        x[count_file_read] = image.astype("float32") / 255
        y[count_file_read] = np.expand_dims(np.ones(label.shape) * (label > 0), axis=-1)

        count_file_read += 1
    return (x, y)
    # return (x, y)
def load_data(dataset, start_num, total_num, size= (256, 256, 4)):
    x = np.zeros([total_num, size, size, 3], dtype=np.float32)
    y = np.zeros([total_num, size, size, 1], dtype=np.uint8)
    count_start_read = 1
    count_file_read = 0
    dataset_path = "G:\\allen\\airs\\" + dataset

    for filename in os.listdir(dataset_path + "\\SI"):
        if count_start_read < start_num:
            count_start_read +=1
            continue
        if count_file_read >= total_num:
            break

        image = cv2.imread(dataset_path + "\\SI\\" + filename)
        label = cv2.imread(dataset_path + "\\label\\" + filename)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_LINEAR)

        x[count_file_read] = image.astype("float32") / 255
        y[count_file_read] = np.expand_dims(np.ones(label.shape) * (label > 0), axis=-1)

        count_file_read += 1
    return (x, y)
    # return (x, y)
# 生成測試資料的影像資料夾
def dataTest_image_save(dataset_path, test_data_start= 56526, size= 256):
    count_file_read = 0

    for filename in os.listdir(dataset_path + "\\SI"):
        if count_file_read < test_data_start:
            count_file_read +=1
            continue
        print("{}. Read file：{}".format(count_file_read + 1, filename))

        image = cv2.imread(dataset_path + "\\SI\\" + filename)
        label = cv2.imread(dataset_path + "\\label\\" + filename)
        label_vis = cv2.imread(dataset_path + "\\label_vis\\" + filename)

        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_LINEAR)
        label_vis = cv2.resize(label_vis, (size, size), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(".\\result\\image\\test_image\\SI\\" + str(count_file_read - test_data_start+1) + "(testOrder)_datasetOrder-" + filename,
                    image)
        cv2.imwrite(".\\result\\image\\test_image\\label\\" + str(count_file_read - test_data_start+1) + "(testOrder)_datasetOrder-" + filename,
                    label)
        cv2.imwrite(".\\result\\image\\test_image\\label_vis\\" + str(count_file_read - test_data_start+1) + "(testOrder)_datasetOrder-" + filename,
                    label_vis)
        count_file_read += 1

def extract_high_result(x, y, excel_file, total_num, extract_index= "iou", threshold= 0.5):
    file = excel.Excel(file_path= excel_file)
    print("Excel file {} is opened.".format(excel_file))

    if extract_index == "iou" or extract_index == "IoU":
        index = file.read_excel(start="b3:b" + str(2 + total_num))
    elif extract_index == "precision" or extract_index == "Precision":
        index = file.read_excel(start="d3:d" + str(2 + total_num))
    elif extract_index == "recall" or extract_index == "Recall":
        index = file.read_excel(start="f3:f" + str(2 + total_num))
    elif extract_index == "f1" or extract_index == "F1":
        index = file.read_excel(start="h3:h" + str(2 + total_num))
    else:
        print("extrac_index：{}. ERROR".format(extract_index))
        raise ValueError
    file.close_excel()

    index = np.array(index)

    print("Index is {}.".format(extract_index))
    low_result = np.array(np.where(index <= threshold))
    re_x = np.delete(x, low_result, axis=0)
    re_y = np.delete(y, low_result, axis=0)

    return (re_x, re_y)

def extract_low_result(x, y, excel_file, total_num, extract_index= "iou", threshold= 0.5):
    file = excel.Excel(file_path= excel_file)
    print("Excel file {} is opened.".format(excel_file))

    if extract_index == "iou" or extract_index == "IoU":
        index = file.read_excel(start="b3:b" + str(2 + total_num))
    elif extract_index == "precision" or extract_index == "Precision":
        index = file.read_excel(start="d3:d" + str(2 + total_num))
    elif extract_index == "recall" or extract_index == "Recall":
        index = file.read_excel(start="f3:f" + str(2 + total_num))
    elif extract_index == "f1" or extract_index == "F1":
        index = file.read_excel(start="h3:h" + str(2 + total_num))
    else:
        print("extrac_index：{}. ERROR".format(extract_index))
        raise ValueError
    file.close_excel()

    index = np.array(index)

    print("Index is {}.".format(extract_index))
    low_result = np.array(np.where(index > threshold))
    re_x = np.delete(x, low_result, axis=0)
    re_y = np.delete(y, low_result, axis=0)
    return (re_x, re_y)

if __name__ == "__main__":
    # 測試資料影像檔案生成
    dataTest_image_save(dataset_path= ".\\self_dataset_stragey1_1024")

    # 生成 npy檔案
    # (x, y) = dataset_read(dataset_path= ".\\self_dataset_stragey1_1024", total= 70658, size= 256)
    #
    # print(x[:49461].shape)
    # np.save(".\\npy\\total-70658_1-1_train_x_49461.npy", x[:49461])
    # np.save(".\\npy\\total-70658_1-1_train_y_49461.npy", y[:49461])
    #
    # print(x[49461:56526].shape)
    # np.save(".\\npy\\total-70658_1-1_valid_x_7065.npy", x[49461:56526])
    # np.save(".\\npy\\total-70658_1-1_valid_y_7065.npy", y[49461:56526])
    # print(x[56526:].shape)
    # np.save(".\\npy\\total-70658_1-1_test_x_14132.npy", x[56526:])
    # np.save(".\\npy\\total-70658_1-1_test_y_14132.npy", y[56526:])