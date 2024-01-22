from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import os
def dataset(img_root, label_root):
    imgs = []
    for i in range(len(img_root)):
        img = os.path.join(img_root[i])
        label = os.path.join(label_root[i])
        imgs.append((img, label))
    return imgs
def get_index():
    x = [i for i in range(107)]

    x = np.array(x)
    rkf = RepeatedKFold(n_splits=5, n_repeats=1,random_state=224587)
    train_index, val_index = [], []
    for train, val in rkf.split(x):
        train_index.append(train)
        val_index.append((val))
    return train_index, val_index

class Five_F_Dataset():
    def __init__(self, img_root, label_root, transform=None, target_transform=None):
        super(Five_F_Dataset,self).__init__()
        imgs = dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        (x_path, y_path) = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return (img_x, img_y)
    def __len__(self):
        return len(self.imgs)
if __name__=="__main__":
    # out = open('dataset.csv')
    # df = pd.read_csv(out)
    # train_index, val_index = get_index()
    # for i in range(5):
    #     train_img, train_lab = [], []
    #     for index in train_index[i]:
    #         train_img.append(df['data'][index])
    #         train_lab.append(df['target'][index])
    #
    #     val_img, val_lab = [], []
    #     for index in val_index[i]:
    #         val_img.append(df['data'][index])
    #         val_lab.append(df['target'][index])
    #     print(val_img)
    #     print(val_lab)
    img_root = 'BUSI/images/'
    label_root = 'BUSI/labels/'

    train_img_root = 'five_folds/fold{}/train/images/'
    train_lab_root = 'five_folds/fold{}/train/labels/'

    test_img_root = 'five_folds/fold{}/test/images/'
    test_lab_root = 'five_folds/fold{}/test/labels/'
    #
    ls = os.listdir(img_root)
    ls_lable = os.listdir(label_root)

    x = np.array([i for i in range(len(ls))])
    rkf = KFold(n_splits=5, shuffle=True)
    train_index, val_index = [], []
    for train, val in rkf.split(x):
        train_index.append(train)
        val_index.append(val)
    for i in range(5):
        for j in train_index[i]:
            if not os.path.exists(train_img_root.format(i)):
                os.makedirs(train_img_root.format(i))
            shutil.copy(img_root + ls[j], train_img_root.format(i))
            """
            用法： shutil.copy(source, destination, *, follow_symlinks = True)
            shutil.copy()Python中的方法用于将源文件的内容复制到目标文件或目录。
            source：代表源文件路径的字符串。
            destination：代表目标文件或目录路径的字符串。
            follow_symlinks(可选)：此参数的默认值为True。如果为False，并且source表示符号链接，则目标将创建为符号链接。
            """
            if not os.path.exists(train_lab_root.format(i)):
                os.makedirs(train_lab_root.format(i))
            shutil.copy(label_root + ls_lable[j], train_lab_root.format(i))

        for k in val_index[i]:
            if not os.path.exists(test_img_root.format(i)):
                os.makedirs(test_img_root.format(i))
            if not os.path.exists( test_lab_root.format(i)):
                os.makedirs( test_lab_root.format(i))
            shutil.copy(img_root + ls[k], test_img_root.format(i))
            shutil.copy(label_root + ls_lable[k], test_lab_root.format(i))

    print(train_index[1])
    print(val_index[1])
    """
    for i in range(5):
        for item in train_index[i]:
           save_root1 = "".format(i+1)
           save_root2 = "".format(i+1)
           if not os.path.exists(save_root1):
               os.mkdir(save_root1)
           if not os.path.exists(save_root2):
               os.mkdir(save_root2)
           shutil.copy(img_root + ls[item], save_root1 + ls[item])
           shutil.copy(label_root + ls[item], save_root2 + ls[item])

        for item in val_index[i]:
            save_root1 = "five_folds/fold{}/test/images/".format(i + 1)
            save_root2 = "five_folds/fold{}/test/labels/".format(i + 1)
            if not os.path.exists(save_root1):
                os.makedirs(save_root1)
            if not os.path.exists(save_root2):
                os.makedirs(save_root2)
            shutil.copy(img_root + ls[item], save_root1 + ls[item])
            shutil.copy(label_root + ls[item], save_root2 + ls[item])
        """
