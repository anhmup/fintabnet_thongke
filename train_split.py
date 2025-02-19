# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
import cv2
import ast


import glob
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets

# %%
SHAPE = 512
H_SHAPE = 720
W_SHAPE = 512

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device to gpu or cpu
device

# %% [markdown]
# # model

# %%
class Resnet_FPN(nn.Module):
    def __init__(self,out_channels = 128):
        super(Resnet_FPN, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.to(device)
        self.layer_1 = nn.Sequential(*list(self.resnet.children())[:4])
        self.layer_2 = self.resnet.layer2
        self.layer_3 = self.resnet.layer3
        self.layer_4 = self.resnet.layer4

        self.lateral4 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if (torch.isnan(x).any()):
            print("error at Resnet_FPN")
        c1 = self.layer_1(x)
        if (torch.any(torch.isnan(c1))):
            print('error resnet ')
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')

        p4 = self.output4(p4)
        p3 = self.output3(p3)
        p2 = self.output2(p2)
        p1 = self.output1(p1)

        return p1, p2, p3, p4

# %%
class transferlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transferlayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

# %%
class mess_passing_H(nn.Module):
    def __init__(self, in_channels = 32, out_channels= 32):
        super(mess_passing_H, self).__init__()
        self.t2d_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[1,9], stride=(1,1), padding=(0,4))
        self.d2t_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[1,9], stride=(1,1), padding=(0,4))
        self.relu = nn.ReLU()
    def forward(self, x):
        nB, c, h, w = x.shape
        feature_list_old = []   
        feature_list_new = []
        for cnt in range(h):
            feature_list_old.append(x[:,:,cnt,:].unsqueeze(dim = 2))
        feature_list_new.append(x[:,:,0,:].unsqueeze(dim = 2))
        feature = self.t2d_layers(feature_list_old[0])
        feature = self.relu(feature) + feature_list_old[1]
        feature_list_new.append(feature)
        for cnt in range(2,h):
            feature =  self.t2d_layers(feature_list_new[cnt-1])
            feature = self.relu(feature) + feature_list_old[cnt]
            feature_list_new.append(feature)
        feature_list_new.append(feature)
        feature_list_old = feature_list_new
        feature_list_new = []
        length = h - 1 
        feature_list_new.append(feature_list_old[length])
        feature = self.d2t_layers(feature_list_old[length])
        feature = self.relu(feature) + feature_list_old[length - 1]
        feature_list_new.append(feature)
        for cnt in range(2, h):
            feature = self.d2t_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[length - cnt]
            feature_list_new.append(feature)
        feature_list_new.reverse()
        processed_feature = torch.stack(feature_list_new,dim = 2)
        processed_feature = processed_feature.squeeze(dim = 3 )
        return processed_feature

# %%
class mess_passing_W(nn.Module): # L -> R
    def __init__(self, in_channels = 32, out_channels= 32):
        super(mess_passing_W, self).__init__()
        self.l2r_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[9,1], stride=(1,1), padding=(4,0))
        self.r2l_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[9,1], stride=(1,1), padding=(4,0))
        self.relu = nn.ReLU()
    def forward(self, x):
        nB, c, h, w = x.shape
        feature_list_old = []
        feature_list_new = []
        for cnt in range(w):
            feature_list_old.append(x[:, :, : , cnt].unsqueeze(dim = 3))
        feature_list_new.append(x[:, :, :, 0].unsqueeze(dim = 3))
        feature = self.l2r_layers(feature_list_old[0])
        feature = self.relu(feature) + feature_list_old[1]
        feature_list_new.append(feature)
        for cnt in range(2, w):
            feature = self.l2r_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[cnt]
            feature_list_new.append(feature)
        feature_list_old = feature_list_new
        feature_list_new = []
        length = w - 1
        feature_list_new.append(feature_list_old[length])
        feature = self.r2l_layers(feature_list_old[length])
        feature = self.relu(feature) + feature_list_old[length - 1]
        feature_list_new.append(feature)
        for cnt in range(2, w):
            feature = self.r2l_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[length - cnt]
            feature_list_new.append(feature)
        feature_list_new.reverse()
        processed_feature = torch.stack(feature_list_new, dim=3)
        processed_feature = torch.squeeze(processed_feature, axis=4)
        return processed_feature

# %%
class sun_branch_1(nn.Module):
    def __init__(self, size_):
        super(sun_branch_1, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

# %%
class sun_branch_2(nn.Module):
    def __init__(self, size_):
        super(sun_branch_2, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

# %%
class sun_branch_3(nn.Module):
    def __init__(self, size_):
        super(sun_branch_3, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

# %%
class updasample_combines(nn.Module):
    def __init__(self ):
        super(updasample_combines, self).__init__()
        self.conv = nn.Conv2d(32, 256, kernel_size=3, stride=1, padding= 'same')
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x_tensor):
        x = F.interpolate(x_tensor, size =(SHAPE,SHAPE), mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# %%
class predict_row(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(predict_row, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        row_ = self.conv(x)
        final_row = self.sigmoid(row_)
        return final_row

# %%
class predict_col(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(predict_col, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        col_ = self.conv(x)
        final_col = self.sigmoid(col_)
        return final_col

# %%
class model_split(nn.Module):
    def __init__(self):
        super(model_split, self).__init__()  # 128 -> 256
        self.backbone = Resnet_FPN()
        self.mess_passing_H = mess_passing_H()
        self.mess_passing_W = mess_passing_W()
        self.predict_row = predict_row(256, 1)
        self.predict_col = predict_col(256, 1)
        self.sun_branch_W_1 = sun_branch_1(size_=(1,2))
        self.sun_branch_W_2 = sun_branch_2(size_=(1,2))
        self.sun_branch_W_3 = sun_branch_3(size_=(1,2))

        self.sun_branch_H_1 = sun_branch_1(size_=(2,1))
        self.sun_branch_H_2 = sun_branch_2(size_=(2,1))
        self.sun_branch_H_3 = sun_branch_3(size_=(2,1))

        self.transfer_layer = transferlayer(128, 64)  # 256 -> 16
        self.updasample_combines_H = updasample_combines()
        self.updasample_combines_W = updasample_combines()  # 256 * 256
    def forward(self, image):
        c2, c3 , c4, c5 = self.backbone(image)
        fm = self.transfer_layer(c2)
        x_H = fm
        x_W = fm
        # for _ in range(3):
        x_H= self.sun_branch_H_1(x_H, x_H.shape[1])
        x_H= self.sun_branch_H_2(x_H, x_H.shape[1])
        x_H= self.sun_branch_H_3(x_H, x_H.shape[1])

        x_W = self.sun_branch_W_1(x_W, x_W.shape[1])
        x_W = self.sun_branch_W_2(x_W, x_W.shape[1])
        x_W = self.sun_branch_W_3(x_W, x_W.shape[1])
        # print(f'x_W : {x_W.shape}')
        col_feature = self.mess_passing_H(x_H)
        col_feature = self.updasample_combines_H(col_feature)  # 256 * 256

        row_feature = self.mess_passing_W(x_W)
        row_feature = self.updasample_combines_W(row_feature)  # 256 * 256
        # print(f'row_feature :{row_feature.shape}, col_feature:{col_feature.shape}')
        final_row = self.predict_row(row_feature)
        final_col = self.predict_col(col_feature)
        return final_row, final_col

# %%
x = torch.rand((1,3,640,512))
x  = x.to(device)
model=  model_split()
model.to(device)
a, b = model(x)
print(a.shape)
print(b.shape)

# %% [markdown]
# # data

# %%
data_path ="C:/Users/SEHC/Desktop/qa/LV/Split/data/"
final_data = pd.read_csv("final_dataframe_fintabnet_split_5.csv")
final_data_sub = final_data[40000:50000]

# %%
import ast
def convert_String_to_array(string_array):
    nested_list = ast.literal_eval(string_array)

    numpy_array_1 = np.array(nested_list[0])
    numpy_array_2 = np.array(nested_list[1])
    # print(numpy_array_1.shape)
    # print(numpy_array_2.shape)
    # numpy_array = np.concatenate((numpy_array_1, numpy_array_2), axis=0)
    return numpy_array_1 , numpy_array_2

# %%
def normalize_img(img):
    #norm_img = (img - img.min()) / (img.max() - img.min())
    norm_img = np.array(img,dtype= np.float32)/255.0
    return norm_img

# %%
def make_image_row(row ,h , w,  new_shape= (512,512)):
  img_row = np.zeros((h, w), dtype=np.uint8)
  mask_img_row = np.where(row ==1 )
  img_row[mask_img_row,:] = 1
  img_row = cv2.resize(img_row, new_shape)
  img_row  = np.array(img_row)
  return img_row

def make_image_col(col,h , w, new_shape= (512,512)):
  img_col = np.zeros((h, w), dtype=np.uint8)
  mask_img_col = np.where(col ==1 )
  img_col[:,mask_img_col] = 1
  img_col = cv2.resize(img_col, new_shape)
  img_col  = np.array(img_col)
  return img_col

# %%
class Data_loader(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
    def __len__(self):
        return len(self.data)
    def make_image_col(self, col, h, w, new_shape= (512,512)):
        img_col = np.zeros((h, w), dtype=np.uint8)
        mask_img_col = np.where(col ==1 )
        img_col[:,mask_img_col] = 1
        img_col = cv2.resize(img_col, new_shape)
        img_col  = np.array(img_col)
        return img_col

    def make_image_row(self, row, h, w, new_shape= (512,512)):
        img_row = np.zeros((h, w), dtype=np.uint8)
        mask_img_row = np.where(row ==1 )
        img_row[mask_img_row,:] = 1
        img_row = cv2.resize(img_row, new_shape)
        img_row  = np.array(img_row)
        return img_row

    def load_data(self, image_path, gt_row, gt_col):
        image_input = Image.open(data_path + image_path + '.jpg')
        image_input = np.array(image_input)
        h,w,c = image_input.shape
        image_input = cv2.resize(image_input, (SHAPE,SHAPE))
        image_input = image_input.astype('float')
        image_input = np.array(image_input,dtype= np.float32)/255.0
        row_image = make_image_row(gt_row, h, w, new_shape=(SHAPE, SHAPE))
        col_image = make_image_col(gt_col, h, w, new_shape=(SHAPE, SHAPE))
        return image_input, row_image, col_image
    def load_data_test(self, image_path, gt_row, gt_col):
        image_input_ori = Image.open(data_path + image_path + '.jpg')
        image_input = np.array(image_input_ori)
        h,w,c = image_input.shape
        print(h,w)
        image_input = cv2.resize(image_input, (W_SHAPE , H_SHAPE))
        image_input = image_input.astype('float')
        image_input = np.array(image_input,dtype= np.float32)/255.0
        row_image = make_image_row(gt_row, h, w, new_shape=(W_SHAPE,H_SHAPE))
        col_image = make_image_col(gt_col, h, w, new_shape=(W_SHAPE,H_SHAPE))
        return image_input, row_image, col_image , np.array(image_input_ori)
    def convert_String_to_array(self, string_array):
        nested_list = ast.literal_eval(string_array)

        numpy_array_1 = np.array(nested_list[0])
        numpy_array_2 = np.array(nested_list[1])
        # print(numpy_array_1.shape)
        # print(numpy_array_2.shape)
        # numpy_array = np.concatenate((numpy_array_1, numpy_array_2), axis=0)
        return numpy_array_1 , numpy_array_2
    def normalize_img(self,img):
    # norm_img = (img - img.min()) / (img.max() - img.min())
        norm_img = np.array(img,dtype= np.float32)/255.0
        return norm_img
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx][0]
        data = self.data.iloc[idx][1]
        data = data[1:-1]
        data_1, data_2 = self.convert_String_to_array(data)
        gt_row = data_1
        gt_row = np.array(gt_row)
        gt_col = data_2
        gt_col = np.array(gt_col)
        image_input, row_image, col_image , image_ori = self.load_data_test(image_path, gt_row, gt_col)
        row_image = torch.from_numpy(row_image)
        row_image = torch.unsqueeze(row_image, dim = 0)
        # row_image = row_image.permute(2,0,1)

        col_image = torch.from_numpy(col_image)
        col_image = torch.unsqueeze(col_image, dim = 0)
        # col_image = row_image.permute(2,0,1)


        image_tensor =  torch.from_numpy(image_input)
        image_tensor = image_tensor.permute(2,0,1)
        return {"image": image_tensor , 'image_ori': image_ori}, {"row": row_image, "col" :col_image}

# %%
data_test = Data_loader(final_data_sub[2000:])
data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)
input_ , output_ = next(iter(data_loader_test))

# %%
input_['image'].shape

output_["row"].shape
input_['image_ori'].shape

# %%
ori_img  = input_['image_ori'][0]
plt.imshow(ori_img)
print(ori_img.shape)

# %%
test_image = input_['image'][0]

test_image =   test_image.permute(1,2,0)
test_image_np = np.array(test_image)

row_image = output_["row"][0]

row_image =   row_image.permute(1,2,0)
row_image_np = np.array(row_image)

fix ,ax = plt.subplots(1,2 ,figsize= (15,5))
ax[0].imshow(test_image_np)
ax[1].imshow(row_image_np)
# plt.show()

# %%
test_image = output_["col"][0]

test_image =   test_image.permute(1,2,0)
test_image_np = np.array(test_image)

plt.imshow(test_image_np)
plt.show()

# %% [markdown]
# # train

# %%
class Data_loader(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
    def __len__(self):
        return len(self.data)
    def make_image_col(self, col, h, w, new_shape= (512,512)):
        img_col = np.zeros((h, w), dtype=np.uint8)
        mask_img_col = np.where(col ==1 )
        img_col[:,mask_img_col] = 1
        img_col = cv2.resize(img_col, new_shape)
        img_col  = np.array(img_col)
        return img_col

    def make_image_row(self, row, h, w, new_shape= (512,512)):
        img_row = np.zeros((h, w), dtype=np.uint8)
        mask_img_row = np.where(row ==1 )
        img_row[mask_img_row,:] = 1
        img_row = cv2.resize(img_row, new_shape)
        img_row  = np.array(img_row)
        return img_row

    def load_data(self, image_path, gt_row, gt_col):
        image_input = Image.open(data_path + image_path + '.jpg')
        image_input = np.array(image_input)
        h,w,c = image_input.shape
        image_input = cv2.resize(image_input, (SHAPE,SHAPE))
        image_input = image_input.astype('float')
        image_input = np.array(image_input,dtype= np.float32)/255.0
        row_image = make_image_row(gt_row, h, w, new_shape=(SHAPE, SHAPE))
        col_image = make_image_col(gt_col, h, w, new_shape=(SHAPE, SHAPE))
        return image_input, row_image, col_image
    def load_data_test(self, image_path, gt_row, gt_col):
        image_input_ori = Image.open(data_path + image_path + '.jpg')
        image_input = np.array(image_input_ori)
        h,w,c = image_input.shape
        print(h,w)
        image_input = cv2.resize(image_input, (W_SHAPE , H_SHAPE))
        image_input = image_input.astype('float')
        image_input = np.array(image_input,dtype= np.float32)/255.0
        row_image = make_image_row(gt_row, h, w, new_shape=(W_SHAPE,H_SHAPE))
        col_image = make_image_col(gt_col, h, w, new_shape=(W_SHAPE,H_SHAPE))
        return image_input, row_image, col_image , np.array(image_input_ori)
    def convert_String_to_array(self, string_array):
        nested_list = ast.literal_eval(string_array)

        numpy_array_1 = np.array(nested_list[0])
        numpy_array_2 = np.array(nested_list[1])
        # print(numpy_array_1.shape)
        # print(numpy_array_2.shape)
        # numpy_array = np.concatenate((numpy_array_1, numpy_array_2), axis=0)
        return numpy_array_1 , numpy_array_2
    def normalize_img(self,img):
    # norm_img = (img - img.min()) / (img.max() - img.min())
        norm_img = np.array(img,dtype= np.float32)/255.0
        return norm_img
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx][0]
        data = self.data.iloc[idx][1]
        data = data[1:-1]
        data_1, data_2 = self.convert_String_to_array(data)
        gt_row = data_1
        gt_row = np.array(gt_row)
        gt_col = data_2
        gt_col = np.array(gt_col)
        image_input, row_image, col_image  = self.load_data(image_path, gt_row, gt_col)
        row_image = torch.from_numpy(row_image)
        row_image = torch.unsqueeze(row_image, dim = 0)
        # row_image = row_image.permute(2,0,1)

        col_image = torch.from_numpy(col_image)
        col_image = torch.unsqueeze(col_image, dim = 0)
        # col_image = row_image.permute(2,0,1)


        image_tensor =  torch.from_numpy(image_input)
        image_tensor = image_tensor.permute(2,0,1)
        return {"image": image_tensor }, {"row": row_image, "col" :col_image}

# %%
data_path ="C:/Users/SEHC/Desktop/qa/LV/Split/data/"
final_data = pd.read_csv("final_dataframe_fintabnet_split_5.csv")
# final_data_sub = final_data[0:15000]

data_frame = final_data[:79900]
data = Data_loader(data_frame)
data_loader = DataLoader(data, batch_size=2, shuffle=True)

# %%
model_ = model_split()
# model_ = nn.DataParallel(model_)
model_.to(device)
# convert_rois_to_boxes = convert_rois_to_boxes.to(device)
optimizer = optim.AdamW(model_.parameters(), lr=0.00001, betas=(0.9, 0.999))

# %%
def dice_loss(y_pred ,y_true, smooth=1e-5):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = torch.sum(y_true * y_pred).sum()
    # sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1,2,3))
    # sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1,2,3))
    dice = 1 - (2 * intersection + smooth) / (y_pred.sum() +  y_true.sum() + smooth)
    return dice

# %%
from tqdm import  tqdm

# %%
# load model

# weights = torch.load("split_25_epoch_again_reverse_512.pth", map_location= device)
# model_.load_state_dict(weights)

# %%
model_name = 'split_model_fintab_25k_'

# %%
def save_check_point(epoch, model_1 , optimizer):
    folder_checkpoint = 'C:/Users/SEHC/Desktop/qa/LV/Split/check_point/'
    checkpoint = {
            'epoch': epoch ,
            'state_dict': model_1.state_dict(),
            'optimizer': optimizer.state_dict()
        }
    torch.save(checkpoint, folder_checkpoint + 'check_point_split_model_fintab_80k' + str(epoch) +'.pth')
    # torch.save(model_1.state_dict(), 'split_model_fintab_80k' + str(epoch) +'.pth')

# %%
model_save_folder_path  = "C:/Users/SEHC/Desktop/qa/LV/Split/model_720_512_shape/"

# %%
num_epochs = 20
loss_list  =[]
loss_row_list  =[]
loss_col_list = []
for epoch in range(num_epochs):
    for i, (inputs, gt) in tqdm(enumerate(data_loader)):
        # print(f'epoch :{epoch}, index {i}')
        # model_1.train(True)
        img_in = inputs['image']
        # bbox_in = inputs['bbox']

        row_gt = gt['row']
        row_gt = row_gt.float()
        # row_gt = row_gt.view(-1)
        col_gt = gt['col']
        col_gt = col_gt.float()
        # col_gt = col_gt.view(-1)
        img_in = img_in.to(device)
        if (torch.isnan(img_in).any()):
            print("error")
            continue

        row_gt = row_gt.to(device)
        col_gt = col_gt.to(device)
        optimizer.zero_grad()
        row_out, col_out = model_(img_in)

        row_loss = nn.BCELoss()(row_out, row_gt)
        col_loss = nn.BCELoss()(col_out, col_gt)
        bce_loss_total = row_loss  + col_loss

        row_dcie_loss = dice_loss(row_out, row_gt)
        col_dcie_loss = dice_loss(col_out, col_gt)
        dcie_loss_total = row_dcie_loss  + col_dcie_loss
        # print(dcie_loss_total)
        total_loss = 0.4*bce_loss_total + 0.6*dcie_loss_total
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_.parameters(), 0.001)
        optimizer.step()


        loss_list.append(total_loss.item())
        loss_row_list.append(row_loss.item())
        loss_col_list.append(col_loss.item())
        del row_loss
        del col_loss
        del  total_loss
        del row_out
        del col_out
    loss_epoch = sum(loss_list)/len(loss_list)
    loss_row_epoch = sum(loss_row_list)/len(loss_row_list)
    loss_col_epoch = sum(loss_col_list)/len(loss_col_list)
    print(f'epoch:{epoch}============ loss: {loss_epoch}      loss row: {loss_row_epoch}      loss col :{loss_col_epoch}')
    loss_list.clear()
    loss_row_list.clear()
    loss_col_list.clear()
    save_check_point(epoch , model_, optimizer)
    torch.save(model_.state_dict(),model_save_folder_path  +"split_model_fintab_80k_720_512_" + str(epoch) +".pth")    
    # if epoch == 10:
    #     torch.save(model_.state_dict(), "split_model_fintab_80k_10.pth")    
    # if epoch == 15:
    #     torch.save(model_.state_dict(), "split_model_fintab_80k_15.pth")
    # if epoch == 19:
    #     torch.save(model_.state_dict(), "split_model_fintab_80k_20.pth")

# %%
num_epochs = 20
loss_list  =[]
loss_row_list  =[]
loss_col_list = []
for epoch in range(20,25):
    for i, (inputs, gt) in tqdm(enumerate(data_loader)):
        # print(f'epoch :{epoch}, index {i}')
        # model_1.train(True)
        img_in = inputs['image']
        # bbox_in = inputs['bbox']

        row_gt = gt['row']
        row_gt = row_gt.float()
        # row_gt = row_gt.view(-1)
        col_gt = gt['col']
        col_gt = col_gt.float()
        # col_gt = col_gt.view(-1)
        img_in = img_in.to(device)
        if (torch.isnan(img_in).any()):
            print("error")
            continue

        row_gt = row_gt.to(device)
        col_gt = col_gt.to(device)
        optimizer.zero_grad()
        row_out, col_out = model_(img_in)

        row_loss = nn.BCELoss()(row_out, row_gt)
        col_loss = nn.BCELoss()(col_out, col_gt)
        bce_loss_total = row_loss  + col_loss

        row_dcie_loss = dice_loss(row_out, row_gt)
        col_dcie_loss = dice_loss(col_out, col_gt)
        dcie_loss_total = row_dcie_loss  + col_dcie_loss
        # print(dcie_loss_total)
        total_loss = 0.4*bce_loss_total + 0.6*dcie_loss_total
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_.parameters(), 0.001)
        optimizer.step()


        loss_list.append(total_loss.item())
        loss_row_list.append(row_loss.item())
        loss_col_list.append(col_loss.item())
        del row_loss
        del col_loss
        del  total_loss
        del row_out
        del col_out
    loss_epoch = sum(loss_list)/len(loss_list)
    loss_row_epoch = sum(loss_row_list)/len(loss_row_list)
    loss_col_epoch = sum(loss_col_list)/len(loss_col_list)
    print(f'epoch:{epoch}============ loss: {loss_epoch}      loss row: {loss_row_epoch}      loss col :{loss_col_epoch}')
    loss_list.clear()
    loss_row_list.clear()
    loss_col_list.clear()
    save_check_point(epoch , model_, optimizer)
    torch.save(model_.state_dict(),model_save_folder_path  +"split_model_fintab_80k_720_512_" + str(epoch) +".pth")    

# %%
checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model_.state_dict(),
    'optimizer': optimizer.state_dict()
}
torch.save(checkpoint, 'check_point_split_model_fintab_80k_20.pth')

# %%
torch.save(model_.state_dict(), "split_25_epoch_mix_data.pth"

# %%
from collections import OrderedDict

# %%
model_2 = model_split()
model_2.to('cpu')
weights = torch.load("C:/Users/SEHC/Desktop/qa/LV/Split/split_model_fintab_25k_15.pth", map_location='cpu')
model_2.load_state_dict(weights)

# %%
data = Data_loader(final_data[1:])
data_loader = DataLoader(data, batch_size=1, shuffle=False)
input_ , output_ = next(iter(data_loader))
a, b = model_2(input_['image'])

# %%
i = 25423

# %%
i = i+1
data = Data_loader(final_data[i:])
data_loader = DataLoader(data, batch_size=1, shuffle=False)
input_ , output_ = next(iter(data_loader))
a, b = model_2(input_['image'])
test_image = input_['image'][0]
print(test_image.shape)
test_image =   test_image.permute(1,2,0)
# test_image_np = np.array(test_image)
row_pred = a[0]
row_pred =   row_pred.permute(1,2,0)
row_pred = row_pred.cpu()
# test_image_np = np.array(test_image)
row_pred = row_pred.detach().numpy()
# plt.imshow(test_image)
# plt.show()

fig, ax = plt.subplots(1,2, figsize=(25, 25))
ax[0].imshow(test_image)
ax[1].imshow(row_pred)

# %%
test_image = b[0]
print(test_image.shape)
test_image =   test_image.permute(1,2,0)
test_image = test_image.cpu()
# test_image_np = np.array(test_image)
test_image_np = test_image.detach().numpy()

plt.imshow(test_image_np)
plt.show()

# %%
loss_fnc = nn.BCELoss()
loss_fnc.to(device)
gt_row = output_['row']
gt_row.to(device)
c = loss_fnc(a, gt_row.float())
c

# %%
gt_row = output_['row'][0]
gt_row =   gt_row.permute(1,2,0)
gt_row = gt_row.cpu()
# test_image_np = np.array(test_image)
gt_row = gt_row.detach().numpy()
fig, ax = plt.subplots(1,2, figsize=(25, 25))
ax[0].imshow(test_image)
ax[1].imshow(gt_row)

# %%
image = Image.open('a.jpg')
image = np.array(image)
# image_tensor = torch.from_numpy(image)
# image_tensor = image_tensor.permute(2,1,0)
# image_tensor = torch.unsqueeze(image_tensor, dim = 0)
# image_tensor.shape
image_input = cv2.resize(image, (512,512))
image_input = image_input.astype('float')
image_input = np.array(image_input,dtype= np.float32)/255.0
image_tensor =  torch.from_numpy(image_input)
image_tensor = image_tensor.permute(2,0,1)
image_tensor = torch.unsqueeze(image_tensor, dim = 0)
image_tensor.shape
a,b = model_2(image_tensor)

# %%
test_image = a[0]
print(test_image.shape)
test_image =   test_image.permute(1,2,0)
test_image = test_image.cpu()
# test_image_np = np.array(test_image)
test_image_np = test_image.detach().numpy()

# plt.imshow(test_image_np)
# plt.show()

# %%
col_img_uint8 = test_image_np.copy()
col_img_uint8 = np.squeeze(col_img_uint8, axis= 2)
plt.imshow(col_img_uint8)
plt.show()
# binary_img = cv2.threshold(col_img_uint8, 127, 255, cv2.THRESH_BINARY)
# binary_img.shape

# %%
reversed_binary_img = 1 - col_img_uint8
mask_1 = reversed_binary_img> 0.5
mask_2 = reversed_binary_img <=0.5
reversed_binary_img[mask_1] = 1 
reversed_binary_img[mask_2] = 0 
plt.imshow(reversed_binary_img, cmap='gray')
plt.show()

# %%
test = reversed_binary_img.copy()
print(test.shape)
mask = np.all(test == 1 ,axis=1)
test[mask == True , :] = 1
test[mask == False, :] = 0
# plt.imshow(test)
# plt.show()

# %%
test = reversed_binary_img.copy()
print(test.shape)
mask = np.all(test == 1 ,axis=1)
test[mask == True, :] = 1
test[mask == False , :] = 0
plt.imshow(test)
plt.show()

# %%
# reversed_binary_img = reversed_binary_img.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
contours, hierarchy = cv2.findContours(test.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fit polynomial curves to each contour
polynomial_fits = []
for i, contour in enumerate(contours):
    # print(i)
    # Extract x and y coordinates from the contour points
    x = contour[:, 0, 1]
    y = contour[:, 0, 0]
    
    # Fit a polynomial of degree 2 (quadratic) to the contour points
    try:
        polynomial_fit = np.polyfit(y, x, 0)
        polynomial_fits.append(polynomial_fit)
    except:
        pass

# Display the polynomial fits
for fit in polynomial_fits:
    print(fit)

# %%
image = Image.open('a.jpg')
image = np.array(image)
image_input = cv2.resize(image, (512,512))

# %%

for line in polynomial_fits:
    cv2.line(image_input, (int(line[0]), 0), ( int(line[0]) , 512), (0,255,0), 1)
    # print(line[0])
plt.imshow(image_input)
plt.show()

# %%
for line in polynomial_fits:
    cv2.line(image_input, (0 , int(line[0])), (512, int(line[0])), (0,255,0), 1)
    # print(line[0])
plt.imshow(image_input)
plt.show()


