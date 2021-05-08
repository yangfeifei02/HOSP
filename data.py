from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import glob, os,csv
import numpy as np
from PIL import Image as pil_image
import pickle
import pickle as pkl
from itertools import islice
from torchvision import transforms
import os.path as osp
from PIL import Image
# from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd


class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition    
        self.data_size = [3, 84, 84]

        # set normalizer  
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),   
                                                 lambda x: np.asarray(x), 
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, 'mini-imagenet/compacted_dataset/','mini_imagenet_%s.pickle' % self.partition)

        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)   

        # for each class  
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):   
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                image_data = image_data.convert('RGB')
                #image_data = np.array(image_data, dtype='float32')
                #image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data


    def get_task_batch(self,
                       num_tasks=80,    
                       num_ways=5, 
                       num_shots=1, 
                       num_queries=1,
                       seed=None): 

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []   
        for _ in range(num_ways * num_shots):   # 5*1
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)  
            support_label.append(label)
        for _ in range(num_ways * num_queries):   
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set   
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])  #len=5
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device)  for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device)  for label in support_label], 1)   #将第二个维度的元素相叠加，形成一个新的tensor  #80*5*3*84*84
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device)  for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]

class TieredImagenetLoader(object):


    def __init__(self, root, partition='train', data_name="tiered-imagenet"):
        super(TieredImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition  
        self.data_size = [3, 84, 84]

        # set normalizer  #标准化
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),   
                                                 lambda x: np.asarray(x),  
                                                 transforms.ToTensor(), 
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        self.data = self.load_dataset()

    def load_dataset(self):
        # set the paths of the csv files
        train_csv = os.path.join(self.root,'tiered-imagenet', 'train.csv')
        val_csv = os.path.join(self.root,'tiered-imagenet', 'val.csv')
        test_csv = os.path.join(self.root,'tiered-imagenet', 'test.csv')

        data_list = []
        e = 0

        if self.partition == "train":
            # store all the classes and images into a dict
            dat = {}
            with open(train_csv) as f_csv:
                f_train = csv.reader(f_csv, delimiter=',')

                for row in f_train:
                    if f_train.line_num == 1:
                        continue
                    img_class, img_name = row
                    # img_name, img_class = row

                    if img_class in dat:
                        path = os.path.join(self.root,'tiered-imagenet/images',img_class,img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)
                    else:

                        dat[img_class] = []
                        path = os.path.join(self.root, 'tiered-imagenet/images', img_class, img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)
            f_csv.close()
            j = 0
            data = {}
            for i in dat.values():
                data[int(j)] = i
                j = j + 1

            # FC100
            # for i in dat.keys():
            #     if isinstance(i, str):
            #         dat[int(i.strip('n'))] = dat.pop(i)
            # for i in dat.keys():
            #     if isinstance(i, str):
            #         dat[int(i.strip('n'))] = dat.pop(i)
            # class_list = dat.keys()



        elif self.partition == "val":
            # store all the classes and images into a dict
            dat = {}
            with open(val_csv) as f_csv:
                f_val = csv.reader(f_csv, delimiter=',')
                for row in f_val:
                    if f_val.line_num == 1:
                        continue
                    img_class , img_name = row

                    if img_class in dat:
                        path = os.path.join(self.root, 'tiered-imagenet/images', img_class, img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)
                    else:

                        dat[img_class] = []

                        path = os.path.join(self.root, 'tiered-imagenet/images', img_class, img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)


            f_csv.close()
            j = 351
            data = {}
            for i in dat.values():
                data[int(j)] = i
                j = j + 1

            # FC100
            # for i in dat.keys():
            #     if isinstance(i, str):
            #         dat[int(i.strip('n'))] = dat.pop(i)
            # for i in dat.keys():
            #     if isinstance(i, str):
            #         dat[int(i.strip('n'))] = dat.pop(i)
            #
            # class_list = dat.keys()


        else:
            # store all the classes and images into a dict
            dat = {}
            with open(test_csv) as f_csv:
                f_test = csv.reader(f_csv, delimiter=',')
                for row in f_test:
                    if f_test.line_num == 1:
                        continue
                    img_class , img_name = row

                    i = 0
                    if img_class in dat:
                        path = os.path.join(self.root, 'tiered-imagenet/images', img_class, img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)
                    else:

                        dat[img_class] = []
                        # data[i] = []
                        path = os.path.join(self.root, 'tiered-imagenet/images', img_class, img_name)
                        img_name = pil_image.open(path)
                        img_name = img_name.convert('RGB')
                        img_name = img_name.resize((84, 84), pil_image.ANTIALIAS)
                        img_name = np.array(img_name, dtype='float32')
                        dat[img_class].append(img_name)
                        # data[i].append(img_name)
                        # i = i + 1
            f_csv.close()
            j = 448
            data = {}
            for i in dat.values():
                data[int(j)] = i
                j = j + 1



        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):   #i_idx在第n类的图像的大小中随机选择，
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                image_data = image_data.convert('RGB')
                #image_data = np.array(image_data, dtype='float32')
                #image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data




    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        # full_class_list = list(self.data.keys())
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]

class Cifar(data.Dataset):
    def __init__(self, root, partition='train'):
        super(Cifar, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 32, 32]
        # set normalizer
        mean_pix = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1,
                                                                        hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])


        # load data
        dataset_path = os.path.join(self.root, 'CIFAR-FS/', 'CIFAR_FS_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            data = u.load()
        self.data = data['data']
        self.labels = data['labels']
        self.label2ind = buildLabelIndex(self.labels)
        self.full_class_list = sorted(self.label2ind.keys())

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)


class CIFARFSLoader:
    """
    The dataloader of DPGN model for MiniImagenet dataset
    """
    def __init__(self, dataset):

        self.dataset = dataset
        self.data_size = dataset.data_size
        self.full_class_list = dataset.full_class_list
        self.label2ind = dataset.label2ind
        self.transform = dataset.transform


    def get_task_batch(self,
                       num_tasks=20,
                       num_ways=5,
                       num_shots=1,
                       num_queries=1,
                       seed = None):
        if seed is not None:
            random.seed(seed)
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)
        # for each task
        for t_idx in range(num_tasks):
            task_class_list = random.sample(self.full_class_list, num_ways)
            # for each sampled class in task
            for c_idx in range(num_ways):
                data_idx = random.sample(self.label2ind[task_class_list[c_idx]], num_shots + num_queries)
                class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx
                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = \
                        self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device)  for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device)  for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device)  for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device)  for label in query_label], 1)
        return support_data, support_label, query_data, query_label


def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds
