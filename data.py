from __future__ import print_function
from torchtools import *
# from torch import *
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
        self.partition = partition    #分区
        self.data_size = [3, 84, 84]

        # set normalizer  #标准化
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),   #依照给定的size随机裁剪，padding填充模块
                                                 lambda x: np.asarray(x),   #lambda表示匿名函数，只能有一个表达式，不用写return,返回值就是该表达式的结果
                                                 transforms.ToTensor(), # PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor，并归一化到[0, 1.0]  range [0, 255] -> [0.0,1.0]
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
        # dataset_path = os.path.join(self.root, 'mini-imagenet\compacted_dataset','\mini_imagenet_%s.pickle' % self.partition) #路径拼接
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)   #必填参数file必须以二进制可读模式打开，即“rb”，其他都为可选参数


        # for each class  类别数=4
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
                       num_tasks=80,     #80
                       num_ways=5,  #类别数5
                       num_shots=1,  #一个类别中的样本数1
                       num_queries=1,  #查询的样本数1
                       seed=None):  #223

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []   #空列表
        for _ in range(num_ways * num_shots):   #5*1
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)  #将data添加到此support_data的末尾，返回一个新对象。 不在此support_data中的列将作为新列添加
            support_label.append(label)
        for _ in range(num_ways * num_queries):   #5*1
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


                # load sample for support set   加载支持集的样本
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


    # pyTorch中和数据读取相关的类都要继承一个基类：torch.utils.data.Dataset
    # 初始化传入参数
    def __init__(self, root, partition='train', data_name="tiered-imagenet"):
        super(TieredImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition  # 分区
        self.data_size = [3, 84, 84]

        # set normalizer  #标准化
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),   #依照给定的size随机裁剪，padding填充模块
                                                 lambda x: np.asarray(x),   #lambda表示匿名函数，只能有一个表达式，不用写return,返回值就是该表达式的结果
                                                 transforms.ToTensor(), # PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor，并归一化到[0, 1.0]  range [0, 255] -> [0.0,1.0]
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        self.data = self.load_dataset()

    def load_dataset(self):
        # set the paths of the csv files
        train_csv = os.path.join(self.root,'tiered-imagenet', 'train.csv')  # os.path.join()将多个路径组合返回
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

            # FC100
            # for i in data.keys():
            #     if isinstance(i, str):
            #         data[int(i.strip('n'))] = data.pop(i)
            # for i in data.keys():
            #     if isinstance(i, str):
            #         data[int(i.strip('n'))] = data.pop(i)
            # class_list = data.keys()


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


# class TieredImagenetLoader(data.Dataset):
#     def __init__(self, root, partition='train'):
#         self.root = root
#         self.partition = partition  # train/val/test
#         # self.preprocess()
#         self.data_size = [3, 84, 84]
#
#         # load data
#         self.data = self.load_dataset()
#         #
#         # if not self._check_exists_():
#         #     self._init_folders_()
#         #     if self.check_decompress():
#         #         self._decompress_()
#         #     self._preprocess_()
#
#
#     def get_image_paths(self, file):
#         images_path, class_names = [], []
#         with open(file, 'r') as f:
#             f.readline()
#             for line in f:
#                 name, class_ = line.split(',')
#                 class_ = class_[0:(len(class_)-1)]
#                 path = self.root + '/tiered-imagenet/images/'+ name +'/' + class_
#                 images_path.append(path)
#                 class_names.append(class_)
#         return class_names, images_path
#
#     def preprocess(self):
#         print('\nPreprocessing Tiered-Imagenet images...')
#         (class_names_train, images_path_train) = self.get_image_paths('%s/tiered-imagenet/train.csv' % self.root)
#         (class_names_test, images_path_test) = self.get_image_paths('%s/tiered-imagenet/test.csv' % self.root)
#         (class_names_val, images_path_val) = self.get_image_paths('%s/tiered-imagenet/val.csv' % self.root)
#
#         keys_train = list(set(class_names_train))
#         keys_test = list(set(class_names_test))
#         keys_val = list(set(class_names_val))
#         label_encoder = {}
#         label_decoder = {}
#         for i in range(len(keys_train)):
#             label_encoder[keys_train[i]] = i
#             label_decoder[i] = keys_train[i]
#         for i in range(len(keys_train), len(keys_train)+len(keys_test)):
#             label_encoder[keys_test[i-len(keys_train)]] = i
#             label_decoder[i] = keys_test[i-len(keys_train)]
#         for i in range(len(keys_train)+len(keys_test), len(keys_train)+len(keys_test)+len(keys_val)):
#             label_encoder[keys_val[i-len(keys_train) - len(keys_test)]] = i
#             label_decoder[i] = keys_val[i-len(keys_train)-len(keys_test)]
#
#         counter = 0
#         train_set = {}
#
#         for class_, path in zip(class_names_train, images_path_train):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#             if label_encoder[class_] not in train_set:
#                 train_set[label_encoder[class_]] = []
#             train_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter "+str(counter) + " from " + str(len(images_path_train)))
#
#         test_set = {}
#         for class_, path in zip(class_names_test, images_path_test):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#
#             if label_encoder[class_] not in test_set:
#                 test_set[label_encoder[class_]] = []
#             test_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter " + str(counter) + " from "+str(len(class_names_test)))
#
#         val_set = {}
#         for class_, path in zip(class_names_val, images_path_val):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#
#             if label_encoder[class_] not in val_set:
#                 val_set[label_encoder[class_]] = []
#             val_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter "+str(counter) + " from " + str(len(class_names_val)))
#
#         partition_count = 0
#         for item in self.chunks(train_set, 20):
#             partition_count = partition_count + 1
#             with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_train_{}.pickle'.format(partition_count)), 'wb') as handle:
#                 pickle.dump(item, handle, protocol=2)
#
#         partition_count = 0
#         for item in self.chunks(test_set, 20):
#             partition_count = partition_count + 1
#             with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_test_{}.pickle'.format(partition_count)), 'wb') as handle:
#                 pickle.dump(item, handle, protocol=2)
#
#         partition_count = 0
#         for item in self.chunks(val_set, 20):
#             partition_count = partition_count + 1
#             with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_val_{}.pickle'.format(partition_count)), 'wb') as handle:
#                 pickle.dump(item, handle, protocol=2)
#
#
#
#         label_encoder = {}
#         keys = list(train_set.keys()) + list(test_set.keys())
#         for id_key, key in enumerate(keys):
#             label_encoder[key] = id_key
#         with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_label_encoder.pickle'), 'wb') as handle:
#             pickle.dump(label_encoder, handle, protocol=2)
#
#         print('Images preprocessed')
#
#     def load_dataset(self):
#         print("Loading dataset")
#         data = {}
#         if self.partition == 'train':
#             num_partition = 18
#         elif self.partition == 'val':
#             num_partition = 5
#         elif self.partition == 'test':
#             num_partition = 8
#
#         partition_count = 0
#         for i in range(num_partition):
#             partition_count = partition_count +1
#             with open(os.path.join(self.root, 'tiered-imagenet/compacted_datasets', 'tiered_imagenet_{}_{}.pickle'.format(self.partition, partition_count)), 'rb') as handle:
#                 data.update(pickle.load(handle))
#
#         # Resize images and normalize
#         for class_ in data:
#             for i in range(len(data[class_])):
#                 image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
#                 image_resized = image2resize.resize((self.data_size[2], self.data_size[1]))
#                 image_resized = np.array(image_resized, dtype='float32')
#
#                 # Normalize
#                 image_resized = np.transpose(image_resized, (2, 0, 1))
#                 image_resized[0, :, :] -= 120.45  # R
#                 image_resized[1, :, :] -= 115.74  # G
#                 image_resized[2, :, :] -= 104.65  # B
#                 image_resized /= 127.5
#
#                 data[class_][i] = image_resized
#
#         print("Num classes " + str(len(data)))
#         num_images = 0
#         for class_ in data:
#             num_images += len(data[class_])
#         print("Num images " + str(num_images))
#         return data
#
#     def chunks(self, data, size=10000):
#         it = iter(data)
#         for i in range(0, len(data), size):
#             yield {k: data[k] for k in islice(it, size)}
#
#     def get_task_batch(self,
#                        num_tasks=5,
#                        num_ways=20,
#                        num_shots=1,
#                        num_queries=1,
#                        seed=None):
#         if seed is not None:
#             random.seed(seed)
#
#         # init task batch data
#         support_data, support_label, query_data, query_label = [], [], [], []
#         for _ in range(num_ways * num_shots):
#             data = np.zeros(shape=[num_tasks] + self.data_size,
#                             dtype='float32')
#             label = np.zeros(shape=[num_tasks],
#                              dtype='float32')
#             support_data.append(data)
#             support_label.append(label)
#         for _ in range(num_ways * num_queries):
#             data = np.zeros(shape=[num_tasks] + self.data_size,
#                             dtype='float32')
#             label = np.zeros(shape=[num_tasks],
#                              dtype='float32')
#             query_data.append(data)
#             query_label.append(label)
#
#         # get full class list in dataset
#         full_class_list = list(self.data.keys())
#
#         # for each task
#         for t_idx in range(num_tasks):
#             # define task by sampling classes (num_ways)
#             task_class_list = random.sample(full_class_list, num_ways)
#
#             # for each sampled class in task
#             for c_idx in range(num_ways):
#                 # sample data for support and query (num_shots + num_queries)
#
#                 class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)
#
#
#                 # load sample for support set
#                 for i_idx in range(num_shots):
#                     # set data
#                     support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
#                     support_label[i_idx + c_idx * num_shots][t_idx] = c_idx
#
#                 # load sample for query set
#                 for i_idx in range(num_queries):
#                     query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
#                     query_label[i_idx + c_idx * num_queries][t_idx] = c_idx
#
#
#
#         # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
#         support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
#         support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
#         query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
#         query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)
#
#         return [support_data, support_label, query_data, query_label]

# class TieredImagenetLoader(object):
#     def __init__(self, root, partition='train'):
#         self.root = root
#         self.dataset = dataset
#         if not self._check_exists_():
#             self._init_folders_()
#             if self.check_decompress():
#                 self._decompress_()
#             self._preprocess_()
#
#     def _init_folders_(self):
#         decompress = False
#         if not os.path.exists(self.root):
#             os.makedirs(self.root)
#         if not os.path.exists(os.path.join(self.root, 'tiered-imagenet')):
#             os.makedirs(os.path.join(self.root, 'tiered-imagenet'))
#             decompress = True
#         if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
#             os.makedirs(os.path.join(self.root, 'compacted_datasets'))
#             decompress = True
#         return decompress
#
#     def check_decompress(self):
#         return os.listdir('%s/tiered-imagenet' % self.root) == []
#
#     def _decompress_(self):
#         print("\nDecompressing Images...")
#         compressed_file = '%s/compressed/tiered-imagenet/images.zip' % self.root
#         if os.path.isfile(compressed_file):
#             os.system('unzip %s -d %s/tiered-imagenet/' % (compressed_file, self.root))
#         else:
#             raise Exception('Missing %s' % compressed_file)
#         print("Decompressed")
#
#     def _check_exists_(self):
#         if not os.path.exists(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_train.pickle')) or not \
#                 os.path.exists(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_test.pickle')):
#             return False
#         else:
#             return True
#
#     def get_image_paths(self, file):
#         images_path, class_names = [], []
#         with open(file, 'r') as f:
#             f.readline()
#             for line in f:
#                 name, class_ = line.split(',')
#                 class_ = class_[0:(len(class_)-1)]
#                 path = self.root + '/tiered-imagenet/images/'+name+'/'+class_
#                 images_path.append(path)
#                 class_names.append(class_)
#         return class_names, images_path
#
#     def _preprocess_(self):
#         print('\nPreprocessing tiered-imagenet images...')
#         (class_names_train, images_path_train) = self.get_image_paths('%s/tiered-imagenet/train.csv' % self.root)
#         (class_names_test, images_path_test) = self.get_image_paths('%s/tiered-imagenet/test.csv' % self.root)
#         (class_names_val, images_path_val) = self.get_image_paths('%s/tiered-imagenet/val.csv' % self.root)
#
#         keys_train = list(set(class_names_train))
#         keys_test = list(set(class_names_test))
#         keys_val = list(set(class_names_val))
#         label_encoder = {}
#         label_decoder = {}
#         for i in range(len(keys_train)):
#             label_encoder[keys_train[i]] = i
#             label_decoder[i] = keys_train[i]
#         for i in range(len(keys_train), len(keys_train)+len(keys_test)):
#             label_encoder[keys_test[i-len(keys_train)]] = i
#             label_decoder[i] = keys_test[i-len(keys_train)]
#         for i in range(len(keys_train)+len(keys_test), len(keys_train)+len(keys_test)+len(keys_val)):
#             label_encoder[keys_val[i-len(keys_train) - len(keys_test)]] = i
#             label_decoder[i] = keys_val[i-len(keys_train)-len(keys_test)]
#
#         counter = 0
#         train_set = {}
#         for class_, path in zip(class_names_train, images_path_train):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#             if label_encoder[class_] not in train_set:
#                 train_set[label_encoder[class_]] = []
#             train_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
#                                                                len(class_names_val)))
#
#         test_set = {}
#         for class_, path in zip(class_names_test, images_path_test):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#
#             if label_encoder[class_] not in test_set:
#                 test_set[label_encoder[class_]] = []
#             test_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter " + str(counter) + " from "+str(len(images_path_train) + len(class_names_test) +
#                                                                len(class_names_val)))
#
#         val_set = {}
#         for class_, path in zip(class_names_val, images_path_val):
#             img = pil_image.open(path)
#             img = img.convert('RGB')
#             img = img.resize((84, 84), pil_image.ANTIALIAS)
#             img = np.array(img, dtype='float32')
#
#             if label_encoder[class_] not in val_set:
#                 val_set[label_encoder[class_]] = []
#             val_set[label_encoder[class_]].append(img)
#             counter += 1
#             if counter % 1000 == 0:
#                 print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
#                                                                len(class_names_val)))
#
#         with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_train.pickle'), 'wb') as handle:
#             pickle.dump(train_set, handle, protocol=2)
#         with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_test.pickle'), 'wb') as handle:
#             pickle.dump(test_set, handle, protocol=2)
#         with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_val.pickle'), 'wb') as handle:
#             pickle.dump(val_set, handle, protocol=2)
#
#         label_encoder = {}
#         keys = list(train_set.keys()) + list(test_set.keys())
#         for id_key, key in enumerate(keys):
#             label_encoder[key] = id_key
#         with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_label_encoder.pickle'), 'wb') as handle:
#             pickle.dump(label_encoder, handle, protocol=2)
#
#         print('Images preprocessed')
#
#     def load_dataset(self, partition, size=(84, 84)):
#         print("Loading dataset")
#         if partition == 'train_val':
#             with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_%s.pickle' % 'train'),
#                       'rb') as handle:
#                 data = pickle.load(handle)
#             with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_%s.pickle' % 'val'),
#                       'rb') as handle:
#                 data_val = pickle.load(handle)
#             data.update(data_val)
#             del data_val
#         else:
#             with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_%s.pickle' % partition),
#                       'rb') as handle:
#                 data = pickle.load(handle)
#
#         with open(os.path.join(self.root, 'compacted_datasets', 'tiered-imagenet_label_encoder.pickle'),
#                   'rb') as handle:
#             label_encoder = pickle.load(handle)
#
#         # Resize images and normalize
#         for class_ in data:
#             for i in range(len(data[class_])):
#                 image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
#                 image_resized = image2resize.resize((size[1], size[0]))
#                 image_resized = np.array(image_resized, dtype='float32')
#
#                 # Normalize
#                 image_resized = np.transpose(image_resized, (2, 0, 1))
#                 image_resized[0, :, :] -= 120.45  # R
#                 image_resized[1, :, :] -= 115.74  # G
#                 image_resized[2, :, :] -= 104.65  # B
#                 image_resized /= 127.5
#
#                 data[class_][i] = image_resized
#
#         print("Num classes " + str(len(data)))
#         num_images = 0
#         for class_ in data:
#             num_images += len(data[class_])
#         print("Num images " + str(num_images))
#         return data, label_encoder

class Cifar(data.Dataset):
    def __init__(self, root, partition='train'):
        super(Cifar, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition    #分区
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
