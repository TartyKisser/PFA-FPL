import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from scipy.sparse import load_npz  # 用于加载 .npz 格式的稀疏矩阵数据
from torchvision import transforms  # 用于图像数据增强和预处理
from torchvision.transforms import functional as F
import random
from PIL import ImageOps

current_path = os.path.dirname(__file__)

path_to_images_dict = {
    'COCO2014': os.path.join(current_path, '../dataset/COCO2014'),
    'VG': os.path.join(current_path, '../dataset/VG'),
}

idx_name = 'COCO2014_idx'


# MLLDataset 继承自 torch.utils.data.Dataset，是一个自定义的数据集类，主要用于多标签学习任务中的数据加载
class MLLDataset(Dataset):
    def __init__(self, dataset_name, phase='train', transform=True, image_size=84):
        self.dataset_name = dataset_name
        self.images = np.load(os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_images.npy'),
                              allow_pickle=True)
        self.labels = load_npz(
            os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_labels.npz')).toarray()
        self.image_transform = get_transform(transform=transform, image_size=image_size)
        self.max_idx = self.images.shape[0]
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < self.labels.shape[0]:
            image = Image.open(os.path.join(path_to_images_dict[self.dataset_name], self.images[index])).convert('RGB')
            image = self.image_transform(image)
            labels = torch.Tensor(self.labels[index])
            sample = {}
            sample['image'] = image
            sample['labels'] = labels
            sample['COCO2014_idx'] = index
        else:
            sample = {}
            sample['image'] = torch.zeros([3, self.image_size, self.image_size])
            sample['labels'] = torch.zeros([self.labels.shape[1]])
            sample['labels'][index - self.max_idx] = 1
            sample['COCO2014_idx'] = index
        return sample


# EpisodeSampler 用于从数据集中创建训练和验证的元任务（episode）
class EpisodeSampler:
    def __init__(self, dataset_name, n_way, n_shot, max_idx,
                 n_query=16, phase='val', iter=100):
        self.dataset_name = dataset_name
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.labels = load_npz(
            os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_labels.npz')).toarray()
        assert self.labels.shape[1] >= self.n_way  # number of ways should not more than number of labels
        self.images = np.load(os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_images.npy'),
                              allow_pickle=True)
        self.label_to_idx_dict = \
            np.load(os.path.join(current_path, idx_name, f'{dataset_name}_' + phase + '_label_to_idx_dict.npy'),
                    allow_pickle=True)[0]
        self.iter = iter
        self.max_idx = max_idx
        self.phase = phase

    def __len__(self):
        return self.iter

    def __iter__(self):
        for _ in range(self.iter):
            # ------------ sample classes ------------
            sampled_class = np.random.choice(np.arange(self.labels.shape[1]), size=self.n_way, replace=False)
            # ------------ sample samples ------------
            support_idx = []
            query_idx = []
            for c in sampled_class:
                idx = np.random.choice(self.label_to_idx_dict[c],
                                       size=np.minimum(self.n_shot + self.n_query, len(self.label_to_idx_dict[c])),
                                       replace=False)
                support_idx.extend(idx[:self.n_shot])
                query_idx.extend(idx[self.n_shot:])

            last_set = list(set(query_idx) - set(support_idx))
            query_idx = np.random.choice(list(last_set), size= self.n_query, replace=False)
            class_idx = self.max_idx + sampled_class
            if self.n_query == 0:
                all_idx = np.concatenate([support_idx, class_idx])
            else:
                all_idx = np.concatenate([support_idx, query_idx, class_idx])
            assert len(all_idx) == self.n_way * self.n_shot + self.n_query + self.n_way
            yield all_idx


def generate_MetaDataset(dataset_name, n_way, n_shot, image_size=224,
                         n_query=16, phase='val', transform=False):
    images = np.load(os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_images.npy'), allow_pickle=True)
    labels = load_npz(os.path.join(current_path, idx_name, f'{dataset_name}_{phase}_labels.npz')).toarray()
    assert labels.shape[1] >= n_way  # number of ways should not more than number of labels
    # ------------ sample classes ------------
    sampled_class = np.random.choice(np.arange(labels.shape[1]), size=n_way, replace=False)
    # ------------ sample samples ------------
    support_idx = []
    query_idx = []
    label_to_idx_dict = \
        np.load(os.path.join(current_path, idx_name, f'{dataset_name}_' + phase + '_label_to_idx_dict.npy'),
                allow_pickle=True)[0]
    for c in sampled_class:
        idx = np.random.choice(label_to_idx_dict[c], size=np.minimum(n_shot + n_query, len(label_to_idx_dict[c])),
                               replace=False)
        support_idx.extend(idx[:n_shot])
        query_idx.extend(idx[n_shot:])
    support_idx = np.array(list(set(support_idx)))
    query_idx = np.array(list(set(query_idx) - set(support_idx)))
    # ------------ load images ------------
    image_transform = get_transform(transform=transform, image_size=image_size)
    x_support = torch.zeros([len(support_idx), 3, image_size, image_size])
    for i, s in enumerate(support_idx):
        image = Image.open(os.path.join(path_to_images_dict[dataset_name], images[s])).convert('RGB')
        x_support[i] = image_transform(image)
    x_query = torch.zeros([len(query_idx), 3, image_size, image_size])
    for i, q in enumerate(query_idx):
        image = Image.open(os.path.join(path_to_images_dict[dataset_name], images[q])).convert('RGB')
        x_query[i] = image_transform(image)
    # ------------ load labels ------------
    y_support = torch.from_numpy(labels[support_idx][:, sampled_class])
    y_query = torch.from_numpy(labels[query_idx][:, sampled_class])
    test_loader = (x_support, y_support, x_query, y_query)
    return test_loader


def get_transform(transform, image_size):
    if transform:
        return transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.3, 1.0)),
            # transforms.RandAugment(),   #服务器支持的torchvision版本不包含该属性
            transforms.ToTensor(),
            RandAugment(num_ops=2, magnitude=9),  # 添加 RandAugment
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    else:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )


from PIL import ImageOps, ImageEnhance, ImageFilter
import torch
import random
from torchvision import transforms


class RandAugment(object):
    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.operations = [
            self.autocontrast,
            self.equalize,
            self.flip,
            self.rotate,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
        ]

    def __call__(self, img):
        ops = random.sample(self.operations, self.num_ops)
        for op in ops:
            img = op(img)
        return img

    def autocontrast(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        img = ImageOps.autocontrast(img)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def equalize(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        img = ImageOps.equalize(img)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def flip(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def rotate(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        angle = random.randint(-self.magnitude, self.magnitude)
        img = img.rotate(angle)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def color(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        factor = random.uniform(1 - self.magnitude / 10, 1 + self.magnitude / 10)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def posterize(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        bits = random.randint(4, 8)  # Randomly choose bits value
        img = ImageOps.posterize(img, bits)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def contrast(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        factor = random.uniform(1 - self.magnitude / 10, 1 + self.magnitude / 10)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def brightness(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        factor = random.uniform(1 - self.magnitude / 10, 1 + self.magnitude / 10)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img

    def sharpness(self, img):
        img = Image.fromarray(img.mul(255).byte().cpu().numpy().transpose(1, 2, 0))  # From Tensor to PIL
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(1 - self.magnitude / 10, 1 + self.magnitude / 10)
        img = enhancer.enhance(factor)
        img = transforms.ToTensor()(img)  # Convert back to Tensor
        return img
