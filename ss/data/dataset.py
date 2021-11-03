import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm
from skimage import segmentation

from ss.data import transform

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def get_sub_list(split, dataset):
    """
    Divide novel and base classes
    :param split: split, range from 0 to 3
    :param dataset: 'pascal' or 'coco'
    :return:
        sub_list: base classes
        sub_val_list: novel classes
    """

    if dataset == 'pascal':
        class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        novel_cls = list(range(5 * split + 1, 5 * split + 6))
        base_cls = list(set(class_list) - set(novel_cls))
    elif dataset == 'coco':
        class_list = list(range(1, 81))
        novel_cls = list(range(split + 1, split + 78, 4))
        base_cls = list(set(class_list) - set(novel_cls))
    else:
        raise ValueError(f'do not support {dataset} dataset')

    return base_cls, novel_cls


def make_dataset(data_root, data_list, sub_cls):
    """
    Generate query list and support dict
    :param data_root: dataroot/
    :param data_list: path to data_list.txt
    :param sub_cls: list of chosen classes
    :return:
        image_label_list:
        list of (image_name, label_name, sub_cls_with_min_area), used to sample query
        sub_class_file_list:
        dict[class] = list of (image_name, label_name), used_to_sample_support
    """
    if not os.path.isfile(data_list):
        raise RuntimeError(f"Image list file do not exist: {data_list}\n")

    list_read = open(data_list).readlines()
    print("Processing data...")

    # initialise sub_class_file_list
    query_list = []
    support_dict = {}
    for sub_c in sub_cls:
        support_dict[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')

        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        _, label, unique_class = get_img_label(label_path=label_name)

        # Shaban uses these lines to remove small objects:
        # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
        #    filtered_item.append(item)
        # which means the mask will be downsampled to 1/32 of the original size and
        # the valid area should be larger than 2,
        # therefore the area in original size should be accordingly larger than 2 * 32 * 32

        sub_cls_with_min_area = []
        for c in unique_class:
            # filter class with min area
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0], target_pix[1]] = 1
            if tmp_label.sum() >= 2 * 32 * 32 and c in sub_cls:
                sub_cls_with_min_area.append(c)
                support_dict[c].append(
                    (image_name, label_name)
                )

        if len(sub_cls_with_min_area) > 0:
            query_list.append(
                (image_name, label_name, sub_cls_with_min_area)
            )

    return query_list, support_dict


def get_img_label(image_path=None, label_path=None):
    """
    :param image_path: str / None
    :param label_path: str
    :return:
        image: np float array of shape (H, W, 3)
        label: np int array of (H, W)
        unique_class: int list, classes exists in the image
    """
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image_path is not None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
    else:
        image = None

    unique_class = list(np.unique(label))
    if 0 in unique_class:
        unique_class.remove(0)
    if 255 in unique_class:
        unique_class.remove(255)
    return image, label, unique_class


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def get_binary_label(label, class_chosen, base_cls, novel_cls):
    """
    Returns:
        label:
            1 represents the chosen class
            2 represents training classes other than the chosen class
            3 represents the validation class
            255 represents ignore labels
            0 represents the rest of pixels
        sem_label: each pixel labeled by its class index
    """
    sem_label = label.copy()
    label[:, :] = 0
    for cls in np.unique(sem_label):
        pix = np.where(sem_label == cls)
        if cls == class_chosen:
            if pix[0].shape[0] > 0:
                label[pix[0], pix[1]] = 1
        elif cls in base_cls:
            label[pix[0], pix[1]] = 2
        elif cls in novel_cls:
            label[pix[0], pix[1]] = 3
        elif cls in [0, 255]:
            label[pix[0], pix[1]] = cls
        else:
            print('encounter class label {} which is neither in train cls neither in val class')
    return label, sem_label


def get_superpixel(image_path, sem_label, base_cls, method, mode):
    """
    Generate superpixel label
    255 -> ignore label
    1 to n -> the n base classes in the image
    0 is reserved for novel class
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if method == 'slic':
        superpixel = segmentation.slic(image, compactness=10, n_segments=100)  # 250
    elif method == 'felzenszwalb':
        superpixel = segmentation.felzenszwalb(image, scale=100, sigma=0.8, min_size=200)
    elif method == 'hed':
        image_name = image_path.split('/')[-1].split('.')[0]
        superpixel = cv2.imread(f'./hed/{image_name}.png', cv2.IMREAD_GRAYSCALE)
        superpixel = np.asarray(superpixel)
    else:
        raise ValueError(f'Do not recognise superpixel method {method}')

    # reserve 0
    superpixel += 1

    if mode == 'train':
        # refine with base cls mask
        base_cls = [c for c in np.unique(sem_label) if c in base_cls]
        superpixel = superpixel + len(base_cls)
        for i, c in enumerate(base_cls):
            superpixel[sem_label == c] = i + 1
        # 255 for ignore label
        superpixel[superpixel == 255] = np.max(superpixel) + 1
        superpixel[sem_label == 255] = 255

    return superpixel


class SemData(Dataset):
    def __init__(self, split=3, shot=1, mode='train',
                 data_root=None, data_list=None, dataset='pascal',
                 superpixel_type='felzenszwalb'):
        assert mode in ['train', 'val']
        assert split in [0, 1, 2, 3]
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root
        self.dataset = dataset
        self.superpixel_type = superpixel_type

        self.base_cls, self.novel_cls = get_sub_list(split, dataset)
        print('base classes: ', self.base_cls)
        print('novel classes: ', self.novel_cls)
        if self.mode == 'train':
            self.sub_cls = self.base_cls
        else:
            self.sub_cls = self.novel_cls

        self.query_list, self.support_dict = make_dataset(data_root, data_list, self.sub_cls)
        assert len(self.support_dict.keys()) == len(self.sub_cls)

        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if self.mode == 'train':
            transform_list = [
                transform.RandScale([0.9, 1.1]),
                transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
                transform.RandomGaussianBlur(),
                transform.RandomHorizontalFlip(),
                transform.Crop(473, crop_type='rand', padding=mean, ignore_label=255)
            ]
        else:
            transform_list = [transform.Resize(size=473)]

        self.transform = transform.Compose(transform_list, mean=mean, std=std)

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, index):
        image_path, label_path, sub_cls_with_min_area = self.query_list[index]
        image, label, unique_class = get_img_label(image_path=image_path, label_path=label_path)

        # sample class
        class_chosen = sub_cls_with_min_area[
            random.randint(0, len(sub_cls_with_min_area) - 1)
        ]

        # sample support
        support_list = self.support_dict[class_chosen]
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(0, len(support_list) - 1)
            support_image_path, support_label_path = support_list[support_idx]
            while (support_image_path == image_path) or (support_idx in support_idx_list):
                support_idx = random.randint(0, len(support_list) - 1)
                support_image_path, support_label_path = support_list[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        # get query
        label, sem_label = get_binary_label(label=label,
                                            class_chosen=class_chosen,
                                            base_cls=self.base_cls,
                                            novel_cls=self.novel_cls)
        superpixel = get_superpixel(image_path=image_path,
                                    sem_label=sem_label,
                                    base_cls=self.base_cls,
                                    method=self.superpixel_type,
                                    mode=self.mode)
        # get support
        support_image_list = []
        support_label_list = []
        for k in range(self.shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image, support_label, _ = get_img_label(support_image_path, support_label_path)
            support_label, _ = get_binary_label(label=support_label,
                                                class_chosen=class_chosen,
                                                base_cls=self.base_cls,
                                                novel_cls=self.novel_cls)
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot                    

        # transform
        raw_label = label.copy()
        image, label, [superpixel], image_original = self.transform(image, label, [superpixel])
        for k in range(self.shot):
            support_image_list[k], support_label_list[k], _, _ = self.transform(
                support_image_list[k], support_label_list[k], None)

        # concatenate supports shot-wise
        s_x = torch.stack(support_image_list, dim=0)
        s_y = torch.stack(support_label_list, dim=0)

        data_dict = dict(
            q_x=image,
            q_y=label,
            s_x=s_x,
            s_y=s_y,
            novel_cls=class_chosen,
            superpixel=superpixel,
            num_cls=len(unique_class),
            image_original=image_original,
            name=label_path.split('/')[-1].split(',')[0]
        )

        if self.mode != 'train':
            data_dict['raw_label'] = raw_label

        return data_dict
