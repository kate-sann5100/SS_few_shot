import torch
from torch.nn import functional as F
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.val = val

    def average(self):
        self.avg = self.sum / self.count


class IoULogger(object):

    def __init__(self, cls_list):
        self.dict_list = []
        self.cls_list = cls_list

    @staticmethod
    def record_single_image(output, target, K=2, ignore_index=255):
        """
        :param output: (2, H, W)
        :param target: (H, W)
        :param K: int, number of classes
        :param ignore_index: int
        :return:
            area_intersection: int
            area_target: int
            area_union: int
        """
        output = F.interpolate(output.unsqueeze(0), target.shape, mode='bilinear', align_corners=True)[0]  # (2, H, W)
        output = torch.argmax(output, dim=0)  # (H, W)
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
        area_output = torch.histc(output, bins=K, min=0, max=K - 1)
        area_target = torch.histc(target, bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target

    def record(
            self,
            predict,
            target,
            novel_cls,
            num_cls
    ):
        """
        predict: (B, 2, H, W)
        target: (B, H, W)
        novel_cls: (B)
        num_cls: (B)
        """
        novel_cls = novel_cls.cpu().numpy()
        num_cls = num_cls.cpu().numpy()

        for p, t in zip(predict, target):
            intersection, union, target_area = self.record_single_image(p, t)
            target_area_percentage = target_area[1] / target.shape[-1] / target.shape[-2]
            intersection = intersection.cpu().numpy()
            union = union.cpu().numpy()
            target_area_percentage = target_area_percentage.cpu().numpy()
            self.dict_list.append(
                {
                    "novel_cls": novel_cls,
                    "num_cls": num_cls,
                    "bg_intersection": intersection[0],
                    "fg_intersection": intersection[1],
                    "bg_union": union[0],
                    "fg_union": union[1],
                    "target_area_percentage": target_area_percentage,
                }
            )

    def df2class_df(self, df):
        class_df = {}
        max_num_classes = 0
        for cls in self.cls_list:
            cls_dict = {}
            cls_df = df.loc[df['novel_cls'] == cls]
            cls_dict['iou'] = cls_df.sum(axis=0)['fg_intersection'] / cls_df.sum(axis=0)['fg_union']
            for num_cls in range(1, 20):
                temp_df = cls_df.loc[cls_df['num_cls'] == num_cls]
                if len(temp_df) == 0:
                    continue
                max_num_classes = max(max_num_classes, num_cls)
                cls_dict[f'{num_cls}_classes'] = temp_df.sum(axis=0)['fg_intersection'] / temp_df.sum(axis=0)[
                    'fg_union']
            class_df[cls] = cls_dict
        class_df = pd.DataFrame(class_df)
        class_df.loc[:, 'mean'] = class_df.mean(axis=1)
        return class_df

    def get_df(self):
        df = pd.DataFrame(self.dict_list)
        df = pd.DataFrame(df)
        class_df = self.df2class_df(df)
        print(class_df)
        return df, class_df
