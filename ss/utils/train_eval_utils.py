import argparse
import logging
import random

import torch

import numpy as np
from torch.utils.data import DataLoader

from ss.data import dataset
from ss.utils import config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pfenet_resnet50.yaml', help='config file')
    parser.add_argument('--split', default=0, dest='split')
    parser.add_argument('--shot', default=1, dest='shot')
    parser.add_argument('--manual_seed', default=321, dest='manual_seed')
    parser.add_argument('--ss', action='store_true')
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.split = int(args.split)
    cfg.ss = args.ss
    cfg.shot = args.shot
    cfg.manual_seed = args.manual_seed
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_train_loader(args):
    train_data = dataset.SemData(
        split=args.split, shot=int(args.shot), mode='train',
        data_root=args.data_root, data_list=args.train_list,
        superpixel_type=args.superpixel_type
    )
    train_loader = DataLoader(
        train_data, batch_size=4, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    return train_loader


def get_val_loader(args):
    val_data = dataset.SemData(
            split=args.split, shot=int(args.shot), mode='val',
            data_root=args.data_root, data_list=args.val_list,
            superpixel_type=args.superpixel_type
        )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    return val_loader


def set_seed(manual_seed):
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)


def get_save_path(args):
    ss = "ss_" if args.ss else ""
    prior = "prior_" if args.prior else ""
    save_path = f"exp/{ss}{prior}{args.model}_{args.backbone}_{args.dataset}_split{args.split}"
    print(f"save path: {save_path}")
    return save_path
