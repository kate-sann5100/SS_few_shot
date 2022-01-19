import pickle

import torch
from torch.backends import cudnn

from ss.model.PFENet import PFENet
from ss.utils.data_collection import IoULogger
from ss.utils.train_eval_utils import get_parser, get_logger, get_val_loader, set_seed, get_save_path


def main():
    args = get_parser()

    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)

    main_worker(args)


def main_worker(argss):
    global args
    args = argss

    if args.model == 'pfenet':
        model = PFENet(backbone=args.backbone, self_supervision=args.ss, prior=args.prior)
    else:
        raise ValueError(f'do not support model {args.model}')

    # initialise logger and writer
    global logger
    logger = get_logger()
    save_path = get_save_path(args)
    logger.info("=> creating model ...")

    model = torch.nn.DataParallel(model.cuda())

    ckpt = f'{save_path}/best_ckpt.pth'
    logger.info("=> loading checkpoint '{}'".format(ckpt))
    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt['state_dict'])

    val_loader = get_val_loader(args)

    df, _ = validate(val_loader, model, args, logger)
    with open(f'{save_path}/{args.shot}shot.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate(val_loader, model, args, logger):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    iou_logger = IoULogger(cls_list=val_loader.dataset.novel_cls)

    set_seed(args.manual_seed)

    model.eval()
    if args.dataset == 'coco':
        test_num = 4000
    else:
        test_num = 1000
    iter_num = 0
    for e in range(10):
        for i, batch in enumerate(val_loader):
            if iter_num - 1 >= test_num:
                break
            iter_num += 1
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            out_dict = model(batch)

            raw_label = batch['raw_label']
            longerside = max(raw_label.size(1), raw_label.size(2))
            backmask = torch.ones(raw_label.size(0), longerside, longerside, device=raw_label.device) * 255
            backmask[0, :raw_label.size(1), :raw_label.size(2)] = raw_label
            target = backmask.clone().long()

            # get binary results
            target[target == 2] = 0
            target[target == 3] = 0
            output = out_dict['out']  # (B, 2, H, W)

            iou_logger.record(
                predict=output, target=target, novel_cls=batch['novel_cls'], num_cls=batch['num_cls']
            )
            if (iter_num + 1) % (test_num / 10) == 0:
                logger.info(f'Test: [{iter_num}/{test_num}]')

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    df, class_df = iou_logger.get_df()
    return df, class_df


if __name__ == '__main__':
    main()
