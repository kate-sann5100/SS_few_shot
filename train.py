import os
import random

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from eval import validate
from ss.model.PFENet import PFENet
from ss.utils.data_collection import AverageMeter
from ss.utils.pfenet_poly_learn_rate import poly_learning_rate
from ss.utils.train_eval_utils import get_parser, get_logger, get_train_loader, get_val_loader, set_seed, get_save_path


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


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

        # get trainable layers and freeze other layers
        trainable_layers = model.comparison_layers
        freeze_layers = model.backbone_layers

        for l in freeze_layers:
            for param in l.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f'do not support model {args.model}')

    # optimiser
    optim_params = [{'params': l.parameters()} for l in trainable_layers]
    optimizer = torch.optim.SGD(
        optim_params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # initialise logger and writer
    global logger, writer
    logger = get_logger()
    save_path = get_save_path(args)
    writer = SummaryWriter(save_path)
    logger.info("=> creating model ...")

    model = torch.nn.DataParallel(model.cuda())

    resume_ckpt = f'{save_path}/last_epoch.pth'
    if os.path.isfile(resume_ckpt):
        logger.info("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        logger.info("=> no checkpoint found at '{}'".format(save_path))
        start_epoch = 0

    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    max_iou = 0.

    for epoch in range(start_epoch, args.epochs):
        set_seed(args.manual_seed + epoch)

        # train
        train_result = train(train_loader, val_loader, model, optimizer, epoch, max_iou, save_path)
        for k, v in train_result.items():
            writer.add_scalar(k + '_train', v, epoch)
        last_ckpt_filename = save_path + '/last_epoch.pth'
        logger.info('Saving checkpoint to: ' + last_ckpt_filename)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f'{save_path}/last_epoch.pth')


def train(train_loader, val_loader, model, optimizer, epoch, max_iou, save_path):

    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    self_supervised_loss_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    max_iter = args.epochs * len(train_loader)
    for i, batch in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        if current_iter % 1000 == 0:
            with torch.no_grad():
                _, class_df = validate(val_loader, model, args,logger)
                class_result_dict = class_df.loc[:, 'mean'].to_dict()
                for k, v in class_result_dict.items():
                    writer.add_scalar(k + '_val', v, current_iter)
                model.train()

                # overwrite best checkpoint
                new_iou = class_result_dict['iou']
                if new_iou > max_iou:
                    max_iou = new_iou
                    filename = save_path + '/train_epoch_' + str(epoch) + '_' + str(new_iou) + '.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename
                    )
        if args.model == 'pfenet':
            poly_learning_rate(
                optimizer, args.base_lr, current_iter, max_iter,
            )
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        out_dict = model(batch)

        # set other training and val classes to background
        target = batch['q_y']
        target[target == 2] = 0
        target[target == 3] = 0

        main_loss = torch.mean(out_dict['main_loss'])
        aux_loss = torch.mean(out_dict['aux_loss'])
        self_supervised_loss = torch.mean(out_dict['self_supervised_loss'])
        loss = main_loss + aux_loss + self_supervised_loss * args.alpha
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        main_loss_meter.update(main_loss.item())
        aux_loss_meter.update(aux_loss.item())
        self_supervised_loss_meter.update(self_supervised_loss.item())
        loss_meter.update(loss.item())

        # log iou
        writer.add_scalar('main_loss_train_batch', main_loss_meter.val, current_iter)
        writer.add_scalar('self_supervised_loss_batch', self_supervised_loss_meter.val, current_iter)

        if (i + 1) % 100 == 0:
            logger.info(f'Epoch: [{epoch + 1}/{args.epochs}][{i + 1}/{len(train_loader)}] ')
            logger.info(f'Loss {loss_meter.val:.4f}')
            logger.info(f'MainLoss {main_loss_meter.val:.4f} ')
            logger.info(f'AuxLoss {aux_loss_meter.val:.4f} ')
            logger.info(f'SelfSupervisedLoss {self_supervised_loss_meter.val:.4f}')
    logger.info('Train result at epoch [{}/{}]: '.format(epoch, args.epochs))
    result = {
        'main_loss': main_loss_meter.avg,
        'self_supervised_loss': self_supervised_loss_meter.avg
    }

    return result


if __name__ == '__main__':
    main()
