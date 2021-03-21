import math

import torch.nn as nn
from torch.optim import SGD

from src.pysot.utils.lr_scheduler import build_lr_scheduler
import src.pysot.core.config as cfg


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    optimizer = SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler


def sample_train(sample_input, model, optimizer, lr_scheduler):

    print("| Starting Sample Train: Forward + Backward pass on sample data.")
    
    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)
    
    cur_lr = lr_scheduler.get_cur_lr()
    print(f"| Start LR : {cur_lr}")
    
    print("| Starting Forward Propagation.")
    outputs = model(sample_input)
    loss = outputs['total_loss']

    if is_valid_number(loss.data.item()):
        print(f"| Starting Backward Propagation | Current Loss: {loss.data.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print("Sample Training over. No error encountered !")

    

    

