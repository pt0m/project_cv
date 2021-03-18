from src.test.test_sample_input import sample_input_array, sample_input_tensor
from src.test.test_sample_train import build_opt_lr, sample_train
from src.pysot.models.model_builder import ModelBuilder
import src.pysot.core.config as cfg
from src.pysot.datasets.data_loader import dataset_loader
from skimage import io
from src.pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from src.tools.train import *


model = ModelBuilder().train().double()
optimizer, scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)

#sample_train(sample_input_tensor, model, optimizer, scheduler)


our_dataset = dataset_loader("bear", "./datasets/sequences-train/")

#for i in range(10):
#    print("train with sample n°", i )
#    dict = data_loader.get_random_dict_tensor()
#    sample_train(dict, model, optimizer, scheduler)


def build_our_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = our_dataset
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader

rank, world_size = dist_init()
train_loader = build_our_data_loader()
dist_model = DistModule(model)

train(train_loader, model, optimizer, scheduler, None)
