from src.test.test_sample_input import sample_input_array, sample_input_tensor
from src.test.test_sample_train import build_opt_lr, sample_train
from src.pysot.models.model_builder import ModelBuilder
import src.pysot.core.config as cfg
from src.pysot.datasets.data_loader import dataset_loader
from skimage import io

im1 = io.imread("bag-024.bmp")
print(im1.shape)
model = ModelBuilder().train().double()
optimizer, scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)

#sample_train(sample_input_tensor, model, optimizer, scheduler)


data_loader = dataset_loader("bear", "./datasets/sequences-train/")

for i in range(10):
    print("train with sample n°", i )
    dict = data_loader.get_random_dict_tensor()
    sample_train(dict, model, optimizer, scheduler)
