from src.test.test_sample_input import sample_input_array, sample_input_tensor
from src.test.test_sample_train import build_opt_lr, sample_train
from src.pysot.models.model_builder import ModelBuilder
import src.pysot.core.config as cfg


model = ModelBuilder().train().double()
optimizer, scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)

sample_train(sample_input_tensor, model, optimizer, scheduler)

