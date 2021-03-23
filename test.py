from src.test.test_sample_input import sample_input_array, sample_input_tensor
from src.test.test_sample_train import build_opt_lr, sample_train
from src.pysot.models.model_builder import ModelBuilder
import src.pysot.core.config as cfg
from src.pysot.datasets.data_loader import dataset_loader
from skimage import io
from src.pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from src.tools.train import *
from tensorboardX import SummaryWriter

def main():
    #iinstaciate model
    model = ModelBuilder().train().double()
    rank, world_size = dist_init()
    train_loader = build_our_data_loader()
    dist_model   = DistModule(model)
    tb_writer    = SummaryWriter(cfg.TRAIN.LOG_DIR)
    model        =torch.nn.DataParallel(model)

    #load parameters from the training
    model.load_state_dict(torch.load("model_save.pth"))

    model.eval()

def load_test_dataset():
    """
    format of the test dataset:

    """
def test():
