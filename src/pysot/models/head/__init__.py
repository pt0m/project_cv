from src.pysot.models.head.mask import MaskCorr, Refine
from src.pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
