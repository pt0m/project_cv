from dataclasses import dataclass

mask_tracker = True

#META_ARC = "siamrpn_r50_l234_dwxcorr"
CUDA     = False

# +------------------------------------------------------------------------+ #
# |                            TRAIN CONFIG NODES                          | #
# +------------------------------------------------------------------------+ #

@dataclass
class LR: pass

@dataclass
class LR_WARMUP: pass

@dataclass
class TRAIN:
    LR        = LR()
    LR_WARMUP = LR_WARMUP()




# +------------------------------------------------------------------------+ #
# |                            DATA CONFIG NODES                           | #
# +------------------------------------------------------------------------+ #

@dataclass
class TEMPLATE: pass

@dataclass
class SEARCH: pass

@dataclass
class DATASET:
    TEMPLATE = TEMPLATE()
    SEARCH   = SEARCH()




# +------------------------------------------------------------------------+ #
# |                           MODEL CONFIG NODES                           | #
# +------------------------------------------------------------------------+ #

@dataclass
class BACKBONE: pass

@dataclass
class ADJUST: pass

@dataclass
class REFINE: pass

@dataclass
class MASK: pass

@dataclass
class RPN: pass

@dataclass
class ANCHOR: pass

@dataclass
class TRACK: pass



# +------------------------------------------------------------------------+ #
# |                              TRAINING CONFIG                           | #
# +------------------------------------------------------------------------+ #

# Anchor Target
# Positive anchor threshold
TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
TRAIN.THR_LOW = 0.3

# Number of negative
TRAIN.NEG_NUM = 16

# Number of positive
TRAIN.POS_NUM = 16

# Number of anchors per images
TRAIN.TOTAL_NUM = 64

TRAIN.EXEMPLAR_SIZE = 127

TRAIN.SEARCH_SIZE = 255

TRAIN.BASE_SIZE = 8

TRAIN.OUTPUT_SIZE = 25

TRAIN.RESUME = ''

TRAIN.PRETRAINED = '../weights/model.pth'
if(mask_tracker == True):
    TRAIN.PRETRAINED = '../weights/model_mask.pth'

TRAIN.LOG_DIR = './logs'

TRAIN.SNAPSHOT_DIR = './snapshot'

TRAIN.EPOCH = 20

TRAIN.START_EPOCH = 0

TRAIN.BATCH_SIZE = 2

TRAIN.NUM_WORKERS = 1

TRAIN.MOMENTUM = 0.9

TRAIN.WEIGHT_DECAY = 0.0001

TRAIN.CLS_WEIGHT = 1.0

TRAIN.LOC_WEIGHT = 1.2

TRAIN.MASK_WEIGHT = 1

TRAIN.PRINT_FREQ = 20

TRAIN.LOG_GRADS = False

TRAIN.GRAD_CLIP = 10.0

TRAIN.BASE_LR = 0.005

TRAIN.LR.BASE_LR = 0.005

TRAIN.LR.TYPE = 'log'

TRAIN.LR.KWARGS = {}

TRAIN.LR_WARMUP.WARMUP = True

TRAIN.LR_WARMUP.TYPE = 'step'

TRAIN.LR_WARMUP.EPOCH = 5

TRAIN.LR_WARMUP.KWARGS = {}


TRAIN_OUTPUT_SIZE = 240*480
# +------------------------------------------------------------------------+ #
# |                                DATA CONFIG                             | #
# +------------------------------------------------------------------------+ #

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
DATASET.TEMPLATE.SHIFT = 4

DATASET.TEMPLATE.SCALE = 0.05

DATASET.TEMPLATE.BLUR = 0.0

DATASET.TEMPLATE.FLIP = 0.0

DATASET.TEMPLATE.COLOR = 1.0

DATASET.SEARCH.SHIFT = 64

DATASET.SEARCH.SCALE = 0.18

DATASET.SEARCH.BLUR = 0.0

DATASET.SEARCH.FLIP = 0.0

DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
DATASET.NEG = 0.2

# improve tracking performance for otb100
DATASET.GRAY = 0.0


# +------------------------------------------------------------------------+ #
# |                              BACKBONE CONFIG                           | #
# +------------------------------------------------------------------------+ #

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
BACKBONE.TYPE = 'resnet50'

# Pretrained backbone weights
BACKBONE.PRETRAINED = '../weights/resnet50.model'

# Train layers
BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
BACKBONE.TRAIN_EPOCH = 10

BACKBONE.KWARGS = {'used_layers': [2, 3, 4]}
if(mask_tracker == True):
    BACKBONE.KWARGS = {'used_layers': [0, 1, 2, 3]}



# +------------------------------------------------------------------------+ #
# |                           ADJUST LAYER CONFIG                          | #
# +------------------------------------------------------------------------+ #

# Adjust layer
ADJUST.ADJUST = True

# Adjust layer type
ADJUST.TYPE = "AdjustAllLayer"

ADJUST.KWARGS = {'in_channels': [1024], 'out_channels': [256]}


# +------------------------------------------------------------------------+ #
# |                                RPN CONFIG                              | #
# +------------------------------------------------------------------------+ #

# RPN type
RPN.TYPE = 'DepthwiseRPN'

RPN.KWARGS = {'anchor_num': 5, 'in_channels': 256, 'out_channels': 256}



# +------------------------------------------------------------------------+ #
# |                                MASK CONFIG                             | #
# +------------------------------------------------------------------------+ #

# Whether to use mask generate segmentation
MASK.MASK = False
if(mask_tracker == True):
    MASK.MASK = True

# Mask type
MASK.TYPE = "MaskCorr"
if(mask_tracker == True):
    MASK.TYPE = "MaskCorr"

MASK.KWARGS = {}
if(mask_tracker == True):
    MASK.KWARGS = {'in_channels': 256, 'hidden': 256, 'out_channels': 3969}

# Mask refine
REFINE.REFINE = False
if(mask_tracker == True):
    REFINE.REFINE = True

# Refine type
REFINE.TYPE = "Refine"




# +------------------------------------------------------------------------+ #
# |                              TRAINING CONFIG                           | #
# +------------------------------------------------------------------------+ #

# Anchor stride
ANCHOR.STRIDE = 8

# Anchor ratios
ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
ANCHOR.SCALES = [8]

# Anchor number
ANCHOR.ANCHOR_NUM = 5




# +------------------------------------------------------------------------+ #
# |                               TRACKER CONFIG                           | #
# +------------------------------------------------------------------------+ #

TRACK.TYPE = 'SiamRPNTracker'
if(mask_tracker == True):
    TRACK.TYPE = 'SiamMaskTracker'

# Scale penalty
TRACK.PENALTY_K = 0.10

# Window influence
TRACK.WINDOW_INFLUENCE = 0.41

# Interpolation learning rate
TRACK.LR = 0.32

# Exemplar size
TRACK.EXEMPLAR_SIZE = 127

# Instance size
TRACK.INSTANCE_SIZE = 255

# Base size
TRACK.BASE_SIZE = 8

# Context amount
TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
TRACK.MASK_THERSHOLD = 0.15

# Mask output size
TRACK.MASK_OUTPUT_SIZE = 127
