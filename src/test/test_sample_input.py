import numpy as np
import torch
import src.pysot.core.config as cfg


template_array  = np.ndarray((cfg.TRAIN.BATCH_SIZE, 3, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE), dtype=np.int64) 
search_array    = np.ndarray((cfg.TRAIN.BATCH_SIZE, 3, cfg.TRAIN.SEARCH_SIZE,   cfg.TRAIN.SEARCH_SIZE),   dtype=np.int64)
bbox            = x1, y1, x2, y2 = 50, 100, 50, 100
cls             = -1 * np.ones((cfg.ANCHOR.ANCHOR_NUM, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
delta           = np.zeros(( 4, cfg.ANCHOR.ANCHOR_NUM, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.float32)
delta_weight    = np.zeros(    (cfg.ANCHOR.ANCHOR_NUM, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.float32)


sample_input_array = {
    'template': template_array,
    'search': search_array,
    'label_cls': cls,
    'label_loc': delta,
    'label_loc_weight': delta_weight,
    'bbox': np.array(bbox)
}

sample_input_tensor = {
    'template': torch.as_tensor(template_array, dtype=torch.double),
    'search': torch.as_tensor(search_array, dtype=torch.double),
    'label_cls': torch.from_numpy(cls),
    'label_loc': torch.from_numpy(delta),
    'label_loc_weight': torch.from_numpy(delta_weight),
    'bbox': torch.from_numpy(np.array(bbox))
}

