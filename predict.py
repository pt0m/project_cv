import cv2
import numpy as np
from tqdm import tqdm
import src.pysot.core.config as cfg
from src.pysot.models.model_builder import ModelBuilder
from src.pysot.tracker.tracker_builder import build_tracker
from src.pysot.utils.bbox import get_axis_aligned_bbox
from src.pysot.utils.model_load import load_pretrain
from src.pysot.utils.bbox import IoU
from test import load_test_dataset


PRETRAINED = True
ROOTDIR    = "../sequences-train/"
NAME       = 'bag'


def load_dataset(rootdir, name):
    lengths = {'bag': 196, 'bear': 26, 'book': 51, 'camel': 90, 'rhino': 90, 'swan': 50}
    return load_test_dataset(name=name, dir=rootdir, nb_images=lengths[name])


def load_model(pretrained):
    model = ModelBuilder().train().float()
    if(pretrained):
        # load pretrained backbone weights
        if cfg.BACKBONE.PRETRAINED:
            print("Loading Backbone Weights")
            load_pretrain(model.backbone, cfg.BACKBONE.PRETRAINED)
        if cfg.TRAIN.PRETRAINED:
            print("Loadind Full Model Weights")
            load_pretrain(model, cfg.TRAIN.PRETRAINED)
    return model.eval()


def dice_score(outputs, targets, ratio=0.5):
    outputs = outputs.flatten()
    targets = targets.flatten()
    outputs[outputs > ratio] = np.float32(1)
    outputs[outputs < ratio] = np.float32(0)    
    return float(2 * (targets * outputs).sum())/float(targets.sum() + outputs.sum())


def predict(model, dataset):
    tracker = build_tracker(model)
    video = dataset[0]
    toc = 0
    pred_bboxes = []
    ious = []
    scores = []
    track_times = []  
    for idx, (img, gt_bbox) in tqdm(enumerate(video), total=len(video)):
        tic = cv2.getTickCount()
        if idx == 0:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            scores.append(0)
            pred_bboxes.append(pred_bbox)
        else:
            outputs = tracker.track(img)
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            scores.append(outputs['best_score'])
            ious.append(IoU(pred_bbox, gt_bbox))
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
    toc /= cv2.getTickFrequency()
    speed = idx / toc
    return pred_bboxes, scores, ious, speed


def main():
    model, dataset = load_model(PRETRAINED), load_dataset(ROOTDIR, NAME)
    pred_bboxes, scores, ious, speed = predict(model, dataset)




if __name__=='__main__': main()



