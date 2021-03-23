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
from src.pysot.tracker.tracker_builder import build_tracker
from src.pysot.utils.bbox import get_axis_aligned_bbox
from src.pysot.utils.model_load import load_pretrain
#from toolkit.datasets import DatasetFactory
#from toolkit.utils.region import vot_overlap, vot_float2str
import cv2

visualization = True

def main():
    #iinstaciate model
    model = ModelBuilder().train().float()
    rank, world_size = dist_init()
    #train_loader = build_our_data_loader()
    #dist_model   = DistModule(model)
    tb_writer    = SummaryWriter(cfg.TRAIN.LOG_DIR)
    #model        = torch.nn.DataParallel(model)


    #load parameters from the training
    #model.load_state_dict(torch.load("model_save.pth"))
    model.eval()

    #load de test dataset
    test_dataset = load_test_dataset()

    #run the test
    test(model,test_dataset)


def load_test_dataset(name ="bag",dir = "./datasets/sequences-train/",nb_images = 25):
    """
    format of the test dataset:
    """
    test_dataset = dataset_loader(name, dir)
    #test_dataset = dataset_loader()
    video = []
    for i in range(nb_images):
        image = test_dataset.__getitem2__(i)
        video.append((image["frame"], image["boundbox"]))
    dataset = [video]
    return dataset

def test(model, dataset):
    # build tracker
    tracker = build_tracker(model)

    #model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    video_name = "video test"
    # OPE tracking
    for v_idx, video in enumerate(dataset):
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if visualization and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video_name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, "TEST", toc, idx / toc))


if __name__ == '__main__':
    main()
