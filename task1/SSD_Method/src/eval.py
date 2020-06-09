import os

from pprint import PrettyPrinter
import numpy
import torch
from utils import *
from datasets import ICDARDataset
from tqdm import tqdm

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = '../ICDAR_Dataset/0325updated.task1train(626p)/'
keep_difficult = False
batch_size = 64
workers = 4
device = torch.device("cpu")
checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
test_img_num = 25

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = ICDARDataset(data_folder,
                                split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model, test_img_num):

    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():

        f1_max = 0
        ap_max = 0
        ar_max = 0

        for min_score in [0.1]:

            for max_overlap in [0.8]:

                # Batches
                f1 = 0
                ap = 0
                ar = 0
                images_num = 0

                for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):

                    if i < test_img_num:

                        images = images.to(device)  # (N, 3, 300, 300)
                        print(images)
                        print(images.size)

                        # Forward prop.
                        predicted_locs, predicted_scores = model(images)

                        # Detect objects in SSD output
                        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                                   min_score=min_score, max_overlap=max_overlap,
                                                                                                   top_k=200, original_image = images, max_OCR_overlap=0.2, max_OCR_pixel=245)
                        # Store this batch's results for mAP calculation
                        boxes = [b.to(device) for b in boxes]
                        labels = [l.to(device) for l in labels]

                        det_boxes.extend(det_boxes_batch)
                        det_labels.extend(det_labels_batch)
                        det_scores.extend(det_scores_batch)
                        true_boxes.extend(boxes)
                        true_labels.extend(labels)

                        f1_0, ap_0, ar_0 = calc_f1(det_boxes_batch[0], boxes[0], iou_thresh=0.5)
                        f1 += f1_0
                        ap += ap_0
                        ar += ar_0
                        images_num += 1

                if f1 / images_num > f1_max:
                    f1_max = f1/ images_num
                    f1_max_par = [min_score, max_overlap]
                print('f1 max:' ,f1_max)
                print('f1 max par:', f1_max_par )
                if ap / images_num > ap_max:
                    ap_max = ap/ images_num
                    ap_max_par = [min_score, max_overlap]
                print('ap max:' ,ap_max)
                print('ap max par:', ap_max_par )
                if ar / images_num > ar_max:
                    ar_max = ar/ images_num
                    ar_max_par = [min_score, max_overlap]
                print('ar max:' ,ar_max)
                print('ar max par:', ar_max_par )


if __name__ == '__main__':
    evaluate(test_loader, model, test_img_num)
