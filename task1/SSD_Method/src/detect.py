import os
import argparse

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="")
parser.add_argument('--name', type=str, default="")

opt = parser.parse_args()
savename = opt.name.split(".")[0]

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load model checkpoint
checkpoint ='C:\\Users\\lenovo\\Desktop\\Class\\Network\\Project\\task1\\SSD_Method\\src\\BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()


def detect(original_image, min_score, max_overlap, top_k, max_OCR_overlap=1.0, max_OCR_ratio=1.0, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                        max_overlap=max_overlap, top_k=top_k, original_image=original_image, max_OCR_overlap=max_OCR_overlap, max_OCR_ratio=max_OCR_ratio)


    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("C:\\Windows\\Fonts\\Sitka.ttc", 15)


    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()

        with open('C:\\Users\\lenovo\\Desktop\\Class\\Network\\Project\\task1\\SSD_Method\\src\\result\\' + savename + '.txt', 'a+') as f:
            f.write(str(int(box_location[0])) + ',' + str(int(box_location[1])) + ',')
            f.write(str(int(box_location[0])) + ',' + str(int(box_location[3])) + ',')
            f.write(str(int(box_location[2])) + ',' + str(int(box_location[1])) + ',')
            f.write(str(int(box_location[2])) + ',' + str(int(box_location[3])) + '\n')
            #f.write(str(box_location)+'\n')

        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])
    del draw
    print(annotated_image)
    return annotated_image


if __name__ == '__main__':
    min_score = 0.1
    max_overlap = 0.9
    max_OCR_overlap = 0.2
    max_OCR_ratio = 1
    top_k = 300

    img_path = opt.path
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    out_image = detect(original_image, min_score=min_score, max_overlap=max_overlap, top_k=top_k, max_OCR_overlap=max_OCR_overlap, max_OCR_ratio=max_OCR_ratio)  # .show()
    img_save_path = 'C:\\Users\\lenovo\\Desktop\\Class\\Network\\Project\\task1\\SSD_Method\\src\\result\\' + savename + ".jpg"
    out_image.save(img_save_path)

