import argparse
import math
import time
import os
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages

from utils.general import check_img_size, non_max_suppression, check_requirements


def inference():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    # device = select_device(opt.device)

    ## Just use CPU for inference
    device=torch.device('cpu')
    ##

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Data Loader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply Non Max Suppression
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)

        try:
            x_coord = ((pred[0][0, 0] + pred[0][0, 2]) / 2) / img.size()[3]
            y_coord = ((pred[0][0, 1] + pred[0][0, 3]) / 2) / img.size()[2]
        except IndexError: # No detections above confidence threshold
            x_coord = 0.0
            y_coord = 0.0

        if opt.test:
            # Append normalized phone coordinates to results
            results.append(f"{path.split('/')[-1]} {x_coord} {y_coord}")
        else:
            # Print normalized phone coordinates
            print(f"{x_coord:0.4f} {y_coord:0.4f}")


def test():
    assert os.path.isfile(opt.labels_path), "labels file does not exist"
    print(f"Testing directory {opt.source}\n")

    file = open(opt.labels_path)
    labels = file.readlines()
    labels.sort(key=sort_labels)  # Sort by image id
    file.close()
    inference()

    results.sort(key=sort_labels)  # Sort by image id

    # Calculate accuracy
    assert len(results) == len(labels), "Number of labels and images does not match"
    epsilon = 0.05
    accuracy = 0
    for i in range(len(results)):
        p = results[i].split(' ')
        l = labels[i].split(' ')

        print(f"{p}\n")
        print(f"{l}\n")

        assert int(p[0].split('.')[0]) == int(l[0].split('.')[0]), "Labels are in different order than results"

        if math.sqrt(((float(p[1]) - float(l[1])) ** 2) + ((float(p[2]) - float(l[2])) ** 2)) < epsilon:
            accuracy += 1
    accuracy /= len(results)

    print(f"Directory: {opt.source} \n"
          f"Number of images: {len(results)} \n"
          f"Error Threshold: {epsilon} normalized radius\n"
          f"Accuracy: {accuracy}")


def sort_labels(line):
    line_fields = line.strip().split()
    image_id = int(line_fields[0].split('.')[0])
    return image_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, default='data/images', help='image path or directory with various images')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing with labeled dataset')
    parser.add_argument('--labels-path', type=str, default='find_phone_task_4/find_phone/labels.txt', help='path to labels file')

    # Changed default settings to achieve phone finding task
    parser.add_argument('--img-size', type=int, default=448, help='inference size (pixels)')
    # Set low confidence threshold since we expect a single phone per image with floor background
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    # Only identify cell phone class in bounding boxes
    parser.add_argument('--classes', nargs='+', type=int, default=67, help='filter by class: --class 67 is cell phone class')

    # Leave default settings
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()

    check_requirements(exclude=('pycocotools', 'thop'))
    results = []

    with torch.no_grad():
        if opt.test:  # Test algorithm with labeled data set
            test()
        else:
            inference()
