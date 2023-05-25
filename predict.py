import argparse
import sys
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import time

from pathlib import Path
from crop.crop_from_csv import cropp

FILE = Path('__file__').absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.functions import to_numpy
from utils.myutils.tools import make_output_dirs
from utils.myutils.deep_learning.pt_tools import load_pt_model
from utils.myutils.deep_learning.onnx_tools import load_onnx_session
from utils.myutils.streams.webcam import load_webcam_dataset


###FUNCTION FOR LEPINOC DETECTION INFERENCE
@torch.no_grad()
def run(weights='models/2families.pt',  # model.pt path(s)
        source='data/',  # file/dir/URL/glob, 0 for webcam
        img_size=640,  # inference size (pixels)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.50,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='prediction_results',  # save results to project/name
        name='exp',  # save results to project/name
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        use_half_precision=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    make_output_dirs(save_dir=save_dir, save_txt=save_txt, save_img=save_img)

    # Initialize
    set_logging()
    device = select_device(device)
    use_half_precision &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    weights_path = weights[0] if isinstance(weights, list) else weights
    suffix = Path(weights_path).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    assert pt or onnx, "Not supported deep learning framework"

    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model, stride, names = load_pt_model(weights, device)
        if use_half_precision:
            model.half()  # to FP16
    else:  # onnx:
        check_requirements(('onnx', 'onnxruntime'))
        session = load_onnx_session(weights_path)

    img_size = check_img_size(img_size, s=stride)  # check image size # set in the arguments
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = load_webcam_dataset(source=source, image_size=img_size, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=img_size, stride=stride, auto=pt)
        bs = 1  # batch_size
        print("Number of images for inference:", len(dataset))
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for img_path, img, im0s, vid_cap in dataset:
        print("STARTING INFERENCE")

        # Choose model and image precision according to specifications
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if use_half_precision else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(img_path).stem, mkdir=True) if visualize else False
            prediction = model(img, augment=augment, visualize=visualize)[0]
            # predicction is a tensor of shape (batch_size, number_of_candidates, 5 + n_classes)
            # where the last entry of size 8 corresponds to: x0, x1, y0, y1, confidence, cl0, cl1, ... clN-1
        else:  # onnx:
            prediction = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # non max suppression
        prediction = non_max_suppression(prediction=prediction, conf_thres=conf_thres, iou_thres=iou_thres,
                                         classes=classes,
                                         agnostic=agnostic_nms, max_det=max_det, multi_label=False)
        # outputs (n,K) tensor per image [xyxy, conf, cls]
        print("Number of detections", prediction[0].shape[0])

        t2 = time_sync()

        # Process predictions
        for i, detections in enumerate(prediction):  # detections per image
            if webcam:  # batch_size >= 1
                p, output_string, im0, frame = img_path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, output_string, im0, frame = img_path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'img' / p.name)  # img.jpg
            txt_xyxy_path = str(save_dir / 'labels/xyxy' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_xywh_path = str(save_dir / 'labels/xywh' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            output_string += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            # Write results
            xyxy_list = []
            if len(detections):
                # Rescale boxes from img_size to im0 size
                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0.shape).round()

                # Print results
                for semantic_class in detections[:, -1].unique():
                    n_detections = (detections[:, -1] == semantic_class).sum()  # detections per class
                    output_string += f"{n_detections} {names[int(semantic_class)]}{'s' * (n_detections > 1)}, "  # add to string

                for *xyxy, confidence, object_class in reversed(detections):
                    x1, y1, x2, y2 = [to_numpy(val) for val in xyxy]
                    bbox_coord = [x1, y1, x2, y2]
                    xyxy_list.append([bbox_coord, to_numpy(confidence), to_numpy(object_class)])

                    #print("on sauvegarde le texte?", save_txt)
                    if save_txt:  # Write to file
                        #print("On sauvegarde les annotations trouvées")
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        semantic_class = int(to_numpy(object_class))
                        line = (semantic_class, *xywh, confidence) if save_conf else (semantic_class, *xywh)

                        with open(txt_xywh_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        print("On sauvegarde les annotations trouvées à ce chemin")
                        with open(txt_xyxy_path + '.txt', 'a') as f:
                            f.write(
                                str(line[0]) + ' ' + str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(int(x2)) + ' ' + str(
                                    int(y2)) + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        semantic_class = int(object_class)  # integer class
                        if hide_labels:
                            label = None
                        else:
                            if hide_conf:
                                label = names[semantic_class]
                            else:
                                label = f'{names[semantic_class]} {confidence:.2f}'

                        annotator.box_label(xyxy, label, color=colors(semantic_class, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[semantic_class] / f'{p.stem}.jpg',
                                         BGR=True)  # originalmente .jpg a modificar en general.py save_one_box

            # Print time (inference + NMS)
            print(f'{output_string}Done. ({t2 - t1:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if len(detections):
                        cv2.imwrite(save_path, im0)
                    else:
                        continue
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            weights_path = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, weights_path, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                        (weights_path, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        output_string = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{output_string}")

    if save_txt:
        detections_summary = summarize_predictions(results_dir=save_dir, names=names)
        print("Global detections' summary")
        print(detections_summary)
        detections_summary.to_csv(os.path.join(save_dir, 'detections_summary.csv'), index=False)

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
    return xyxy_list, save_dir, save_crop, source



def image_detections_summary(file_path, names):
    file = Path(file_path).stem

    df = pd.read_csv(file_path, sep=" ", header=None)
    df.columns = ["class", "x", "y", "w", "h"]
    frequencies = df["class"].value_counts()
    frequencies = frequencies.to_frame()
    frequencies.reset_index(inplace=True)
    frequencies.columns = ['class', file]

    def get_name(row):
        return names[row['class']]

    frequencies["image"] = frequencies.apply(func=lambda row: names[row['class']], axis=1)
    img_detections = frequencies[['image', file]].set_index('image').T
    return img_detections


def summarize_predictions(results_dir: str, names):
    detections_dir = os.path.join(results_dir, 'labels/xywh')
    global_detections = []
    for file in os.listdir(detections_dir):
        file_path = os.path.join(detections_dir, file)
        image_detections = image_detections_summary(file_path, names)
        global_detections.append(image_detections)
    global_summary = pd.concat(global_detections).fillna(0).astype(int)
    global_summary["image"] = global_summary.index
    return global_summary[["image"] + names]


def parse_opt(): #tous les arguments à rentrer
    print("parsing arguments for Lepinoc detection")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/Workspace/Repos/b00786574@essec.edu/noe_yolov5//models/lepidoptera.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/lucien/projet_lepinoc/data/test_detect', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--img_size', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold, default 0.25')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold, default 0.5')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image, default 1000')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default = True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='prediction_results', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--use_half_precision', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))  # function from utils
    boxlist, save_dir, save_crop, source = run(**vars(opt))
    cropp()

    #Pour passer à la partie classification avec un resnet exécuter une fonciton après celle-ci
    