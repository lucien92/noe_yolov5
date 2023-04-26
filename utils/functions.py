import numpy as np
import torch
import os

import supervisely_lib as sly
import time
from utils.general import xywh2xyxy
import torchvision
from models.experimental import attempt_load
import onnxruntime as rt


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def removeduplicate(it):
    seen = []
    for x in it:
        if x not in seen:
            yield x
            seen.append(x)


def construct_model_meta(model, CONFIDENCE):
    # names = names
    names = model.module.names if hasattr(model, 'module') else model.names

    colors = None
    if hasattr(model, 'module') and hasattr(model.module, 'colors'):
        colors = model.module.colors
    elif hasattr(model, 'colors'):
        colors = model.colors
    else:
        colors = []
        for i in range(len(names)):
            colors.append(sly.color.generate_rgb(exist_colors=colors))

    obj_classes = [sly.ObjClass(name, sly.Rectangle, color) for name, color in zip(names, colors)]
    tags = [sly.TagMeta(CONFIDENCE, sly.TagValueType.ANY_NUMBER)]

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                           tag_metas=sly.TagMetaCollection(tags))
    return meta


def non_max_suppression_sly(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False,
                            labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.as_tensor(prediction)
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def onnx_inference(path_to_onnx_saved_model):
    onnx_model = rt.InferenceSession(path_to_onnx_saved_model)
    input_name = onnx_model.get_inputs()[0].name
    label_name = onnx_model.get_outputs()[0].name
    return onnx_model, input_name, label_name


def sliding_window_approach(model_, image, **kwargs):
    conf_threshold = kwargs['conf_threshold'] if 'conf_threshold' in kwargs else 0.25
    iou_threshold = kwargs['iou_threshold'] if 'iou_threshold' in kwargs else 0.45
    agnostic = kwargs['agnostic'] if 'agnostic' in kwargs else False
    native = kwargs['native'] if 'native' in kwargs else True
    sliding_window_step = kwargs['sliding_window_step'] if 'sliding_window_step' in kwargs else 320  # None
    input_iamge_size = kwargs['input_iamge_size'] if 'input_iamge_size' in kwargs else 640  # None

    if isinstance(model_, tuple):
        onnx_model, input_name, label_name = model_
    img_h, img_w = image.shape[-2:]
    try:
        sw_h, sw_w = model_.img_size
    except:
        assert input_iamge_size is not None, 'For torchScript and ONNX models input image size should be passed!'
        sw_h, sw_w = input_iamge_size

    if sliding_window_step:
        # print(sliding_window_step)
        sws_h, sws_w = sliding_window_step
    else:
        sws_h = (img_h - sw_h + 1) // 4
        sws_w = (img_w - sw_w + 1) // 4

    possible_height_steps = (img_h - sw_h + 1) // sws_h
    possible_width_steps = (img_w - sw_w + 1) // sws_w

    candidates = []

    for w in range(possible_width_steps + 1):
        for h in range(possible_height_steps + 1):
            top = h * sws_h
            left = w * sws_w
            bot = top + sw_h
            right = left + sw_w
            cropped_image = image[..., top:bot, left:right].unsqueeze(0) / 255
            if not isinstance(model_, tuple):
                inf_res = model_(cropped_image)[0]
            else:
                inf_res = onnx_model.run([label_name], {input_name: to_numpy(cropped_image).astype(np.float32)})[0]

            inf_res = inf_res[inf_res[..., 4] > conf_threshold]
            inf_res[:, 0] += left
            inf_res[:, 1] += top
            if native:
                inf_res = inf_res if len(inf_res.shape) == 3 else np.expand_dims(inf_res, axis=0)
                inf_res = non_max_suppression_sly(inf_res,
                                                  conf_thres=conf_threshold,
                                                  iou_thres=iou_threshold,
                                                  agnostic=agnostic)[0]
            candidates.append(inf_res)

    if isinstance(candidates[0], np.ndarray):
        candidates = [torch.as_tensor(element) for element in candidates]  # if not isinstance(element, torch.Tensor)
    detections = torch.cat(candidates).unsqueeze_(0)

    if not native:
        detections = non_max_suppression_sly(detections, conf_thres=conf_threshold, iou_thres=iou_threshold,
                                             agnostic=agnostic)
    return detections


def prepare_model(weights_path, **kwargs):
    path_to_saved_model = weights_path
    if 'pt' in weights_path:
        if 'torchscript' in weights_path:
            model = torch.jit.load(path_to_saved_model)
            return model
        else:
            model = attempt_load(weights=path_to_saved_model)  # , map_location=device
            kwargs['model'] = model
            return model, kwargs
    if 'onnx' in weights_path:
        model = onnx_inference(path_to_saved_model)
        kwargs['model'] = model
        return model, kwargs


def infer_torch_model(torch_script_model, tensor):
    # simple inference for torchScript:
    torch_script_model_inference = torch_script_model(tensor)[0]
    return torch_script_model_inference


def infer_onnx_model(onnx_model, tensor):
    # simple inference for ONNX:
    onnx_model, input_name, label_name = onnx_model
    onnx_model_inference = onnx_model.run([label_name], {input_name: to_numpy(tensor).astype(np.float32)})[0]
    return onnx_model_inference
