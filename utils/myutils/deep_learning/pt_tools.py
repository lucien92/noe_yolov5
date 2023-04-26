from pathlib import Path

import torch

from models.experimental import attempt_load


def load_pt_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    return model, stride, names


def load_classifier(device, classifier_path: str = 'resnet50.pt'):
    modelc = load_classifier(name=Path(classifier_path).stem, n=2)  # initialize
    modelc.load_state_dict(torch.load(classifier_path, map_location=device)['model']).to(device).eval()
    return modelc