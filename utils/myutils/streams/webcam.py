from utils.datasets import LoadStreams


def load_webcam_dataset(source, image_size, stride, auto):
    dataset = LoadStreams(source, img_size=image_size, stride=stride, auto=auto)
    return dataset
