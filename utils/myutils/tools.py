from pathlib import Path


def make_output_dirs(save_dir: Path, save_txt: bool, save_img: bool) -> None:
    (save_dir / 'img' if save_img else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels/xywh' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels/xyxy' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    return


