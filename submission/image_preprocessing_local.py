"""Local image preprocessing for Task 1.

Features:
- Uses local `dataset/` folder (no HF/datasets or torch dependency)
- Resize to 224x224, grayscale, gaussian blur, normalize, simple augmentations
- Anomaly filtering: too dark and small object (contour area ratio)
- Exposes `run_with_params` for programmatic use and a CLI.
"""
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import shutil
import argparse

IMAGE_SIZE = (224, 224)

def is_too_dark(pil_img, threshold=35):
    arr = np.array(pil_img.convert('L'))
    return arr.mean() < threshold

def has_small_object(pil_img, min_ratio=0.03):
    cv_img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = pil_img.width * pil_img.height
    return (area / img_area) < min_ratio

def normalize_uint8(pil_img):
    arr = np.array(pil_img).astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return Image.fromarray(arr.astype(np.uint8))

def augment(pil_img):
    outs = []
    outs.append(('orig', pil_img.copy()))
    outs.append(('flip_h', ImageOps.mirror(pil_img)))
    outs.append(('rot-10', pil_img.rotate(-10, resample=Image.BILINEAR)))
    outs.append(('rot10', pil_img.rotate(10, resample=Image.BILINEAR)))
    # color jitter via enhance
    enh = ImageEnhance.Brightness(pil_img)
    outs.append(('bright+10', enh.enhance(1.1)))
    return outs

def process_single(img_path, out_dir, save_samples_dir=None, idx=0, params=None):
    pil = Image.open(img_path).convert('RGB')
    pil = pil.resize(IMAGE_SIZE, Image.LANCZOS)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=1))
    pil_gray = pil.convert('L')
    pil_norm = normalize_uint8(pil_gray)

    # anomaly filters
    if params and params.get('filter_dark') and is_too_dark(pil_norm, params.get('min_brightness',35)):
        return []
    if params and params.get('filter_small') and has_small_object(pil, params.get('min_obj_ratio',0.03)):
        return []

    aug = augment(pil_norm)
    saved = []
    stem = Path(img_path).stem
    for tag, img in aug:
        name = f"{stem}_{tag}.png"
        path = Path(out_dir) / name
        img.save(path)
        saved.append(path)

    # save a representative sample
    if save_samples_dir:
        sample_path = Path(save_samples_dir) / f"{stem}_sample.png"
        shutil.copy(saved[0], sample_path)

    return saved

def run_with_params(input_dir='dataset', output_dir='preprocessed', save_samples_dir='submission/samples',
                    filter_dark=True, min_brightness=35, filter_small=True, min_obj_ratio=0.03):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    save_samples_dir = Path(save_samples_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    if not all_images:
        print('No images found in', input_dir)
        return []

    params = {'filter_dark':filter_dark, 'min_brightness':min_brightness, 'filter_small':filter_small, 'min_obj_ratio':min_obj_ratio}
    saved_files = []
    for i,p in enumerate(all_images):
        res = process_single(p, output_dir, save_samples_dir, i, params)
        saved_files.extend(res)

    print(f"Processed {len(all_images)} inputs; saved {len(saved_files)} output images in {output_dir}")
    return saved_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='dataset')
    parser.add_argument('--output-dir', default='preprocessed')
    parser.add_argument('--save-samples', default='submission/samples')
    parser.add_argument('--no-filter-dark', dest='filter_dark', action='store_false')
    parser.add_argument('--no-filter-small', dest='filter_small', action='store_false')
    args = parser.parse_args()

    run_with_params(input_dir=args.input_dir, output_dir=args.output_dir, save_samples_dir=args.save_samples,
                    filter_dark=args.filter_dark, filter_small=args.filter_small)

if __name__ == '__main__':
    main()
