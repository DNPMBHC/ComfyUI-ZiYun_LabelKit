# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from PIL import ImageOps, Image
try:
    import pillow_jxl  # noqa: F401
    jxl = True
except ImportError:
    jxl = False
import comfy
import numpy as np
import logging
import re

# 内联 utils（无需 libs/ 目录）
def extract_first_number(s):
    match = re.search(r'\d+', str(s))
    return int(match.group()) if match else float('inf')

sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)"
]

def sort_by(items, base_path='.', method=None):
    def fullpath(x): return os.path.join(base_path, x)

    def get_timestamp(path):
        try:
            return os.path.getmtime(path)
        except FileNotFoundError:
            return float('-inf')

    if method == "Alphabetical (ASC)":
        return sorted(items)
    elif method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    elif method == "Numerical (ASC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    else:
        return items

def _basename_no_ext(path_or_name: str) -> str:
    """返回文件名（不含扩展名）。"""
    try:
        name = os.path.basename(str(path_or_name))
        base = os.path.splitext(name)[0]
        return base
    except Exception:
        return str(path_or_name)

class BatchLoadImagesWithNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT")
    RETURN_NAMES = ("IMAGES", "MASKS", "FILE PATHS", "FILE NAMES", "COUNT")
    OUTPUT_IS_LIST = (True, True, True, True, False)

    FUNCTION = "load_images"
    CATEGORY = "ZiYun/工具箱"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get('load_always'):
            return float("NaN")
        return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None):
        if not os.path.isdir(directory):
            logging.warning(f"Directory '{directory}' not found. Returning empty outputs.")
            return self._empty_output()

        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            logging.warning(f"No files in '{directory}'. Returning empty outputs.")
            return self._empty_output()

        # Filter by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if jxl:
            valid_extensions.append('.jxl')
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        if not dir_files:
            logging.warning(f"No valid image files in '{directory}'. Returning empty outputs.")
            return self._empty_output()

        # Sort and slice
        dir_files = sort_by(dir_files, directory, sort_method)
        dir_files = [os.path.join(directory, x) for x in dir_files][start_index:]

        images = []
        masks = []
        file_paths = []
        file_names = []
        image_count = 0
        limit_images = image_load_cap > 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                rgb = i.convert("RGB")
                h, w = rgb.size[1], rgb.size[0]
                image = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).unsqueeze(0)
                if 'A' in i.getbands():
                    mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask_np)
                else:
                    mask = torch.zeros((h, w), dtype=torch.float32)
                images.append(image)
                masks.append(mask)
                file_paths.append(str(image_path))
                file_names.append(_basename_no_ext(image_path))
                image_count += 1
            except Exception as e:
                logging.warning(f"Failed to load '{image_path}': {e}. Skipping.")
                continue

        if image_count == 0:
            return self._empty_output()

        return (images, masks, file_paths, file_names, image_count)

    def _empty_output(self):
        return ([], [], [], [], 0)
