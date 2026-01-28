# -*- coding: utf-8 -*-

from .BatchLoadImagesWithNames import BatchLoadImagesWithNames

NODE_CLASS_MAPPINGS = {
    "BatchLoadImagesWithNames": BatchLoadImagesWithNames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLoadImagesWithNames": "Batch Load Images With Names",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]