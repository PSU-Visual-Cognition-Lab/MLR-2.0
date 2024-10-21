# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import namedtuple

import numpy as np
from PIL import Image, ImageDraw


def bounding_box(z_where, x_size):
    """This doesn't take into account interpolation, but it's close
    enough to be usable."""
    w = x_size / z_where.s
    h = x_size / z_where.s
    xtrans = -z_where.x / z_where.s * x_size / 2.0
    ytrans = -z_where.y / z_where.s * x_size / 2.0
    x = (x_size - w) / 2 + xtrans  # origin is top left
    y = (x_size - h) / 2 + ytrans
    return (x, y), w, h


def arr2img(arr):
    # arr is expected to be a 3d array with shape (3, 64, 64) and floats in [0, 1]
    
    # Ensure the array is in the expected shape
    assert arr.shape[0] == 3 and arr.shape[1] == 64 and arr.shape[2] == 64, "Array shape must be (3, 64, 64)"
    
    # Convert the array to uint8
    arr = (arr * 255).astype(np.uint8)
    
    # Split the array into three separate channels
    r = Image.frombuffer("L", (64, 64), arr[0].tostring(), "raw", "L", 0, 1)
    g = Image.frombuffer("L", (64, 64), arr[1].tostring(), "raw", "L", 0, 1)
    b = Image.frombuffer("L", (64, 64), arr[2].tostring(), "raw", "L", 0, 1)
    
    # Merge the channels back into a single image
    img = Image.merge("RGB", (r, g, b))
    
    return img


def img2arr(img):
    # Ensure the image is in RGB mode
    img = img.convert("RGB")
    
    # Convert image to NumPy array
    arr = np.array(img, dtype=np.uint8)
    
    # Transpose the array to have the shape (3, 64, 64)
    arr = arr.transpose((2, 0, 1))
    
    # Normalize the array values to [0, 1]
    arr = arr / 255.0
    
    return arr



def colors(k):
    return [(255, 0, 0), (0, 255, 0), (0, 0, 255)][k % 3]


def draw_one(imgarr, z_arr):
    # Note that this clipping makes the visualisation somewhat
    # misleading, as it incorrectly suggests objects occlude one
    # another.
    clipped = np.clip(imgarr.detach().cpu().numpy(), 0, 1)
    img = arr2img(clipped).convert("RGB")
    draw = ImageDraw.Draw(img)
    for k, z in enumerate(z_arr):
        # It would be better to use z_pres to change the opacity of
        # the bounding boxes, but I couldn't make that work with PIL.
        # Instead this darkens the color, and skips boxes altogether
        # when z_pres==0.
        if z.pres > 0:
            (x, y), w, h = bounding_box(z, imgarr.size(0))
            color = tuple(map(lambda c: int(c * z.pres), colors(k)))
            draw.rectangle([x, y, x + w, y + h], outline=color)
    is_relaxed = any(z.pres != math.floor(z.pres) for z in z_arr)
    fmtstr = "{:.1f}" if is_relaxed else "{:.0f}"
    draw.text((0, 0), fmtstr.format(sum(z.pres for z in z_arr)), fill="white")
    return img2arr(img)


def draw_many(imgarrs, z_arr):
    # canvases is expected to be a (n,w,h) numpy array
    # z_where_arr is expected to be a list of length n
    return [draw_one(imgarr, z) for (imgarr, z) in zip(imgarrs.cpu(), z_arr)]


z_obj = namedtuple("z", "s,x,y,pres")


# Map a tensor of latents (as produced by latents_to_tensor) to a list
# of z_obj named tuples.
def tensor_to_objs(latents):
    return [[z_obj._make(step) for step in z] for z in latents]