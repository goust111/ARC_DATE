from __future__ import annotations

import random

import numpy as np
from arc_utils import crop_bbox, find_components, paste


BG = 0
GREY = 5

ARROWS = {
    "up": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    "down": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)},
    "left": {(0, 1), (1, 0), (1, 1), (2, 1), (1, 2)},
    "right": {(0, 1), (1, 1), (2, 1), (1, 2), (1, 3)},
}


def _norm(coords: np.ndarray) -> set[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return {(int(r - r0), int(c - c0)) for r, c in coords}


def _arrow_dir(comp: dict) -> str:
    shape = _norm(comp["coords"])
    # generator uses only these four canonical arrow masks
    if shape == {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}:
        return "up"
    if shape == {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)}:
        return "down"
    if shape == {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}:
        return "left"
    return "right"


def _side(arrow_bbox: tuple[int, int, int, int], obj_bbox: tuple[int, int, int, int]) -> str:
    ar0, ac0, ar1, ac1 = arrow_bbox
    or0, oc0, or1, oc1 = obj_bbox
    if ac0 > oc1:
        return "right"
    if ac1 < oc0:
        return "left"
    if ar0 > or1:
        return "down"
    return "up"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored object so its current arrow-side facing becomes the arrow's pointed direction.

    Input structure:
    - Black background with monochrome irregular objects and separate grey arrows.
    - Each arrow sits one cell away from its nearest object on one of the four sides.
    - Arrow direction is one of up, down, left, or right.

    Rule:
    - Pair every arrow with its nearest colored object.
    - Treat the arrow's position relative to that object as the object's current facing side.
    - Rotate the object in 90-degree steps so that current facing becomes the arrow's pointed direction.
    - Place the rotated object back at the original bbox center and erase the arrow.

    Key insight:
    - The arrow gives two independent signals: its location identifies the current facing, while its glyph orientation identifies the target facing.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    arrows = [c for c in find_components(g, bg=BG, colors=[GREY], connectivity=4, monochromatic=True) if int(c["area"]) >= 5]
    objects = [c for c in find_components(g, bg=BG, connectivity=4, monochromatic=True) if int(c["color"]) != GREY]
    order = ["up", "right", "down", "left"]
    for obj in objects:
        patch = crop_bbox(g, obj["bbox"])
        best = min(arrows, key=lambda a: abs(_center(a["bbox"])[0] - _center(obj["bbox"])[0]) + abs(_center(a["bbox"])[1] - _center(obj["bbox"])[1]))
        current = _side(best["bbox"], obj["bbox"])
        target = _arrow_dir(best)
        steps = (order.index(target) - order.index(current)) % 4
        rotated = np.rot90(patch, -steps)
        cr, cc = _center(obj["bbox"])
        r0 = int(round(cr - (rotated.shape[0] - 1) / 2))
        c0 = int(round(cc - (rotated.shape[1] - 1) / 2))
        paste(out, rotated, r0, c0, transparent=BG)
    return out


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _make_patch(size: int, color: int) -> np.ndarray:
    p = np.zeros((size, size), dtype=int)
    p[:, 0] = color
    p[-1, :] = color
    p[size // 2, 1:] = color
    return p


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(1, 3))
    occupied = np.zeros_like(g, dtype=bool)
    arrow_masks = {
        "up": np.array([[0, GREY, 0], [GREY, GREY, GREY], [0, GREY, 0]], dtype=int),
        "down": np.array([[0, GREY, 0], [GREY, GREY, GREY], [0, GREY, 0], [0, GREY, 0]], dtype=int),
        "left": np.array([[0, GREY, 0], [GREY, GREY, GREY], [0, GREY, 0]], dtype=int),
        "right": np.array([[0, GREY, 0, 0], [0, GREY, GREY, GREY], [0, GREY, 0, 0]], dtype=int),
    }
    for color in colors:
        for _ in range(100):
            size = random.choice([3, 5])
            patch = _make_patch(size, color)
            r0 = random.randint(2, h - size - 3)
            c0 = random.randint(2, w - size - 3)
            side = random.choice(["up", "down", "left", "right"])
            target = random.choice(["up", "down", "left", "right"])
            am = arrow_masks[target]
            if side == "up":
                ar, ac = r0 - am.shape[0] - 1, c0 + size // 2 - am.shape[1] // 2
            elif side == "down":
                ar, ac = r0 + size + 1, c0 + size // 2 - am.shape[1] // 2
            elif side == "left":
                ar, ac = r0 + size // 2 - am.shape[0] // 2, c0 - am.shape[1] - 1
            else:
                ar, ac = r0 + size // 2 - am.shape[0] // 2, c0 + size + 1
            if ar < 0 or ac < 0 or ar + am.shape[0] > h or ac + am.shape[1] > w:
                continue
            view = occupied[r0 : r0 + size, c0 : c0 + size]
            if np.any(view & (patch != 0)):
                continue
            av = occupied[ar : ar + am.shape[0], ac : ac + am.shape[1]]
            if np.any(av & (am != 0)):
                continue
            g[r0 : r0 + size, c0 : c0 + size][patch != 0] = color
            g[ar : ar + am.shape[0], ac : ac + am.shape[1]][am != 0] = GREY
            occupied[r0 : r0 + size, c0 : c0 + size] |= patch != 0
            occupied[ar : ar + am.shape[0], ac : ac + am.shape[1]] |= am != 0
            break
    return g
