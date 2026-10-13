import random
from collections import deque

import numpy as np


BG = 8
PALETTE = [c for c in range(10) if c != BG]
ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _components(grid: np.ndarray):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ORTHO4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            crop = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
            for rr, cc in cells:
                crop[rr - r0, cc - c0] = color
            out.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "crop": crop,
                }
            )
    return out


def _blank_corner_count(crop: np.ndarray) -> int:
    return sum(
        int(crop[r, c] == BG)
        for r, c in ((0, 0), (0, crop.shape[1] - 1), (crop.shape[0] - 1, 0), (crop.shape[0] - 1, crop.shape[1] - 1))
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _components(g)
    assert objects

    items = []
    for obj in objects:
        crop = obj["crop"]
        k = _blank_corner_count(crop)
        r0, c0, _r1, _c1 = obj["bbox"]
        items.append((k, r0, c0, crop))

    items.sort(key=lambda item: (-item[0], item[1], item[2]))

    out_h = max(crop.shape[0] for _k, _r0, _c0, crop in items)
    out_w = sum(crop.shape[1] for _k, _r0, _c0, crop in items) + len(items) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for _k, _r0, _c0, crop in items:
        h, w = crop.shape
        out[out_h - h : out_h, c : c + w] = crop
        c += w + 1
    return out


def _rotate_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, turns)


def _shape_k0() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    return np.ones((h, w), dtype=bool)


def _shape_k1() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    mask = np.ones((h, w), dtype=bool)
    mask[0, 0] = False
    return _rotate_mask(mask, random.randrange(4))


def _shape_k2() -> np.ndarray:
    if random.random() < 0.5:
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, w // 2] = True
        return _rotate_mask(mask, random.randrange(4))

    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, : max(2, w - 1)] = True
    mask[:, max(1, w // 2)] = True
    mask[h - 1, max(1, w // 2) :] = True
    return _rotate_mask(mask, random.randrange(4))


def _shape_k3() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    anchor_col = w - 2
    mask[0, : w - 1] = True
    mask[:, anchor_col] = True
    return _rotate_mask(mask, random.randrange(4))


def _shape_k4() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    if random.random() < 0.5 and h >= 5 and w >= 5:
        mask[1 : h - 1, w // 2] = True
        mask[h // 2, 1 : w - 1] = True
    return mask


def _make_shape(k: int) -> np.ndarray:
    makers = {
        0: _shape_k0,
        1: _shape_k1,
        2: _shape_k2,
        3: _shape_k3,
        4: _shape_k4,
    }
    mask = makers[k]()
    assert _blank_corner_count(np.where(mask, 1, BG)) == k
    return mask


def _placeable(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    for r in range(top - 1, top + h + 1):
        for c in range(left - 1, left + w + 1):
            if 0 <= r < gh and 0 <= c < gw and int(grid[r, c]) != BG:
                mr, mc = r - top, c - left
                if 0 <= mr < h and 0 <= mc < w:
                    if not mask[mr, mc]:
                        return False
                else:
                    return False
    return True


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _k_sequence(count: int) -> list[int]:
    base = [0, 1, 2, 3, 4]
    random.shuffle(base)
    seq = []
    while len(seq) < count:
        seq.extend(base)
        random.shuffle(base)
    seq = seq[:count]
    if len(set(seq)) == 1:
        seq[-1] = (seq[-1] + 1) % 5
    return seq


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.randint(3, 5)
    ks = _k_sequence(obj_count)
    masks = [_make_shape(k) for k in ks]

    max_h = max(mask.shape[0] for mask in masks)
    total_w = sum(mask.shape[1] for mask in masks) + 2 * (obj_count - 1)
    grid_h = random.randint(max_h + 4, max_h + 12)
    grid_w = random.randint(total_w + 4, total_w + 12)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    colors = random.sample(PALETTE, obj_count)
    order = list(range(obj_count))
    random.shuffle(order)

    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        placed = False
        for _ in range(80):
            top = random.randint(1, grid_h - h - 1)
            left = random.randint(1, grid_w - w - 1)
            if _placeable(grid, top, left, mask):
                _paint_mask(grid, top, left, mask, colors[idx])
                placed = True
                break
        if not placed:
            return None

    comps = _components(grid)
    if len(comps) != obj_count:
        return None
    found_ks = sorted(_blank_corner_count(comp["crop"]) for comp in comps)
    if found_ks != sorted(ks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
