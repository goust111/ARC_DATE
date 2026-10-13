import random
from collections import defaultdict, deque

import numpy as np


def _components(grid):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not seen[nr, nc]
                        and grid[nr, nc] != 0
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rr = [p[0] for p in cells]
            cc = [p[1] for p in cells]
            comps.append(
                {
                    "cells": cells,
                    "color": color,
                    "top": min(rr),
                    "left": min(cc),
                    "height": max(rr) - min(rr) + 1,
                    "width": max(cc) - min(cc) + 1,
                    "count": len(cells),
                }
            )
    return comps


def _mask_for(comp):
    mask = np.zeros((comp["height"], comp["width"]), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - comp["top"], c - comp["left"]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Merge each smaller source shape into its same-size larger slot shape.

    Input structure:
    The grid has a black background and an even number of small non-black
    4-connected objects.  Each bounding-box size appears on exactly two
    objects; those two objects have different colors and slightly different
    filled-cell patterns inside the same height and width.

    Rule:
    Pair objects by equal bounding-box dimensions.  In each pair, the object
    with fewer pixels is the source and the object with more pixels is the
    slot.  Align their masks by the top-left corner of their bounding boxes.
    Keep all slot pixels in the slot's original color, and fill only source
    pixels whose aligned slot position is empty using the source color.  Remove
    the source object from the output.

    Key insight:
    The transformation depends on three separate steps: pair by bounding-box
    size, infer source versus slot from pixel count, then overlay only the
    source-only cells into the slot's holes.
    """
    out = np.zeros_like(grid)
    groups = defaultdict(list)
    for comp in _components(grid):
        groups[(comp["height"], comp["width"])].append(comp)

    for pair in groups.values():
        if len(pair) != 2:
            continue
        a, b = pair
        if a["count"] < b["count"]:
            source, slot = a, b
        else:
            source, slot = b, a

        source_mask = _mask_for(source)
        slot_mask = _mask_for(slot)
        sr, sc = slot["top"], slot["left"]
        out[sr : sr + slot["height"], sc : sc + slot["width"]][slot_mask] = slot[
            "color"
        ]
        fill_mask = source_mask & ~slot_mask
        out[sr : sr + slot["height"], sc : sc + slot["width"]][fill_mask] = source[
            "color"
        ]
    return out


def _base_mask(height, width, corner):
    mask = np.zeros((height, width), dtype=bool)
    if corner == 0:
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == 1:
        mask[0, :] = True
        mask[:, width - 1] = True
    elif corner == 2:
        mask[height - 1, :] = True
        mask[:, 0] = True
    else:
        mask[height - 1, :] = True
        mask[:, width - 1] = True
    return mask


def _adjacent_to(mask, blocked):
    height, width = mask.shape
    cells = []
    for r in range(height):
        for c in range(width):
            if mask[r, c] or blocked[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _grow(mask, blocked, count):
    added = []
    for _ in range(count):
        cells = _adjacent_to(mask, blocked)
        if not cells:
            break
        r, c = random.choice(cells)
        mask[r, c] = True
        blocked[r, c] = True
        added.append((r, c))
    return added


def _make_pair_masks(height, width):
    common = _base_mask(height, width, random.randrange(4))
    source = common.copy()
    slot = common.copy()
    blocked = common.copy()

    source_extra = random.randint(1, 2)
    slot_extra = source_extra + random.randint(1, 3)
    _grow(source, blocked, source_extra)
    _grow(slot, blocked, slot_extra)

    if slot.sum() <= source.sum() or not np.any(source & ~slot):
        return _make_pair_masks(height, width)
    return source, slot


def _can_place(occupied, top, left, height, width):
    rows, cols = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(rows, top + height + 1)
    c1 = min(cols, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _mark_place(occupied, top, left, height, width):
    rows, cols = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(rows, top + height + 1)
    c1 = min(cols, left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _place_object(grid, occupied, mask, color):
    height, width = mask.shape
    rows, cols = grid.shape
    for _ in range(300):
        top = random.randint(0, rows - height)
        left = random.randint(0, cols - width)
        if _can_place(occupied, top, left, height, width):
            patch = grid[top : top + height, left : left + width]
            patch[mask] = color
            _mark_place(occupied, top, left, height, width)
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(200):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        pair_count = random.randint(2, 4)
        all_dims = [(h, w) for h in range(4, 8) for w in range(4, 8)]
        random.shuffle(all_dims)
        dims = all_dims[:pair_count]
        colors = list(range(1, 10))
        grid = np.zeros((rows, cols), dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        ok = True

        for height, width in dims:
            source_mask, slot_mask = _make_pair_masks(height, width)
            source_color, slot_color = random.sample(colors, 2)
            objects = [(source_mask, source_color), (slot_mask, slot_color)]
            random.shuffle(objects)
            for mask, color in objects:
                if not _place_object(grid, occupied, mask, color):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return grid
    return generate_input()
