import random
from collections import deque

import numpy as np


BLACK = 0


def _components(grid):
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [x for x, _ in cells]
            cs = [y for _, y in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "top": min(rs),
                    "left": min(cs),
                    "height": max(rs) - min(rs) + 1,
                    "width": max(cs) - min(cs) + 1,
                    "area": len(cells),
                }
            )
    return comps


def _mask_for(comp, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - comp["top"], c - comp["left"]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary colored fragments and render each completed rectangle.

    Input structure:
    The background is black.  Every non-black object is a 4-connected single
    color component, and object colors are distinct.  The objects appear in an
    even count; each pair is made from two shapes that become complementary
    masks after each shape is aligned to the top-left corner of its own bounding
    box.

    Rule:
    Use the maximum object bounding-box height and width as the common target
    rectangle.  Normalize every object into that target-sized boolean mask.
    Match two objects when their masks are exact complements.  For every match,
    fill one target rectangle with each object's original color on the cells
    owned by that object's mask.  Order the completed rectangles by the original
    row, then column, of the smaller-area object in the pair, and place the
    rectangles in one compact row separated by one black column.

    Key insight:
    The useful signal is shape complementarity after bounding-box alignment, not
    the colors or the input positions of nearby fragments.
    """
    comps = _components(grid)
    if not comps:
        return np.zeros((0, 0), dtype=grid.dtype)

    height = max(comp["height"] for comp in comps)
    width = max(comp["width"] for comp in comps)
    objs = []
    for index, comp in enumerate(comps):
        item = dict(comp)
        item["index"] = index
        item["mask"] = _mask_for(comp, height, width)
        objs.append(item)

    unused = set(range(len(objs)))
    pairs = []
    while unused:
        i = min(unused)
        unused.remove(i)
        partner = None
        for j in sorted(unused):
            if np.array_equal(objs[i]["mask"], ~objs[j]["mask"]):
                partner = j
                break
        if partner is None:
            raise ValueError("unpaired fragment")
        unused.remove(partner)
        a, b = objs[i], objs[partner]
        small = min((a, b), key=lambda x: (x["area"], x["top"], x["left"], x["index"]))
        pairs.append(((small["top"], small["left"], small["index"]), a, b))

    pairs.sort(key=lambda x: x[0])
    out_width = len(pairs) * width + max(0, len(pairs) - 1)
    out = np.zeros((height, out_width), dtype=grid.dtype)
    for pos, (_, a, b) in enumerate(pairs):
        start = pos * (width + 1)
        block = out[:, start : start + width]
        block[a["mask"]] = a["color"]
        block[b["mask"]] = b["color"]
    return out


def _connected(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(coords[0])
    q = deque([start])
    seen = {start}
    rows, cols = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _usable_pair_mask(mask):
    other = ~mask
    return (
        0 < int(mask.sum()) < mask.size
        and mask[0].any()
        and mask[:, 0].any()
        and other[0].any()
        and other[:, 0].any()
        and _connected(mask)
        and _connected(other)
    )


def _random_pair_mask(height, width):
    cells = [(r, c) for r in range(height) for c in range(width)]
    total = height * width
    for _ in range(3000):
        target = random.randint(max(2, total // 4), min(total - 2, (3 * total) // 4))
        start = random.choice(cells)
        chosen = {start}
        frontier = set()
        r, c = start
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width:
                frontier.add((nr, nc))
        while frontier and len(chosen) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.add((nr, nc))
        if len(chosen) != target:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _usable_pair_mask(mask):
            return mask

    mask = np.zeros((height, width), dtype=bool)
    mask[0, : width - 1] = True
    mask[: height - 1, 0] = True
    return mask


def _normalized_key(mask):
    coords = np.argwhere(mask)
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)
    trimmed = np.zeros(mask.shape, dtype=bool)
    piece = mask[top : bottom + 1, left : right + 1]
    trimmed[: piece.shape[0], : piece.shape[1]] = piece
    return trimmed.tobytes()


def _find_place(grid, mask, margin=1):
    rows, cols = grid.shape
    coords = np.argwhere(mask)
    box_h = int(coords[:, 0].max()) + 1
    box_w = int(coords[:, 1].max()) + 1
    options = []
    for top in range(rows - box_h + 1):
        for left in range(cols - box_w + 1):
            r0 = max(0, top - margin)
            r1 = min(rows, top + box_h + margin)
            c0 = max(0, left - margin)
            c1 = min(cols, left + box_w + margin)
            if np.any(grid[r0:r1, c0:c1] != BLACK):
                continue
            options.append((top, left))
    if not options:
        return None
    return random.choice(options)


def generate_input() -> np.ndarray:
    height = random.randint(3, 6)
    width = random.randint(3, 6)
    max_pairs_by_output = (30 + 1) // (width + 1)
    pair_count = random.randint(2, min(4, max_pairs_by_output))

    used = set()
    pairs = []
    while len(pairs) < pair_count:
        mask = _random_pair_mask(height, width)
        if random.choice((False, True)):
            mask = mask.T
        if mask.shape != (height, width):
            mask = mask.T
        if random.choice((False, True)):
            mask = ~mask
        if not _usable_pair_mask(mask):
            continue
        key_a = _normalized_key(mask)
        key_b = _normalized_key(~mask)
        if key_a in used or key_b in used:
            continue
        used.add(key_a)
        used.add(key_b)
        pairs.append((mask, ~mask))

    total_objects = pair_count * 2
    colors = random.sample(range(1, 10), total_objects)
    slot_h = height + 2
    slot_w = width + 2
    layouts = []
    for slot_rows in range(1, 31):
        for slot_cols in range(1, 31):
            if (
                slot_rows * slot_cols >= total_objects
                and slot_rows * slot_h <= 30
                and slot_cols * slot_w <= 30
            ):
                layouts.append((slot_rows, slot_cols))
    slot_rows, slot_cols = random.choice(layouts)
    base_rows = slot_rows * slot_h
    base_cols = slot_cols * slot_w
    grid_rows = random.randint(max(12, base_rows), 30)
    grid_cols = random.randint(max(12, base_cols), 30)
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    row_offset = random.randint(0, grid_rows - base_rows)
    col_offset = random.randint(0, grid_cols - base_cols)

    objects = []
    color_index = 0
    for first, second in pairs:
        pair_masks = [first, second]
        if random.choice((False, True)):
            pair_masks.reverse()
        for mask in pair_masks:
            objects.append((mask, colors[color_index]))
            color_index += 1
    random.shuffle(objects)

    slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    random.shuffle(slots)
    for (mask, color), (slot_r, slot_c) in zip(objects, slots):
        top = row_offset + slot_r * slot_h + random.randint(0, 1)
        left = col_offset + slot_c * slot_w + random.randint(0, 1)
        for r, c in np.argwhere(mask):
            grid[top + int(r), left + int(c)] = color
    return grid
