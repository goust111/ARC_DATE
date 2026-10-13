import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_instruction_run(grid):
    h, w = grid.shape
    best = None
    for r in range(h - 1, -1, -1):
        c = 0
        while c < w:
            if grid[r, c] in (BLACK, GREY):
                c += 1
                continue
            start = c
            vals = []
            while c < w and grid[r, c] not in (BLACK, GREY):
                vals.append(int(grid[r, c]))
                c += 1
            if len(vals) >= 2 and len(set(vals)) == len(vals):
                candidate = (r, start, vals)
                if best is None or r > best[0] or (r == best[0] and len(vals) > len(best[2])):
                    best = candidate
        if best is not None and best[0] == r:
            break
    if best is None:
        raise ValueError("instruction sequence not found")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Stack same-shaped colored objects at the grey anchor in instruction-line order.

    Input structure:
    The grid has black background, one bottom horizontal instruction segment made of
    unique non-black, non-grey colors, one grey single-pixel anchor, and one small
    object in each instruction color. The colored objects share the same shape.

    Rule:
    Read the instruction segment from left to right. Remove the instruction segment,
    the grey anchor, and the original scattered objects. For each instruction color,
    translate that color's object without rotating or recoloring it. The first
    object is placed so its bounding-box bottom center aligns with the anchor; each
    later object is placed directly above the previous object's bounding box with no
    empty row between the two boxes.

    Key insight:
    The object colors alone do not determine the output order; the bottom color
    segment is a visible program whose left-to-right order controls the stack.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    seq_r, seq_c, order = _find_instruction_run(grid)

    anchor_positions = np.argwhere(grid == GREY)
    if len(anchor_positions) != 1:
        raise ValueError("expected exactly one grey anchor")
    anchor_r, anchor_c = map(int, anchor_positions[0])

    seq_mask = np.zeros(grid.shape, dtype=bool)
    seq_mask[seq_r, seq_c:seq_c + len(order)] = True

    next_bottom = anchor_r
    for color in order:
        cells = np.argwhere((grid == color) & (~seq_mask))
        if len(cells) == 0:
            raise ValueError("missing object for instruction color")

        min_r = int(cells[:, 0].min())
        min_c = int(cells[:, 1].min())
        norm = cells - np.array([min_r, min_c])
        bbox_bottom = int(norm[:, 0].max())
        bbox_right = int(norm[:, 1].max())
        center_c = bbox_right // 2

        top_r = next_bottom - bbox_bottom
        left_c = anchor_c - center_c
        for rr, cc in norm:
            tr = top_r + int(rr)
            tc = left_c + int(cc)
            if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                out[tr, tc] = color
        next_bottom = top_r - 1

    return out


def _make_shape():
    kind = random.choice(["rect", "t", "u", "cross", "step"])
    if kind == "rect":
        height = random.randint(2, 4)
        width = random.choice([3, 5])
        return [(r, c) for r in range(height) for c in range(width)]

    if kind == "t":
        width = random.choice([3, 5])
        height = random.randint(3, 4)
        center = width // 2
        return [(0, c) for c in range(width)] + [(r, center) for r in range(1, height)]

    if kind == "u":
        width = random.choice([3, 5])
        height = random.randint(3, 4)
        cells = [(height - 1, c) for c in range(width)]
        for r in range(height - 1):
            cells.append((r, 0))
            cells.append((r, width - 1))
        return cells

    if kind == "cross":
        arm = random.choice([1, 2])
        size = arm * 2 + 1
        center = arm
        return [(center, c) for c in range(size)] + [(r, center) for r in range(size)]

    height = random.randint(3, 5)
    width = random.choice([3, 5])
    cells = []
    for r in range(height):
        limit = 1 + min(r, width - 1)
        for c in range(limit):
            cells.append((r, c))
    return cells


def _normalize_shape(cells):
    arr = np.array(cells, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return sorted((int(r), int(c)) for r, c in arr)


def _transform_shape(cells):
    arr = np.array(cells, dtype=int)
    if random.random() < 0.5:
        arr[:, 1] = arr[:, 1].max() - arr[:, 1]
    if random.random() < 0.5:
        arr[:, 0] = arr[:, 0].max() - arr[:, 0]
    return _normalize_shape(arr)


def _can_place(shape, top, left, h, w, forbidden):
    for r, c in shape:
        rr = top + r
        cc = left + c
        if rr < 0 or rr >= h - 1 or cc < 0 or cc >= w:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in forbidden:
                    return False
    return True


def _mark_forbidden(shape, top, left, forbidden):
    for r, c in shape:
        rr = top + r
        cc = left + c
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                forbidden.add((nr, nc))


def generate_input() -> np.ndarray:
    for _ in range(500):
        count = random.randint(3, 6)
        colors = random.sample(COLORS, count)
        shape = _transform_shape(_make_shape())
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        center_c = (shape_w - 1) // 2

        min_h = count * shape_h + 3
        if min_h > 30:
            continue
        height = random.randint(min_h, 30)
        width = random.randint(max(10, shape_w + count + 3), 30)

        anchor_r_min = count * shape_h - 1
        anchor_r_max = height - 3
        if anchor_r_min > anchor_r_max:
            continue
        anchor_r = random.randint(anchor_r_min, anchor_r_max)
        anchor_c = random.randint(center_c, width - (shape_w - center_c))

        seq_start = random.randint(0, width - count)
        grid = np.zeros((height, width), dtype=int)
        grid[-1, seq_start:seq_start + count] = colors
        grid[anchor_r, anchor_c] = GREY

        forbidden = set()
        for c in range(seq_start, seq_start + count):
            forbidden.add((height - 1, c))
        for nr in range(anchor_r - 1, anchor_r + 2):
            for nc in range(anchor_c - 1, anchor_c + 2):
                forbidden.add((nr, nc))

        placed = []
        for color in colors:
            found = False
            for _trial in range(250):
                top = random.randint(0, height - 2 - shape_h)
                left = random.randint(0, width - shape_w)
                if _can_place(shape, top, left, height, width, forbidden):
                    placed.append((color, top, left))
                    _mark_forbidden(shape, top, left, forbidden)
                    found = True
                    break
            if not found:
                break
        if len(placed) != count:
            continue

        for color, top, left in placed:
            for r, c in shape:
                grid[top + r, left + c] = color
        return grid

    raise RuntimeError("could not generate a valid input")
