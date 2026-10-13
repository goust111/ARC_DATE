from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
FRAME = 5
GEN_RETRIES = 400


def _full_separator_rows(g: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == color)]


def _full_separator_cols(g: np.ndarray, color: int) -> list[int]:
    return [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == color)]


def _region_boxes(g: np.ndarray, frame: int) -> list[tuple[int, int, int, int]]:
    rows = _full_separator_rows(g, frame)
    cols = _full_separator_cols(g, frame)
    assert len(rows) >= 2 and len(cols) >= 2, "need a framed row/col partition"
    assert rows[0] == 0 and rows[-1] == g.shape[0] - 1, "outer frame rows required"
    assert cols[0] == 0 and cols[-1] == g.shape[1] - 1, "outer frame cols required"

    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in zip(rows, rows[1:]):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in zip(cols, cols[1:]):
            if c1 - c0 <= 1:
                continue
            boxes.append((int(r0 + 1), int(r1 - 1), int(c0 + 1), int(c1 - 1)))
    assert boxes, "must contain at least one enclosed region"
    return boxes


def _style_step(count: int) -> int | None:
    if count == 1:
        return 1
    if count == 2:
        return 2
    if count == 3:
        return 3
    return None


def _paint_segment(out: np.ndarray, coords: list[tuple[int, int]], color: int, step: int) -> None:
    for idx, (r, c) in enumerate(coords):
        if idx % step == 0:
            out[r, c] = color
        else:
            out[r, c] = FRAME


def _paint_border(out: np.ndarray, box: tuple[int, int, int, int], color: int, count: int) -> None:
    step = _style_step(count)
    if step is None:
        return

    r0, r1, c0, c1 = box
    top = r0 - 1
    bottom = r1 + 1
    left = c0 - 1
    right = c1 + 1

    _paint_segment(out, [(top, c) for c in range(left, right + 1)], color, step)
    _paint_segment(out, [(bottom, c) for c in range(left, right + 1)], color, step)
    _paint_segment(out, [(r, left) for r in range(top, bottom + 1)], color, step)
    _paint_segment(out, [(r, right) for r in range(top, bottom + 1)], color, step)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    out = g.copy()
    boxes = sorted(_region_boxes(g, FRAME), key=lambda box: (box[0], box[2]))

    for r0, r1, c0, c1 in boxes:
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_mask = (region != BG) & (region != FRAME)
        marker_vals = region[marker_mask]

        if marker_vals.size == 0:
            continue
        colors = {int(v) for v in marker_vals}
        assert len(colors) == 1, "each region must use one marker color"
        color = next(iter(colors))
        count = int(marker_vals.size)

        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[marker_mask] = BG
        _paint_border(out, (r0, r1, c0, c1), color, count)

    return out


def _sample_partition(total_inside: int, pieces: int, rng: random.Random) -> list[int]:
    assert total_inside >= pieces * 3
    cuts = sorted(rng.sample(range(3, total_inside - 2), pieces - 1))
    vals = []
    prev = 0
    for cut in cuts + [total_inside]:
        vals.append(cut - prev)
        prev = cut
    if min(vals) < 3:
        return _sample_partition(total_inside, pieces, rng)
    return vals


def _choose_counts(n_regions: int, rng: random.Random) -> list[int]:
    base = [1, 2, 3, 4]
    counts = []
    while len(counts) < n_regions:
        bag = base[:]
        rng.shuffle(bag)
        counts.extend(bag)
    counts = counts[:n_regions]
    rng.shuffle(counts)
    return counts


def _region_positions(shape: tuple[int, int], count: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    return cells[:count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_regions = rng.randint(2, 4)
    col_regions = rng.randint(2, 4)
    inside_h = rng.randint(row_regions * 3, row_regions * 6)
    inside_w = rng.randint(col_regions * 3, col_regions * 6)

    row_sizes = _sample_partition(inside_h, row_regions, rng)
    col_sizes = _sample_partition(inside_w, col_regions, rng)
    h = inside_h + row_regions + 1
    w = inside_w + col_regions + 1

    g = np.full((h, w), BG, dtype=int)
    row_lines = [0]
    cursor = 1
    for size in row_sizes:
        cursor += size
        row_lines.append(cursor)
        cursor += 1

    col_lines = [0]
    cursor = 1
    for size in col_sizes:
        cursor += size
        col_lines.append(cursor)
        cursor += 1

    g[row_lines, :] = FRAME
    g[:, col_lines] = FRAME

    boxes = sorted(_region_boxes(g, FRAME), key=lambda box: (box[0], box[2]))
    counts = _choose_counts(len(boxes), rng)

    palette = [c for c in range(10) if c not in (BG, FRAME)]
    rng.shuffle(palette)
    if len(boxes) > len(palette):
        colors = [palette[i % len(palette)] for i in range(len(boxes))]
        rng.shuffle(colors)
    else:
        colors = palette[: len(boxes)]

    for (r0, r1, c0, c1), count, color in zip(boxes, counts, colors):
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        if region_h * region_w < count:
            return None
        pts = _region_positions((region_h, region_w), count, rng)
        for rr, cc in pts:
            g[r0 + rr, c0 + cc] = color

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
