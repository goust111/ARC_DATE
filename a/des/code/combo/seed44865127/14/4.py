from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _marker_sequence(g: np.ndarray, bg: int) -> list[int]:
    marker_cols = np.flatnonzero(g[0] != bg)
    assert 3 <= len(marker_cols) <= 5, "marker must contain 3-5 pixels"
    assert np.array_equal(marker_cols, np.arange(int(marker_cols[0]), int(marker_cols[-1]) + 1))
    colors = [int(g[0, c]) for c in marker_cols]
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    return colors


def _single_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one object pixel"

    start = tuple(int(x) for x in coords[0])
    stack = [start]
    visited = {start}
    cells: list[tuple[int, int]] = []
    h, w = mask.shape

    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))

    assert len(cells) == int(mask.sum()), "each marker color must form exactly one 4-connected object"
    return np.asarray(cells, dtype=int)


def _extract_object(body: np.ndarray, color: int) -> dict[str, object]:
    coords = _single_component(body == color)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    rel = coords - np.array([r0, c0], dtype=int)
    return {
        "top": int(r0 + 1),
        "left": int(c0),
        "width": int(c1 - c0 + 1),
        "rel": rel,
    }


def _paint_object(out: np.ndarray, obj: dict[str, object], left: int, color: int) -> None:
    top = int(obj["top"])
    rel = np.asarray(obj["rel"], dtype=int)
    rows = top + rel[:, 0]
    cols = int(left) + rel[:, 1]
    assert rows.min() >= 0 and rows.max() < out.shape[0]
    assert cols.min() >= 0 and cols.max() < out.shape[1]
    out[rows, cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack marker-matched objects left to right, then shift their colors by one step.

    Input structure:
    - The background is the dominant color.
    - The top row contains one contiguous marker segment of 3-5 distinct colors.
    - Below it, each marker color appears in exactly one monochrome 4-connected object.

    Rule:
    - Read the marker colors from left to right.
    - Take the corresponding objects in that order.
    - Repack them from left to right with one background column between bounding boxes.
    - To keep the fixed canvas deterministic, the repacked row positions stay with their
      original objects and packing starts at the leftmost original object column.
    - Recolor each object to the next marker color cyclically and erase the marker row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 4 and g.shape[1] >= 6

    bg = _most_common_color(g)
    sequence = _marker_sequence(g, bg)
    body = g[1:, :]

    body_colors = {int(v) for v in np.unique(body) if int(v) != bg}
    assert body_colors == set(sequence), "body colors must match the marker colors exactly"

    objects = {color: _extract_object(body, color) for color in sequence}
    ordered = [objects[color] for color in sequence]

    pack_left = min(int(obj["left"]) for obj in ordered)
    packed_width = sum(int(obj["width"]) for obj in ordered) + len(ordered) - 1
    assert pack_left + packed_width <= g.shape[1], "packed objects must fit in the canvas"

    out = np.full_like(g, bg)
    next_left = pack_left
    for idx, obj in enumerate(ordered):
        new_color = sequence[(idx + 1) % len(sequence)]
        _paint_object(out, obj, next_left, new_color)
        next_left += int(obj["width"]) + 1

    return out


# --------------- generator ---------------

OBJECT_COUNT_CHOICES = (3, 4, 5)
MAX_BBOX_SIDE = 4
MAX_SHAPE_AREA = 7
HEIGHT_EXTRA_RANGE = (4, 8)
WIDTH_EXTRA_RANGE = (3, 8)
PACK_LEFT_MAX = 3
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not bool(mask[r, c]) and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(mask[r, c]) and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]) and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~outside)))


def _random_shape() -> np.ndarray:
    for _ in range(100):
        h = random.randint(1, MAX_BBOX_SIDE)
        w = random.randint(1, MAX_BBOX_SIDE)
        if h == 1 and w == 1:
            continue

        max_area = min(MAX_SHAPE_AREA, h * w)
        if max_area < 2:
            continue
        target_area = random.randint(2, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {start}

        while int(mask.sum()) < target_area:
            base_r, base_c = random.choice(tuple(frontier))
            candidates = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    candidates.append((nr, nc))
            if not candidates:
                frontier.remove((base_r, base_c))
                if not frontier:
                    break
                continue
            chosen = random.choice(candidates)
            mask[chosen] = True
            frontier.add(chosen)

        if int(mask.sum()) != target_area:
            continue

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        mask = mask[rows][:, cols]
        if mask.shape == (1, 1):
            continue
        if _has_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a connected hole-free shape")


def _shape_object(mask: np.ndarray, top: int, left: int) -> dict[str, object]:
    rel = np.argwhere(mask)
    return {
        "top": int(top),
        "left": int(left),
        "width": int(mask.shape[1]),
        "rel": rel.astype(int),
    }


def _paint_sequence(grid: np.ndarray, start_col: int, colors: list[int]) -> None:
    for offset, color in enumerate(colors):
        grid[0, start_col + offset] = int(color)


def _build_expected(
    shape_masks: list[np.ndarray],
    tops: list[int],
    sequence: list[int],
    pack_left: int,
    grid_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    next_left = pack_left
    for idx, mask in enumerate(shape_masks):
        obj = _shape_object(mask, tops[idx], next_left)
        _paint_object(out, obj, next_left, sequence[(idx + 1) % len(sequence)])
        next_left += int(mask.shape[1]) + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(OBJECT_COUNT_CHOICES)
    shape_masks = [_random_shape() for _ in range(count)]
    heights = [int(mask.shape[0]) for mask in shape_masks]
    widths = [int(mask.shape[1]) for mask in shape_masks]

    packed_width = sum(widths) + count - 1
    pack_left = random.randint(0, PACK_LEFT_MAX)
    width_extra = random.randint(*WIDTH_EXTRA_RANGE)
    height_extra = random.randint(*HEIGHT_EXTRA_RANGE)
    grid_h = max(2 + max(heights) + height_extra, 8)
    grid_w = pack_left + packed_width + width_extra

    if grid_w < count + 3:
        return None

    tops = [random.randint(2, grid_h - h) for h in heights]

    gap_extras = np.random.multinomial(width_extra, np.full(count, 1.0 / count)).astype(int).tolist()
    scatter_order = list(range(count))
    random.shuffle(scatter_order)
    if scatter_order == list(range(count)):
        return None

    lefts = [0] * count
    cursor = pack_left
    for pos, idx in enumerate(scatter_order):
        lefts[idx] = cursor
        cursor += widths[idx]
        if pos < count - 1:
            cursor += 1 + int(gap_extras[pos])

    bg, *rest = random.sample(PALETTE, count + 1)
    sequence = rest
    marker_start = random.randint(0, grid_w - count)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    _paint_sequence(inp, marker_start, sequence)

    for idx, mask in enumerate(shape_masks):
        obj = _shape_object(mask, tops[idx], lefts[idx])
        _paint_object(inp, obj, lefts[idx], sequence[idx])

    expected = _build_expected(
        shape_masks=shape_masks,
        tops=tops,
        sequence=sequence,
        pack_left=pack_left,
        grid_shape=inp.shape,
        bg=bg,
    )

    input_order = sorted(range(count), key=lambda i: lefts[i])
    if input_order == list(range(count)):
        return None
    if _most_common_color(inp) != bg:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sequence-marker input")
