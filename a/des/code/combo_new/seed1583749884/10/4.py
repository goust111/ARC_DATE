from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 4
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]

GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
MARGIN_TO_EDGE = 2
PLACEMENT_TRIES = 80
GEN_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _frame_bbox(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(comp) != 2 * (h + w) - 4:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(sub == color, border):
        return None
    return (r0, c0, r1, c1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    top = max(r0, 0)
    bottom = min(r1, h - 1)
    left = max(c0, 0)
    right = min(c1, w - 1)
    if top > bottom or left > right:
        return
    if 0 <= r0 < h:
        grid[r0, left : right + 1] = color
    if 0 <= r1 < h:
        grid[r1, left : right + 1] = color
    if 0 <= c0 < w:
        grid[top : bottom + 1, c0] = color
    if 0 <= c1 < w:
        grid[top : bottom + 1, c1] = color


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _branch_count(branch: int) -> int:
    if branch == 0:
        return 0
    if branch == 1:
        return random.choice([1, 2])
    return random.choice([3, 4])


def _sample_counts(frame_count: int) -> list[int]:
    counts = [0, _branch_count(1), _branch_count(2)]
    while len(counts) < frame_count:
        counts.append(_branch_count(random.randrange(3)))
    random.shuffle(counts)
    return counts


def _sample_bbox(
    height: int,
    width: int,
    marker_count: int,
    reserved: np.ndarray,
    inner_side_cap: int,
) -> tuple[int, int, int, int] | None:
    max_inner_h = min(inner_side_cap, height - 2 * MARGIN_TO_EDGE - 2)
    max_inner_w = min(inner_side_cap, width - 2 * MARGIN_TO_EDGE - 2)
    if max_inner_h < 1 or max_inner_w < 1:
        return None

    for _ in range(PLACEMENT_TRIES):
        inner_h = random.randint(1, max_inner_h)
        min_inner_w = max(1, (marker_count + inner_h - 1) // inner_h)
        if min_inner_w > max_inner_w:
            continue
        inner_w = random.randint(min_inner_w, max_inner_w)
        frame_h = inner_h + 2
        frame_w = inner_w + 2

        r0_min = MARGIN_TO_EDGE
        c0_min = MARGIN_TO_EDGE
        r0_max = height - frame_h - MARGIN_TO_EDGE
        c0_max = width - frame_w - MARGIN_TO_EDGE
        if r0_max < r0_min or c0_max < c0_min:
            continue

        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        rr0, cc0, rr1, cc1 = _expand_bbox(bbox, 2)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return bbox
    return None


def _sample_markers(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return random.sample(interior, count)


def _build_grids(
    shape: tuple[int, int],
    frames: list[tuple[int, tuple[int, int, int, int], int, list[tuple[int, int]]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG_COLOR, dtype=int)
    out = np.full(shape, BG_COLOR, dtype=int)
    for color, bbox, count, markers in frames:
        _draw_frame(inp, bbox, color)
        for r, c in markers:
            inp[r, c] = MARKER_COLOR

        if count == 0:
            continue
        _draw_frame(out, bbox, color)
        if count >= 3:
            _draw_frame(out, _expand_bbox(bbox, 1), color)
    return inp, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Change each rectangular frame according to how many marker pixels it encloses.

    Input structure:
    - The background is one dominant color.
    - Several non-background colors form one-pixel-thick closed rectangular frames.
    - Remaining non-background pixels are marker dots placed strictly inside frame interiors.

    Rule:
    - Detect every rectangular frame and count the marker pixels inside its open interior.
    - If a frame contains 0 markers, erase it; if it contains 1 or 2 markers, keep it unchanged; if it contains 3 or more markers, add a same-colored outer ring one cell farther out.
    - Remove all marker dots from the output.

    Key insight:
    - The interior dots are not output objects; they act as control tokens that decide whether the surrounding frame disappears, stays thin, or becomes thicker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _dominant_color(g)

    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    frame_mask = np.zeros((h, w), dtype=bool)
    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(g == color):
            bbox = _frame_bbox(g, color, comp)
            if bbox is None:
                continue
            frames.append((color, bbox))
            for r, c in comp:
                frame_mask[r, c] = True

    assert frames, "expected at least one rectangular frame"
    marker_mask = (g != bg) & ~frame_mask

    out = np.full_like(g, bg)
    for color, bbox in frames:
        r0, c0, r1, c1 = bbox
        marker_count = int(marker_mask[r0 + 1 : r1, c0 + 1 : c1].sum())
        if marker_count == 0:
            continue
        _draw_frame(out, bbox, color)
        if marker_count >= 3:
            _draw_frame(out, _expand_bbox(bbox, 1), color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    area = height * width
    if area >= 500:
        frame_count = random.randint(3, 5)
    elif area >= 300:
        frame_count = random.randint(3, 4)
    else:
        frame_count = 3

    inner_side_cap = max(3, 6 - max(0, frame_count - 3))
    counts = _sample_counts(frame_count)
    colors = random.sample(FRAME_COLORS, frame_count)
    reserved = np.zeros((height, width), dtype=bool)

    frames: list[tuple[int, tuple[int, int, int, int], int, list[tuple[int, int]]]] = []
    for color, count in zip(colors, counts):
        bbox = _sample_bbox(height, width, count, reserved, inner_side_cap)
        if bbox is None:
            return None
        markers = _sample_markers(bbox, count)
        frames.append((color, bbox, count, markers))

    inp, expected = _build_grids((height, width), frames)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
