from __future__ import annotations
from collections import defaultdict

import numpy as np

from arc_utils import find_components, get_bg
import random


def _cells(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _solid_completion_offsets(
    g: np.ndarray,
    source: frozenset[tuple[int, int]],
    target: frozenset[tuple[int, int]],
    bg: int,
) -> list[tuple[int, int]]:
    height, width = g.shape
    offsets: list[tuple[int, int]] = []

    for dr in range(1 - height, height):
        for dc in range(1 - width, width):
            moved = frozenset((r + dr, c + dc) for r, c in source)
            if any(r < 0 or r >= height or c < 0 or c >= width for r, c in moved):
                continue
            if moved & target:
                continue
            if any(int(g[r, c]) != int(bg) for r, c in moved):
                continue

            union = moved | target
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            rect_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            if len(union) == rect_area:
                offsets.append((dr, dc))

    return offsets


def _completion_candidate(
    g: np.ndarray,
    bg: int,
) -> tuple[int, frozenset[tuple[int, int]], int, int]:
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for component in components:
        by_color[int(component["color"])].append(component)

    candidates: list[tuple[int, frozenset[tuple[int, int]], int, int]] = []
    for color, parts in by_color.items():
        if len(parts) != 2:
            continue

        left_edge = [part for part in parts if int(part["bbox"][1]) == 0]
        anchors = [part for part in parts if int(part["bbox"][1]) != 0]
        if len(left_edge) != 1 or len(anchors) != 1:
            continue

        source = _cells(left_edge[0]["coords"])
        target = _cells(anchors[0]["coords"])
        for dr, dc in _solid_completion_offsets(g, source, target, bg):
            candidates.append((int(color), source, dr, dc))

    assert len(candidates) == 1, "expected one left-edge object that completes a same-color rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the left-edge fragment into its same-colored anchor fragment to complete a solid rectangle.

    Input structure
    - The grid has a dominant background color.
    - Exactly one foreground color is split into two 4-connected objects: one
      touches the left edge, and the other is an anchor away from the left edge.
    - The translated left-edge object and the anchor can form one filled rectangle.
    Rule
    - Find the unique split color whose left-edge object can be translated onto
      background cells beside the other same-colored object so their union is a
      solid axis-aligned rectangle.
    - Clear the left-edge object's original cells to the background.
    - Paint that object at the unique completing offset, preserving all other cells.
    Key insight
    - The active color is identified by object structure and rectangle completion,
      not by a fixed color id; the same-colored anchor object stays in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    color, source, dr, dc = _completion_candidate(g, bg)

    out = g.copy()
    for r, c in source:
        out[r, c] = int(bg)
    for r, c in source:
        rr, cc = r + dr, c + dc
        assert int(out[rr, cc]) == int(bg), "translated source must land on background cells"
        out[rr, cc] = int(color)

    return out

# --------------- generator (o5x) ---------------

# Canvas and object size bounds.
GRID_SIZE_RANGE = (8, 16)
OBJECT_HEIGHT_RANGE = (3, 5)
RECT_WIDTH_RANGE = (4, 7)
LEFT_PATCH_CANVAS_PAD = 3

# Layout and shape constraints.
ROW_GAP_MIN = 1
MIN_ROW_WIDTH = 1
SOURCE_RIGHT_PAD = 2
DST_WIDTH_RESERVED_COLS = 1

# Sampling controls.
PALETTE_COLORS = tuple(range(10))
NONSPLIT_ABOVE_CHOICES = (True, False)
MAX_GENERATION_ATTEMPTS = 300


def _gen_symmetric_widths(height: int, width: int, *, min_width: int = MIN_ROW_WIDTH,
                          varied: bool = False, include_max: bool = False) -> list[int]:
    """Sample row widths and mirror them vertically."""
    half = (height + 1) // 2
    vals = [random.randint(min_width, width) for _ in range(half)]
    if include_max:
        vals[random.randrange(half)] = width
    if varied and width > min_width and len(set(vals + vals[: height // 2][::-1])) == 1:
        j = random.randrange(half)
        vals[j] = min(width, vals[j] + 1) if vals[j] < width else vals[j] - 1
    return vals + vals[: height // 2][::-1]


def _right_anchored_mask(widths: list[int], patch_width: int) -> np.ndarray:
    """Return a mask whose true cells fill each row from the right edge."""
    row_widths = np.asarray(widths, dtype=int)[:, None]
    cols = np.arange(patch_width)[None, :]
    return cols >= patch_width - row_widths


def _gen_right_shape_with_max_width(height: int, patch_width: int, max_row_width: int,
                                    color: int, bg: int) -> np.ndarray:
    patch = np.full((height, patch_width), bg, dtype=int)
    widths = _gen_symmetric_widths(height, max_row_width, varied=True, include_max=True)
    patch[_right_anchored_mask(widths, patch_width)] = color
    return patch


def _gen_pick_layout(n: int) -> tuple[int, int, int, int, int] | None:
    h_min, h_max = OBJECT_HEIGHT_RANGE
    pairs = [
        (rh, nh)
        for rh in range(h_min, h_max + 1)
        for nh in range(h_min, h_max + 1)
        if rh + nh + ROW_GAP_MIN <= n
    ]
    if not pairs:
        return None
    rect_h, nonsplit_h = random.choice(pairs)
    min_w, max_w = RECT_WIDTH_RANGE[0], min(RECT_WIDTH_RANGE[1], n - LEFT_PATCH_CANVAS_PAD)
    if max_w < min_w:
        return None
    rect_w = random.randint(min_w, max_w)

    place_nonsplit_above = random.choice(NONSPLIT_ABOVE_CHOICES)
    if place_nonsplit_above:
        max_rect_top = n - rect_h
        min_rect_top = nonsplit_h + ROW_GAP_MIN
        if min_rect_top > max_rect_top:
            return None
        rect_top = random.randint(min_rect_top, max_rect_top)
        nonsplit_top = random.randint(0, rect_top - nonsplit_h - ROW_GAP_MIN)
    else:
        max_rect_top = n - rect_h - nonsplit_h - ROW_GAP_MIN
        if max_rect_top < 0:
            return None
        rect_top = random.randint(0, max_rect_top)
        nonsplit_top = random.randint(rect_top + rect_h + ROW_GAP_MIN, n - nonsplit_h)
    return rect_h, rect_w, rect_top, nonsplit_h, nonsplit_top


def _gen_split_patches(rect_h: int, rect_w: int, min_dst_width: int,
                       split_color: int, bg: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    dst_widths = _gen_symmetric_widths(rect_h, rect_w - 1, min_width=min_dst_width, varied=True)
    dst = np.full((rect_h, rect_w), bg, dtype=int)
    src = np.full((rect_h, rect_w), bg, dtype=int)
    dst_mask = _right_anchored_mask(dst_widths, rect_w)
    dst[dst_mask] = split_color
    src[~dst_mask] = split_color
    return dst, src, dst_widths


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)
    layout = _gen_pick_layout(n)
    if layout is None:
        return None
    rect_h, rect_w, rect_top, nonsplit_h, nonsplit_top = layout

    colors = list(PALETTE_COLORS)
    bg = random.choice(colors)
    colors.remove(bg)
    split_color, nonsplit_color = random.sample(colors, 2)
    min_dst_width = max(MIN_ROW_WIDTH, 2 * rect_w - n + SOURCE_RIGHT_PAD)
    if min_dst_width > rect_w - DST_WIDTH_RESERVED_COLS:
        return None

    dst, src, dst_widths = _gen_split_patches(rect_h, rect_w, min_dst_width, split_color, bg)
    nonsplit = _gen_right_shape_with_max_width(
        nonsplit_h,
        rect_w,
        max(dst_widths),
        nonsplit_color,
        bg,
    )
    split_area = rect_h * rect_w
    nonsplit_area = int(np.count_nonzero(nonsplit == nonsplit_color))
    bg_area = n * n - split_area - nonsplit_area
    if bg_area <= max(split_area, nonsplit_area):
        return None

    rect_left = n - rect_w
    out = np.full((n, n), bg, dtype=int)
    out[rect_top : rect_top + rect_h, rect_left:n] = split_color
    out[nonsplit_top : nonsplit_top + nonsplit_h, rect_left:n] = nonsplit

    inp = out.copy()
    inp[rect_top : rect_top + rect_h, rect_left:n] = bg
    inp[rect_top : rect_top + rect_h, rect_left:n] = dst
    # Opaque paste of src preserves the original small-canvas overlap behavior.
    inp[rect_top : rect_top + rect_h, 0:rect_w] = src

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fe45cba4 input")
