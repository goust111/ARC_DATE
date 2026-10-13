from __future__ import annotations

import random

import numpy as np


ARC_MAX = 30
BG = 0
COLOR_POOL = tuple(range(1, 10))
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 6)
HINT_GAP_RANGE = (2, 4)
HINT_STEP_RANGE = (2, 3)
SLACK_RANGE = (0, 4)
GEN_RETRIES = 400
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0, "expected at least one pixel in bbox mask"
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _find_hint_column(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    nonbg_cols = np.where(np.any(grid != bg, axis=0))[0]
    assert nonbg_cols.size > 0, "expected at least one non-background column"
    hint_col = int(nonbg_cols[-1])
    order = [int(v) for v in grid[:, hint_col].tolist() if int(v) != bg]
    assert len(order) >= 2, "expected at least two layers in the hint column"
    assert len(order) == len(set(order)), "hint colors must be unique"
    return hint_col, order


def _restore_rectangles(
    scene: np.ndarray, order: list[int], bg: int
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    bboxes: list[tuple[int, int, int, int]] = []
    sizes: set[tuple[int, int]] = set()

    for color in order:
        mask = scene == color
        assert mask.any(), f"hint color {color} must appear in the stack"
        box = _bbox(mask)
        bboxes.append(box)
        sizes.add((box[2] - box[0] + 1, box[3] - box[1] + 1))

    assert len(sizes) == 1, "all restored rectangles must share one size"
    rect_h, rect_w = next(iter(sizes))

    recon = np.full(scene.shape, bg, dtype=int)
    for color, (r0, c0, r1, c1) in zip(order, bboxes, strict=False):
        recon[r0 : r1 + 1, c0 : c1 + 1] = color
    assert np.array_equal(recon, scene), "restored full rectangles must reproduce the stack"

    return bboxes, (rect_h, rect_w)


def _expected_output(order: list[int], rect_h: int, rect_w: int, bg: int) -> np.ndarray:
    out_w = len(order) * rect_w + (len(order) - 1)
    out = np.full((rect_h, out_w), bg, dtype=int)
    for i, color in enumerate(order):
        c0 = i * (rect_w + 1)
        out[:, c0 : c0 + rect_w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped solid rectangles from a depth-hint column and unpack them left-to-right by depth.

    Input structure:
    - The background is black.
    - Several same-sized solid monochrome rectangles overlap in the main scene area, with higher layers hiding one corner of lower layers.
    - The overlap follows one diagonal drift, so every covered rectangle still leaves a visible L-shaped fragment whose bounding box is its original full rectangle.
    - A right-side hint column contains one pixel per rectangle; reading these colors from top to bottom gives the layer order from bottommost to topmost.

    Rule:
    - Read the right-side hint column to obtain the depth order.
    - Ignore that hint column, then for each hinted color find the visible fragment of that layer in the scene and restore the full rectangle by taking the fragment's bounding box.
    - Check that repainting those restored rectangles in the hinted depth order exactly recreates the observed occlusion pattern.
    - Output the restored full rectangles as solid blocks arranged horizontally from bottom layer to top layer, with one black separator column between neighbors.

    Key insight:
    - The hint column resolves the z-order, and the diagonal occlusion pattern guarantees that each buried layer can be fully recovered from the bounding box of its remaining visible L-shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    hint_col, order = _find_hint_column(g, bg)
    scene = g[:, :hint_col].copy()

    _bboxes, (rect_h, rect_w) = _restore_rectangles(scene, order, bg)
    return _expected_output(order, rect_h, rect_w, bg)


def _hint_rows(height: int, n_layers: int) -> list[int] | None:
    steps = [random.randint(*HINT_STEP_RANGE) for _ in range(n_layers - 1)]
    span = 1 + sum(steps)
    if span > height:
        return None
    start = random.randint(0, height - span)
    rows = [start]
    for step in steps:
        rows.append(rows[-1] + step)
    return rows


def _relative_anchors(
    n_layers: int, rect_h: int, rect_w: int, direction: tuple[int, int]
) -> list[tuple[int, int]]:
    dr_sign, dc_sign = direction
    anchors = [(0, 0)]
    r = 0
    c = 0
    for _ in range(n_layers - 1):
        r += dr_sign * random.randint(1, rect_h - 1)
        c += dc_sign * random.randint(1, rect_w - 1)
        anchors.append((r, c))

    min_r = min(r0 for r0, _c0 in anchors)
    min_c = min(c0 for _r0, c0 in anchors)
    return [(r0 - min_r, c0 - min_c) for r0, c0 in anchors]


def _render_stack(
    height: int,
    width: int,
    anchors: list[tuple[int, int]],
    colors: list[int],
    rect_h: int,
    rect_w: int,
    stack_top: int,
    stack_left: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for color, (r0, c0) in zip(colors, anchors, strict=False):
        rr = stack_top + r0
        cc = stack_left + c0
        grid[rr : rr + rect_h, cc : cc + rect_w] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(*RECT_COUNT_RANGE)
    rect_h = random.randint(*RECT_SIDE_RANGE)
    rect_w = random.randint(*RECT_SIDE_RANGE)
    if n_layers * rect_w + (n_layers - 1) > ARC_MAX:
        return None

    direction = random.choice(DIRECTIONS)
    colors = random.sample(COLOR_POOL, n_layers)
    anchors = _relative_anchors(n_layers, rect_h, rect_w, direction)

    stack_h = max(r0 for r0, _c0 in anchors) + rect_h
    stack_w = max(c0 for _r0, c0 in anchors) + rect_w
    hint_gap = random.randint(*HINT_GAP_RANGE)
    width = stack_w + hint_gap + 1 + random.randint(*SLACK_RANGE)
    if width > ARC_MAX:
        return None

    min_height = max(stack_h, 1 + (n_layers - 1) * HINT_STEP_RANGE[0])
    height = min_height + random.randint(*SLACK_RANGE)
    if height > ARC_MAX:
        return None

    hint_rows = _hint_rows(height, n_layers)
    if hint_rows is None:
        return None

    max_stack_top = height - stack_h
    max_stack_left = width - 1 - hint_gap - stack_w
    assert max_stack_left >= 0
    if max_stack_top < 0:
        return None

    stack_top = random.randint(0, max_stack_top)
    stack_left = random.randint(0, max_stack_left)
    grid = _render_stack(
        height=height,
        width=width,
        anchors=anchors,
        colors=colors,
        rect_h=rect_h,
        rect_w=rect_w,
        stack_top=stack_top,
        stack_left=stack_left,
    )

    hint_col = width - 1
    stack_right = stack_left + stack_w - 1
    assert hint_col - stack_right >= hint_gap + 1
    for row, color in zip(hint_rows, colors, strict=False):
        assert grid[row, hint_col] == BG
        grid[row, hint_col] = color

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != BG:
        return None

    scene = grid[:, :hint_col]
    if any(int((scene == color).sum()) == 0 for color in colors):
        return None
    if not any(int((scene == color).sum()) < rect_h * rect_w for color in colors[:-1]):
        return None

    expected = _expected_output(colors, rect_h, rect_w, BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
