from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
LAYER_RANGE = (1, 5)
INNER_HEIGHT_RANGE = (1, 8)
INNER_WIDTH_RANGE = (1, 8)
OUTER_MARGIN_RANGE = (3, 7)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _most_common_color(arr: np.ndarray) -> int:
    vals = np.asarray(arr, dtype=int).ravel()
    return int(np.bincount(vals, minlength=10).argmax())


def _frame_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, tuple[int, int, int, int]] | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return None

    return next(iter(colors)), (r0, c0, r1, c1)


def _corner_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    assert h >= 2 and w >= 2, "grid must fit four 2x2 corner blocks"
    tl = _most_common_color(grid[:2, :2])
    tr = _most_common_color(grid[:2, w - 2 : w])
    bl = _most_common_color(grid[h - 2 : h, :2])
    br = _most_common_color(grid[h - 2 : h, w - 2 : w])
    return tl, tr, bl, br


def _quadrant_fill(out: np.ndarray, bbox: tuple[int, int, int, int], colors: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= 1 and inner_w >= 1, "innermost frame must enclose a non-empty area"

    top_h = (inner_h + 1) // 2
    left_w = (inner_w + 1) // 2
    bottom_r0 = inner_r0 + top_h
    right_c0 = inner_c0 + left_w

    tl, tr, bl, br = colors
    out[inner_r0 : inner_r0 + top_h, inner_c0 : inner_c0 + left_w] = tl
    out[inner_r0 : inner_r0 + top_h, right_c0 : inner_c0 + inner_w] = tr
    out[bottom_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + left_w] = bl
    out[bottom_r0 : inner_r0 + inner_h, right_c0 : inner_c0 + inner_w] = br


def _fill_gap(
    out: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
) -> None:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox

    ring = np.zeros_like(out, dtype=bool)
    ring[or0 + 1 : or1, oc0 + 1 : oc1] = True
    ring[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    ring &= out == int(bg)
    out[ring] = int(color)


def _detect_frames(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid != bg):
        frame = _frame_component(grid, comp)
        if frame is not None:
            frames.append(frame)

    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda item: (item[1][2] - item[1][0] + 1) * (item[1][3] - item[1][1] + 1))

    for idx in range(1, len(frames)):
        inner_bbox = frames[idx - 1][1]
        outer_bbox = frames[idx][1]
        assert outer_bbox[0] < inner_bbox[0] and outer_bbox[1] < inner_bbox[1]
        assert outer_bbox[2] > inner_bbox[2] and outer_bbox[3] > inner_bbox[3]

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested rectangular frames, fill the innermost cavity by corner quadrants,
    and fill each inter-frame gap using the color of the immediately outer frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _detect_frames(g, bg)
    out = g.copy()

    _quadrant_fill(out, frames[0][1], _corner_colors(g))
    for idx in range(1, len(frames)):
        outer_color, outer_bbox = frames[idx]
        inner_bbox = frames[idx - 1][1]
        _fill_gap(out, outer_bbox, inner_bbox, outer_color, bg)

    return out


def _sample_dim_with_parity(rng: random.Random, bounds: tuple[int, int]) -> int:
    low, high = bounds
    parity = rng.randrange(2)
    options = [value for value in range(low, high + 1) if value % 2 == parity]
    if not options:
        options = list(range(low, high + 1))
    return rng.choice(options)


def _corner_pattern(rng: random.Random, colors: list[int]) -> tuple[int, int, int, int]:
    mode = rng.choice(("distinct", "paired_diag", "paired_rows", "all_same"))
    if mode == "distinct":
        tl, tr, bl, br = rng.sample(colors, 4)
        return tl, tr, bl, br
    if mode == "paired_diag":
        a, b = rng.sample(colors, 2)
        return a, b, b, a
    if mode == "paired_rows":
        a, b = rng.sample(colors, 2)
        return a, a, b, b
    fill = rng.choice(colors)
    return fill, fill, fill, fill


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray:
    layers = rng.randint(*LAYER_RANGE)
    inner_h = _sample_dim_with_parity(rng, INNER_HEIGHT_RANGE)
    inner_w = _sample_dim_with_parity(rng, INNER_WIDTH_RANGE)
    inner_frame_h = inner_h + 2
    inner_frame_w = inner_w + 2
    outer_h = inner_frame_h + 4 * (layers - 1)
    outer_w = inner_frame_w + 4 * (layers - 1)

    top_pad = rng.randint(*OUTER_MARGIN_RANGE)
    bottom_pad = rng.randint(*OUTER_MARGIN_RANGE)
    left_pad = rng.randint(*OUTER_MARGIN_RANGE)
    right_pad = rng.randint(*OUTER_MARGIN_RANGE)

    grid_h = outer_h + top_pad + bottom_pad
    grid_w = outer_w + left_pad + right_pad
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_colors = rng.sample(list(ALL_COLORS), layers)
    corner_palette = [color for color in ALL_COLORS if color not in frame_colors]
    corners = _corner_pattern(rng, corner_palette)

    grid[:2, :2] = corners[0]
    grid[:2, grid_w - 2 : grid_w] = corners[1]
    grid[grid_h - 2 : grid_h, :2] = corners[2]
    grid[grid_h - 2 : grid_h, grid_w - 2 : grid_w] = corners[3]

    outer_bbox = (top_pad, left_pad, top_pad + outer_h - 1, left_pad + outer_w - 1)
    for idx in range(layers):
        inset = 2 * (layers - 1 - idx)
        bbox = (
            outer_bbox[0] + inset,
            outer_bbox[1] + inset,
            outer_bbox[2] - inset,
            outer_bbox[3] - inset,
        )
        _paint_frame(grid, bbox, frame_colors[idx])

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        frames = _detect_frames(grid, BG)
        if len(frames) < 1:
            continue
        if np.array_equal(main(grid), grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
