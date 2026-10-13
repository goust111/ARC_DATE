from __future__ import annotations

import numpy as np

from arc_utils import Color, paste
import random


MIN_SIDE = 4
TARGET = Color.YELLOW


def _right_runs(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    runs = np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if int(g[r, c]) == int(g[r, c + 1]):
                runs[r, c] = runs[r, c + 1] + 1
    return runs


def _maximal_uniform_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    runs = _right_runs(g)
    rects: set[tuple[int, int, int, int, int]] = set()

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            max_w = w - c0
            for r1 in range(r0, h):
                if int(g[r1, c0]) != color:
                    break
                max_w = min(max_w, int(runs[r1, c0]))
                rect_h = r1 - r0 + 1
                if rect_h >= MIN_SIDE and max_w >= MIN_SIDE:
                    rects.add((r0, c0, r1, c0 + max_w - 1, color))

    maximal: list[tuple[int, int, int, int, int]] = []
    for rect in rects:
        r0, c0, r1, c1, color = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        contained = False
        for other in rects:
            if other == rect:
                continue
            or0, oc0, or1, oc1, other_color = other
            if other_color != color:
                continue
            other_area = (or1 - or0 + 1) * (oc1 - oc0 + 1)
            if other_area > area and or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1:
                contained = True
                break
        if not contained:
            maximal.append(rect)

    return sorted(maximal)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every large solid monochrome rectangle hidden in the noisy grid to yellow.

    Input structure
    A 2D grid with multicolor noise and no pre-existing yellow pixels; the meaningful
    objects are one or more axis-aligned solid rectangles of a single color.

    Rule
    Find all maximal same-color rectangles whose height and width are both at least
    4, and recolor exactly those rectangle cells to yellow.

    Key insight
    The hidden objects are the only large uniform rectangular regions; maximality
    excludes smaller subrectangles inside them while the side-length threshold
    ignores ordinary short runs in the noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert not np.any(g == TARGET)

    rects = _maximal_uniform_rectangles(g)
    assert rects

    out = g.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1, color in rects:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert r1 - r0 + 1 >= MIN_SIDE
        assert c1 - c0 + 1 >= MIN_SIDE
        assert np.all(patch == color)
        assert not np.any(occupied[r0 : r1 + 1, c0 : c1 + 1])
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0 : c1 + 1] = TARGET

    return out

# --------------- generator (o5x) ---------------
# Canvas and colors.
_GEN_H, _GEN_W = 30, 30
_GEN_SHAPE = (_GEN_H, _GEN_W)
_GEN_PALETTE = (1, 2, 3, 6, 8)
_GEN_TARGET = 4

# Rectangle sampling and placement.
_GEN_RECT_COUNT_RANGE = (2, 4)
_GEN_RECT_HEIGHT_RANGE = (4, 7)
_GEN_RECT_WIDTH_RANGE = (5, 10)
_GEN_MARGIN, _GEN_CLEARANCE = 2, 1

# Noise cleanup and rejection sampling.
_GEN_UNIFORM_BLOCK, _GEN_NOISE_REPAIR_PASSES = 3, 40
_GEN_MAX_ATTEMPTS = 300

_GenRect = tuple[int, int, int, int, int]


def _gen_padded_region(r0: int, c0: int, r1: int, c1: int) -> tuple[slice, slice]:
    return (
        slice(max(0, r0 - _GEN_CLEARANCE), min(_GEN_H, r1 + _GEN_CLEARANCE + 1)),
        slice(max(0, c0 - _GEN_CLEARANCE), min(_GEN_W, c1 + _GEN_CLEARANCE + 1)),
    )


def _gen_place_rectangles() -> list[_GenRect]:
    count = random.randint(*_GEN_RECT_COUNT_RANGE)
    dims = [
        (random.randint(*_GEN_RECT_HEIGHT_RANGE), random.randint(*_GEN_RECT_WIDTH_RANGE))
        for _ in range(count)
    ]
    colors = random.sample(_GEN_PALETTE, count)
    specs = list(zip(dims, colors))
    random.shuffle(specs)
    occupied = np.zeros(_GEN_SHAPE, dtype=bool)
    rects: list[_GenRect] = []
    for (height, width), color in specs:
        choices = []
        max_r0 = _GEN_H - _GEN_MARGIN - height
        max_c0 = _GEN_W - _GEN_MARGIN - width
        for r0 in range(_GEN_MARGIN, max_r0 + 1):
            for c0 in range(_GEN_MARGIN, max_c0 + 1):
                r1 = r0 + height - 1
                c1 = c0 + width - 1
                rows, cols = _gen_padded_region(r0, c0, r1, c1)
                if not np.any(occupied[rows, cols]):
                    choices.append((r0, c0))
        if not choices:
            return []
        r0, c0 = random.choice(choices)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        rows, cols = _gen_padded_region(r0, c0, r1, c1)
        occupied[rows, cols] = True
        rects.append((r0, c0, r1, c1, color))
    return rects


def _gen_forbidden_colors(
    rects: list[_GenRect], rect_mask: np.ndarray
) -> list[list[set[int]]]:
    forbidden = [[set() for _ in range(_GEN_W)] for _ in range(_GEN_H)]
    for r0, c0, r1, c1, color in rects:
        rows, cols = _gen_padded_region(r0, c0, r1, c1)
        for r in range(rows.start, rows.stop):
            for c in range(cols.start, cols.stop):
                if not rect_mask[r, c]:
                    forbidden[r][c].add(color)
    return forbidden


def _gen_choices(forbidden: list[list[set[int]]], r: int, c: int) -> list[int]:
    return [color for color in _GEN_PALETTE if color not in forbidden[r][c]]


def _gen_uniform_noise_blocks(grid: np.ndarray, rect_mask: np.ndarray) -> list[tuple[int, int]]:
    blocks = []
    for r in range(_GEN_H - _GEN_UNIFORM_BLOCK + 1):
        for c in range(_GEN_W - _GEN_UNIFORM_BLOCK + 1):
            if np.any(rect_mask[r : r + _GEN_UNIFORM_BLOCK, c : c + _GEN_UNIFORM_BLOCK]):
                continue
            patch = grid[r : r + _GEN_UNIFORM_BLOCK, c : c + _GEN_UNIFORM_BLOCK]
            if np.all(patch == int(patch[0, 0])):
                blocks.append((r, c))
    return blocks


def _gen_fill_noise(
    out: np.ndarray, rect_mask: np.ndarray, forbidden: list[list[set[int]]]
) -> bool:
    for r in range(_GEN_H):
        for c in range(_GEN_W):
            if rect_mask[r, c]:
                continue
            out[r, c] = random.choice(_gen_choices(forbidden, r, c))

    for _ in range(_GEN_NOISE_REPAIR_PASSES):
        blocks = _gen_uniform_noise_blocks(out, rect_mask)
        if not blocks:
            return True
        for r0, c0 in blocks:
            uniform_color = int(out[r0, c0])
            candidates = []
            for r in range(r0, r0 + _GEN_UNIFORM_BLOCK):
                for c in range(c0, c0 + _GEN_UNIFORM_BLOCK):
                    choices = [
                        color
                        for color in _gen_choices(forbidden, r, c)
                        if color != uniform_color
                    ]
                    if choices:
                        candidates.append((r, c, choices))
            if not candidates:
                return False
            r, c, choices = random.choice(candidates)
            out[r, c] = random.choice(choices)
    return not _gen_uniform_noise_blocks(out, rect_mask)


def _gen_paste_rect(dst: np.ndarray, rect: _GenRect, color: int) -> None:
    r0, c0, r1, c1, _ = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    paste(dst, np.full((height, width), color, dtype=int), r0, c0)


def _gen_render_target_grid(rects: list[_GenRect]) -> tuple[np.ndarray, np.ndarray]:
    rect_mask = np.zeros(_GEN_SHAPE, dtype=bool)
    out = np.zeros(_GEN_SHAPE, dtype=int)
    for rect in rects:
        r0, c0, r1, c1, _ = rect
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        _gen_paste_rect(out, rect, _GEN_TARGET)
    return out, rect_mask


def _gen_render_input_grid(out: np.ndarray, rects: list[_GenRect]) -> np.ndarray:
    inp = out.copy()
    for rect in rects:
        _gen_paste_rect(inp, rect, rect[4])
    return inp


def _gen_oracle_matches(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    rects = _gen_place_rectangles()
    if not rects:
        return None

    out, rect_mask = _gen_render_target_grid(rects)
    forbidden = _gen_forbidden_colors(rects, rect_mask)
    if not _gen_fill_noise(out, rect_mask, forbidden):
        return None
    inp = _gen_render_input_grid(out, rects)
    if not _gen_oracle_matches(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid 25094a63 grid")
