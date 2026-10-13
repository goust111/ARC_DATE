import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rect_frame(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    return set(comp) == _perimeter_cells((r0, c0, r1, c1))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _monochrome_components(grid, color):
            if _is_rect_frame(comp):
                bbox = _bbox(comp)
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                frames.append({"color": color, "bbox": bbox, "cells": comp, "area": area})
    frames.sort(key=lambda frame: frame["area"])
    assert len(frames) >= 2, "expected at least two nested frames"
    for inner, outer in zip(frames, frames[1:]):
        assert _strictly_contains(outer["bbox"], inner["bbox"]), "frames must be strictly nested"
    return frames


def _read_palette(grid: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    _, _, _, outer_c1 = outer_bbox
    mapping: dict[int, int] = {}
    for r in range(grid.shape[0]):
        cols = [int(c) for c in np.flatnonzero(grid[r] != BG) if int(c) > outer_c1]
        if not cols:
            continue
        assert len(cols) == 2 and cols[1] == cols[0] + 1, "palette rows must be 2 adjacent pixels"
        mapping[int(grid[r, cols[0]])] = int(grid[r, cols[1]])
    assert mapping, "palette mapping must be non-empty"
    return mapping


def _fill_annulus(
    out: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    mask = np.zeros_like(out, dtype=bool)
    mask[or0 + 1 : or1, oc0 + 1 : oc1] = True
    mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    out[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gap between nested rectangular frames by propagating the core color through a frame-keyed palette.

    Input structure:
    - The background is black.
    - On the left is a stack of nested one-pixel-thick rectangular frames, with black gaps of varying widths between consecutive frames.
    - Inside the innermost frame sits a smaller solid rectangular core object of one color.
    - On the right is a vertical palette of horizontal 2-pixel pairs: the left pixel matches a frame color and the right pixel gives that frame's outward fill color.

    Rule:
    - Preserve every frame in place and erase the right-side palette to black.
    - Fill the entire interior of the innermost frame with the core object's color.
    - For each gap between frame i and the next outer frame, read the palette entry keyed by frame i's color and fill that gap with the mapped color.
    - Leave everything outside the outermost frame black.

    Key insight:
    - The palette is indexed by frame colors, so the solver must recover the nesting order of the rectangular borders before assigning each annular gap its mapped fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    inner_bbox = frames[0]["bbox"]
    outer_bbox = frames[-1]["bbox"]
    palette = _read_palette(g, outer_bbox)

    out = np.full_like(g, BG)
    for frame in frames:
        color = int(frame["color"])
        for r, c in frame["cells"]:
            out[r, c] = color

    ir0, ic0, ir1, ic1 = inner_bbox
    core_window = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
    core_colors = np.unique(core_window[core_window != BG])
    assert core_colors.size == 1, "expected a unique core-object color"
    core_color = int(core_colors[0])
    out[ir0 + 1 : ir1, ic0 + 1 : ic1] = core_color

    for inner, outer in zip(frames, frames[1:]):
        inner_color = int(inner["color"])
        assert inner_color in palette, "missing palette entry for an inner frame color"
        _fill_annulus(out, outer["bbox"], inner["bbox"], int(palette[inner_color]))

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _perimeter_cells(bbox):
        grid[r, c] = color


def _expected_output(
    shape: tuple[int, int],
    frame_bboxes: list[tuple[int, int, int, int]],
    frame_colors: list[int],
    core_color: int,
    gap_fills: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for bbox, color in zip(frame_bboxes, frame_colors):
        _draw_frame(out, bbox, color)

    ir0, ic0, ir1, ic1 = frame_bboxes[0]
    out[ir0 + 1 : ir1, ic0 + 1 : ic1] = core_color
    for i in range(len(frame_bboxes) - 1):
        _fill_annulus(out, frame_bboxes[i + 1], frame_bboxes[i], gap_fills[i])
    return out


def _sample_layout(rng: random.Random, layers: int):
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 5)
    bboxes = [(0, 0, inner_h + 1, inner_w + 1)]
    for _ in range(layers - 1):
        prev = bboxes[-1]
        top_gap = rng.randint(1, 2)
        bottom_gap = rng.randint(1, 2)
        left_gap = rng.randint(1, 2)
        right_gap = rng.randint(1, 2)
        bboxes.append(
            (
                prev[0] - (top_gap + 1),
                prev[1] - (left_gap + 1),
                prev[2] + (bottom_gap + 1),
                prev[3] + (right_gap + 1),
            )
        )

    min_r = min(b[0] for b in bboxes)
    min_c = min(b[1] for b in bboxes)
    norm = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in bboxes]
    outer = norm[-1]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    return norm, inner_h, inner_w, outer_h, outer_w


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    layers = rng.choice([2, 3, 4])
    frame_bboxes_local, inner_h, inner_w, outer_h, outer_w = _sample_layout(rng, layers)

    palette_rows = layers - 1
    palette_step = rng.choice([1, 2])
    palette_h = 1 + (palette_rows - 1) * palette_step

    min_h = max(outer_h + 2, palette_h + 2)
    min_w = outer_w + 5
    if min_h > 30 or min_w > 30:
        return None

    height = rng.randint(min_h, min(30, min_h + 4))
    width = rng.randint(min_w, min(30, min_w + 4))

    palette_left = width - 3
    max_obj_c0 = palette_left - outer_w - 1
    if max_obj_c0 < 1:
        return None

    obj_r0 = rng.randint(1, height - outer_h - 1)
    obj_c0 = rng.randint(1, max_obj_c0)
    pal_r0 = rng.randint(1, height - palette_h - 1)

    color_pool = rng.sample(COLORS, 2 * layers)
    frame_colors = color_pool[:layers]
    gap_fills = color_pool[layers : 2 * layers - 1]
    core_color = color_pool[-1]

    frame_bboxes = [
        (r0 + obj_r0, c0 + obj_c0, r1 + obj_r0, c1 + obj_c0)
        for r0, c0, r1, c1 in frame_bboxes_local
    ]

    grid = np.full((height, width), BG, dtype=int)
    for bbox, color in zip(frame_bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    inner_bbox = frame_bboxes[0]
    ir0, ic0, ir1, ic1 = inner_bbox
    core_h = rng.randint(1, inner_h - 2)
    core_w = rng.randint(1, inner_w - 2)
    core_r0 = rng.randint(ir0 + 2, ir1 - core_h - 1)
    core_c0 = rng.randint(ic0 + 2, ic1 - core_w - 1)
    grid[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = core_color

    entries = list(zip(frame_colors[:-1], gap_fills))
    rng.shuffle(entries)
    for idx, (key_color, dst_color) in enumerate(entries):
        rr = pal_r0 + idx * palette_step
        grid[rr, palette_left] = key_color
        grid[rr, palette_left + 1] = dst_color

    expected = _expected_output(grid.shape, frame_bboxes, frame_colors, core_color, gap_fills)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        case = _build_case(rng)
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
