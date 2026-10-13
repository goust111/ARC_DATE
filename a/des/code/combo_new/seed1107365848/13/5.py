from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_COLOR = 0
MARKER_COLOR = 4
FRAME_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
LAYER_OPTIONS = (3, 4)
MARKER_COUNT_OPTIONS = (1, 2, 3)
GAP_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (5, 10)
INNER_WIDTH_RANGE = (5, 10)
TOTAL_MARGIN_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 300


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return set(coords) == _frame_cells(bbox)


def _parse_frames_and_markers(grid: np.ndarray) -> tuple[list[dict[str, object]], list[tuple[int, int]], int]:
    bg = _background(grid)
    non_bg_colors = [int(color) for color in np.unique(grid) if int(color) != bg]

    frames: list[dict[str, object]] = []
    marker_components: list[list[tuple[int, int]]] = []
    for color in non_bg_colors:
        for comp in _components_mask(grid == color):
            if _is_rect_frame(comp):
                frames.append(
                    {
                        "color": int(color),
                        "bbox": _bbox(comp),
                        "cells": set(comp),
                    }
                )
            else:
                marker_components.append(comp)

    assert 3 <= len(frames) <= 4, "expected 3 or 4 nested rectangular frames"
    assert marker_components, "expected marker pixels inside the innermost frame"
    assert all(len(comp) == 1 for comp in marker_components), "markers must be singleton pixels"

    frames.sort(key=lambda frame: frame["bbox"])  # outer frames have smaller top-left corners
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "frames must be strictly nested"

    marker_points = [comp[0] for comp in marker_components]
    marker_colors = {int(grid[r, c]) for r, c in marker_points}
    assert len(marker_colors) == 1, "all markers must share one color"
    marker_color = next(iter(marker_colors))

    ir0, ic0, ir1, ic1 = frames[-1]["bbox"]
    for r, c in marker_points:
        assert ir0 < r < ir1 and ic0 < c < ic1, "markers must be inside the innermost frame"

    return frames, marker_points, marker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill odd-depth frame annuli with checkerboards and project the innermost markers outward to the first odd border they hit.

    Input structure:
    - A black background contains 3 or 4 one-pixel-thick nested rectangular frames, each in a different color.
    - Consecutive frames are separated by at least a two-cell empty gap, and the deepest frame encloses 1-3 singleton marker pixels.
    - The marker pixels all share one color and no other non-frame objects are present.

    Rule:
    - Number the frames from outermost to innermost.
    - For every odd-numbered frame that has a next inner frame, fill the annulus between the two frames with that odd frame's color in a checkerboard pattern, starting with a colored cell at the annulus top-left corner.
    - Leave every even-numbered annulus black, and do not fill the interior of the innermost frame just because it is odd-numbered.
    - For each innermost marker, extend four cardinal rays outward, repainting every crossed cell until the first odd-numbered frame border in that direction is reached.

    Key insight:
    - The same parity test controls both subproblems: odd layers are the only ones that create checkerboard annuli, and odd borders are also the only ray terminals.
    """
    g = np.asarray(grid, dtype=int)
    frames, marker_points, marker_color = _parse_frames_and_markers(g)
    bg = _background(g)
    out = np.array(g, copy=True)

    for layer_idx, (outer, inner) in enumerate(zip(frames, frames[1:]), start=1):
        if layer_idx % 2 == 0:
            continue
        r0, c0, r1, c1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        anchor_r, anchor_c = r0 + 1, c0 + 1
        color = int(outer["color"])

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                    continue
                if (r - anchor_r + c - anchor_c) % 2 == 0:
                    out[r, c] = color
                else:
                    out[r, c] = bg

    odd_border_cells: set[tuple[int, int]] = set()
    for layer_idx, frame in enumerate(frames, start=1):
        if layer_idx % 2 == 1:
            odd_border_cells.update(frame["cells"])

    for mr, mc in marker_points:
        for dr, dc in DIRS4:
            r, c = mr, mc
            while True:
                r += dr
                c += dc
                assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "ray must hit an odd border before leaving the grid"
                out[r, c] = marker_color
                if (r, c) in odd_border_cells:
                    break

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _sample_marker_points(
    bbox: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if len(cells) < count:
        return None

    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    want_unique_rows = interior_h >= count
    want_unique_cols = interior_w >= count

    for _ in range(80):
        choice = random.sample(cells, count)
        if want_unique_rows and len({r for r, _ in choice}) < count:
            continue
        if want_unique_cols and len({c for _, c in choice}) < count:
            continue
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for i, (r1, c1) in enumerate(choice) for r2, c2 in choice[i + 1 :]):
            continue
        return choice

    return random.sample(cells, count)


def _sample_bboxes() -> tuple[list[tuple[int, int, int, int]], int, int] | None:
    layers = random.choice(LAYER_OPTIONS)
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)
    gaps = [random.randint(*GAP_RANGE) for _ in range(layers - 1)]
    deltas = [gap + 1 for gap in gaps]
    pad = sum(deltas)

    outer_h = inner_h + 2 * pad
    outer_w = inner_w + 2 * pad
    if outer_h > 28 or outer_w > 28:
        return None

    total_v_margin = random.randint(*TOTAL_MARGIN_RANGE)
    total_h_margin = random.randint(*TOTAL_MARGIN_RANGE)
    grid_h = outer_h + total_v_margin
    grid_w = outer_w + total_h_margin
    if grid_h > 30 or grid_w > 30:
        return None

    top = random.randint(1, total_v_margin - 1)
    left = random.randint(1, total_h_margin - 1)
    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1

    bboxes = [(r0, c0, r1, c1)]
    for delta in deltas:
        r0 += delta
        c0 += delta
        r1 -= delta
        c1 -= delta
        bboxes.append((r0, c0, r1, c1))
    return bboxes, grid_h, grid_w


def _expected_marker_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == MARKER_COLOR))


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_bboxes()
    if sampled is None:
        return None
    bboxes, h, w = sampled

    grid = np.full((h, w), BG_COLOR, dtype=int)

    frame_colors = random.sample(FRAME_COLOR_POOL, len(bboxes))
    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    marker_count = random.choice(MARKER_COUNT_OPTIONS)
    marker_points = _sample_marker_points(bboxes[-1], marker_count)
    if marker_points is None:
        return None
    for r, c in marker_points:
        grid[r, c] = MARKER_COLOR

    if _background(grid) != BG_COLOR:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if _expected_marker_count(grid) != marker_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame projection input")
