from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(cells: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "size": len(cells),
                    "bbox": _bbox(cells),
                }
            )
    return comps


def _square_size(comp: dict) -> int | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w:
        return None
    if comp["size"] != h * w:
        return None
    if h not in (1, 2, 3):
        return None
    return h


def _infer_scene(grid: np.ndarray) -> tuple[int, dict, Coord, list[dict]]:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)

    candidates: list[tuple[int, int, int]] = []
    for anchor_idx, anchor_comp in enumerate(comps):
        if anchor_comp["size"] != 1:
            continue
        ar, ac = anchor_comp["cells"][0]
        touch_counts: Counter[int] = Counter()
        for nr, nc in _neighbors4(ar, ac):
            if _in_bounds(grid.shape, nr, nc):
                neighbor_color = int(grid[nr, nc])
                if neighbor_color != bg and neighbor_color != anchor_comp["color"]:
                    touch_counts[neighbor_color] += 1
        for source_idx, source_comp in enumerate(comps):
            if source_comp["size"] <= 1:
                continue
            contact = touch_counts[source_comp["color"]]
            if contact > 0:
                candidates.append((contact, anchor_idx, source_idx))

    assert candidates, "expected one singleton anchor touching the source"
    best_contact = max(contact for contact, _, _ in candidates)
    best = [(a, s) for contact, a, s in candidates if contact == best_contact]
    assert len(best) == 1, "anchor/source assignment must be unique"
    anchor_idx, source_idx = best[0]
    anchor_comp = comps[anchor_idx]
    source_comp = comps[source_idx]
    anchor = anchor_comp["cells"][0]

    markers: list[dict] = []
    seen_colors: set[int] = set()
    for idx, comp in enumerate(comps):
        if idx in (anchor_idx, source_idx):
            continue
        size = _square_size(comp)
        assert size is not None, "every non-source object must be a 1x1, 2x2, or 3x3 solid square"
        assert comp["color"] not in seen_colors, "marker colors must be unique"
        seen_colors.add(comp["color"])
        marker = dict(comp)
        marker["square_size"] = size
        markers.append(marker)

    assert len(markers) >= 3, "expected several marker blocks"
    return bg, source_comp, anchor, markers


def _relative_stamp_cells(offsets: list[Coord], scale: int) -> list[Coord]:
    cells: set[Coord] = set()
    for dr, dc in offsets:
        base_r = scale * dr
        base_c = scale * dc
        for rr in range(scale):
            for cc in range(scale):
                cells.add((base_r + rr, base_c + cc))
    return sorted(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp scaled, recolored copies of a source shape from its anchor onto square markers.

    Input structure:
    - A mostly empty background contains one monochrome multi-pixel source shape and one isolated anchor pixel in a different color.
    - The anchor is the unique singleton that has the strongest 4-neighbor contact with the source color.
    - Elsewhere, several solid square marker blocks of side 1, 2, or 3 use distinct colors different from the source and anchor.

    Rule:
    - Record every source pixel as a coordinate offset from the anchor.
    - For each marker block of side k, take the marker's top-left cell as a new origin and multiply every recorded offset by k.
    - At each scaled offset, draw a solid k-by-k block in the marker's own color, then erase the original source, anchor, and marker blocks.

    Key insight:
    - Each marker block encodes three instructions at once: its top-left corner is the stamp origin, its side length is the scale factor, and its color is the recolor target.
    """
    g = np.asarray(grid, dtype=int)
    bg, source_comp, anchor, markers = _infer_scene(g)
    offsets = sorted((r - anchor[0], c - anchor[1]) for r, c in source_comp["cells"])

    out = np.full_like(g, bg)
    for marker in markers:
        scale = marker["square_size"]
        r0, c0, _, _ = marker["bbox"]
        for dr, dc in _relative_stamp_cells(offsets, scale):
            rr = r0 + dr
            cc = c0 + dc
            assert _in_bounds(g.shape, rr, cc), "stamp must stay in bounds"
            if out[rr, cc] not in (bg, marker["color"]):
                raise AssertionError("different marker copies must not overlap")
            out[rr, cc] = marker["color"]
    return out


_BG = 0
_ALL_COLORS = tuple(range(1, 10))
_GRID_MIN = 22
_GRID_MAX = 30
_MARKER_COUNT_CHOICES = (3, 4)
_MARKER_SIZES = (1, 2, 3)
_SOURCE_DIM_RANGE = (3, 5)
_GEN_ATTEMPTS = 400


def _reserve_cells(mask: np.ndarray, cells: list[Coord] | set[Coord], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                mask[rr, cc] = True


def _try_place_relative(mask: np.ndarray, rel_cells: list[Coord], pad: int = 1) -> tuple[int, int] | None:
    min_r = min(r for r, _ in rel_cells)
    max_r = max(r for r, _ in rel_cells)
    min_c = min(c for _, c in rel_cells)
    max_c = max(c for _, c in rel_cells)
    candidates = [
        (r0, c0)
        for r0 in range(-min_r, mask.shape[0] - max_r)
        for c0 in range(-min_c, mask.shape[1] - max_c)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        placed = [(r0 + r, c0 + c) for r, c in rel_cells]
        if any(mask[r, c] for r, c in placed):
            continue
        _reserve_cells(mask, placed, pad=pad)
        return r0, c0
    return None


def _grow_connected_shape(height: int, width: int, target: int) -> set[Coord]:
    cells: set[Coord] = {(0, 1), (1, 0), (1, 1)}
    while len(cells) < target:
        frontier: set[Coord] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) != (0, 0) and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        frontier_list = sorted(frontier)
        weights = [1 + (r == height - 1) + (c == width - 1) for r, c in frontier_list]
        cells.add(random.choices(frontier_list, weights=weights, k=1)[0])
    return cells


def _generate_source_offsets() -> list[Coord]:
    for _ in range(200):
        height = random.randint(*_SOURCE_DIM_RANGE)
        width = random.randint(*_SOURCE_DIM_RANGE)
        max_target = min(height * width - 1, 8)
        target = random.randint(5, max_target)
        cells = _grow_connected_shape(height, width, target)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or min(cols) != 0 or max(rows) != height - 1 or max(cols) != width - 1:
            continue
        if len(cells) == height * width - 1:
            continue
        if all(r in (0, height - 1) or c in (0, width - 1) for r, c in cells):
            continue
        sign_r, sign_c = random.choice(((1, 1), (1, -1), (-1, 1), (-1, -1)))
        offsets = sorted((sign_r * r, sign_c * c) for r, c in cells)
        return offsets
    raise RuntimeError("failed to generate a non-degenerate source shape")


def _build_expected_output(
    shape: tuple[int, int],
    offsets: list[Coord],
    markers: list[tuple[int, Coord, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for scale, origin, color in markers:
        for dr, dc in _relative_stamp_cells(offsets, scale):
            rr = origin[0] + dr
            cc = origin[1] + dc
            if out[rr, cc] not in (0, color):
                raise AssertionError("marker copies overlap")
            out[rr, cc] = color
    return out


def _place_markers_and_output(
    grid_shape: tuple[int, int], offsets: list[Coord], marker_specs: list[tuple[int, int]]
) -> tuple[list[tuple[int, Coord, int]], np.ndarray] | None:
    reserved = np.zeros(grid_shape, dtype=bool)
    markers: list[tuple[int, Coord, int]] = []
    for scale, color in sorted(marker_specs, reverse=True):
        rel_cells = _relative_stamp_cells(offsets, scale)
        origin = _try_place_relative(reserved, rel_cells, pad=1)
        if origin is None:
            return None
        markers.append((scale, origin, color))
    out = _build_expected_output(grid_shape, offsets, markers)
    return markers, out


def _place_source(
    grid_shape: tuple[int, int], offsets: list[Coord], blocked: np.ndarray
) -> Coord | None:
    rel_cells = [(0, 0)] + offsets
    return _try_place_relative(blocked, rel_cells, pad=1)


def _render_input(
    grid_shape: tuple[int, int],
    source_color: int,
    anchor_color: int,
    anchor: Coord,
    offsets: list[Coord],
    markers: list[tuple[int, Coord, int]],
) -> np.ndarray:
    grid = np.full(grid_shape, _BG, dtype=int)
    grid[anchor] = anchor_color
    for dr, dc in offsets:
        grid[anchor[0] + dr, anchor[1] + dc] = source_color
    for scale, (r0, c0), color in markers:
        grid[r0 : r0 + scale, c0 : c0 + scale] = color
    return grid


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    offsets: list[Coord],
    marker_specs: list[tuple[int, Coord, int]],
) -> bool:
    try:
        bg, source_comp, anchor, markers = _infer_scene(inp)
    except AssertionError:
        return False

    if bg != _BG:
        return False
    inferred_offsets = sorted((r - anchor[0], c - anchor[1]) for r, c in source_comp["cells"])
    if inferred_offsets != sorted(offsets):
        return False

    inferred_sizes = sorted(marker["square_size"] for marker in markers)
    required_sizes = sorted(scale for scale, _, _ in marker_specs)
    if inferred_sizes != required_sizes:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    if np.any(np.isin(expected, [int(inp[anchor]), source_comp["color"]])):
        return False

    source_area = len(offsets)
    for scale, _origin, color in marker_specs:
        if int(np.count_nonzero(expected == color)) != source_area * scale * scale:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        offsets = _generate_source_offsets()
        marker_count = random.choice(_MARKER_COUNT_CHOICES)
        sizes = [1, 2, 3]
        if marker_count == 4:
            sizes.append(random.choice(_MARKER_SIZES))

        color_count = 2 + marker_count
        colors = random.sample(_ALL_COLORS, color_count)
        source_color = colors[0]
        anchor_color = colors[1]
        marker_colors = colors[2:]
        marker_specs_raw = list(zip(sizes, marker_colors))

        grid_shape = (
            random.randint(_GRID_MIN, _GRID_MAX),
            random.randint(_GRID_MIN, _GRID_MAX),
        )
        placed = _place_markers_and_output(grid_shape, offsets, marker_specs_raw)
        if placed is None:
            continue
        markers, expected = placed

        blocked = np.zeros(grid_shape, dtype=bool)
        for scale, origin, _color in markers:
            marker_cells = [(origin[0] + dr, origin[1] + dc) for dr in range(scale) for dc in range(scale)]
            _reserve_cells(blocked, marker_cells, pad=1)
        anchor = _place_source(grid_shape, offsets, blocked)
        if anchor is None:
            continue

        inp = _render_input(grid_shape, source_color, anchor_color, anchor, offsets, markers)
        if _validate_generated(inp, expected, offsets, markers):
            return inp
    raise RuntimeError("failed to generate a valid input")
