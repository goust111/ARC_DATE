from __future__ import annotations

import random

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4

ASYMMETRIC = 0
SINGLE_AXIS = 1
DUAL_AXIS = 2

OUTLINE_COLORS = {
    ASYMMETRIC: RED,
    SINGLE_AXIS: YELLOW,
    DUAL_AXIS: GREEN,
}

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _color_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = np.array([r for r, _ in coords], dtype=int)
    cols = np.array([c for _, c in coords], dtype=int)
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def _classify_mask(mask: np.ndarray) -> int:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and vertical:
        return DUAL_AXIS
    if horizontal or vertical:
        return SINGLE_AXIS
    return ASYMMETRIC


def _paint_outline(
    out: np.ndarray,
    inp: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
) -> None:
    h, w = inp.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)

    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if int(inp[r, c]) == bg:
                out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each monochrome 4-connected object's outer bbox ring by symmetry type.

    Asymmetric objects get a red ring, single-axis symmetric objects get a
    yellow ring, and objects symmetric across both horizontal and vertical
    axes get a green ring. Original pixels are preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)

    comps = _color_components(g, bg)
    assert comps

    out = g.copy()
    for coords in comps:
        mask, bbox = _coords_to_mask(coords)
        kind = _classify_mask(mask)
        _paint_outline(out, g, bbox, OUTLINE_COLORS[kind], bg)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (3, 5)
PLACEMENT_ATTEMPTS = 250
SHAPE_ATTEMPTS = 200
GEN_ATTEMPTS = 400

INPUT_COLORS = (0, 1, 5, 6, 7, 8, 9)
TYPE_BOUNDS = {
    ASYMMETRIC: {"rows": (2, 6), "cols": (2, 6), "cells": (3, 12)},
    SINGLE_AXIS: {"rows": (2, 7), "cols": (2, 7), "cells": (4, 14)},
    DUAL_AXIS: {"rows": (3, 7), "cols": (3, 7), "cells": (5, 15)},
}


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_bool_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.35:
        out = out.T
    return np.array(out, dtype=bool, copy=True)


def _grow_connected_cells(height: int, width: int, target: int) -> set[tuple[int, int]]:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in NEIGHBORS4
        if 0 <= seed[0] + dr < height and 0 <= seed[1] + dc < width
    }

    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    return cells


def _symmetry_orbits(height: int, width: int, keep_h: bool, keep_v: bool) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []

    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            if keep_h:
                orbit.add((height - 1 - r, c))
            if keep_v:
                orbit.add((r, width - 1 - c))
            if keep_h and keep_v:
                orbit.add((height - 1 - r, width - 1 - c))
            frozen = tuple(sorted(orbit))
            out.append(frozen)
            seen.update(frozen)

    return out


def _orbit_adjacency(orbits: list[tuple[tuple[int, int], ...]]) -> list[set[int]]:
    adj = [set() for _ in orbits]
    for i in range(len(orbits)):
        for j in range(i + 1, len(orbits)):
            touching = False
            for r0, c0 in orbits[i]:
                for r1, c1 in orbits[j]:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        touching = True
                        break
                if touching:
                    break
            if touching:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _grow_from_orbits(
    height: int,
    width: int,
    keep_h: bool,
    keep_v: bool,
    min_cells: int,
    max_cells: int,
    kind: int,
) -> np.ndarray | None:
    orbits = _symmetry_orbits(height, width, keep_h, keep_v)
    adj = _orbit_adjacency(orbits)

    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(min_cells, max_cells)
        seed = random.randrange(len(orbits))
        chosen = {seed}
        area = len(orbits[seed])
        frontier = set(adj[seed])

        while True:
            candidates = [idx for idx in frontier if area + len(orbits[idx]) <= max_cells]
            if area >= target and (not candidates or random.random() < 0.45):
                break
            if not candidates:
                break
            idx = random.choice(candidates)
            chosen.add(idx)
            area += len(orbits[idx])
            frontier.update(adj[idx])
            frontier.difference_update(chosen)

        mask = np.zeros((height, width), dtype=bool)
        for idx in chosen:
            for r, c in orbits[idx]:
                mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < min_cells or int(mask.sum()) > max_cells:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if not _mask_connected(mask):
            continue
        if _classify_mask(mask) != kind:
            continue
        return _random_transform(mask)

    return None


def _make_asymmetric_mask() -> np.ndarray | None:
    bounds = TYPE_BOUNDS[ASYMMETRIC]
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*bounds["rows"])
        width = random.randint(*bounds["cols"])
        max_cells = min(bounds["cells"][1], height * width - 1)
        min_cells = min(bounds["cells"][0], max_cells)
        if max_cells < min_cells:
            continue
        target = random.randint(min_cells, max_cells)
        cells = _grow_connected_cells(height, width, target)
        mask = _cells_to_bool_mask(cells)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if _classify_mask(mask) != ASYMMETRIC:
            continue
        return _random_transform(mask)
    return None


def _make_single_axis_mask() -> np.ndarray | None:
    bounds = TYPE_BOUNDS[SINGLE_AXIS]
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*bounds["rows"])
        width = random.randint(*bounds["cols"])
        max_cells = min(bounds["cells"][1], height * width - 1)
        min_cells = min(bounds["cells"][0], max_cells)
        if max_cells < min_cells:
            continue
        axis = random.choice(("h", "v"))
        mask = _grow_from_orbits(
            height,
            width,
            keep_h=axis == "h",
            keep_v=axis == "v",
            min_cells=min_cells,
            max_cells=max_cells,
            kind=SINGLE_AXIS,
        )
        if mask is not None:
            return mask
    return None


def _make_dual_axis_mask() -> np.ndarray | None:
    bounds = TYPE_BOUNDS[DUAL_AXIS]
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*bounds["rows"])
        width = random.randint(*bounds["cols"])
        max_cells = min(bounds["cells"][1], height * width - 1)
        min_cells = min(bounds["cells"][0], max_cells)
        if max_cells < min_cells:
            continue
        mask = _grow_from_orbits(
            height,
            width,
            keep_h=True,
            keep_v=True,
            min_cells=min_cells,
            max_cells=max_cells,
            kind=DUAL_AXIS,
        )
        if mask is not None:
            return mask
    return None


def _make_mask(kind: int) -> np.ndarray | None:
    if kind == ASYMMETRIC:
        return _make_asymmetric_mask()
    if kind == SINGLE_AXIS:
        return _make_single_axis_mask()
    return _make_dual_axis_mask()


def _sample_kinds(count: int) -> list[int]:
    if random.random() < 0.65:
        kinds = [ASYMMETRIC, SINGLE_AXIS, DUAL_AXIS]
        kinds.extend(random.choice((ASYMMETRIC, SINGLE_AXIS, DUAL_AXIS)) for _ in range(count - 3))
    else:
        kinds = random.sample((ASYMMETRIC, SINGLE_AXIS, DUAL_AXIS), 2)
        kinds.extend(random.choice((ASYMMETRIC, SINGLE_AXIS, DUAL_AXIS)) for _ in range(count - 2))
    random.shuffle(kinds)
    return kinds


def _place_masks(grid_shape: tuple[int, int], masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    forbidden = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        if mh + 2 > h or mw + 2 > w:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, h - mh - 1)
            c0 = random.randint(1, w - mw - 1)
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(h, r0 + mh + 2)
            cc1 = min(w, c0 + mw + 2)
            if forbidden[rr0:rr1, cc0:cc1].any():
                continue
            forbidden[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            break
        else:
            return None

    return [pos for pos in placements if pos is not None]


def _render_input(
    height: int,
    width: int,
    bg: int,
    colors: list[int],
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, mask, (r0, c0) in zip(colors, masks, placements, strict=True):
        mh, mw = mask.shape
        patch = grid[r0 : r0 + mh, c0 : c0 + mw]
        patch[mask] = color
    return grid


def _expected_output(
    inp: np.ndarray,
    bg: int,
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
    kinds: list[int],
) -> np.ndarray:
    out = inp.copy()
    for mask, (r0, c0), kind in zip(masks, placements, kinds, strict=True):
        mh, mw = mask.shape
        bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
        _paint_outline(out, inp, bbox, OUTLINE_COLORS[kind], bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(INPUT_COLORS)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample([c for c in INPUT_COLORS if c != bg], count)
    planned_kinds = _sample_kinds(count)

    masks: list[np.ndarray] = []
    actual_kinds: list[int] = []
    for kind in planned_kinds:
        mask = _make_mask(kind)
        if mask is None:
            return None
        actual = _classify_mask(mask)
        if actual != kind:
            return None
        masks.append(mask)
        actual_kinds.append(actual)

    placements = _place_masks((height, width), masks)
    if placements is None:
        return None

    inp = _render_input(height, width, bg, colors, masks, placements)
    comps = _color_components(inp, bg)
    if len(comps) != count:
        return None
    if _dominant_color(inp) != bg:
        return None

    expected = _expected_output(inp, bg, masks, placements, actual_kinds)
    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-outline input")
