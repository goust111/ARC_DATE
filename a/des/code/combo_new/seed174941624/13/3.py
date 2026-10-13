from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRECTIONS = ("up", "down", "left", "right")
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MIN_REGION_SIZE = 6
MAX_REGION_SIZE = 12
MAX_GENERATION_ATTEMPTS = 300
MAX_REGION_ATTEMPTS = 120

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((c, -r) for r, c in cells))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((r, -c) for r, c in cells))


def _build_shapes() -> list[dict]:
    variants: dict[tuple[tuple[int, int], ...], dict] = {}
    for base in BASE_SHAPES:
        shape = _normalize_shape(base)
        for _ in range(4):
            for candidate in (shape, _flip_shape(shape)):
                if candidate not in variants:
                    max_r = max(r for r, _ in candidate)
                    max_c = max(c for _, c in candidate)
                    variants[candidate] = {
                        "cells": candidate,
                        "height": max_r + 1,
                        "width": max_c + 1,
                    }
            shape = _rotate_shape(shape)
    return list(variants.values())


SHAPES = _build_shapes()


def _quadrants(h_sep: int, v_sep: int, height: int, width: int) -> list[tuple[str, slice, slice]]:
    return [
        ("tl", slice(0, h_sep), slice(0, v_sep)),
        ("tr", slice(0, h_sep), slice(v_sep + 1, width)),
        ("bl", slice(h_sep + 1, height), slice(0, v_sep)),
        ("br", slice(h_sep + 1, height), slice(v_sep + 1, width)),
    ]


def _marker_cells(region_h: int, region_w: int, quadrant: str, direction: str) -> tuple[tuple[int, int], ...]:
    top_half = quadrant in ("tl", "tr")
    left_half = quadrant in ("tl", "bl")

    if direction == "up":
        row = 0
        col = 0 if left_half else region_w - 2
        return ((row, col), (row, col + 1))
    if direction == "down":
        row = region_h - 1
        col = 0 if left_half else region_w - 2
        return ((row, col), (row, col + 1))
    if direction == "left":
        col = 0
        row = 0 if top_half else region_h - 2
        return ((row, col), (row + 1, col))

    col = region_w - 1
    row = 0 if top_half else region_h - 2
    return ((row, col), (row + 1, col))


def _inflate(cells: tuple[tuple[int, int], ...], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    mask[rr, cc] = True
    return mask


def _extract_components(region: np.ndarray) -> list[dict]:
    height, width = region.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(region[r, c])
            if seen[r, c] or color in (BLACK, GREY, MAROON):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(region[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def _front_coord(bbox: tuple[int, int, int, int], direction: str) -> int:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return r0
    if direction == "down":
        return r1
    if direction == "left":
        return c0
    return c1


def _orth_interval_bbox(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction in ("up", "down"):
        return (c0, c1)
    return (r0, r1)


def _orth_interval_cells(cells: tuple[tuple[int, int], ...], direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        values = [c for _, c in cells]
    else:
        values = [r for r, _ in cells]
    return (min(values), max(values))


def _front_gap_cells(cells: tuple[tuple[int, int], ...], direction: str, height: int, width: int) -> int:
    if direction == "up":
        return min(r for r, _ in cells)
    if direction == "down":
        return height - 1 - max(r for r, _ in cells)
    if direction == "left":
        return min(c for _, c in cells)
    return width - 1 - max(c for _, c in cells)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _has_ambiguous_front_tie(components: list[dict], direction: str) -> bool:
    for i, first in enumerate(components):
        for second in components[i + 1 :]:
            if _front_coord(first["bbox"], direction) != _front_coord(second["bbox"], direction):
                continue
            if _intervals_overlap(
                _orth_interval_bbox(first["bbox"], direction),
                _orth_interval_bbox(second["bbox"], direction),
            ):
                return True
    return False


def _settle_key(component: dict, direction: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    if direction == "up":
        return (r0, c0, r1, c1)
    if direction == "down":
        return (-r1, c0, -r0, c1)
    if direction == "left":
        return (c0, r0, c1, r1)
    return (-c1, r0, -c0, r1)


def _settle_region(region: np.ndarray, direction: str, components: list[dict] | None = None) -> tuple[np.ndarray, list[dict]]:
    patch = np.asarray(region, dtype=int)
    height, width = patch.shape
    comps = _extract_components(patch) if components is None else components
    occupied = np.zeros((height, width), dtype=bool)
    out = np.full((height, width), BLACK, dtype=int)
    step = np.array(DIR_STEPS[direction], dtype=int)
    infos: list[dict] = []

    for component in sorted(comps, key=lambda comp: _settle_key(comp, direction)):
        coords = component["coords"].astype(int).copy()
        shift = 0
        stop_reason = "boundary"

        while True:
            nxt = coords + step
            if (
                np.any(nxt[:, 0] < 0)
                or np.any(nxt[:, 0] >= height)
                or np.any(nxt[:, 1] < 0)
                or np.any(nxt[:, 1] >= width)
            ):
                stop_reason = "boundary"
                break
            if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                stop_reason = "object"
                break
            coords = nxt
            shift += 1

        out[coords[:, 0], coords[:, 1]] = int(component["color"])
        occupied[coords[:, 0], coords[:, 1]] = True
        infos.append(
            {
                "color": int(component["color"]),
                "coords": coords,
                "shift": shift,
                "reason": stop_reason,
            }
        )

    return out, infos


def _marker_direction(region: np.ndarray) -> str:
    coords = np.argwhere(region == MAROON)
    height, width = region.shape
    assert coords.shape[0] == 2, "each quadrant must contain exactly one 2-cell marker"

    rows = {int(r) for r, _ in coords}
    cols = {int(c) for _, c in coords}
    if len(rows) == 1:
        row = next(iter(rows))
        assert abs(int(coords[0, 1]) - int(coords[1, 1])) == 1
        if row == 0:
            return "up"
        assert row == height - 1
        return "down"

    assert len(cols) == 1
    col = next(iter(cols))
    assert abs(int(coords[0, 0]) - int(coords[1, 0])) == 1
    if col == 0:
        return "left"
    assert col == width - 1
    return "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each quadrant's colored objects toward a local gravity edge encoded by a small maroon bar.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column divide the canvas into four rectangular quadrants.
    - Each quadrant contains exactly one maroon 2-cell bar on one of its sides; the touched side encodes that quadrant's gravity direction.
    - Each quadrant also contains several monochrome 4-connected colored objects, and those objects never use black, grey, or maroon.

    Rule:
    - Detect the separator row and column, then process the four quadrants independently.
    - Read each maroon bar: a horizontal bar on the top or bottom means gravity up or down, while a vertical bar on the left or right means gravity left or right.
    - Erase the maroon marker and slide every colored object rigidly along that direction.
    - Move the frontmost object first; each object stops when one more step would leave the quadrant or collide with an already settled object.
    - Keep the grey separator lines unchanged.

    Key insight:
    - The puzzle is four separate gravity chambers, and the small maroon corner bars are direction codes rather than movable content.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    grey_rows = [r for r in range(height) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(width) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1
    h_sep = grey_rows[0]
    v_sep = grey_cols[0]

    out = g.copy()
    seen_directions: set[str] = set()
    for _name, row_slice, col_slice in _quadrants(h_sep, v_sep, height, width):
        region = g[row_slice, col_slice]
        direction = _marker_direction(region)
        seen_directions.add(direction)
        settled, _infos = _settle_region(region, direction)
        out[row_slice, col_slice] = settled

    assert len(seen_directions) == 4
    return out


def _candidate_positions(shape: dict, height: int, width: int, blocked: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    positions: list[tuple[tuple[int, int], ...]] = []
    for top in range(height - shape["height"] + 1):
        for left in range(width - shape["width"] + 1):
            cells = tuple((top + dr, left + dc) for dr, dc in shape["cells"])
            if any(blocked[r, c] for r, c in cells):
                continue
            positions.append(cells)
    return positions


def _generate_region(region_h: int, region_w: int, quadrant: str, direction: str, require_chain: bool) -> np.ndarray | None:
    marker = _marker_cells(region_h, region_w, quadrant, direction)
    base = np.full((region_h, region_w), BLACK, dtype=int)
    for r, c in marker:
        base[r, c] = MAROON
    base_blocked = _inflate(marker, region_h, region_w)
    fitting_shapes = [shape for shape in SHAPES if shape["height"] <= region_h and shape["width"] <= region_w]

    count_choices = (2, 3) if require_chain else (1, 2, 3)
    for _ in range(MAX_REGION_ATTEMPTS):
        patch = base.copy()
        blocked = base_blocked.copy()
        object_count = random.choice(count_choices)
        colors = random.sample(OBJECT_COLORS, object_count)
        placed_intervals: list[tuple[int, int]] = []
        chain_lane: int | None = None
        ok = True

        for index in range(object_count):
            shape = random.choice(fitting_shapes)
            candidates = _candidate_positions(shape, region_h, region_w, blocked)
            if not candidates:
                ok = False
                break

            movable = [
                cells
                for cells in candidates
                if _front_gap_cells(cells, direction, region_h, region_w) >= 1
            ]
            if movable:
                candidates = movable

            if require_chain and chain_lane is not None:
                overlap_candidates = [
                    cells
                    for cells in candidates
                    if _orth_interval_cells(cells, direction)[0]
                    <= chain_lane
                    <= _orth_interval_cells(cells, direction)[1]
                ]
                if overlap_candidates:
                    candidates = overlap_candidates

            if not require_chain and placed_intervals:
                disjoint = [
                    cells
                    for cells in candidates
                    if all(
                        not _intervals_overlap(_orth_interval_cells(cells, direction), interval)
                        for interval in placed_intervals
                    )
                ]
                if disjoint and random.random() < 0.8:
                    candidates = disjoint

            cells = random.choice(candidates)
            for r, c in cells:
                patch[r, c] = colors[index]
            blocked |= _inflate(cells, region_h, region_w)
            interval = _orth_interval_cells(cells, direction)
            placed_intervals.append(interval)
            if require_chain and chain_lane is None:
                chain_lane = random.randint(interval[0], interval[1])

        if not ok:
            continue

        components = _extract_components(patch)
        if len(components) != object_count:
            continue
        if _has_ambiguous_front_tie(components, direction):
            continue

        settled, infos = _settle_region(patch, direction, components)
        if sum(int(info["shift"] > 0) for info in infos) == 0:
            continue
        if require_chain and not any(info["reason"] == "object" for info in infos):
            continue
        if np.array_equal(patch, settled):
            continue
        return patch

    return None


def _sample_region_sizes() -> tuple[int, int, int, int]:
    top_h = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    bottom_h = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    left_w = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    right_w = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    return top_h, bottom_h, left_w, right_w


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        top_h, bottom_h, left_w, right_w = _sample_region_sizes()
        height = top_h + 1 + bottom_h
        width = left_w + 1 + right_w
        if height > 30 or width > 30:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        h_sep = top_h
        v_sep = left_w
        grid[h_sep, :] = GREY
        grid[:, v_sep] = GREY

        assignments = dict(
            zip(
                ("tl", "tr", "bl", "br"),
                random.sample(DIRECTIONS, len(DIRECTIONS)),
            )
        )
        chain_regions = set(random.sample(("tl", "tr", "bl", "br"), 2))

        built = True
        for quadrant, row_slice, col_slice in _quadrants(h_sep, v_sep, height, width):
            direction = assignments[quadrant]
            require_chain = quadrant in chain_regions
            region = _generate_region(
                row_slice.stop - row_slice.start,
                col_slice.stop - col_slice.start,
                quadrant,
                direction,
                require_chain,
            )
            if region is None:
                built = False
                break
            grid[row_slice, col_slice] = region

        if not built:
            continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue
        if np.any(out == MAROON):
            continue
        if not np.all(out[h_sep, :] == GREY) or not np.all(out[:, v_sep] == GREY):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
