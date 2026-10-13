from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
MARKER = 9
REGION_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS8 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
OBJECT_COUNT_CHOICES = (2, 3, 4)
COPY_COUNT_CHOICES = (1, 2, 3)
SHAPE_BBOX_H_RANGE = (3, 6)
SHAPE_BBOX_W_RANGE = (3, 6)
SHAPE_AREA_RANGE = (8, 18)
SHAPE_ATTEMPTS = 150
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rounded_mean(values: np.ndarray) -> int:
    return int(np.floor(float(values.mean()) + 0.5))


def _closest_direction(delta_r: int, delta_c: int) -> tuple[int, int] | None:
    if delta_r == 0 and delta_c == 0:
        return None

    best_dir: tuple[int, int] | None = None
    best_score = float("-inf")
    second_score = float("-inf")
    for dr, dc in DIRS8:
        score = (delta_r * dr + delta_c * dc) / ((dr * dr + dc * dc) ** 0.5)
        if score > best_score + 1e-9:
            second_score = best_score
            best_score = score
            best_dir = (dr, dc)
        elif score > second_score:
            second_score = score
    if best_score - second_score <= 1e-9:
        return None
    return best_dir


def _copy_positions(
    top: int,
    left: int,
    stamp_h: int,
    stamp_w: int,
    direction: tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]]:
    step_r = direction[0] * stamp_h
    step_c = direction[1] * stamp_w
    positions: list[tuple[int, int]] = []
    cur_r = top + step_r
    cur_c = left + step_c
    while 0 <= cur_r <= grid_h - stamp_h and 0 <= cur_c <= grid_w - stamp_w:
        positions.append((cur_r, cur_c))
        cur_r += step_r
        cur_c += step_c
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the maroon marker inside each region to choose a direction, then stamp the
    region's tight-cropped shape outward in that direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    components = []
    for coords in _components_from_mask(g != BACKGROUND):
        colors = Counter(int(g[r, c]) for r, c in coords)
        assert MARKER in colors, "each region must contain exactly one maroon marker"
        base_colors = [color for color in colors if color not in (BACKGROUND, MARKER)]
        assert len(base_colors) == 1, "each region must have one non-marker color"
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == MARKER]
        assert len(marker_cells) == 1, "each region must contain exactly one marker pixel"

        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, c0 = int(rs.min()), int(cs.min())
        r1, c1 = int(rs.max()), int(cs.max())
        components.append((r0, c0, r1, c1, coords, base_colors[0], marker_cells[0]))

    components.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    grid_h, grid_w = g.shape
    for r0, c0, r1, c1, coords, base_color, (mr, mc) in components:
        center_r = _rounded_mean(coords[:, 0])
        center_c = _rounded_mean(coords[:, 1])
        direction = _closest_direction(mr - center_r, mc - center_c)
        assert direction is not None, "marker must not sit on the rounded centroid"

        stamp_h = r1 - r0 + 1
        stamp_w = c1 - c0 + 1
        mask = np.zeros((stamp_h, stamp_w), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        for top, left in _copy_positions(r0, c0, stamp_h, stamp_w, direction, grid_h, grid_w):
            patch = out[top : top + stamp_h, left : left + stamp_w]
            fill_mask = mask & (patch == BACKGROUND)
            patch[fill_mask] = int(base_color)

    return out


def _marker_candidates(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    center_r = _rounded_mean(coords[:, 0])
    center_c = _rounded_mean(coords[:, 1])
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        direction = _closest_direction(int(r) - center_r, int(c) - center_c)
        if direction is None:
            continue
        candidates.setdefault(direction, []).append((int(r), int(c)))
    return candidates


def _best_marker(
    cells: list[tuple[int, int]],
    direction: tuple[int, int],
    mask: np.ndarray,
) -> tuple[int, int]:
    coords = np.argwhere(mask)
    center_r = _rounded_mean(coords[:, 0])
    center_c = _rounded_mean(coords[:, 1])

    def key_fn(cell: tuple[int, int]) -> tuple[int, int, int, int]:
        r, c = cell
        dr = r - center_r
        dc = c - center_c
        projection = dr * direction[0] + dc * direction[1]
        distance = abs(dr) + abs(dc)
        return (projection, distance, r, c)

    return max(cells, key=key_fn)


def _make_shape_for_direction(
    rng: random.Random,
    direction: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = rng.randint(*SHAPE_BBOX_H_RANGE)
        box_w = rng.randint(*SHAPE_BBOX_W_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        if max_area < SHAPE_AREA_RANGE[0]:
            continue
        target_area = rng.randint(SHAPE_AREA_RANGE[0], max_area)

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if int(mask.sum()) < SHAPE_AREA_RANGE[0]:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        if len(_components_from_mask(mask)) != 1:
            continue

        candidates = _marker_candidates(mask)
        if direction not in candidates:
            continue
        marker = _best_marker(candidates[direction], direction, mask)
        return mask, marker

    raise RuntimeError("failed to generate a directed region shape")


def _all_cells(
    local_cells: list[tuple[int, int]],
    top: int,
    left: int,
    copies: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for base_r, base_c in [(top, left), *copies]:
        for dr, dc in local_cells:
            cells.add((base_r + dr, base_c + dc))
    return cells


def _expand8(cells: set[tuple[int, int]], grid_h: int, grid_w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    expanded.add((nr, nc))
    return expanded


def _build_instance(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    if object_count > len(REGION_COLORS):
        return None

    colors = rng.sample(REGION_COLORS, object_count)
    directions = [rng.choice(DIRS8) for _ in range(object_count)]
    if object_count >= 2 and len(set(directions)) == 1:
        directions[0] = rng.choice([d for d in DIRS8 if d != directions[0]])

    specs = []
    for color, direction in zip(colors, directions):
        mask, marker = _make_shape_for_direction(rng, direction)
        specs.append(
            {
                "color": color,
                "direction": direction,
                "copies_target": rng.choice(COPY_COUNT_CHOICES),
                "mask": mask,
                "marker": marker,
                "local_cells": [tuple(map(int, cell)) for cell in np.argwhere(mask)],
            }
        )

    specs.sort(
        key=lambda spec: (
            spec["mask"].shape[0] * spec["mask"].shape[1] * (spec["copies_target"] + 1),
            len(spec["local_cells"]) * (spec["copies_target"] + 1),
        ),
        reverse=True,
    )

    occupied_output: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()
    total_expected_added = 0

    for spec in specs:
        mask = spec["mask"]
        stamp_h, stamp_w = mask.shape
        direction = spec["direction"]
        candidates = []
        for top in range(grid_h - stamp_h + 1):
            for left in range(grid_w - stamp_w + 1):
                copies = _copy_positions(top, left, stamp_h, stamp_w, direction, grid_h, grid_w)
                if len(copies) != spec["copies_target"]:
                    continue
                cells = _all_cells(spec["local_cells"], top, left, copies)
                if any(cell in forbidden_output for cell in cells):
                    continue
                candidates.append((top, left, copies, cells))

        if not candidates:
            return None

        top, left, copies, cells = rng.choice(candidates)
        spec["top"] = top
        spec["left"] = left
        spec["copies"] = copies
        spec["all_cells"] = cells

        occupied_output |= cells
        forbidden_output |= _expand8(cells, grid_h, grid_w)
        total_expected_added += len(spec["local_cells"]) * len(copies)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        color = spec["color"]
        for dr, dc in spec["local_cells"]:
            grid[top + dr, left + dc] = color
        mr, mc = spec["marker"]
        grid[top + mr, left + mc] = MARKER

    comps = _components_from_mask(grid != BACKGROUND)
    if len(comps) != object_count:
        return None

    out = main(grid)
    changed_bg = int(np.count_nonzero((grid == BACKGROUND) & (out != BACKGROUND)))
    if changed_bg != total_expected_added:
        return None
    if np.any(out[grid != BACKGROUND] != grid[grid != BACKGROUND]):
        return None
    if changed_bg == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
