from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
TOTAL_OBJECT_OPTIONS = (4, 5)
MAX_OBJECTS_PER_FAMILY = 3
AXIS_MARGIN = 6
AXIS_GAP2 = 4
MASK_RETRIES = 80
MASK_AREA_CAP = 12
PAIR_RETRIES = 60
OUTER_RETRIES = 500

NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return max(counts.items(), key=lambda item: item[1])[0]


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _edge_name(h: int, w: int, r: int, c: int) -> str:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("marker must lie on the border")


def _extract_marker_pairs(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    border = _border_mask(g.shape)
    colors = sorted(int(v) for v in np.unique(g[border]) if int(v) != int(bg))
    pairs: dict[int, dict[str, object]] = {}
    for color in colors:
        pts = [tuple(map(int, rc)) for rc in np.argwhere((g == int(color)) & border)]
        assert len(pts) == 2, "each marker color must appear exactly twice on the border"
        e1 = _edge_name(h, w, pts[0][0], pts[0][1])
        e2 = _edge_name(h, w, pts[1][0], pts[1][1])
        edges = {e1, e2}
        if edges == {"top", "bottom"}:
            kind = "tb"
            axis2 = int(pts[0][1] + pts[1][1])
        elif edges == {"left", "right"}:
            kind = "lr"
            axis2 = int(pts[0][0] + pts[1][0])
        else:
            raise AssertionError("marker pair must lie on opposite borders")
        pairs[int(color)] = {
            "color": int(color),
            "kind": kind,
            "axis2": int(axis2),
            "markers": tuple(sorted(pts)),
        }
    return pairs


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r0 in range(1, h - 1):
        for c0 in range(1, w - 1):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue
            color = int(g[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in NEIGH4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            objects.append({"color": color, "cells": tuple(sorted(cells))})
    return objects


def _reflect_cells(cells: tuple[tuple[int, int], ...], pair: dict[str, object]) -> tuple[tuple[int, int], ...]:
    axis2 = int(pair["axis2"])
    if pair["kind"] == "tb":
        return tuple(sorted((int(r), int(axis2 - c)) for r, c in cells))
    return tuple(sorted((int(axis2 - r), int(c)) for r, c in cells))


def _bbox_center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    center_r = (min(rows) + max(rows)) / 2.0
    center_c = (min(cols) + max(cols)) / 2.0
    return center_r, center_c


def _nearest_recolor(
    pair: dict[str, object],
    pairs: dict[int, dict[str, object]],
    reflected: tuple[tuple[int, int], ...],
) -> int:
    center_r, center_c = _bbox_center(reflected)
    if pair["kind"] == "tb":
        needed_kind = "lr"
        target_coord = center_r
    else:
        needed_kind = "tb"
        target_coord = center_c

    candidates = [
        other
        for other in pairs.values()
        if other["kind"] == needed_kind and int(other["color"]) != int(pair["color"])
    ]
    assert candidates, "need another marker family for recoloring"

    ranked = sorted(
        ((abs(int(other["axis2"]) / 2.0 - target_coord), int(other["color"]), other) for other in candidates),
        key=lambda item: item[0],
    )
    if len(ranked) > 1:
        assert abs(ranked[0][0] - ranked[1][0]) > 1e-9, "nearest recolor marker must be unique"
    return int(ranked[0][2]["color"])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    pairs = _extract_marker_pairs(g, bg)
    objects = _find_objects(g, bg)

    assert objects, "expected at least one interior object"
    assert any(pair["kind"] == "tb" for pair in pairs.values())
    assert any(pair["kind"] == "lr" for pair in pairs.values())

    out = g.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = bg

    target_occupancy: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for obj in objects:
        color = int(obj["color"])
        assert color in pairs, "each object color must match a border marker pair"
        pair = pairs[color]
        reflected = _reflect_cells(obj["cells"], pair)
        for r, c in reflected:
            assert 1 <= r < h - 1 and 1 <= c < w - 1, "reflected object must stay inside the interior"
        overlap = target_occupancy.intersection(reflected)
        assert not overlap, "reflected objects must not overlap"
        target_occupancy.update(reflected)
        placements.append((reflected, _nearest_recolor(pair, pairs, reflected)))

    for cells, new_color in placements:
        for r, c in cells:
            out[r, c] = int(new_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior object with the border marker pair of the same color,
    reflect the object across that pair's midpoint-defined axis, then recolor it
    using the nearest marker pair from the perpendicular border family.
    """
    return _solve(grid)


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _sample_opposite_pairs(
    count: int,
    axis_limit: int,
    other_limit: int,
    colors: list[int],
    kind: str,
) -> list[dict[str, object]] | None:
    axis_lo = 2 * AXIS_MARGIN
    axis_hi = 2 * (axis_limit - 1 - AXIS_MARGIN)
    axis_values = _sample_spaced_values(axis_lo, axis_hi, count, AXIS_GAP2)
    if axis_values is None:
        return None

    used_first: set[int] = set()
    used_second: set[int] = set()
    pairs: list[dict[str, object]] = []
    for color, axis2 in zip(colors, axis_values):
        coord_lo = max(1, axis2 - (axis_limit - 1))
        coord_hi = min(axis_limit - 1, axis2 - 1)
        choices = [
            first
            for first in range(coord_lo, coord_hi + 1)
            if first not in used_first and (axis2 - first) not in used_second
        ]
        if not choices:
            return None

        aligned = axis2 // 2 if axis2 % 2 == 0 else None
        if aligned is not None and aligned in choices and aligned not in used_second and random.random() < 0.5:
            first = aligned
        else:
            first = random.choice(choices)
        second = axis2 - first
        used_first.add(int(first))
        used_second.add(int(second))

        if kind == "tb":
            markers = ((0, int(first)), (other_limit - 1, int(second)))
        else:
            markers = ((int(first), 0), (int(second), other_limit - 1))
        pairs.append(
            {
                "color": int(color),
                "kind": kind,
                "axis2": int(axis2),
                "markers": tuple(sorted(markers)),
            }
        )
    return pairs


def _random_mask() -> np.ndarray:
    for _ in range(MASK_RETRIES):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        min_area = max(5, (h + w) // 2)
        max_area = min(MASK_AREA_CAP, h * w - 1)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in NEIGH4:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGH4:
                nr = cell[0] + dr
                nc = cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if bool(mask.all()):
            continue
        if mask.any(axis=0).sum() < 2 or mask.any(axis=1).sum() < 2:
            continue
        return mask
    raise RuntimeError("failed to sample an irregular object mask")


def _mark_blocked(blocked: np.ndarray, cells: tuple[tuple[int, int], ...]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _translated_cells(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...]:
    rel = np.argwhere(mask)
    return tuple(sorted((int(top + r), int(left + c)) for r, c in rel))


def _placement_candidates(
    mask: np.ndarray,
    pair: dict[str, object],
    pairs: dict[int, dict[str, object]],
    blocked: np.ndarray,
) -> tuple[tuple[int, int], ...] | None:
    h, w = blocked.shape
    mh, mw = mask.shape
    side_order = ["low", "high"]
    random.shuffle(side_order)

    positions = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)]
    random.shuffle(positions)

    for side in side_order:
        for top, left in positions:
            cells = _translated_cells(mask, top, left)
            if pair["kind"] == "tb":
                if side == "low" and not all(2 * c < int(pair["axis2"]) for _, c in cells):
                    continue
                if side == "high" and not all(2 * c > int(pair["axis2"]) for _, c in cells):
                    continue
            else:
                if side == "low" and not all(2 * r < int(pair["axis2"]) for r, _ in cells):
                    continue
                if side == "high" and not all(2 * r > int(pair["axis2"]) for r, _ in cells):
                    continue

            reflected = _reflect_cells(cells, pair)
            if set(cells).intersection(reflected):
                continue
            if any(blocked[r, c] for r, c in cells):
                continue
            if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in reflected):
                continue
            if any(blocked[r, c] for r, c in reflected):
                continue

            try:
                _nearest_recolor(pair, pairs, reflected)
            except AssertionError:
                continue
            return cells
    return None


def _build_scene() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    total = random.choice(TOTAL_OBJECT_OPTIONS)

    if total == 4:
        n_tb = 2
    else:
        n_tb = random.choice([2, 3])
    n_lr = total - n_tb
    if n_tb < 2 or n_lr < 2 or n_tb > MAX_OBJECTS_PER_FAMILY or n_lr > MAX_OBJECTS_PER_FAMILY:
        return None

    colors = random.sample(PALETTE, total)
    tb_colors = colors[:n_tb]
    lr_colors = colors[n_tb:]

    tb_pairs = _sample_opposite_pairs(n_tb, w, h, tb_colors, "tb")
    lr_pairs = _sample_opposite_pairs(n_lr, h, w, lr_colors, "lr")
    if tb_pairs is None or lr_pairs is None:
        return None

    all_pairs = tb_pairs + lr_pairs
    pair_map = {int(pair["color"]): pair for pair in all_pairs}

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    for pair in all_pairs:
        for r, c in pair["markers"]:
            grid[r, c] = int(pair["color"])
        _mark_blocked(blocked, pair["markers"])

    pair_order = all_pairs[:]
    random.shuffle(pair_order)
    placed = 0
    for pair in pair_order:
        placed_here = False
        for _ in range(PAIR_RETRIES):
            mask = _random_mask()
            cells = _placement_candidates(mask, pair, pair_map, blocked)
            if cells is None:
                continue
            for r, c in cells:
                grid[r, c] = int(pair["color"])
            _mark_blocked(blocked, cells)
            _mark_blocked(blocked, _reflect_cells(cells, pair))
            placed += 1
            placed_here = True
            break
        if not placed_here:
            return None

    if placed != total:
        return None

    try:
        pairs = _extract_marker_pairs(grid, BG)
        objects = _find_objects(grid, BG)
        if len(pairs) != total or len(objects) != total:
            return None
        out = _solve(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
