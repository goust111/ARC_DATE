from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

PALETTE = (RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA)
COLOR_ORDER = (RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA)
SPECIAL_MIX = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = {"R": "L", "L": "R", "U": "D", "D": "U"}
STEP = {"R": (0, 1), "L": (0, -1), "U": (-1, 0), "D": (1, 0)}

# Tip + two-cell shaft + two tail wings.
ARROW_OFFSETS = {
    "R": ((0, 0), (0, -1), (0, -2), (-1, -2), (1, -2)),
    "L": ((0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)),
    "U": ((0, 0), (1, 0), (2, 0), (2, -1), (2, 1)),
    "D": ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1)),
}

SPECIAL_PAIRS = tuple(tuple(sorted(pair)) for pair in SPECIAL_MIX)
FALLBACK_PAIRS = tuple(
    pair
    for pair in (
        tuple(sorted((a, b)))
        for i, a in enumerate(PALETTE)
        for b in PALETTE[i + 1 :]
    )
    if frozenset(pair) not in SPECIAL_MIX
)

PAIR_COUNT_CHOICES = (1, 2, 3)
PAIR_COUNT_WEIGHTS = (3, 4, 3)
MIN_SIDE = 16
MAX_SIDE = 26
MIN_HALF_DISTANCE = 3
MAX_HALF_DISTANCE = 6
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _color_rank(color: int) -> int:
    return COLOR_ORDER.index(int(color))


def _mix_colors(color_a: int, color_b: int) -> int:
    combo = frozenset((int(color_a), int(color_b)))
    if combo in SPECIAL_MIX:
        return int(SPECIAL_MIX[combo])
    return min((int(color_a), int(color_b)), key=_color_rank)


def _arrow_cells(tip: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append((color, coords))
    return components


def _detect_arrow(color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    pts = set(coords)
    matches: list[tuple[tuple[int, int], str]] = []
    for tip in pts:
        tr, tc = tip
        for direction, offsets in ARROW_OFFSETS.items():
            if {(tr + dr, tc + dc) for dr, dc in offsets} == pts:
                matches.append((tip, direction))
    assert len(matches) == 1, f"invalid arrow shape for color {color}"
    tip, direction = matches[0]
    return {"color": int(color), "tip": tip, "dir": direction, "cells": pts}


def _pair_candidates(grid: np.ndarray, arrows: list[dict[str, object]], idx: int) -> list[int]:
    arrow = arrows[idx]
    tip_r, tip_c = arrow["tip"]  # type: ignore[assignment]
    direction = str(arrow["dir"])
    dr, dc = STEP[direction]
    matches: list[int] = []

    for j, other in enumerate(arrows):
        if j == idx:
            continue
        if str(other["dir"]) != OPPOSITE[direction]:
            continue
        other_r, other_c = other["tip"]  # type: ignore[assignment]
        delta_r = int(other_r - tip_r)
        delta_c = int(other_c - tip_c)

        if direction in {"R", "L"}:
            if delta_r != 0:
                continue
            if delta_c * dc <= 0:
                continue
            step_count = abs(delta_c)
            path = [int(grid[tip_r, tip_c + step * dc]) for step in range(1, step_count)]
        else:
            if delta_c != 0:
                continue
            if delta_r * dr <= 0:
                continue
            step_count = abs(delta_r)
            path = [int(grid[tip_r + step * dr, tip_c]) for step in range(1, step_count)]

        if all(value == BG for value in path):
            matches.append(j)

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g)
    assert comps, "expected at least one arrow"
    assert len(comps) % 2 == 0, "arrows must come in pairs"

    arrows = []
    for color, coords in comps:
        assert len(coords) == 5, f"component of color {color} is not a 5-cell arrow"
        arrows.append(_detect_arrow(color, coords))

    partner_of: dict[int, int] = {}
    for idx in range(len(arrows)):
        candidates = _pair_candidates(g, arrows, idx)
        assert len(candidates) == 1, f"arrow {idx} has ambiguous pairing"
        partner_of[idx] = candidates[0]

    pair_indices: list[tuple[int, int]] = []
    used: set[int] = set()
    for idx, partner in partner_of.items():
        assert partner_of.get(partner) == idx, "pairing must be reciprocal"
        if idx in used or partner in used:
            continue
        used.add(idx)
        used.add(partner)
        pair_indices.append(tuple(sorted((idx, partner))))

    assert len(pair_indices) * 2 == len(arrows), "every arrow must belong to exactly one pair"

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    for idx_a, idx_b in pair_indices:
        arrow_a = arrows[idx_a]
        arrow_b = arrows[idx_b]
        tip_a_r, tip_a_c = arrow_a["tip"]  # type: ignore[assignment]
        tip_b_r, tip_b_c = arrow_b["tip"]  # type: ignore[assignment]

        if tip_a_r == tip_b_r:
            c0, c1 = sorted((int(tip_a_c), int(tip_b_c)))
            assert (c0 + c1) % 2 == 0, "horizontal trajectories must meet on a cell"
            center = (int(tip_a_r), (c0 + c1) // 2)
        else:
            r0, r1 = sorted((int(tip_a_r), int(tip_b_r)))
            assert tip_a_c == tip_b_c, "paired arrows must align in one row or one column"
            assert (r0 + r1) % 2 == 0, "vertical trajectories must meet on a cell"
            center = ((r0 + r1) // 2, int(tip_a_c))

        mix_color = _mix_colors(int(arrow_a["color"]), int(arrow_b["color"]))
        for cell in _cross_cells(center):
            r, c = cell
            assert _in_bounds(g.shape, r, c), "collision mark would leave the grid"
            assert int(g[r, c]) == BG, "collision mark must be drawn only on background"
            assert cell not in painted, "collision marks must be disjoint"
            out[r, c] = mix_color
            painted.add(cell)

    return out


def _is_special_pair(pair: tuple[int, int]) -> bool:
    return frozenset(pair) in SPECIAL_MIX


def _choose_color_pairs(pair_count: int) -> list[tuple[int, int]]:
    target_special = random.random() < 0.5

    if pair_count == 1:
        pool = SPECIAL_PAIRS if target_special else FALLBACK_PAIRS
        return [random.choice(pool)]

    for _ in range(200):
        if target_special:
            special = random.choice(SPECIAL_PAIRS)
            remaining = [color for color in PALETTE if color not in special]
            need = 2 * (pair_count - 1)
            random.shuffle(remaining)
            if need > len(remaining):
                continue
            extra_colors = remaining[:need]
            for _ in range(100):
                random.shuffle(extra_colors)
                extras = [tuple(sorted(extra_colors[2 * i : 2 * i + 2])) for i in range(pair_count - 1)]
                if all(not _is_special_pair(pair) for pair in extras):
                    pairs = [tuple(sorted(special)), *extras]
                    random.shuffle(pairs)
                    return pairs
        else:
            subset = random.sample(PALETTE, 2 * pair_count)
            for _ in range(100):
                random.shuffle(subset)
                pairs = [tuple(sorted(subset[2 * i : 2 * i + 2])) for i in range(pair_count)]
                if all(not _is_special_pair(pair) for pair in pairs):
                    return pairs

    raise RuntimeError("failed to select color pairs")


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _pair_geometry(
    shape: tuple[int, int],
    center: tuple[int, int],
    orientation: str,
    half_distance: int,
) -> dict[str, object] | None:
    h, w = shape
    r, c = center

    if orientation == "H":
        left_tip = (r, c - half_distance)
        right_tip = (r, c + half_distance)
        arrows = [
            {"tip": left_tip, "dir": "R"},
            {"tip": right_tip, "dir": "L"},
        ]
    else:
        top_tip = (r - half_distance, c)
        bottom_tip = (r + half_distance, c)
        arrows = [
            {"tip": top_tip, "dir": "D"},
            {"tip": bottom_tip, "dir": "U"},
        ]

    arrow_cells = set()
    for arrow in arrows:
        cells = _arrow_cells(arrow["tip"], arrow["dir"])
        if any(not _in_bounds(shape, ar, ac) for ar, ac in cells):
            return None
        arrow_cells |= cells

    cross = _cross_cells(center)
    if any(not _in_bounds(shape, ar, ac) for ar, ac in cross):
        return None

    all_cells = arrow_cells | cross
    if len(all_cells) != 15:
        return None

    if orientation == "H":
        if not (1 <= r <= h - 2):
            return None
    else:
        if not (1 <= c <= w - 2):
            return None

    return {
        "center": center,
        "orientation": orientation,
        "half_distance": half_distance,
        "arrows": arrows,
        "arrow_cells": arrow_cells,
        "cross": cross,
        "all_cells": all_cells,
    }


def _candidate_geometries(
    shape: tuple[int, int],
    orientation: str,
    blocked: set[tuple[int, int]],
    used_horizontal_rows: set[int],
    used_vertical_cols: set[int],
) -> list[dict[str, object]]:
    h, w = shape
    candidates: list[dict[str, object]] = []

    for half_distance in range(MIN_HALF_DISTANCE, MAX_HALF_DISTANCE + 1):
        if orientation == "H":
            for r in range(1, h - 1):
                if r in used_horizontal_rows:
                    continue
                for c in range(half_distance + 2, w - half_distance - 2):
                    geom = _pair_geometry(shape, (r, c), orientation, half_distance)
                    if geom is None:
                        continue
                    if geom["all_cells"] & blocked:  # type: ignore[operator]
                        continue
                    candidates.append(geom)
        else:
            for c in range(1, w - 1):
                if c in used_vertical_cols:
                    continue
                for r in range(half_distance + 2, h - half_distance - 2):
                    geom = _pair_geometry(shape, (r, c), orientation, half_distance)
                    if geom is None:
                        continue
                    if geom["all_cells"] & blocked:  # type: ignore[operator]
                        continue
                    candidates.append(geom)

    return candidates


def _build_grid() -> np.ndarray | None:
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    min_side = max(MIN_SIDE, 14 + 2 * pair_count)
    h = random.randint(min_side, MAX_SIDE)
    w = random.randint(min_side, MAX_SIDE)

    pairs = _choose_color_pairs(pair_count)
    blocked: set[tuple[int, int]] = set()
    used_horizontal_rows: set[int] = set()
    used_vertical_cols: set[int] = set()
    placed: list[dict[str, object]] = []

    for pair in pairs:
        orientation = random.choice(("H", "V"))
        candidates = _candidate_geometries((h, w), orientation, blocked, used_horizontal_rows, used_vertical_cols)
        if not candidates:
            orientation = "V" if orientation == "H" else "H"
            candidates = _candidate_geometries((h, w), orientation, blocked, used_horizontal_rows, used_vertical_cols)
        if not candidates:
            return None

        geom = random.choice(candidates)
        colors = list(pair)
        random.shuffle(colors)
        arrows = list(geom["arrows"])  # type: ignore[arg-type]
        for arrow, color in zip(arrows, colors):
            arrow["color"] = int(color)
        geom["arrows"] = arrows
        placed.append(geom)

        blocked |= _expand_cells(geom["all_cells"], (h, w))  # type: ignore[arg-type]
        center_r, center_c = geom["center"]  # type: ignore[assignment]
        if geom["orientation"] == "H":
            used_horizontal_rows.add(int(center_r))
        else:
            used_vertical_cols.add(int(center_c))

    grid = np.zeros((h, w), dtype=int)
    for geom in placed:
        for arrow in geom["arrows"]:  # type: ignore[index]
            color = int(arrow["color"])
            for r, c in _arrow_cells(arrow["tip"], arrow["dir"]):
                if grid[r, c] != BG:
                    return None
                grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
