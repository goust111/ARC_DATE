from __future__ import annotations

import numpy as np


BG = 0
GREY = 5

COLD_FAMILY = frozenset({1, 8})
WARM_FAMILY = frozenset({4, 7})
DISTRACTOR_FAMILY = frozenset({2, 3})

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _king_neighbors(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _decode_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    grey_cells = [tuple(map(int, coord)) for coord in np.argwhere(grid == GREY)]
    assert len(grey_cells) == 4, "expected exactly four grey marker cells"
    grey_set = set(grey_cells)

    degrees = {
        cell: sum((nbr in grey_set) for nbr in _neighbors4(*cell))
        for cell in grey_cells
    }
    centers = [cell for cell, deg in degrees.items() if deg == 3]
    assert len(centers) == 1, "expected a unique marker center"
    center = centers[0]

    neighbors = [nbr for nbr in _neighbors4(*center) if nbr in grey_set]
    same_row = [nbr for nbr in neighbors if nbr[0] == center[0]]
    same_col = [nbr for nbr in neighbors if nbr[1] == center[1]]
    if len(same_row) == 2 and len(same_col) == 1:
        control = same_col[0]
    elif len(same_col) == 2 and len(same_row) == 1:
        control = same_row[0]
    else:
        raise AssertionError("grey cells do not form a valid marker bar plus control pixel")

    direction = (center[0] - control[0], center[1] - control[1])
    assert abs(direction[0]) + abs(direction[1]) == 1
    return center, direction, grey_set


def _group_pairs(grid: np.ndarray, direction: tuple[int, int]) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    coords = [tuple(map(int, coord)) for coord in np.argwhere((grid != BG) & (grid != GREY))]
    assert coords, "expected foreground pair pixels"

    groups: dict[int, list[tuple[int, int]]] = {}
    if dr != 0:
        for r, c in coords:
            groups.setdefault(c, []).append((r, c))
    else:
        for r, c in coords:
            groups.setdefault(r, []).append((r, c))

    pairs = list(groups.values())
    assert pairs and all(len(pair) == 2 for pair in pairs), "expected exactly one pair per active line"
    return pairs


def _family(colors: set[int]) -> str:
    if colors <= COLD_FAMILY:
        return "cold"
    if colors <= WARM_FAMILY:
        return "warm"
    return "distractor"


def _projection(cell: tuple[int, int], direction: tuple[int, int]) -> int:
    return cell[0] * direction[0] + cell[1] * direction[1]


def _apply_rule(
    grid: np.ndarray,
    direction: tuple[int, int],
    *,
    move_all_pairs: bool = False,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    dr, dc = direction

    for pair in _group_pairs(out, direction):
        colors = {int(out[cell]) for cell in pair}
        movable = move_all_pairs or _family(colors) in {"cold", "warm"}
        if not movable:
            continue

        upstream = max(pair, key=lambda cell: _projection(cell, direction))
        downstream = min(pair, key=lambda cell: _projection(cell, direction))
        new_pos = (upstream[0] - dr, upstream[1] - dc)

        assert new_pos != upstream
        assert _in_bounds(out.shape, *new_pos)
        assert out[new_pos] == BG or new_pos == downstream

        moving_color = int(out[downstream])
        out[downstream] = BG
        out[new_pos] = moving_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a grey marker bar to choose a global attraction direction, then slide only same-hue pairs until adjacent.

    Input structure:
    - The grid has a black background and exactly one grey marker made of a 3-cell horizontal or vertical bar plus one control pixel touching the bar center orthogonally.
    - Additional foreground pixels appear as isolated 2-pixel pairs; every pair lies on one row or one column according to the global attraction axis.
    - Blue/teal and yellow/orange pairs are movable same-hue pairs, while red/green pairs are distractors that must stay unchanged.

    Rule:
    - Decode the attraction direction as the direction from the marker-bar center away from the grey control pixel.
    - Group the non-grey foreground pixels into 2-pixel pairs along the row/column axis implied by that direction.
    - For every movable same-hue pair, keep the upstream pixel fixed and slide the downstream pixel along the attraction direction until the two become 4-neighbors.
    - Leave every red/green distractor pair unchanged and preserve the full grey marker structure.

    Key insight:
    - The puzzle is not a fixed left-shift task: the grey marker supplies a global direction, and that direction decides both which pixel in each pair is downstream and where it must stop.
    """
    g = np.asarray(grid, dtype=int)
    _center, direction, _grey_cells = _decode_marker(g)
    return _apply_rule(g, direction)


def _marker_cells(
    shape: tuple[int, int],
    direction_name: str,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    if direction_name == "up":
        r = int(rng.integers(1, min(3, h - 1)))
        c = int(rng.integers(2, w - 2))
        center = (r, c)
        cells = [(r, c - 1), center, (r, c + 1), (r + 1, c)]
    elif direction_name == "down":
        r = int(rng.integers(max(1, h - 3), h - 1))
        c = int(rng.integers(2, w - 2))
        center = (r, c)
        cells = [(r, c - 1), center, (r, c + 1), (r - 1, c)]
    elif direction_name == "left":
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(1, min(3, w - 1)))
        center = (r, c)
        cells = [(r - 1, c), center, (r + 1, c), (r, c + 1)]
    elif direction_name == "right":
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(max(1, w - 3), w - 1))
        center = (r, c)
        cells = [(r - 1, c), center, (r + 1, c), (r, c - 1)]
    else:
        raise ValueError(direction_name)

    assert all(_in_bounds(shape, r0, c0) for r0, c0 in cells)
    return cells, center


def _can_place_structure(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied:
            return False
        for nbr in _king_neighbors(r, c):
            if nbr in occupied and nbr not in cell_set:
                return False
    return True


def _pair_cells(
    shape: tuple[int, int],
    direction: tuple[int, int],
    line_key: int,
    distance: int,
    offset: int,
) -> list[tuple[int, int]]:
    h, w = shape
    if direction[0] != 0:
        r0 = offset
        r1 = offset + distance
        assert 0 <= r0 < r1 < h
        return [(r0, line_key), (r1, line_key)]
    c0 = offset
    c1 = offset + distance
    assert 0 <= c0 < c1 < w
    return [(line_key, c0), (line_key, c1)]


def _paint_pair(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    colors: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    shuffled = list(colors)
    rng.shuffle(shuffled)
    for (r, c), color in zip(cells, shuffled):
        grid[r, c] = int(color)


def _sample_pair_specs(rng: np.random.Generator) -> list[tuple[str, tuple[int, int]]]:
    specs = [
        ("cold", (1, 8)),
        ("warm", (4, 7)),
        ("distractor", (2, 3)),
    ]
    extras = [
        ("cold", (1, 8)),
        ("warm", (4, 7)),
        ("distractor", (2, 3)),
    ]
    if rng.random() < 0.8:
        specs.append(extras[int(rng.integers(0, len(extras)))])
    rng.shuffle(specs)
    return specs


def _line_candidates(
    shape: tuple[int, int],
    direction: tuple[int, int],
    center: tuple[int, int],
) -> list[int]:
    h, w = shape
    if direction[0] != 0:
        cols = [c for c in range(w) if abs(c - center[1]) >= 2]
        return cols
    rows = [r for r in range(h) if abs(r - center[0]) >= 2]
    return rows


def _try_place_pairs(
    grid: np.ndarray,
    direction: tuple[int, int],
    center: tuple[int, int],
    occupied: set[tuple[int, int]],
    pair_specs: list[tuple[str, tuple[int, int]]],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    line_keys = _line_candidates(grid.shape, direction, center)
    rng.shuffle(line_keys)
    if len(line_keys) < len(pair_specs):
        return False

    available = list(line_keys)
    for _label, colors in pair_specs:
        placed = False
        rng.shuffle(available)
        for line_key in list(available):
            distances = list(range(2, 7))
            rng.shuffle(distances)
            for distance in distances:
                axis_limit = h if direction[0] != 0 else w
                if distance >= axis_limit:
                    continue
                offsets = list(range(axis_limit - distance))
                rng.shuffle(offsets)
                for offset in offsets:
                    cells = _pair_cells(grid.shape, direction, line_key, distance, offset)
                    if not _can_place_structure(grid.shape, cells, occupied):
                        continue
                    _paint_pair(grid, cells, colors, rng)
                    occupied.update(cells)
                    available.remove(line_key)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _pair_summary(grid: np.ndarray, direction: tuple[int, int]) -> list[dict[str, object]]:
    summary = []
    for pair in _group_pairs(grid, direction):
        cells = sorted(pair)
        colors = tuple(int(grid[cell]) for cell in cells)
        if direction[0] != 0:
            gap = abs(cells[0][0] - cells[1][0]) - 1
        else:
            gap = abs(cells[0][1] - cells[1][1]) - 1
        summary.append(
            {
                "cells": cells,
                "colors": colors,
                "family": _family(set(colors)),
                "gap": int(gap),
            }
        )
    return summary


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        center, direction, grey_cells = _decode_marker(grid)
        assert len(grey_cells) == 4
        out = main(grid)
        assert out.shape == grid.shape
        assert np.array_equal(out[list(zip(*grey_cells))[0], list(zip(*grey_cells))[1]], np.full(4, GREY))

        pairs = _pair_summary(grid, direction)
        families = {item["family"] for item in pairs}
        assert {"cold", "warm", "distractor"} <= families
        assert any(item["gap"] >= 2 for item in pairs)
        assert np.any(out != grid), "at least one movable pair must change"

        opposite = (-direction[0], -direction[1])
        opposite_out = _apply_rule(grid, opposite)
        assert not np.array_equal(out, opposite_out), "direction must matter"

        no_class_filter = _apply_rule(grid, direction, move_all_pairs=True)
        assert not np.array_equal(out, no_class_filter), "distractors must matter"

        if direction[0] != 0:
            aligned_bar = sum(1 for r, c in grey_cells if r == center[0])
            assert aligned_bar == 3
        else:
            aligned_bar = sum(1 for r, c in grey_cells if c == center[1])
            assert aligned_bar == 3
    except Exception:
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 18))
    w = int(rng.integers(12, 18))
    grid = np.full((h, w), BG, dtype=int)

    direction_name = str(rng.choice(np.array(list(DIRS.keys()), dtype=object)))
    direction = DIRS[direction_name]
    marker_cells, center = _marker_cells(grid.shape, direction_name, rng)
    for r, c in marker_cells:
        grid[r, c] = GREY
    occupied = set(marker_cells)

    pair_specs = _sample_pair_specs(rng)
    if not _try_place_pairs(grid, direction, center, occupied, pair_specs, rng):
        return None

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
