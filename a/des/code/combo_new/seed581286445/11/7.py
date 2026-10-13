from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
GREY = 5
ORTHOGONAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200

_RNG = np.random.default_rng()


def _is_4_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHOGONAL_DIRS:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _infer_tower(color: int, cells: set[tuple[int, int]]) -> dict[str, object]:
    assert 2 <= len(cells) <= 5, "a tower must contain 2 to 5 pixels"
    assert _is_4_connected(cells), "tower pixels must be 4-connected"

    if len(cells) == 2:
        a, b = sorted(cells)
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "one-arm tower must be a domino"
        # A 2-cell tower has no visible unique center, so we use a canonical
        # top/left endpoint; the generator only emits one-arm towers in that form.
        center = a
        arms = ((b[0] - a[0], b[1] - a[1]),)
        return {
            "color": int(color),
            "center": center,
            "cells": cells,
            "arms": arms,
            "level": 1,
        }

    candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for center in sorted(cells):
        arms = []
        ok = True
        for r, c in cells:
            if (r, c) == center:
                continue
            dr = r - center[0]
            dc = c - center[1]
            if abs(dr) + abs(dc) != 1:
                ok = False
                break
            arms.append((dr, dc))
        if ok and len(set(arms)) == len(cells) - 1:
            candidates.append((center, tuple(sorted(arms))))

    assert len(candidates) == 1, "tower shape must be a partial radius-1 cross"
    center, arms = candidates[0]
    return {
        "color": int(color),
        "center": center,
        "cells": cells,
        "arms": arms,
        "level": len(arms),
    }


def _scan_key(tower: dict[str, object]) -> tuple[int, int, int]:
    center = tower["center"]
    assert isinstance(center, tuple)
    return int(tower["level"]), int(center[0]), int(center[1])


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    cells = [(r1, c) for c in _inclusive_range(c1, c2)]
    cells.extend((r, c2) for r in _inclusive_range(r1, r2))
    return cells


def _extract_towers(grid: np.ndarray) -> list[dict[str, object]]:
    towers = []
    for color in map(int, np.unique(grid)):
        if color == BLACK:
            continue
        cells = set(map(tuple, np.argwhere(grid == color)))
        towers.append(_infer_tower(color, cells))
    assert len(towers) >= 2, "expected at least two towers"
    return towers


def _render_towers(towers: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for tower in towers:
        color = int(tower["color"])
        for r, c in tower["cells"]:
            grid[int(r), int(c)] = color
    return grid


def _render_output_from_towers(
    towers: list[dict[str, object]],
    shape: tuple[int, int],
) -> np.ndarray:
    out = _render_towers(towers, shape)
    ordered = sorted(towers, key=_scan_key)
    for left, right in zip(ordered, ordered[1:]):
        for r, c in _path_cells(left["center"], right["center"]):
            if out[int(r), int(c)] == BLACK:
                out[int(r), int(c)] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect partial-cross signal towers in arm-count order with grey elbow paths.

    Input structure:
    - The background is black.
    - Each foreground color forms one isolated tower: a center pixel plus 1 to 4 same-color orthogonal arm pixels at distance 1.
    - Different towers use different colors and there is empty space between towers.

    Rule:
    - For each tower, locate its center and count the occupied orthogonal arm directions to get its level.
    - Sort towers by level ascending; break ties by raster scan order of their centers (left-to-right within a row, then top-to-bottom).
    - For each consecutive pair in that order, draw a grey path from the first center: move horizontally to the second center's column, then vertically to the second center's row.
    - Keep tower pixels unchanged and only paint the connection onto background cells.

    Key insight:
    - The ordering key is the hidden number of occupied arms, not color or proximity, and every bridge follows the same horizontal-then-vertical elbow rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    towers = _extract_towers(g)
    return _render_output_from_towers(towers, g.shape)


def _tower_patterns(level: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    assert 1 <= level <= 4
    if level == 1:
        return (((0, 1),), ((1, 0),))
    return tuple(itertools.combinations(ORTHOGONAL_DIRS, level))


def _tower_cells(center: tuple[int, int], arms: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells = {center}
    for dr, dc in arms:
        cells.add((center[0] + dr, center[1] + dc))
    return cells


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _sample_levels() -> tuple[list[int], bool]:
    if bool(_RNG.integers(0, 2)):
        count = int(_RNG.integers(4, 8))
        while True:
            levels = list(map(int, _RNG.integers(1, 5, size=count)))
            if len(set(levels)) < len(levels):
                return levels, True
    count = int(_RNG.integers(3, 5))
    levels = list(map(int, _RNG.choice(np.array([1, 2, 3, 4], dtype=int), size=count, replace=False)))
    _RNG.shuffle(levels)
    return levels, False


def _place_towers(
    shape: tuple[int, int],
    levels: list[int],
    colors: list[int],
) -> list[dict[str, object]] | None:
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    towers: list[dict[str, object] | None] = [None] * len(levels)
    order = list(map(int, _RNG.permutation(len(levels))))
    order.sort(key=lambda i: -levels[i])

    for idx in order:
        level = int(levels[idx])
        patterns = _tower_patterns(level)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            arms = patterns[int(_RNG.integers(0, len(patterns)))]
            center = (int(_RNG.integers(1, h - 1)), int(_RNG.integers(1, w - 1)))
            cells = _tower_cells(center, arms)
            if any(blocked[r, c] for r, c in cells):
                continue
            towers[idx] = {
                "color": int(colors[idx]),
                "center": center,
                "cells": cells,
                "arms": arms,
                "level": level,
            }
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            return None

    return [tower for tower in towers if tower is not None]


def _paths_avoid_third_party_towers(towers: list[dict[str, object]]) -> bool:
    ordered = sorted(towers, key=_scan_key)
    occupied_by: dict[tuple[int, int], int] = {}
    for idx, tower in enumerate(towers):
        for cell in tower["cells"]:
            occupied_by[cell] = idx

    for left, right in zip(ordered, ordered[1:]):
        if left["center"][0] == right["center"][0] or left["center"][1] == right["center"][1]:
            return False
        allow = {occupied_by[left["center"]], occupied_by[right["center"]]}
        allow.update(occupied_by[cell] for cell in left["cells"])
        allow.update(occupied_by[cell] for cell in right["cells"])
        for cell in _path_cells(left["center"], right["center"]):
            owner = occupied_by.get(cell)
            if owner is not None and owner not in allow:
                return False
    return True


def _has_tie_pattern(levels: list[int], expect_ties: bool) -> bool:
    has_ties = len(set(levels)) < len(levels)
    return has_ties if expect_ties else not has_ties


def _gen_attempt() -> np.ndarray | None:
    levels, expect_ties = _sample_levels()
    if not _has_tie_pattern(levels, expect_ties):
        return None

    count = len(levels)
    colors = list(map(int, _RNG.choice(np.array(TOWER_COLORS, dtype=int), size=count, replace=False)))

    min_h = max(12, 3 * count + 4)
    min_w = max(12, 3 * count + 4)
    h = int(_RNG.integers(min_h, min(31, min_h + 7)))
    w = int(_RNG.integers(min_w, min(31, min_w + 7)))

    towers = _place_towers((h, w), levels, colors)
    if towers is None:
        return None
    if not _paths_avoid_third_party_towers(towers):
        return None

    inp = _render_towers(towers, (h, w))
    out = _render_output_from_towers(towers, (h, w))

    if np.array_equal(inp, out):
        return None
    if int(np.count_nonzero(out == GREY)) < len(towers) - 1:
        return None
    if np.unique(inp).size != count + 1:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
