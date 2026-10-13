from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

OTHER_COLORS = (RED, GREEN, YELLOW, MAGENTA, ORANGE)
AXIS_KINDS = ("row", "col", "diag", "anti")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 16
GRID_MAX = 24
NON_AZURE_COUNT_RANGE = (3, 5)
ERASE_COUNT_OPTIONS = (1, 2)
AZURE_AREA_RANGE = (4, 8)
OTHER_AREA_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _bbox_size(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = {(r, c)}
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.add((nr, nc))
            comps.append(comp)

    return comps


def _all_components(grid: np.ndarray) -> list[tuple[int, set[tuple[int, int]]]]:
    comps: list[tuple[int, set[tuple[int, int]]]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BLACK):
        for comp in _components_of_color(grid, color):
            comps.append((color, comp))
    return comps


def _reflect_point(axis: tuple[str, int], r: int, c: int) -> tuple[int, int]:
    kind, index = axis
    if kind == "row":
        return 2 * index - r, c
    if kind == "col":
        return r, 2 * index - c
    if kind == "diag":
        return c + index, r - index
    if kind == "anti":
        return index - c, index - r
    raise AssertionError(f"unknown axis kind: {kind}")


def _reflect_cells(
    axis: tuple[str, int], cells: set[tuple[int, int]], shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    h, w = shape
    reflected = {_reflect_point(axis, r, c) for r, c in cells}
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in reflected):
        return None
    return reflected


def _side_value(axis: tuple[str, int], r: int, c: int) -> int:
    kind, index = axis
    if kind == "row":
        return r - index
    if kind == "col":
        return c - index
    if kind == "diag":
        return r - c - index
    if kind == "anti":
        return r + c - index
    raise AssertionError(f"unknown axis kind: {kind}")


def _classify_object(axis: tuple[str, int], cells: set[tuple[int, int]]) -> str:
    values = [_side_value(axis, r, c) for r, c in cells]
    if any(v == 0 for v in values):
        return "erase"
    has_pos = any(v > 0 for v in values)
    has_neg = any(v < 0 for v in values)
    if has_pos and has_neg:
        return "erase"
    return "mirror"


def _axis_candidates(shape: tuple[int, int]) -> list[tuple[str, int]]:
    h, w = shape
    axes: list[tuple[str, int]] = []
    axes.extend(("row", r) for r in range(h))
    axes.extend(("col", c) for c in range(w))
    axes.extend(("diag", k) for k in range(-(w - 1), h))
    axes.extend(("anti", k) for k in range(h + w - 1))
    return axes


def _find_hidden_axis(grid: np.ndarray) -> tuple[str, int]:
    azure = _components_of_color(grid, AZURE)
    assert len(azure) == 2, "expected exactly two azure objects"
    first, second = azure
    candidates = []
    for axis in _axis_candidates(grid.shape):
        reflected = _reflect_cells(axis, first, grid.shape)
        if reflected == second:
            candidates.append(axis)
    assert len(candidates) == 1, "azure pair must define a unique hidden axis"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    axis = _find_hidden_axis(g)

    out = g.copy()
    to_draw: list[tuple[int, set[tuple[int, int]]]] = []
    for color, comp in _all_components(g):
        if color == AZURE:
            continue
        status = _classify_object(axis, comp)
        if status == "erase":
            for r, c in comp:
                out[r, c] = BLACK
            continue

        reflected = _reflect_cells(axis, comp, g.shape)
        assert reflected is not None
        to_draw.append((color, reflected))

    for color, cells in to_draw:
        for r, c in cells:
            current = int(out[r, c])
            assert current in (BLACK, color), "mirrored copy cannot overwrite another object"
            out[r, c] = color

    return out


def _random_connected_shape(
    area: int,
    max_h: int,
    max_w: int,
    irregular: bool,
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            options = list(frontier)
            random.shuffle(options)
            placed = False
            for cand in options:
                trial = set(cells)
                trial.add(cand)
                norm = set(_normalize(trial))
                height, width = _bbox_size(norm)
                if height <= max_h and width <= max_w:
                    cells = norm
                    placed = True
                    break
            if not placed:
                break

        if len(cells) != area:
            continue

        height, width = _bbox_size(cells)
        if height == 1 or width == 1:
            continue
        if irregular and area == height * width:
            continue
        return tuple(sorted(cells))
    return None


def _translated(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in shape}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _choose_axis(h: int, w: int) -> tuple[str, int]:
    kind = random.choice(AXIS_KINDS)
    if kind == "row":
        return kind, random.randint(4, h - 5)
    if kind == "col":
        return kind, random.randint(4, w - 5)
    if kind == "diag":
        choices = [k for k in range(-(w - 1), h) if 5 <= sum(1 for r in range(h) if 0 <= r - k < w)]
        return kind, random.choice(choices)
    choices = [k for k in range(h + w - 1) if 5 <= sum(1 for r in range(h) if 0 <= k - r < w)]
    return kind, random.choice(choices)


def _fit_pair_candidates(
    shape: tuple[tuple[int, int], ...],
    axis: tuple[str, int],
    grid_shape: tuple[int, int],
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = grid_shape
    sh, sw = _bbox_size(shape)
    out: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = _translated(shape, r0, c0)
            if _classify_object(axis, cells) != "mirror":
                continue
            reflected = _reflect_cells(axis, cells, grid_shape)
            if reflected is None or cells & reflected:
                continue
            out.append((cells, reflected))
    return out


def _fit_erase_candidates(
    shape: tuple[tuple[int, int], ...],
    axis: tuple[str, int],
    grid_shape: tuple[int, int],
    mode: str,
) -> list[set[tuple[int, int]]]:
    h, w = grid_shape
    sh, sw = _bbox_size(shape)
    out: list[set[tuple[int, int]]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = _translated(shape, r0, c0)
            values = [_side_value(axis, r, c) for r, c in cells]
            if mode == "touch":
                if not any(v == 0 for v in values):
                    continue
            elif mode == "cross":
                if not (any(v < 0 for v in values) and any(v > 0 for v in values)):
                    continue
            else:
                if _classify_object(axis, cells) != "erase":
                    continue
            out.append(cells)
    return out


def _pick_non_overlapping(
    options: list[tuple[set[tuple[int, int]], ...]],
    reserved: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], ...] | None:
    shuffled = list(options)
    random.shuffle(shuffled)
    for item in shuffled:
        union = set().union(*item)
        if any(cell in reserved for cell in union):
            continue
        return item
    return None


def _reserve_footprints(
    reserved: set[tuple[int, int]],
    footprints: tuple[set[tuple[int, int]], ...],
    grid_shape: tuple[int, int],
) -> None:
    for cells in footprints:
        reserved.update(_expanded(cells, grid_shape))


def _place_azure_pair(
    axis: tuple[str, int],
    grid_shape: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        area = random.randint(*AZURE_AREA_RANGE)
        shape = _random_connected_shape(area, 4, 4, irregular=True)
        if shape is None:
            continue
        options = _fit_pair_candidates(shape, axis, grid_shape)
        placed = _pick_non_overlapping(options, reserved, grid_shape)
        if placed is None:
            continue
        return placed
    return None


def _place_one_sided(
    axis: tuple[str, int],
    grid_shape: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        area = random.randint(*OTHER_AREA_RANGE)
        shape = _random_connected_shape(area, 5, 5, irregular=False)
        if shape is None:
            continue
        options = _fit_pair_candidates(shape, axis, grid_shape)
        placed = _pick_non_overlapping(options, reserved, grid_shape)
        if placed is None:
            continue
        return placed
    return None


def _place_erased(
    axis: tuple[str, int],
    grid_shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    mode: str,
) -> set[tuple[int, int]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        area = random.randint(*OTHER_AREA_RANGE)
        shape = _random_connected_shape(area, 5, 5, irregular=False)
        if shape is None:
            continue
        options = [(cells,) for cells in _fit_erase_candidates(shape, axis, grid_shape, mode)]
        placed = _pick_non_overlapping(options, reserved, grid_shape)
        if placed is None:
            continue
        return placed[0]
    return None


def _build_grid(
    grid_shape: tuple[int, int],
    azure_pair: tuple[set[tuple[int, int]], set[tuple[int, int]]],
    one_sided: list[tuple[int, set[tuple[int, int]]]],
    erased: list[tuple[int, set[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for cells in azure_pair:
        for r, c in cells:
            grid[r, c] = AZURE
    for color, cells in one_sided + erased:
        for r, c in cells:
            assert grid[r, c] == BLACK
            grid[r, c] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        axis = _find_hidden_axis(grid)
    except AssertionError:
        return False

    comps = _all_components(grid)
    azure_count = sum(1 for color, _ in comps if color == AZURE)
    if azure_count != 2:
        return False

    non_azure = [(color, comp) for color, comp in comps if color != AZURE]
    if not (NON_AZURE_COUNT_RANGE[0] <= len(non_azure) <= NON_AZURE_COUNT_RANGE[1]):
        return False
    colors = [color for color, _ in non_azure]
    if len(colors) != len(set(colors)):
        return False
    if any(color not in OTHER_COLORS for color in colors):
        return False

    erase_count = 0
    for _color, comp in non_azure:
        if _classify_object(axis, comp) == "erase":
            erase_count += 1
    if erase_count not in ERASE_COUNT_OPTIONS:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    axis = _choose_axis(h, w)
    reserved: set[tuple[int, int]] = set()

    azure_pair = _place_azure_pair(axis, (h, w), reserved)
    if azure_pair is None:
        return None
    _reserve_footprints(reserved, azure_pair, (h, w))

    object_count = random.randint(*NON_AZURE_COUNT_RANGE)
    erase_count = random.choice(ERASE_COUNT_OPTIONS)
    mirror_count = object_count - erase_count
    colors = list(OTHER_COLORS)
    random.shuffle(colors)
    erase_colors = colors[:erase_count]
    mirror_colors = colors[erase_count:erase_count + mirror_count]

    one_sided: list[tuple[int, set[tuple[int, int]]]] = []
    for color in mirror_colors:
        placed = _place_one_sided(axis, (h, w), reserved)
        if placed is None:
            return None
        cells, reflected = placed
        one_sided.append((color, cells))
        _reserve_footprints(reserved, (cells, reflected), (h, w))

    erased: list[tuple[int, set[tuple[int, int]]]] = []
    erase_modes = [random.choice(("touch", "cross")) for _ in range(erase_count)]
    if erase_count == 2 and erase_modes[0] == erase_modes[1]:
        erase_modes[1] = "touch" if erase_modes[0] == "cross" else "cross"
    for color, mode in zip(erase_colors, erase_modes):
        placed = _place_erased(axis, (h, w), reserved, mode)
        if placed is None:
            fallback = _place_erased(axis, (h, w), reserved, "either")
            if fallback is None:
                return None
            placed = fallback
        erased.append((color, placed))
        _reserve_footprints(reserved, (placed,), (h, w))

    grid = _build_grid((h, w), azure_pair, one_sided, erased)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
