from __future__ import annotations

import random

import numpy as np


BG = 0
TILE_FG = 1
INVERTED_BG = 2
KEY_FG = 3
SEP = 5

TILE_SIZE = 4
ROW_BANDS = 2
COL_BANDS = 3
ROW_SEPARATOR = 1
COL_SEPARATOR = 1
KEY_SEPARATOR = 2

GRID_H = ROW_BANDS * TILE_SIZE + ROW_SEPARATOR
GRID_W = COL_BANDS * TILE_SIZE + (COL_BANDS - 1) * COL_SEPARATOR + KEY_SEPARATOR + TILE_SIZE
MAX_GENERATION_ATTEMPTS = 300

_MODE_CYCLE = ("scale1", "scale2", "mixed2", "mixed3")
_GEN_CALL_COUNT = 0

_KEY_SHAPES = (
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
)

_LINE3_SHAPES = (
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0)}),
)

_AREA12_SHAPES = (
    frozenset((r, c) for r in range(3) for c in range(4)),
    frozenset((r, c) for r in range(4) for c in range(3)),
)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected at least one index"
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def _coords_of_color(block: np.ndarray, color: int) -> frozenset[tuple[int, int]]:
    return frozenset(tuple(map(int, rc)) for rc in np.argwhere(block == color))


def _translate(coords: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in coords)


def _normalize(coords: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def _fits_tile(coords: frozenset[tuple[int, int]]) -> bool:
    return all(0 <= r < TILE_SIZE and 0 <= c < TILE_SIZE for r, c in coords)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(coords: frozenset[tuple[int, int]]) -> bool:
    if not coords:
        return False
    seen = {next(iter(coords))}
    stack = list(seen)
    while stack:
        cell = stack.pop()
        for nb in _neighbors4(cell):
            if nb in coords and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(coords)


def _scale2(coords: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        expanded.update(
            {
                (2 * r, 2 * c),
                (2 * r, 2 * c + 1),
                (2 * r + 1, 2 * c),
                (2 * r + 1, 2 * c + 1),
            }
        )
    return frozenset(expanded)


def _tile_slices() -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    row_runs = [(0, 3), (5, 8)]
    col_runs = [(0, 3), (5, 8), (10, 13)]
    key_cols = (16, 19)
    return row_runs, col_runs, key_cols


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]

    free_rows = [r for r in range(grid.shape[0]) if r not in set(grey_rows)]
    free_cols = [c for c in range(grid.shape[1]) if c not in set(grey_cols)]
    row_runs = _runs(free_rows)
    col_runs = _runs(free_cols)

    assert grey_rows == [4], "expected one full separator row"
    assert grey_cols == [4, 9, 14, 15], "expected two inner separator columns and one double key separator"
    assert row_runs == [(0, 3), (5, 8)], "expected two 4x4 tile row bands"
    assert col_runs == [(0, 3), (5, 8), (10, 13), (16, 19)], "expected three tile columns plus one key column"

    key_col_run = col_runs[-1]
    key_row_candidates = [
        idx
        for idx, (r0, r1) in enumerate(row_runs)
        if np.any(grid[r0 : r1 + 1, key_col_run[0] : key_col_run[1] + 1] == KEY_FG)
    ]
    assert len(key_row_candidates) == 1, "expected exactly one 4x4 key region"
    return row_runs, col_runs[:3], key_col_run, int(key_row_candidates[0])


def _classify_tiles(
    grid: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[int, int],
    int,
    frozenset[tuple[int, int]],
    list[str | None],
    list[frozenset[tuple[int, int]]],
]:
    row_runs, col_runs, key_cols, key_row_index = _parse_layout(grid)
    kr0, kr1 = row_runs[key_row_index]
    kc0, kc1 = key_cols
    key_block = grid[kr0 : kr1 + 1, kc0 : kc1 + 1]
    key_coords = _coords_of_color(key_block, KEY_FG)
    assert key_coords, "key region must contain a green shape"
    assert _is_connected(key_coords), "key shape must be 4-connected"
    scaled_key = _scale2(key_coords)

    classes: list[str | None] = []
    tile_coords: list[frozenset[tuple[int, int]]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            coords = _coords_of_color(tile, TILE_FG)
            assert coords, "each tile must contain a blue shape"
            assert _is_connected(coords), "each tile shape must be 4-connected"
            tile_coords.append(coords)

            scale1 = coords == key_coords
            scale2 = coords == scaled_key
            assert not (scale1 and scale2), "a tile cannot simultaneously be a 1x and 2x match"
            classes.append("1x" if scale1 else "2x" if scale2 else None)

    return row_runs, col_runs, key_cols, key_row_index, key_coords, classes, tile_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert occupancy only inside those tiles whose blue shape matches the green key shape at 1x or 2x scale.

    Input structure:
    - A full grey row and four full grey columns partition the canvas into six 4x4 blue-pattern tiles on the left and one 4x4 key area on the right.
    - Each tile contains a 4-connected blue shape on black background.
    - Exactly one of the two possible key slots contains a 4-connected green shape on black background.

    Rule:
    - Read the green key shape as a set of occupied local coordinates inside its 4x4 key region.
    - For each 4x4 tile, read the blue occupied local coordinates and compare them against the key at two allowed scales:
      exact 1x equality, or the 2x expansion produced by replacing every key cell with its 2x2 block.
    - If a tile matches at either scale, invert occupancy inside that tile only: blue cells become black and black cells become red.
    - Leave every non-matching tile, every separator line, and the entire key area unchanged.

    Key insight:
    - Shape matching is only a trigger predicate: the matched tiles are not extracted or recolored directly; instead the match decides where the local occupancy inversion is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_runs, col_runs, _key_cols, _key_row_index, _key_coords, classes, _tile_coords = _classify_tiles(g)

    out = g.copy()
    idx = 0
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            if classes[idx] is not None:
                out[r0 : r1 + 1, c0 : c1 + 1] = _invert_tile_contents(out[r0 : r1 + 1, c0 : c1 + 1])
            idx += 1
    return out


def _place_normalized_shape(norm: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    dr = random.randint(0, TILE_SIZE - max_r - 1)
    dc = random.randint(0, TILE_SIZE - max_c - 1)
    return _translate(norm, dr, dc)


def _random_connected_shape(
    area: int,
    forbidden_exact: set[frozenset[tuple[int, int]]],
    forbidden_norm: set[frozenset[tuple[int, int]]],
) -> frozenset[tuple[int, int]]:
    for _ in range(200):
        shape = {(
            random.randint(0, TILE_SIZE - 1),
            random.randint(0, TILE_SIZE - 1),
        )}
        while len(shape) < area:
            frontier = {
                nb
                for cell in shape
                for nb in _neighbors4(cell)
                if 0 <= nb[0] < TILE_SIZE and 0 <= nb[1] < TILE_SIZE and nb not in shape
            }
            if not frontier:
                break
            shape.add(random.choice(tuple(frontier)))
        coords = frozenset(shape)
        if len(coords) != area or not _is_connected(coords):
            continue
        if coords in forbidden_exact or _normalize(coords) in forbidden_norm:
            continue
        return coords
    raise RuntimeError("failed to sample a connected distractor shape")


def _make_shift_distractor(key_coords: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    shifts = []
    for dr in range(TILE_SIZE):
        for dc in range(TILE_SIZE):
            if dr == 0 and dc == 0:
                continue
            moved = _translate(key_coords, dr, dc)
            if _fits_tile(moved):
                shifts.append(moved)
    assert shifts, "key shape should admit at least one non-zero translation"
    return random.choice(shifts)


def _make_same_area_distractor(
    key_coords: frozenset[tuple[int, int]],
    forbidden_exact: set[frozenset[tuple[int, int]]],
) -> frozenset[tuple[int, int]]:
    forbidden_norm = {_normalize(key_coords)}
    for _ in range(100):
        candidate = _place_normalized_shape(random.choice(_LINE3_SHAPES))
        if candidate in forbidden_exact or _normalize(candidate) in forbidden_norm:
            continue
        return candidate
    raise RuntimeError("failed to place same-area distractor")


def _make_area12_distractor(
    scaled_key: frozenset[tuple[int, int]],
    forbidden_exact: set[frozenset[tuple[int, int]]],
) -> frozenset[tuple[int, int]]:
    forbidden_norm = {_normalize(scaled_key)}
    for _ in range(100):
        candidate = _place_normalized_shape(random.choice(_AREA12_SHAPES))
        if candidate in forbidden_exact or _normalize(candidate) in forbidden_norm:
            continue
        return candidate
    raise RuntimeError("failed to place area-12 distractor")


def _invert_tile_contents(tile: np.ndarray) -> np.ndarray:
    blue = tile == TILE_FG
    black = tile == BG
    out = tile.copy()
    out[blue] = BG
    out[black] = INVERTED_BG
    return out


def _expected_output(grid: np.ndarray, classes: list[str | None]) -> np.ndarray:
    out = grid.copy()
    row_runs, col_runs, _key_cols = _tile_slices()
    idx = 0
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            if classes[idx] is not None:
                out[r0 : r1 + 1, c0 : c1 + 1] = _invert_tile_contents(out[r0 : r1 + 1, c0 : c1 + 1])
            idx += 1
    return out


def _render_case(
    key_coords: frozenset[tuple[int, int]],
    key_row_index: int,
    tile_payloads: list[tuple[str | None, frozenset[tuple[int, int]]]],
) -> tuple[np.ndarray, list[str | None]]:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    grid[4, :] = SEP
    for c in (4, 9, 14, 15):
        grid[:, c] = SEP

    row_runs, col_runs, key_cols = _tile_slices()
    classes: list[str | None] = []
    idx = 0
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            label, coords = tile_payloads[idx]
            classes.append(label)
            for r, c in coords:
                grid[r0 + r, c0 + c] = TILE_FG
            idx += 1

    kr0, kr1 = row_runs[key_row_index]
    kc0, _kc1 = key_cols
    for r, c in key_coords:
        grid[kr0 + r, kc0 + c] = KEY_FG

    return grid, classes


def _validate_generated(grid: np.ndarray, expected_classes: list[str | None], mode: str) -> None:
    assert grid.shape == (GRID_H, GRID_W)

    row_runs, col_runs, key_cols, key_row_index, key_coords, classes, tile_coords = _classify_tiles(grid)
    assert classes == expected_classes, "parsed matches must equal the intended layout"

    match_count = sum(label is not None for label in classes)
    assert match_count in (2, 3), "the puzzle requires 2-3 matching tiles"
    if mode == "scale1":
        assert set(label for label in classes if label is not None) == {"1x"}
    elif mode == "scale2":
        assert set(label for label in classes if label is not None) == {"2x"}
    else:
        labels = {label for label in classes if label is not None}
        assert labels == {"1x", "2x"}, "mixed modes must contain both scales"

    scaled_key = _scale2(key_coords)
    unmatched = [(coords, _normalize(coords), len(coords)) for coords, label in zip(tile_coords, classes) if label is None]
    assert any(norm == _normalize(key_coords) and coords != key_coords for coords, norm, _ in unmatched), (
        "need a translated same-shape distractor to rule out crop-based matching"
    )
    assert any(area == len(key_coords) and norm != _normalize(key_coords) for _, norm, area in unmatched), (
        "need a same-area different-shape distractor"
    )
    assert any(area == len(scaled_key) and coords != scaled_key for coords, _, area in unmatched), (
        "need a same-area 2x-scale distractor"
    )
    unmatched_norms = [norm for _, norm, _ in unmatched]
    assert len(unmatched_norms) == len(set(unmatched_norms)), "non-matching tile shapes should be distinct"

    expected = _expected_output(grid, expected_classes)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the intended rule"

    kr0, kr1 = row_runs[key_row_index]
    kc0, kc1 = key_cols
    assert np.array_equal(actual[kr0 : kr1 + 1, kc0 : kc1 + 1], grid[kr0 : kr1 + 1, kc0 : kc1 + 1]), (
        "the key region must stay unchanged"
    )
    for idx, ((coords, label), (r0, r1, c0, c1)) in enumerate(
        zip(
            zip(tile_coords, classes),
            [(rr0, rr1, cc0, cc1) for rr0, rr1 in row_runs for cc0, cc1 in col_runs],
        )
    ):
        inp_tile = grid[r0 : r1 + 1, c0 : c1 + 1]
        out_tile = actual[r0 : r1 + 1, c0 : c1 + 1]
        if label is None:
            assert np.array_equal(inp_tile, out_tile), f"unmatched tile {idx} must remain unchanged"
        else:
            assert np.array_equal(out_tile, _invert_tile_contents(inp_tile)), f"matched tile {idx} must be inverted"


def _generate_case(mode: str) -> tuple[np.ndarray, list[str | None]]:
    key_coords = random.choice(_KEY_SHAPES)
    key_row_index = random.randint(0, 1)
    scaled_key = _scale2(key_coords)

    shift = _make_shift_distractor(key_coords)
    used_exact = {key_coords, scaled_key, shift}

    same_area = _make_same_area_distractor(key_coords, used_exact)
    used_exact.add(same_area)

    area12 = _make_area12_distractor(scaled_key, used_exact)
    used_exact.add(area12)

    forbidden_norm = {
        _normalize(key_coords),
        _normalize(scaled_key),
        _normalize(shift),
        _normalize(same_area),
        _normalize(area12),
    }
    random_area = random.choice((4, 5, 6, 7, 8, 9, 10))
    random_distractor = _random_connected_shape(random_area, used_exact, forbidden_norm)
    used_exact.add(random_distractor)

    if mode == "scale1":
        matched_count = random.choice((2, 3))
        payloads = [("1x", key_coords) for _ in range(matched_count)]
    elif mode == "scale2":
        matched_count = random.choice((2, 3))
        payloads = [("2x", scaled_key) for _ in range(matched_count)]
    elif mode == "mixed2":
        payloads = [("1x", key_coords), ("2x", scaled_key)]
    else:
        if random.random() < 0.5:
            payloads = [("1x", key_coords), ("1x", key_coords), ("2x", scaled_key)]
        else:
            payloads = [("1x", key_coords), ("2x", scaled_key), ("2x", scaled_key)]

    distractors = [
        (None, shift),
        (None, same_area),
        (None, area12),
        (None, random_distractor),
    ]
    tile_payloads = payloads + distractors[: 6 - len(payloads)]
    assert len(tile_payloads) == 6
    random.shuffle(tile_payloads)
    return _render_case(key_coords, key_row_index, tile_payloads)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _MODE_CYCLE[_GEN_CALL_COUNT % len(_MODE_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected_classes = _generate_case(mode)
        try:
            _validate_generated(grid, expected_classes, mode)
            return grid
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid {mode} puzzle instance")
