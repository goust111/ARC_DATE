from __future__ import annotations

import random

import numpy as np


DARK_BG = 0
LIGHT_BG = 5
YELLOW = 4

ANCHOR_COLORS = (1, 2, 3, 6, 7, 8, 9)
PAIR_TYPES = ("row", "col", "l_hv", "l_vh")

GRID_H_MIN = 10
GRID_H_MAX = 16
GRID_W_MIN = 10
GRID_W_MAX = 16

PAIR_COUNT_CHOICES = (2, 3, 4)
PAIR_COUNT_WEIGHTS = (1, 2, 3)

SINGLE_PAIR_RETRIES = 250
CROSS_PAIR_RETRIES = 250
GEN_RETRIES = 400


def _make_checkerboard(height: int, width: int, dark_parity: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    dark_mask = (rows + cols) % 2 == dark_parity
    return np.where(dark_mask, DARK_BG, LIGHT_BG).astype(int)


def _infer_dark_parity(grid: np.ndarray) -> int:
    rows = np.arange(grid.shape[0])[:, None]
    cols = np.arange(grid.shape[1])[None, :]
    bg_mask = (grid == DARK_BG) | (grid == LIGHT_BG)

    score_even = np.count_nonzero(bg_mask & (((rows + cols) % 2 == 0) & (grid == DARK_BG)))
    score_even += np.count_nonzero(bg_mask & (((rows + cols) % 2 == 1) & (grid == LIGHT_BG)))

    score_odd = np.count_nonzero(bg_mask & (((rows + cols) % 2 == 1) & (grid == DARK_BG)))
    score_odd += np.count_nonzero(bg_mask & (((rows + cols) % 2 == 0) & (grid == LIGHT_BG)))

    dark_parity = 0 if score_even >= score_odd else 1
    expected = _make_checkerboard(grid.shape[0], grid.shape[1], dark_parity)
    assert np.array_equal(grid[bg_mask], expected[bg_mask]), "background must be a black/gray checkerboard"
    return int(dark_parity)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(int(r1), int(c)) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(int(r), int(c1)) for r in range(r1, r2 + step, step)]
    raise AssertionError("segment must be horizontal or vertical")


def _path_for_pair(
    p1: tuple[int, int],
    p2: tuple[int, int],
    dark_parity: int,
) -> list[tuple[int, int]]:
    start, end = sorted((p1, p2))
    r1, c1 = start
    r2, c2 = end

    if r1 == r2:
        return _segment(start, end)
    if c1 == c2:
        return _segment(start, end)

    if (r1 + c1) % 2 == dark_parity:
        corner = (r1, c2)
        first = _segment(start, corner)
        second = _segment(corner, end)
    else:
        corner = (r2, c1)
        first = _segment(start, corner)
        second = _segment(corner, end)
    return first + second[1:]


def _anchor_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (DARK_BG, LIGHT_BG)]
    assert YELLOW not in colors, "yellow is reserved for overlaps in the output"
    assert 2 <= len(colors) <= 4, "expected 2-4 anchor pairs"

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in colors:
        points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(points) == 2, "each anchor color must appear exactly twice"
        pairs[int(color)] = (points[0], points[1])
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every same-colored anchor pair across a checkerboard background.

    Straight-aligned pairs get straight segments. Non-aligned pairs get an L path;
    the lexicographically earlier anchor chooses horizontal-then-vertical when it
    sits on a dark square, otherwise vertical-then-horizontal. Overlaps between
    different colors turn yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dark_parity = _infer_dark_parity(g)
    pairs = _anchor_pairs(g)

    claims: dict[tuple[int, int], set[int]] = {}
    for color, (p1, p2) in pairs.items():
        for cell in _path_for_pair(p1, p2, dark_parity):
            claims.setdefault(cell, set()).add(int(color))

    out = g.copy()
    for (r, c), colors in claims.items():
        out[r, c] = YELLOW if len(colors) > 1 else next(iter(colors))
    return out


def _build_spec(
    pair_type: str,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
    dark_parity: int,
) -> dict[str, object]:
    path = _path_for_pair(start, end, dark_parity)
    return {
        "type": pair_type,
        "color": int(color),
        "points": (tuple(start), tuple(end)),
        "path": tuple(path),
        "path_set": set(path),
    }


def _place_crossing_row_col(
    height: int,
    width: int,
    row_color: int,
    col_color: int,
    dark_parity: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    del dark_parity
    for _ in range(CROSS_PAIR_RETRIES):
        cross_r = random.randint(1, height - 2)
        cross_c = random.randint(1, width - 2)

        if min(cross_c, width - 1 - cross_c, cross_r, height - 1 - cross_r) < 1:
            continue

        left = random.randint(1, cross_c)
        right = random.randint(1, width - 1 - cross_c)
        up = random.randint(1, cross_r)
        down = random.randint(1, height - 1 - cross_r)

        row_start = (cross_r, cross_c - left)
        row_end = (cross_r, cross_c + right)
        col_start = (cross_r - up, cross_c)
        col_end = (cross_r + down, cross_c)

        row_spec = _build_spec("row", row_start, row_end, row_color, 0)
        col_spec = _build_spec("col", col_start, col_end, col_color, 0)
        overlap = row_spec["path_set"] & col_spec["path_set"]
        if overlap != {(cross_r, cross_c)}:
            continue
        return row_spec, col_spec
    return None


def _sample_single_pair(
    pair_type: str,
    color: int,
    height: int,
    width: int,
    dark_parity: int,
    used_paths: set[tuple[int, int]],
    used_anchors: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(SINGLE_PAIR_RETRIES):
        if pair_type == "row":
            row = random.randint(0, height - 1)
            c1 = random.randint(0, width - 3)
            c2 = random.randint(c1 + 2, width - 1)
            start = (row, c1)
            end = (row, c2)
        elif pair_type == "col":
            col = random.randint(0, width - 1)
            r1 = random.randint(0, height - 3)
            r2 = random.randint(r1 + 2, height - 1)
            start = (r1, col)
            end = (r2, col)
        elif pair_type == "l_hv":
            r1 = random.randint(0, height - 3)
            r2 = random.randint(r1 + 2, height - 1)
            c1 = random.randint(0, width - 1)
            c2 = random.randint(0, width - 1)
            if abs(c2 - c1) < 2:
                continue
            start = (r1, c1)
            end = (r2, c2)
            if (r1 + c1) % 2 != dark_parity:
                continue
        elif pair_type == "l_vh":
            r1 = random.randint(0, height - 3)
            r2 = random.randint(r1 + 2, height - 1)
            c1 = random.randint(0, width - 1)
            c2 = random.randint(0, width - 1)
            if abs(c2 - c1) < 2:
                continue
            start = (r1, c1)
            end = (r2, c2)
            if (r1 + c1) % 2 == dark_parity:
                continue
        else:
            raise AssertionError(f"unknown pair type: {pair_type}")

        spec = _build_spec(pair_type, start, end, color, dark_parity)
        anchors = set(spec["points"])
        path_set = spec["path_set"]
        if anchors & used_anchors:
            continue
        if path_set & used_paths:
            continue
        if path_set & used_anchors:
            continue
        return spec
    return None


def _choose_pair_types() -> list[str]:
    count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    return random.sample(PAIR_TYPES, k=int(count))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_H_MIN, GRID_H_MAX)
    width = random.randint(GRID_W_MIN, GRID_W_MAX)
    dark_parity = random.randint(0, 1)
    pair_types = _choose_pair_types()
    colors = random.sample(ANCHOR_COLORS, k=len(pair_types))

    used_paths: set[tuple[int, int]] = set()
    used_anchors: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    want_cross = "row" in pair_types and "col" in pair_types and random.random() < 0.7
    color_for_type = {pair_type: color for pair_type, color in zip(pair_types, colors, strict=False)}

    if want_cross:
        cross_specs = _place_crossing_row_col(
            height,
            width,
            color_for_type["row"],
            color_for_type["col"],
            dark_parity,
        )
        if cross_specs is None:
            return None
        for spec in cross_specs:
            specs.append(spec)
            used_paths |= set(spec["path_set"])
            used_anchors |= set(spec["points"])

    remaining_types = list(pair_types)
    if want_cross:
        remaining_types.remove("row")
        remaining_types.remove("col")
    random.shuffle(remaining_types)

    for pair_type in remaining_types:
        spec = _sample_single_pair(
            pair_type,
            color_for_type[pair_type],
            height,
            width,
            dark_parity,
            used_paths,
            used_anchors,
        )
        if spec is None:
            return None
        specs.append(spec)
        used_paths |= set(spec["path_set"])
        used_anchors |= set(spec["points"])

    grid = _make_checkerboard(height, width, dark_parity)
    for spec in specs:
        color = int(spec["color"])
        p1, p2 = spec["points"]
        grid[p1] = color
        grid[p2] = color

    out = main(grid)
    has_yellow = bool(np.any(out == YELLOW))
    if want_cross != has_yellow:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
