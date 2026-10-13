from __future__ import annotations

import numpy as np


PAIR_COLORS = (1, 2, 3, 4)
NEXT_COLOR = {1: 2, 2: 3, 3: 4, 4: 1}
CROSS_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRY_LIMIT = 400

_GEN_CALL_COUNT = 0


def _cross_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in CROSS_DELTAS]


def _cross_box_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}


def _pair_line_cells(pair: dict) -> set[tuple[int, int]]:
    (r1, c1), (r2, c2) = pair["endpoints"]
    cells: set[tuple[int, int]] = set()
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            cells.add((r1, c))
    else:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            cells.add((r, c1))
    return cells


def _pair_reserved_cells(pair: dict) -> set[tuple[int, int]]:
    return _pair_line_cells(pair) | _cross_box_cells(pair["midpoint"])


def _discover_pairs(grid: np.ndarray, *, strict_layout: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != 0)
    assert 2 <= len(colors) <= 4, "expected 2-4 foreground colors"
    assert set(colors).issubset(PAIR_COLORS), "foreground colors must come from 1-4"

    pairs: list[dict] = []
    endpoint_rows_by_color: dict[int, set[int]] = {}
    endpoint_cols_by_color: dict[int, set[int]] = {}

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, "each foreground color must mark exactly one pixel pair"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))

        same_row = a[0] == b[0]
        same_col = a[1] == b[1]
        assert same_row ^ same_col, "each pair must be horizontal or vertical"

        if same_row:
            left, right = sorted((a, b), key=lambda rc: rc[1])
            gap = right[1] - left[1]
            midpoint = (left[0], (left[1] + right[1]) // 2)
            assert np.all(g[left[0], left[1] + 1 : right[1]] == 0), "pair interior must be black"
            orientation = "h"
        else:
            top, bottom = sorted((a, b), key=lambda rc: rc[0])
            gap = bottom[0] - top[0]
            midpoint = ((top[0] + bottom[0]) // 2, top[1])
            assert np.all(g[top[0] + 1 : bottom[0], top[1]] == 0), "pair interior must be black"
            orientation = "v"

        assert gap % 2 == 0 and gap >= 6, "pair spacing must be even and leave room for a centered cross"
        assert 1 <= midpoint[0] < g.shape[0] - 1 and 1 <= midpoint[1] < g.shape[1] - 1, "cross must fit"

        pair = {
            "color": color,
            "orientation": orientation,
            "endpoints": (a, b),
            "midpoint": midpoint,
            "gap": gap,
        }
        for endpoint in pair["endpoints"]:
            assert endpoint not in _cross_box_cells(midpoint) - {midpoint}, "endpoints must stay outside the cross box"

        pairs.append(pair)
        endpoint_rows_by_color[color] = {a[0], b[0]}
        endpoint_cols_by_color[color] = {a[1], b[1]}

    if strict_layout:
        reserved: list[set[tuple[int, int]]] = []
        for i, pair_i in enumerate(pairs):
            rows_i = endpoint_rows_by_color[pair_i["color"]]
            cols_i = endpoint_cols_by_color[pair_i["color"]]
            box_i = _cross_box_cells(pair_i["midpoint"])
            reserved_i = _pair_reserved_cells(pair_i)

            for pair_j in pairs[i + 1 :]:
                rows_j = endpoint_rows_by_color[pair_j["color"]]
                cols_j = endpoint_cols_by_color[pair_j["color"]]
                box_j = _cross_box_cells(pair_j["midpoint"])

                assert rows_i.isdisjoint(rows_j), "different pairs must not share endpoint rows"
                assert cols_i.isdisjoint(cols_j), "different pairs must not share endpoint columns"
                assert box_i.isdisjoint(box_j), "3x3 cross boxes must not overlap"

            reserved.append(reserved_i)

        for i, cells_i in enumerate(reserved):
            for cells_j in reserved[i + 1 :]:
                assert cells_i.isdisjoint(cells_j), "pairs must be visually separated"

    return sorted(pairs, key=lambda pair: pair["color"])


def _render_expected(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for pair in pairs:
        color = int(pair["color"])
        (r1, c1), (r2, c2) = pair["endpoints"]
        if r1 == r2:
            c0, c1 = sorted((c1, c2))
            out[r1, c0 + 1 : c1] = color
        else:
            r0, r1 = sorted((r1, r2))
            out[r0 + 1 : r1, c1] = color

    for pair in pairs:
        cross_color = NEXT_COLOR[int(pair["color"])]
        for r, c in _cross_cells(pair["midpoint"]):
            out[r, c] = cross_color
        for r, c in pair["endpoints"]:
            out[r, c] = int(pair["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each aligned same-color pixel pair, then cover its midpoint with a cyclically recolored 3x3 cross.

    Input structure:
    - A black-background grid contains 2-4 foreground colors chosen from blue, red, green, and yellow.
    - Each foreground color appears exactly twice as a single-pixel pair.
    - Within a pair, the two pixels share one row or one column, are separated by an even distance, and leave room for a centered 3x3 cross.

    Rule:
    - For every same-color pair, fill all black cells strictly between the two endpoints along their shared row or column using the pair color.
    - Compute the midpoint of that segment and place a plus-sign cross there: center, up, down, left, and right.
    - Color the cross with the cyclic successor of the pair color: 1->2, 2->3, 3->4, 4->1.
    - The cross overwrites any line cells underneath it, but the two original endpoints remain in their original color.

    Key insight:
    - The endpoints carry three separate cues at once: geometry defines the midpoint, alignment defines the line direction, and the endpoint color determines a different cross color through a fixed cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    pairs = _discover_pairs(g, strict_layout=True)
    return _render_expected(g, pairs)


def _generation_plan(rng: np.random.Generator) -> list[tuple[int, str]]:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    if call_index < 2:
        colors = list(rng.permutation(np.array(PAIR_COLORS)))
        orientations = ["h", "h", "v", "v"]
        rng.shuffle(orientations)
        return list(zip(colors, orientations))

    pair_count = int(rng.choice([2, 3, 4], p=[0.25, 0.35, 0.40]))
    colors = list(rng.choice(PAIR_COLORS, size=pair_count, replace=False))

    if pair_count == 2:
        patterns = (["h", "v"], ["h", "h"], ["v", "v"])
        weights = np.array([0.5, 0.25, 0.25], dtype=float)
        orientations = list(patterns[int(rng.choice(len(patterns), p=weights))])
    else:
        orientations = [str(rng.choice(["h", "v"])) for _ in range(pair_count)]
        if len(set(orientations)) == 1 and rng.random() < 0.7:
            flip_index = int(rng.integers(pair_count))
            orientations[flip_index] = "v" if orientations[flip_index] == "h" else "h"

    rng.shuffle(colors)
    rng.shuffle(orientations)
    return list(zip(colors, orientations))


def _candidate_placements(
    shape: tuple[int, int],
    orientation: str,
    used_rows: set[int],
    used_cols: set[int],
    reserved_cells: set[tuple[int, int]],
) -> list[dict]:
    h, w = map(int, shape)
    candidates: list[dict] = []

    if orientation == "h":
        for r in range(1, h - 1):
            if r in used_rows:
                continue
            for c0 in range(0, w - 6):
                if c0 in used_cols:
                    continue
                for c1 in range(c0 + 6, w):
                    if c1 in used_cols:
                        continue
                    if (c1 - c0) % 2 != 0:
                        continue
                    pair = {
                        "orientation": "h",
                        "endpoints": ((r, c0), (r, c1)),
                        "midpoint": (r, (c0 + c1) // 2),
                    }
                    if _pair_reserved_cells(pair) & reserved_cells:
                        continue
                    candidates.append(pair)
    else:
        for c in range(1, w - 1):
            if c in used_cols:
                continue
            for r0 in range(0, h - 6):
                if r0 in used_rows:
                    continue
                for r1 in range(r0 + 6, h):
                    if r1 in used_rows:
                        continue
                    if (r1 - r0) % 2 != 0:
                        continue
                    pair = {
                        "orientation": "v",
                        "endpoints": ((r0, c), (r1, c)),
                        "midpoint": ((r0 + r1) // 2, c),
                    }
                    if _pair_reserved_cells(pair) & reserved_cells:
                        continue
                    candidates.append(pair)

    return candidates


def _independent_semantic_check(grid: np.ndarray) -> bool:
    try:
        pairs = _discover_pairs(grid, strict_layout=True)
    except AssertionError:
        return False

    out = _render_expected(grid, pairs)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, solved):
        return False
    if np.array_equal(out, grid):
        return False

    for pair in pairs:
        color = int(pair["color"])
        mid_r, mid_c = pair["midpoint"]
        cross_color = NEXT_COLOR[color]
        for cell in _cross_cells(pair["midpoint"]):
            if int(out[cell]) != cross_color:
                return False
        for cell in pair["endpoints"]:
            if int(out[cell]) != color:
                return False
        if pair["orientation"] == "h":
            left_c, right_c = sorted((pair["endpoints"][0][1], pair["endpoints"][1][1]))
            if mid_c - left_c < 3 or right_c - mid_c < 3:
                return False
            if not np.all(out[mid_r, left_c + 1 : mid_c - 1] == color):
                return False
            if not np.all(out[mid_r, mid_c + 2 : right_c] == color):
                return False
        else:
            top_r, bottom_r = sorted((pair["endpoints"][0][0], pair["endpoints"][1][0]))
            if mid_r - top_r < 3 or bottom_r - mid_r < 3:
                return False
            if not np.all(out[top_r + 1 : mid_r - 1, mid_c] == color):
                return False
            if not np.all(out[mid_r + 2 : bottom_r, mid_c] == color):
                return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    plan = _generation_plan(rng)
    pair_count = len(plan)
    min_size = 15 if pair_count == 4 else 13
    h = int(rng.integers(min_size, 20))
    w = int(rng.integers(min_size, 20))

    grid = np.zeros((h, w), dtype=int)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    reserved_cells: set[tuple[int, int]] = set()
    placed_pairs: list[dict] = []

    for color, orientation in rng.permutation(np.array(plan, dtype=object)):
        candidates = _candidate_placements(grid.shape, str(orientation), used_rows, used_cols, reserved_cells)
        if not candidates:
            return None

        pair = candidates[int(rng.integers(len(candidates)))]
        pair["color"] = int(color)
        placed_pairs.append(pair)

        for r, c in pair["endpoints"]:
            grid[r, c] = int(color)

        used_rows.update(r for r, _c in pair["endpoints"])
        used_cols.update(c for _r, c in pair["endpoints"])
        reserved_cells |= _pair_reserved_cells(pair)

    if len({pair["orientation"] for pair in placed_pairs}) == 1 and len(placed_pairs) >= 3 and rng.random() < 0.35:
        return None
    if not _independent_semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
