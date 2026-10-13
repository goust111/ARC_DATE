import random

import numpy as np


BG = 0
DIAL_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _all_neighbors(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_dial(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if np.all(grid[r0 : r0 + 3, c0 : c0 + 3] != BG):
                candidates.append((r0, c0))
    assert len(candidates) == 1, f"expected exactly one solid 3x3 dial, got {candidates}"
    return candidates[0]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return [(r1, c) for c in range(lo, hi + 1)]
    assert c1 == c2
    lo, hi = sorted((r1, r2))
    return [(r, c1) for r in range(lo, hi + 1)]


def _parse_pairs(grid: np.ndarray, r0: int, c0: int):
    dial_rows = range(r0, r0 + 3)
    dial_cols = range(c0, c0 + 3)
    outside = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if not (r0 <= int(r) <= r0 + 2 and c0 <= int(c) <= c0 + 2)
    }

    pairs = []
    used = set()

    for r in dial_rows:
        left = sorted(c for rr, c in outside if rr == r and c < c0)
        right = sorted(c for rr, c in outside if rr == r and c > c0 + 2)
        if not left and not right:
            continue
        assert len(left) == 1 and len(right) == 1, "each dial row can host at most one horizontal pair"
        lc, rc = left[0], right[0]
        assert (lc + rc) % 2 == 0
        mid_c = (lc + rc) // 2
        assert c0 <= mid_c <= c0 + 2
        p1 = (r, lc)
        p2 = (r, rc)
        pairs.append((p1, p2, (r, mid_c)))
        used.add(p1)
        used.add(p2)

    for c in dial_cols:
        top = sorted(r for r, cc in outside if cc == c and r < r0)
        bottom = sorted(r for r, cc in outside if cc == c and r > r0 + 2)
        if not top and not bottom:
            continue
        assert len(top) == 1 and len(bottom) == 1, "each dial column can host at most one vertical pair"
        tr, br = top[0], bottom[0]
        assert (tr + br) % 2 == 0
        mid_r = (tr + br) // 2
        assert r0 <= mid_r <= r0 + 2
        p1 = (tr, c)
        p2 = (br, c)
        pairs.append((p1, p2, (mid_r, c)))
        used.add(p1)
        used.add(p2)

    assert pairs, "expected at least one pair"
    assert used == outside, "every non-dial foreground pixel must belong to exactly one pair"
    return pairs


def _render_output(grid: np.ndarray, r0: int, c0: int, pairs) -> np.ndarray:
    out = np.zeros_like(grid)
    dial = grid[r0 : r0 + 3, c0 : c0 + 3]
    for p1, p2, mid in pairs:
        color = int(dial[mid[0] - r0, mid[1] - c0])
        for cell in _segment_cells(p1, p2):
            current = int(out[cell])
            assert current in (BG, color), "different-colored lines cannot overlap without an order rule"
            out[cell] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each opposite-side pixel pair and recolor the whole segment from the dial cell at its midpoint.

    Input structure:
    - The background is black.
    - One solid 3x3 color dial appears somewhere away from the border; each of its nine cells is colored.
    - Every other foreground object is a single pixel, and each pair lies on one dial row or one dial column with its midpoint landing inside the dial.

    Rule:
    - Find every horizontal or vertical opposite-side pair whose midpoint falls on the 3x3 dial.
    - Draw the complete straight segment between the pair endpoints.
    - Color the entire segment, including endpoints and midpoint, with the color already present in the dial cell at that midpoint; erase every dial cell not covered by any drawn segment.

    Key insight:
    - The endpoint colors are decoys: the only color source is the dial entry addressed by the geometric midpoint of each pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    r0, c0 = _find_dial(g)
    pairs = _parse_pairs(g, r0, c0)
    return _render_output(g, r0, c0, pairs)


def _isolated(points: list[tuple[int, int]]) -> bool:
    point_set = set(points)
    for r, c in points:
        for nr, nc in _all_neighbors(r, c):
            if (nr, nc) in point_set:
                return False
    return True


def _far_from_existing(candidate: tuple[int, int], existing: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for er, ec in existing:
        if max(abs(r - er), abs(c - ec)) <= 1:
            return False
    return True


def _sample_horizontal_pairs(h: int, w: int, r0: int, c0: int):
    row_ids = sorted(random.sample(range(3), 2))
    chosen = []
    endpoints = []
    for row_id in row_ids:
        row = r0 + row_id
        options = []
        mid_choices = random.sample(range(3), 3)
        for mid_id in mid_choices:
            mid_c = c0 + mid_id
            min_d = max(mid_c - (c0 - 2), (c0 + 4) - mid_c)
            max_d = min(mid_c, (w - 1) - mid_c)
            for d in range(min_d, max_d + 1):
                left = (row, mid_c - d)
                right = (row, mid_c + d)
                if _far_from_existing(left, endpoints) and _far_from_existing(right, endpoints):
                    options.append((mid_id, left, right))
        if not options:
            return None
        mid_id, left, right = random.choice(options)
        chosen.append((left, right, (row, c0 + mid_id)))
        endpoints.extend([left, right])
    return chosen


def _sample_vertical_pairs(h: int, w: int, r0: int, c0: int):
    col_ids = sorted(random.sample(range(3), 2))
    chosen = []
    endpoints = []
    for col_id in col_ids:
        col = c0 + col_id
        options = []
        mid_choices = random.sample(range(3), 3)
        for mid_id in mid_choices:
            mid_r = r0 + mid_id
            min_d = max(mid_r - (r0 - 2), (r0 + 4) - mid_r)
            max_d = min(mid_r, (h - 1) - mid_r)
            for d in range(min_d, max_d + 1):
                top = (mid_r - d, col)
                bottom = (mid_r + d, col)
                if _far_from_existing(top, endpoints) and _far_from_existing(bottom, endpoints):
                    options.append((mid_id, top, bottom))
        if not options:
            return None
        mid_id, top, bottom = random.choice(options)
        chosen.append((top, bottom, (r0 + mid_id, col)))
        endpoints.extend([top, bottom])
    return chosen


def _make_input_from_pairs(h: int, w: int, r0: int, c0: int, dial: np.ndarray, pairs) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r0 + 3, c0 : c0 + 3] = dial
    for p1, p2, mid in pairs:
        line_color = int(dial[mid[0] - r0, mid[1] - c0])
        choices = [color for color in DIAL_COLORS if color != line_color]
        grid[p1] = random.choice(choices)
        grid[p2] = random.choice(choices)
    return grid


def _covered_dial_cells(output: np.ndarray, r0: int, c0: int) -> int:
    return int(np.count_nonzero(output[r0 : r0 + 3, c0 : c0 + 3]))


def _validate_generated_instance(grid: np.ndarray, expected: np.ndarray):
    r0, c0 = _find_dial(grid)
    pairs = _parse_pairs(grid, r0, c0)
    outside = [
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if not (r0 <= int(r) <= r0 + 2 and c0 <= int(c) <= c0 + 2)
    ]
    assert len(outside) == 4
    assert _isolated(outside)
    for r, c in outside:
        assert r < r0 - 1 or r > r0 + 3 or c < c0 - 1 or c > c0 + 3
    produced = main(grid)
    assert np.array_equal(produced, expected)
    covered = _covered_dial_cells(produced, r0, c0)
    assert 0 < covered < 9
    assert len(np.unique(produced[produced != BG])) == 2
    assert len(pairs) == 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 24)
    w = random.randint(11, 24)
    r0 = random.randint(4, h - 7)
    c0 = random.randint(4, w - 7)
    dial = np.array(random.sample(DIAL_COLORS, 9), dtype=int).reshape(3, 3)
    mode = random.choice(("horizontal", "vertical"))
    if mode == "horizontal":
        pairs = _sample_horizontal_pairs(h, w, r0, c0)
    else:
        pairs = _sample_vertical_pairs(h, w, r0, c0)
    if pairs is None:
        return None

    grid = _make_input_from_pairs(h, w, r0, c0, dial, pairs)
    expected = _render_output(grid, r0, c0, pairs)
    _validate_generated_instance(grid, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid midpoint-dial puzzle")
