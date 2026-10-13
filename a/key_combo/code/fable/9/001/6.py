from __future__ import annotations

import random
from collections import Counter

import numpy as np


_BG = 0
_PALETTE = tuple(range(1, 10))
_QUADRANT_TO_SIGN = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}
_GEN_CALL_COUNT = 0
_GEN_RETRY_LIMIT = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _balance_candidates(counts: np.ndarray) -> list[int]:
    counts = np.asarray(counts, dtype=int)
    prefix_before = np.cumsum(counts) - counts
    total = int(counts.sum())
    return [
        int(i)
        for i in range(len(counts))
        if int(prefix_before[i]) == total - int(prefix_before[i]) - int(counts[i])
    ]


def _balance_index(mask: np.ndarray, axis: int) -> int:
    counts = np.count_nonzero(mask, axis=axis)
    candidates = _balance_candidates(counts)
    assert len(candidates) == 1, "expected a unique balance line"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Find the invisible balanced cross-crease, then fold all quadrants onto one corner by mirror overlay.

    Input structure:
    - A dominant background contains sparse multicolored foreground pixels.
    - There is exactly one balance row and one balance column such that the
      strict foreground counts above/below and left/right are equal.
    - The crease row and crease column are not drawn, and mirrored quadrant
      cells never conflict after folding.

    Rule:
    - Identify the unique balance row and balance column from foreground counts.
    - Treat their intersection as the folding origin, and map each foreground
      cell to coordinates given by its absolute row and column distances from
      that origin.
    - Return the transparent overlay on a canvas of size
      `(max_row_distance + 1, max_col_distance + 1)`.

    Key insight:
    - The fold is defined by mass balance rather than by geometric center or a
      visible marker; once the hidden cross is found, folding is just absolute
      distance remapping from all four quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    fg = g != bg
    assert np.any(fg), "input must contain foreground pixels"

    crease_r = _balance_index(fg, axis=1)
    crease_c = _balance_index(fg, axis=0)

    row_counts = np.count_nonzero(fg, axis=1)
    col_counts = np.count_nonzero(fg, axis=0)
    assert int(row_counts[crease_r]) == 0, "crease row must be invisible"
    assert int(col_counts[crease_c]) == 0, "crease column must be invisible"

    coords = np.argwhere(fg)
    row_dist = np.abs(coords[:, 0] - crease_r)
    col_dist = np.abs(coords[:, 1] - crease_c)
    out = np.full((int(row_dist.max()) + 1, int(col_dist.max()) + 1), bg, dtype=int)

    for (r, c), dr, dc in zip(coords, row_dist, col_dist):
        color = int(g[r, c])
        if out[dr, dc] != bg:
            assert out[dr, dc] == color, "folded cells must not conflict"
        out[dr, dc] = color

    return out


def _mode_for_call() -> tuple[bool, bool]:
    # Cycle through centered/off-center geometry modes so early samples expose all branches.
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1
    return (
        mode in (1, 3),  # row centered
        mode in (2, 3),  # col centered
    )


def _pick_positions(
    fold_h: int, fold_w: int, extra_pairs_a: int, extra_pairs_b: int
) -> list[tuple[int, int, str]]:
    anchors = [
        (1, 1, "TL"),
        (1, 2, "BR"),
        (2, 1, "TR"),
        (2, 2, "BL"),
    ]
    used = {(r, c) for r, c, _label in anchors}
    free = [(r, c) for r in range(1, fold_h) for c in range(1, fold_w) if (r, c) not in used]
    random.shuffle(free)

    placed = anchors[:]
    idx = 0
    for pair_labels, pair_count in ((("TL", "BR"), extra_pairs_a), (("TR", "BL"), extra_pairs_b)):
        for _ in range(pair_count):
            if idx + 2 > len(free):
                raise ValueError("not enough folded positions")
            pos_a = free[idx]
            pos_b = free[idx + 1]
            idx += 2
            labels = list(pair_labels)
            random.shuffle(labels)
            placed.append((pos_a[0], pos_a[1], labels[0]))
            placed.append((pos_b[0], pos_b[1], labels[1]))
    return placed


def _choose_side_lengths(required_a: int, required_b: int, centered: bool) -> tuple[int, int]:
    if centered:
        side = max(required_a, required_b) + random.randint(0, 2)
        return side, side

    a = required_a + random.randint(0, 3)
    b = required_b + random.randint(0, 3)
    if a == b:
        if random.random() < 0.5:
            a += 1
        else:
            b += 1
    return a, b


def _colorized_cells(labeled_positions: list[tuple[int, int, str]]) -> list[tuple[int, int, str, int]]:
    palette_size = random.randint(3, min(6, len(_PALETTE)))
    palette = random.sample(_PALETTE, palette_size)
    cells = [(dr, dc, label, random.choice(palette)) for dr, dc, label in labeled_positions]
    if len({color for *_rest, color in cells}) == 1:
        dr, dc, label, _color = cells[-1]
        cells[-1] = (dr, dc, label, palette[-1])
    return cells


def _render_input(
    top: int, bottom: int, left: int, right: int, cells: list[tuple[int, int, str, int]]
) -> np.ndarray:
    grid = np.full((top + bottom + 1, left + right + 1), _BG, dtype=int)
    crease_r = top
    crease_c = left
    for dr, dc, label, color in cells:
        sr, sc = _QUADRANT_TO_SIGN[label]
        r = crease_r + sr * dr
        c = crease_c + sc * dc
        assert grid[r, c] == _BG
        grid[r, c] = color
    return grid


def _render_expected(cells: list[tuple[int, int, str, int]]) -> np.ndarray:
    out_h = max(dr for dr, _dc, _label, _color in cells) + 1
    out_w = max(dc for _dr, dc, _label, _color in cells) + 1
    out = np.full((out_h, out_w), _BG, dtype=int)
    for dr, dc, _label, color in cells:
        assert out[dr, dc] == _BG
        out[dr, dc] = color
    return out


def _valid_structure(inp: np.ndarray, expected: np.ndarray) -> bool:
    fg = inp != _BG
    row_counts = np.count_nonzero(fg, axis=1)
    col_counts = np.count_nonzero(fg, axis=0)
    row_candidates = _balance_candidates(row_counts)
    col_candidates = _balance_candidates(col_counts)
    if len(row_candidates) != 1 or len(col_candidates) != 1:
        return False

    crease_r = row_candidates[0]
    crease_c = col_candidates[0]
    if int(row_counts[crease_r]) != 0 or int(col_counts[crease_c]) != 0:
        return False

    quadrants_seen = set()
    for r, c in np.argwhere(fg):
        if r < crease_r and c < crease_c:
            quadrants_seen.add("TL")
        elif r < crease_r and c > crease_c:
            quadrants_seen.add("TR")
        elif r > crease_r and c < crease_c:
            quadrants_seen.add("BL")
        elif r > crease_r and c > crease_c:
            quadrants_seen.add("BR")
    if quadrants_seen != set(_QUADRANT_TO_SIGN):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt(row_centered: bool, col_centered: bool) -> np.ndarray | None:
    fold_h = random.randint(5, 8)
    fold_w = random.randint(5, 8)
    extra_pairs_a = random.randint(1, 4)
    extra_pairs_b = random.randint(1, 4)
    total_cells = 4 + 2 * (extra_pairs_a + extra_pairs_b)
    if total_cells > (fold_h - 1) * (fold_w - 1):
        return None

    labeled_positions = _pick_positions(fold_h, fold_w, extra_pairs_a, extra_pairs_b)
    cells = _colorized_cells(labeled_positions)

    req_top = max(dr for dr, _dc, label, _color in cells if label in {"TL", "TR"})
    req_bottom = max(dr for dr, _dc, label, _color in cells if label in {"BL", "BR"})
    req_left = max(dc for _dr, dc, label, _color in cells if label in {"TL", "BL"})
    req_right = max(dc for _dr, dc, label, _color in cells if label in {"TR", "BR"})

    top, bottom = _choose_side_lengths(req_top, req_bottom, row_centered)
    left, right = _choose_side_lengths(req_left, req_right, col_centered)

    inp = _render_input(top, bottom, left, right, cells)
    expected = _render_expected(cells)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _valid_structure(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    row_centered, col_centered = _mode_for_call()
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt(row_centered, col_centered)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid balanced-fold input")
