import random
from collections import Counter

import numpy as np


BG = 0
HORIZONTAL_COLORS = (1, 2, 3)
VERTICAL_COLORS = (4, 6, 7)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def _covered_positions(length: int, limit: int) -> list[int]:
    cells: list[int] = []
    for start in range(2 * int(length), int(limit), 2 * int(length)):
        stop = min(start + int(length), int(limit))
        cells.extend(range(start, stop))
    return cells


def _find_horizontal_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bars: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(int(h)):
        color = int(grid[r, 0])
        if color == bg:
            continue
        length = 0
        while length < int(w) and int(grid[r, length]) == color:
            length += 1
        if length >= 2:
            bars.append((int(r), int(length), int(color)))
    bars.sort()
    return bars


def _find_vertical_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bars: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for c in range(int(w)):
        color = int(grid[0, c])
        if color == bg:
            continue
        length = 0
        while length < int(h) and int(grid[length, c]) == color:
            length += 1
        if length >= 2:
            bars.append((int(c), int(length), int(color)))
    bars.sort()
    return bars


def _paint_masks(
    grid: np.ndarray,
    horizontals: list[tuple[int, int, int]],
    verticals: list[tuple[int, int, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    h_mask = np.zeros((h, w), dtype=bool)
    v_mask = np.zeros((h, w), dtype=bool)
    h_fill = np.zeros((h, w), dtype=int)
    v_fill = np.zeros((h, w), dtype=int)

    for r, length, color in horizontals:
        for start in range(2 * length, int(w), 2 * length):
            stop = min(start + length, int(w))
            h_mask[r, start:stop] = True
            h_fill[r, start:stop] = int(color)

    for c, length, color in verticals:
        for start in range(2 * length, int(h), 2 * length):
            stop = min(start + length, int(h))
            v_mask[start:stop, c] = True
            v_fill[start:stop, c] = int(color)

    out = np.array(grid, dtype=int, copy=True)
    bg_mask = out == int(bg)
    only_h = bg_mask & h_mask & ~v_mask
    only_v = bg_mask & v_mask & ~h_mask
    both = bg_mask & h_mask & v_mask

    out[only_h] = h_fill[only_h]
    out[only_v] = v_fill[only_v]

    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    choose_h = (rr + cc) % 2 == 0
    out[both & choose_h] = h_fill[both & choose_h]
    out[both & ~choose_h] = v_fill[both & ~choose_h]
    return out, h_mask, v_mask, h_fill, v_fill


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    horizontals = _find_horizontal_bars(g, bg)
    verticals = _find_vertical_bars(g, bg)
    return _paint_masks(g, horizontals, verticals, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Extend left-anchored horizontal bars and top-anchored vertical bars, weaving only at periodic crossings.

    Input structure:
    - A black background contains 2-3 monochrome horizontal bars whose left end touches column 0.
    - It also contains 1-2 monochrome vertical bars whose top end touches row 0.
    - Horizontal and vertical source bars are disjoint, have length 2-4, and use disjoint color sets.

    Rule:
    - Each horizontal bar of length L repeats to the right as alternating L-cell gaps and L-cell colored segments, starting after the first L-cell gap.
    - Each vertical bar of length L repeats downward the same way.
    - Pixels reached by only one extension take that bar's color; pixels reached by both extensions weave by parity: even (r+c) uses the horizontal color, odd uses the vertical color.
    - Original source-bar pixels stay unchanged even if an extension would otherwise pass through them.

    Key insight:
    - The bar length controls both copied segment size and gap size, so the only ambiguity is at true crossing pixels, where coordinate parity decides which direction wins.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _sample_extra_row(h: int, min_row: int, used: set[int], blocked: set[int]) -> int | None:
    preferred = [
        r
        for r in range(int(min_row), int(h))
        if r not in used and r not in blocked and all(abs(r - x) >= 2 for x in used)
    ]
    if preferred:
        return int(random.choice(preferred))
    relaxed = [
        r
        for r in range(int(min_row), int(h))
        if r not in used and all(abs(r - x) >= 1 for x in used)
    ]
    if relaxed:
        return int(random.choice(relaxed))
    return None


def _scene_is_valid(grid: np.ndarray, require_protected: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    if bg != BG:
        return False

    horizontals = _find_horizontal_bars(g, bg)
    verticals = _find_vertical_bars(g, bg)
    if not (2 <= len(horizontals) <= 3 and 1 <= len(verticals) <= 2):
        return False

    h_colors = [color for _, _, color in horizontals]
    v_colors = [color for _, _, color in verticals]
    if len(set(h_colors)) != len(h_colors) or len(set(v_colors)) != len(v_colors):
        return False
    if set(h_colors) & set(v_colors):
        return False
    if not set(h_colors).issubset(HORIZONTAL_COLORS):
        return False
    if not set(v_colors).issubset(VERTICAL_COLORS):
        return False
    if any(not 2 <= length <= 4 for _, length, _ in horizontals + verticals):
        return False

    for r, h_len, _ in horizontals:
        for c, v_len, _ in verticals:
            if r < v_len and c < h_len:
                return False

    out, h_mask, v_mask, _h_fill, _v_fill = _solve(g)
    bg_mask = g == bg
    overlap = bg_mask & h_mask & v_mask
    if int(overlap.sum()) < 2:
        return False

    parity_values = {(int(r) + int(c)) % 2 for r, c in np.argwhere(overlap)}
    if parity_values != {0, 1}:
        return False

    if not np.any(bg_mask & h_mask & ~v_mask):
        return False
    if not np.any(bg_mask & v_mask & ~h_mask):
        return False
    if np.array_equal(g, out):
        return False

    protected = bool(np.any((g != bg) & (h_mask | v_mask)))
    if require_protected and not protected:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    n_h = 3 if random.random() < 0.55 else 2
    n_v = 2 if random.random() < 0.6 else 1
    require_protected = n_v == 2 and random.random() < 0.7

    weave_h_len = random.randint(2, 4)
    weave_v_len = random.randint(2, 4)
    h_lengths = [weave_h_len, weave_h_len]
    while len(h_lengths) < n_h:
        h_lengths.append(random.randint(2, 4))

    if n_v == 1:
        v_lengths = [weave_v_len]
    elif require_protected:
        v_lengths = [weave_v_len, weave_v_len]
    else:
        v_lengths = [weave_v_len, random.randint(2, 4)]

    min_row = 2 * max(v_lengths)
    if min_row >= h:
        return None

    weave_rows = [r for r in _covered_positions(weave_v_len, h) if r >= min_row]
    even_rows = [r for r in weave_rows if r % 2 == 0]
    odd_rows = [r for r in weave_rows if r % 2 == 1]
    if not even_rows or not odd_rows:
        return None
    r0 = random.choice(even_rows)
    r1 = random.choice(odd_rows)
    if r0 == r1:
        return None

    rows = [int(r0), int(r1)]
    blocked_rows = set().union(*[set(_covered_positions(length, h)) for length in v_lengths])
    if n_h == 3:
        extra_row = _sample_extra_row(h, min_row, set(rows), blocked_rows)
        if extra_row is None:
            return None
        rows.append(extra_row)

    weave_cols = _covered_positions(weave_h_len, w)
    if not weave_cols:
        return None
    c_weave = int(random.choice(weave_cols))

    if n_v == 1:
        cols = [c_weave]
    elif require_protected:
        protected_cols = [c for c in range(1, min(weave_h_len, 4)) if c != c_weave]
        if not protected_cols:
            return None
        cols = [int(random.choice(protected_cols)), c_weave]
    else:
        covered = set(weave_cols)
        candidates = [
            c
            for c in range(1, w - 1)
            if c != c_weave and abs(c - c_weave) >= 2 and c not in covered
        ]
        if not candidates:
            candidates = [c for c in range(1, w - 1) if c != c_weave and abs(c - c_weave) >= 2]
        if not candidates:
            return None
        cols = [c_weave, int(random.choice(candidates))]

    grid = np.zeros((h, w), dtype=int)
    h_colors = random.sample(HORIZONTAL_COLORS, n_h)
    v_colors = random.sample(VERTICAL_COLORS, n_v)

    for row, length, color in zip(rows, h_lengths, h_colors):
        grid[int(row), : int(length)] = int(color)
    for col, length, color in zip(cols, v_lengths, v_colors):
        if np.any(grid[: int(length), int(col)] != BG):
            return None
        grid[: int(length), int(col)] = int(color)

    if not _scene_is_valid(grid, require_protected=require_protected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
