from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
GRAY = 5
TEAL = 8
PALETTE = [0, 1, 3, 4, 6, 7, 9]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _is_isolated_pixel(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != int(bg):
                return False
    return True


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 1 <= h <= 30 and 1 <= w <= 30, "grid must stay within ARC limits"
    assert w >= 6 and h >= 6, "grid is too small for the described layout"

    bg = get_bg(g)
    slot_col = w - 2
    assert np.all(g[:, slot_col] == bg), "penultimate column must be the empty result slot"

    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.shape[0] == 1, "expected exactly one gray seed pixel"
    seed_r, seed_c = map(int, gray_coords[0])
    assert seed_c != slot_col, "seed pixel cannot occupy the result slot"
    assert _is_isolated_pixel(g, seed_r, seed_c, bg), "gray seed must be isolated"

    non_bg_colors = set(map(int, np.unique(g))) - {int(bg)}
    assert non_bg_colors <= {RED, GRAY, TEAL}, "unexpected non-background color"
    assert RED in non_bg_colors and TEAL in non_bg_colors, "both operand colors must be present"

    bars: list[tuple[int, int, int]] = []
    for c in range(w):
        if c == slot_col:
            continue
        col = g[:, c]
        fg_rows = np.flatnonzero(col != bg)
        if fg_rows.size == 0:
            continue
        colors = set(map(int, col[fg_rows]))
        if colors == {GRAY}:
            assert fg_rows.size == 1 and int(fg_rows[0]) == seed_r and c == seed_c
            continue

        assert colors <= {RED, TEAL}, "bars may only use teal or red"
        assert len(colors) == 1, "each bar column must be monochrome"
        color = int(next(iter(colors)))
        start = int(fg_rows[0])
        assert np.array_equal(fg_rows, np.arange(start, h)), "bars must be bottom-aligned contiguous columns"
        bars.append((c, color, h - start))

    assert bars, "expected at least one operand bar"

    covered = np.zeros_like(g, dtype=bool)
    covered[seed_r, seed_c] = True
    for c, color, height in bars:
        covered[h - height :, c] = True
        assert int(g[h - height, c]) == color
    assert np.array_equal(covered, g != bg), "input must contain only bars plus the gray seed"

    teal_total = sum(height for _c, color, height in bars if color == TEAL)
    red_total = sum(height for _c, color, height in bars if color == RED)
    k = teal_total - red_total
    assert k >= 1, "teal total height must exceed red total height"
    assert k <= h, "result bar must fit in the canvas"

    inner_color = TEAL if teal_total >= red_total else RED
    return {
        "grid": g,
        "bg": bg,
        "shape": (h, w),
        "slot_col": slot_col,
        "seed": (seed_r, seed_c),
        "bars": bars,
        "k": k,
        "inner_color": inner_color,
    }


def _render_output(spec: dict) -> np.ndarray:
    g = spec["grid"]
    h, _w = spec["shape"]
    slot_col = int(spec["slot_col"])
    seed_r, seed_c = map(int, spec["seed"])
    k = int(spec["k"])
    inner_color = int(spec["inner_color"])
    other_color = RED if inner_color == TEAL else TEAL

    out = g.copy()
    out[h - k :, slot_col] = GRAY

    for radius in range(1, k + 1):
        color = inner_color if radius % 2 == 1 else other_color
        for dr in range(-radius, radius + 1):
            dc = radius - abs(dr)
            for sign in (-1, 1):
                rr = seed_r + dr
                cc = seed_c + sign * dc
                if in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == int(spec["bg"]):
                    out[rr, cc] = color
            if dc == 0:
                cc = seed_c
                rr = seed_r + dr
                if in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == int(spec["bg"]):
                    out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn bar-height subtraction into a grey result column and matching concentric diamond outlines.

    Input structure:
    - A dominant background contains several one-cell-wide bottom-aligned vertical bars.
    - Teal bars contribute positively, red bars contribute negatively.
    - The penultimate column is an empty result slot, and exactly one isolated gray seed pixel appears elsewhere.
    - No gray result bar and no diamond outlines are present yet.

    Rule:
    - Compute k = total teal-bar height minus total red-bar height, guaranteed to be at least 1.
    - Fill the penultimate column from the bottom upward with k gray cells.
    - Around the gray seed, draw the clipped Manhattan-distance diamond outlines of radii 1 through k.
    - The innermost ring starts with the dominant operand color, and the ring colors alternate outward.
    - Preserve every original bar and the original gray seed pixel.

    Key insight:
    - The same arithmetic value is realized twice: once as the height of the new gray result bar and once as the number of concentric diamond rings; the operand-color dominance still matters because it sets the alternation start.
    """
    return _render_output(_parse_input(grid))


def _random_composition(total: int, parts: int, cap: int) -> list[int]:
    assert parts >= 1 and parts <= total <= parts * cap
    vals = [1] * parts
    remaining = total - parts
    for i in range(parts - 1):
        slots_left = parts - i - 1
        lo = max(0, remaining - slots_left * (cap - 1))
        hi = min(cap - 1, remaining)
        add = random.randint(lo, hi)
        vals[i] += add
        remaining -= add
    vals[-1] += remaining
    random.shuffle(vals)
    return vals


def _alternating_colors(n_teal: int, n_red: int) -> list[int]:
    assert abs(n_teal - n_red) <= 1 and n_teal >= 1 and n_red >= 1
    if n_teal > n_red:
        start = TEAL
    elif n_red > n_teal:
        start = RED
    else:
        start = random.choice([TEAL, RED])

    counts = {TEAL: n_teal, RED: n_red}
    seq: list[int] = []
    prev = None
    while counts[TEAL] or counts[RED]:
        preferred = start if prev != start and counts[start] > 0 else (RED if start == TEAL else TEAL)
        if prev == preferred or counts[preferred] == 0:
            preferred = RED if preferred == TEAL else TEAL
        assert counts[preferred] > 0
        seq.append(preferred)
        counts[preferred] -= 1
        prev = preferred
    return seq


def _choose_counts() -> tuple[int, int]:
    options = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4)]
    return random.choice(options)


def _seed_row_choices(h: int, k: int) -> list[int]:
    choices: list[int] = []
    if k <= h - 1:
        low, high = k, h - 1 - k
        if low <= high:
            choices.extend([r for r in range(low, high + 1)] * 3)
    if k >= 2:
        choices.extend(range(1, min(h - 1, k)))
        choices.extend(range(max(1, h - k), h - 1))
    if not choices:
        choices.extend(range(1, h - 1))
    return choices


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    slot_col = w - 2

    seed_r, seed_c = map(int, np.argwhere(g == GRAY)[0])
    teal_total = 0
    red_total = 0
    for c in range(w):
        if c == slot_col:
            continue
        col = g[:, c]
        if np.all(col == bg):
            continue
        if np.count_nonzero(col == GRAY):
            continue
        rows = np.flatnonzero(col != bg)
        if rows.size == 0:
            continue
        height = int(rows.size)
        color = int(col[rows[0]])
        if color == TEAL:
            teal_total += height
        elif color == RED:
            red_total += height

    k = teal_total - red_total
    out = g.copy()
    out[h - k :, slot_col] = GRAY
    for radius in range(1, k + 1):
        color = TEAL if radius % 2 == 1 else RED
        for r in range(h):
            for c in range(w):
                if abs(r - seed_r) + abs(c - seed_c) == radius and int(out[r, c]) == int(bg):
                    out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    n_teal, n_red = _choose_counts()
    m = n_teal + n_red
    k = random.randint(1, 9)

    min_w = max(12, m + 2 * k + 6)
    if min_w > 30:
        return None
    w = random.randint(min_w, 30)
    slot_col = w - 2

    min_h = max(10, k + 3)
    h = random.randint(min_h, 30)
    max_bar_height = h - 2

    red_upper = min(n_red * max_bar_height, n_teal * max_bar_height - k, 2 * h)
    if red_upper < n_red:
        return None
    red_total = random.randint(n_red, red_upper)
    teal_total = red_total + k

    teal_heights = _random_composition(teal_total, n_teal, max_bar_height)
    red_heights = _random_composition(red_total, n_red, max_bar_height)

    seed_c_min = m + k + 2
    seed_c_max = slot_col - k - 2
    if seed_c_min > seed_c_max:
        return None
    seed_c = random.randint(seed_c_min, seed_c_max)
    seed_r = random.choice(_seed_row_choices(h, k))

    left_limit = seed_c - k - 2
    if left_limit < m:
        return None
    bar_cols = sorted(random.sample(range(1, left_limit + 1), m))
    color_seq = _alternating_colors(n_teal, n_red)

    teal_pool = teal_heights[:]
    red_pool = red_heights[:]
    random.shuffle(teal_pool)
    random.shuffle(red_pool)

    g = np.full((h, w), bg, dtype=int)
    for col, color in zip(bar_cols, color_seq):
        height = teal_pool.pop() if color == TEAL else red_pool.pop()
        g[h - height :, col] = color

    g[seed_r, seed_c] = GRAY

    try:
        parsed = _parse_input(g)
    except AssertionError:
        return None

    out = main(g)
    expected = _independent_expected(g)
    if not np.array_equal(out, expected):
        return None

    if get_bg(g) != bg:
        return None
    if parsed["k"] != k:
        return None
    if np.array_equal(out, g):
        return None
    if not np.any(out == RED):
        return None
    if not np.any(out == TEAL):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
