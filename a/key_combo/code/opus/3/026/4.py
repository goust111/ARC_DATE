from __future__ import annotations

import random

import numpy as np


BG = 0
BAR_PALETTE = (1, 2, 3, 4)
BAR_COUNT_CHOICES = (3, 4)
HEIGHT_RANGE = (3, 8)
LEFT_MARGIN_RANGE = (1, 3)
RIGHT_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (2, 6)
WIDTH_CAP = 24
HEIGHT_CAP = 18
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400
FORCE_TIE_PROBABILITY = 0.4


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    H, W = g.shape

    bar_cols = [int(c) for c in np.flatnonzero(g[0] != BG)]
    assert 3 <= len(bar_cols) <= 4

    seen = np.zeros_like(g, dtype=bool)
    bars = []
    for c in bar_cols:
        color = int(g[0, c])
        h = 0
        while h < H and int(g[h, c]) == color:
            h += 1
        assert h > 0
        assert np.all(g[:h, c] == color)
        assert np.all(g[h:, c] == BG)
        bars.append({"col": c, "height": h, "color": color})
        seen[:h, c] = True

    assert len({bar["color"] for bar in bars}) == len(bars)
    assert np.array_equal(g != BG, seen), "input must contain only top-aligned one-cell-wide bars"
    bars.sort(key=lambda bar: bar["col"])
    return bars


def _contributors_for_column(bars: list[dict], col: int) -> list[tuple[int, int]]:
    contributors = []
    for i, bar in enumerate(bars):
        if col == bar["col"]:
            continue
        fill_h = int(bar["height"]) - abs(int(col) - int(bar["col"]))
        if fill_h > 0:
            contributors.append((i, fill_h))
    return contributors


def _winner_table(shape: tuple[int, int], bars: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    H, W = map(int, shape)
    best_h = np.zeros(W, dtype=int)
    best_i = np.full(W, -1, dtype=int)

    # Bars are processed left-to-right, so equal heights automatically keep the left bar.
    for i, bar in enumerate(bars):
        col = int(bar["col"])
        height = int(bar["height"])
        for c in range(max(0, col - height + 1), min(W, col + height)):
            if c == col:
                continue
            fill_h = height - abs(c - col)
            if fill_h > best_h[c]:
                best_h[c] = fill_h
                best_i[c] = i
    return best_h, best_i


def _solve_from_bars(shape: tuple[int, int], bars: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = map(int, shape)
    bars = sorted(bars, key=lambda bar: bar["col"])
    best_h, best_i = _winner_table((H, W), bars)

    out = np.zeros((H, W), dtype=int)
    n = len(bars)
    for c in range(W):
        winner = int(best_i[c])
        if winner < 0:
            continue
        fill_color = int(bars[(winner + 1) % n]["color"])
        out[: min(H, int(best_h[c])), c] = fill_color

    for bar in bars:
        out[: int(bar["height"]), int(bar["col"])] = int(bar["color"])
    return out, best_h, best_i


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each top-aligned vertical bar sideways with a height gradient while borrowing the next bar's color cyclically.

    Input structure:
    - The grid is a black background with 3 or 4 non-black vertical bars.
    - Every bar is one cell wide, starts on row 0, has a single color, and continues downward for its full height.
    - Bar colors are distinct and bars are read from left to right by their column positions.

    Rule:
    - Keep every original bar pixel unchanged.
    - For each bar, extend to both sides: at horizontal distance d, fill the top h-d cells, where h is that bar's height.
    - The extension color is not the bar's own color; it is the color of the bar immediately to its right in cyclic left-to-right order.
    - When several bars extend into the same column, keep the contribution with larger fill height; if heights tie, keep the leftmost bar's contribution.

    Key insight:
    - Shape and color come from different sources: the current bar controls the triangular height profile, but the visible extension color is borrowed from its cyclic right neighbor.
    """
    bars = _extract_bars(np.asarray(grid, dtype=int))
    out, _best_h, _best_i = _solve_from_bars(grid.shape, bars)
    return out


def _gap_bounds(h1: int, h2: int) -> tuple[int, int]:
    lo = max(3, h1 // 2 + 1, h2 // 2 + 1)
    hi = min(6, h1 + h2 - 2)
    return lo, hi


def _render_input(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    H, W = map(int, shape)
    grid = np.zeros((H, W), dtype=int)
    for bar in bars:
        grid[: int(bar["height"]), int(bar["col"])] = int(bar["color"])
    return grid


def _sample_scene(rng: random.Random) -> tuple[np.ndarray, list[dict], bool] | None:
    n_bars = rng.choice(BAR_COUNT_CHOICES)
    want_tie = rng.random() < FORCE_TIE_PROBABILITY

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        heights = [rng.randint(*HEIGHT_RANGE) for _ in range(n_bars)]
        tie_pair = None
        if want_tie:
            tie_pair = rng.randrange(n_bars - 1)
            tie_height = rng.randint(4, HEIGHT_RANGE[1])
            heights[tie_pair] = tie_height
            heights[tie_pair + 1] = tie_height

        if len(set(heights)) == 1:
            continue

        gaps = []
        ok = True
        for i in range(n_bars - 1):
            lo, hi = _gap_bounds(heights[i], heights[i + 1])
            if lo > hi:
                ok = False
                break
            if want_tie and i == tie_pair:
                choices = [g for g in range(lo, hi + 1) if g % 2 == 0 and heights[i] - g // 2 > 0]
                if not choices:
                    ok = False
                    break
                gaps.append(rng.choice(choices))
            else:
                gaps.append(rng.randint(lo, hi))
        if not ok:
            continue

        left_margin = rng.randint(*LEFT_MARGIN_RANGE)
        cols = [left_margin]
        for gap in gaps:
            cols.append(cols[-1] + gap)

        right_margin = rng.randint(*RIGHT_MARGIN_RANGE)
        min_W = cols[-1] + right_margin + 1
        if min_W > WIDTH_CAP:
            continue
        max_W = min(WIDTH_CAP, min_W + rng.randint(0, 4))
        if min_W > max_W:
            continue
        W = rng.randint(max(12, min_W), max_W) if max(12, min_W) <= max_W else min_W

        max_bar_h = max(heights)
        min_H = max(10, max_bar_h + 2)
        max_H = min(HEIGHT_CAP, max_bar_h + rng.randint(*BOTTOM_MARGIN_RANGE))
        if min_H > max_H:
            continue
        H = rng.randint(min_H, max_H)

        colors = rng.sample(list(BAR_PALETTE), n_bars)
        bars = [{"col": cols[i], "height": heights[i], "color": colors[i]} for i in range(n_bars)]
        return _render_input((H, W), bars), bars, want_tie
    return None


def _visible_extension_pixels(bars: list[dict], best_h: np.ndarray, best_i: np.ndarray, idx: int) -> int:
    hidden_by_bar = {int(bar["col"]): int(bar["height"]) for bar in bars}
    total = 0
    for col, winner in enumerate(best_i.tolist()):
        if int(winner) != idx:
            continue
        total += max(0, int(best_h[col]) - hidden_by_bar.get(int(col), 0))
    return total


def _visible_unique_columns(bars: list[dict], best_h: np.ndarray, best_i: np.ndarray, idx: int) -> int:
    hidden_by_bar = {int(bar["col"]): int(bar["height"]) for bar in bars}
    count = 0
    for col, winner in enumerate(best_i.tolist()):
        if int(winner) != idx:
            continue
        if int(best_h[col]) <= hidden_by_bar.get(int(col), 0):
            continue
        if len(_contributors_for_column(bars, col)) == 1:
            count += 1
    return count


def _analyze_columns(bars: list[dict]) -> tuple[int, int]:
    W = max(int(bar["col"]) for bar in bars) + max(int(bar["height"]) for bar in bars) + 1
    strict_overlap = 0
    tie_overlap = 0
    for col in range(W):
        contributors = _contributors_for_column(bars, col)
        if len(contributors) < 2:
            continue
        heights = sorted((fill_h for _i, fill_h in contributors), reverse=True)
        if heights[0] == heights[1]:
            tie_overlap += 1
        else:
            strict_overlap += 1
    return strict_overlap, tie_overlap


def _semantic_check(inp: np.ndarray, bars: list[dict], out: np.ndarray, best_h: np.ndarray, best_i: np.ndarray, want_tie: bool) -> bool:
    try:
        parsed = _extract_bars(inp)
    except AssertionError:
        return False
    if parsed != bars:
        return False

    strict_overlap, tie_overlap = _analyze_columns(bars)
    if strict_overlap == 0:
        return False
    if want_tie and tie_overlap == 0:
        return False

    if np.count_nonzero(out) <= np.count_nonzero(inp):
        return False
    if np.count_nonzero(out) >= int(out.size * 0.6):
        return False

    for i in range(len(bars)):
        if _visible_extension_pixels(bars, best_h, best_i, i) < 2:
            return False
        if _visible_unique_columns(bars, best_h, best_i, i) == 0:
            return False

    for col in range(inp.shape[1]):
        contributors = _contributors_for_column(bars, col)
        if not contributors:
            continue
        winner = int(best_i[col])
        assert winner >= 0
        max_h = max(fill_h for _i, fill_h in contributors)
        leftmost_max = min(i for i, fill_h in contributors if fill_h == max_h)
        if winner != leftmost_max or int(best_h[col]) != max_h:
            return False
        visible_color = int(bars[(winner + 1) % len(bars)]["color"])
        hidden = next((int(bar["height"]) for bar in bars if int(bar["col"]) == col), 0)
        if int(best_h[col]) > hidden:
            if not np.all(out[hidden:int(best_h[col]), col] == visible_color):
                return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_scene(rng)
        if sampled is None:
            continue
        inp, bars, want_tie = sampled
        out, best_h, best_i = _solve_from_bars(inp.shape, bars)
        if not _semantic_check(inp, bars, out, best_h, best_i, want_tie):
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid multi-bar cyclic-borrow scene")
