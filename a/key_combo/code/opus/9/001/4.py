from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

LEFT_COLORS = (GREEN, YELLOW, PINK, ORANGE)
GEN_RETRY_LIMIT = 500


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "input must contain exactly one full-height grey separator"
    return cols[0]


def _bar_height(col: np.ndarray, color: int) -> int:
    rows = np.flatnonzero(col == color)
    assert rows.size > 0, "bar column must contain colored cells"
    r0 = int(rows[0])
    height = int(rows.size)
    assert np.all(col[:r0] == BLACK), "bars must have black background above them"
    assert np.all(col[r0:] == color), "bars must be solid and bottom-aligned"
    return height


def _shift_from_bars(region: np.ndarray) -> int:
    total = 0
    seen = 0
    for c in range(region.shape[1]):
        col = region[:, c]
        non_black = np.unique(col[col != BLACK])
        if non_black.size == 0:
            continue
        assert non_black.size == 1, "each operand column must be monochromatic"
        color = int(non_black[0])
        assert color in (TEAL, RED), "only teal/red bars are allowed in the arithmetic zone"
        seen += 1
        height = _bar_height(col, color)
        total += height if color == TEAL else -height
    assert seen >= 2, "arithmetic zone must contain multiple bars"
    assert total > 0, "shift must be positive"
    return total


def _infer_tile(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    visible = np.argwhere(region != BLACK)
    assert visible.size > 0, "periodic panel must have visible cells"

    for p in range(1, h + 1):
        for q in range(1, w + 1):
            tile = np.full((p, q), -1, dtype=int)
            ok = True
            for r, c in visible:
                rr, cc = int(r) % p, int(c) % q
                value = int(region[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = value
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                return tile

    raise AssertionError("no complete periodic tile fits the visible cells")


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None] % tile.shape[0]
    cols = np.arange(w)[None, :] % tile.shape[1]
    return tile[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the left periodic panel using a tile whose row phase is shifted by the bar arithmetic result.

    Input structure:
    - The grid is split by a single full-height grey separator column.
    - Left of the separator is a rectangular periodic color table made from
      green/yellow/pink/orange cells, with some cells replaced by black holes.
    - Right of the separator is a black arithmetic zone containing bottom-aligned
      one-cell-wide teal and red vertical bars.

    Rule:
    - Sum teal bar heights and subtract red bar heights to obtain a positive row
      offset S.
    - Infer the smallest periodic tile consistent with all non-black cells in the
      left panel, requiring every tile phase to be visible at least once.
    - Circularly shift that tile downward by S rows, then repaint the entire left
      panel from the shifted tile by row/column phase.
    - Return only the repaired left panel.

    Key insight:
    - The left side reveals the base repeating tile, but the right-side bar
      arithmetic tells you that the repair must use a row-shifted phase of that
      tile rather than the standard unshifted completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    tile = _infer_tile(left)
    shift = _shift_from_bars(right)
    assert 0 < shift < tile.shape[0], "shift must be strictly between 0 and the row period"
    shifted_tile = np.roll(tile, shift=shift, axis=0)
    return _repeat_tile(shifted_tile, left.shape)


def _minimal_row_period(tile: np.ndarray) -> int:
    h = tile.shape[0]
    for p in range(1, h + 1):
        ok = True
        for r in range(h):
            if not np.array_equal(tile[r], tile[r % p]):
                ok = False
                break
        if ok:
            return p
    raise AssertionError("unreachable")


def _minimal_col_period(tile: np.ndarray) -> int:
    w = tile.shape[1]
    for q in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(tile[:, c], tile[:, c % q]):
                ok = False
                break
        if ok:
            return q
    raise AssertionError("unreachable")


def _sample_tile(p: int, q: int) -> np.ndarray | None:
    colors = random.sample(LEFT_COLORS, k=random.randint(2, 3))
    tile = np.array([[random.choice(colors) for _ in range(q)] for _ in range(p)], dtype=int)
    if len(set(map(int, tile.flatten()))) < 2:
        return None
    if _minimal_row_period(tile) != p:
        return None
    if _minimal_col_period(tile) != q:
        return None
    return tile


def _bounded_composition(total: int, parts: int, upper: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None
    out: list[int] = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        lo = max(1, remaining - slots * upper)
        hi = min(upper, remaining - slots)
        out.append(random.randint(lo, hi))
        remaining -= out[-1]
    random.shuffle(out)
    return out


def _sample_bars(shift: int, max_height: int) -> list[tuple[int, int]] | None:
    bar_count = random.randint(3, 5)
    teal_count = random.randint(1, bar_count - 1)
    red_count = bar_count - teal_count

    red_sum_lo = max(red_count, teal_count - shift)
    red_sum_hi = min(red_count * max_height, teal_count * max_height - shift)
    if red_sum_lo > red_sum_hi:
        return None

    red_sum = random.randint(red_sum_lo, red_sum_hi)
    teal_sum = red_sum + shift
    teal_heights = _bounded_composition(teal_sum, teal_count, max_height)
    red_heights = _bounded_composition(red_sum, red_count, max_height)
    if teal_heights is None or red_heights is None:
        return None

    bars = [(TEAL, h) for h in teal_heights] + [(RED, h) for h in red_heights]
    random.shuffle(bars)
    return bars


def _phase_counts(shape: tuple[int, int], p: int, q: int) -> np.ndarray:
    h, w = shape
    counts = np.zeros((p, q), dtype=int)
    for r in range(h):
        for c in range(w):
            counts[r % p, c % q] += 1
    return counts


def _sample_holes(shape: tuple[int, int], p: int, q: int) -> np.ndarray | None:
    h, w = shape
    hidden = np.zeros((h, w), dtype=bool)
    visible_counts = _phase_counts(shape, p, q)
    target = random.randint(max(5, (h * w) // 7), max(6, (h * w) // 3))

    rect_attempts = random.randint(2, 5)
    for _ in range(rect_attempts):
        rh = random.randint(1, min(3, h))
        rw = random.randint(1, min(3, w))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        cells = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw) if not hidden[r, c]]
        if not cells:
            continue
        take = np.zeros((p, q), dtype=int)
        for r, c in cells:
            take[r % p, c % q] += 1
        if np.any(visible_counts - take < 1):
            continue
        hidden[r0 : r0 + rh, c0 : c0 + rw] = True
        visible_counts -= take
        if int(hidden.sum()) >= target:
            break

    order = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(order)
    for r, c in order:
        if int(hidden.sum()) >= target:
            break
        if hidden[r, c]:
            continue
        rr, cc = r % p, c % q
        if visible_counts[rr, cc] <= 1:
            continue
        hidden[r, c] = True
        visible_counts[rr, cc] -= 1

    if int(hidden.sum()) < max(4, (h * w) // 9):
        return None
    if np.any(visible_counts < 1):
        return None
    return hidden


def _build_input(
    base_tile: np.ndarray,
    shift: int,
    left_shape: tuple[int, int],
    bars: list[tuple[int, int]],
    holes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_full = _repeat_tile(base_tile, left_shape)
    left_inp = left_full.copy()
    left_inp[holes] = BLACK

    h, w_left = left_shape
    right_w = 2 * len(bars) + 1
    grid = np.zeros((h, w_left + 1 + right_w), dtype=int)
    grid[:, :w_left] = left_inp
    grid[:, w_left] = GREY
    for i, (color, height) in enumerate(bars):
        col = w_left + 1 + 2 * i
        grid[h - height :, col] = color

    shifted = _repeat_tile(np.roll(base_tile, shift=shift, axis=0), left_shape)
    return grid, shifted


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, p: int, q: int, shift: int) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    sep = _find_separator_col(inp)
    left = inp[:, :sep]
    base_tile = _infer_tile(left)
    if base_tile.shape != (p, q):
        return False
    standard = _repeat_tile(base_tile, left.shape)
    if np.array_equal(standard, expected):
        return False
    for wrong_shift in range(base_tile.shape[0]):
        if wrong_shift == shift:
            continue
        wrong = _repeat_tile(np.roll(base_tile, shift=wrong_shift, axis=0), left.shape)
        if np.array_equal(wrong, expected):
            return False
    right = inp[:, sep + 1 :]
    if _shift_from_bars(right) != shift:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    p = random.randint(2, 4)
    q = random.randint(2, 4)
    tile = _sample_tile(p, q)
    if tile is None:
        return None

    left_h = random.randint(max(7, p + 3), 14)
    left_w = random.randint(max(7, q + 3), 15)
    shift = random.randint(1, p - 1)
    max_bar_height = left_h - 1
    bars = _sample_bars(shift, max_bar_height)
    if bars is None:
        return None

    holes = _sample_holes((left_h, left_w), p, q)
    if holes is None:
        return None

    inp, expected = _build_input(tile, shift, (left_h, left_w), bars, holes)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(inp, expected, p, q, shift):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
