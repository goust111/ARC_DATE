from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
Gate = tuple[int, int, int, int, int]  # (top_row, c0, c1, top_color, bottom_color)


def _find_gates(g: np.ndarray) -> list[Gate]:
    """Detect aligned two-row horizontal gate bars on a black background."""
    h, w = g.shape
    gates: list[Gate] = []

    for r in range(1, h - 1):
        c = 0
        while c < w:
            top_color = int(g[r, c])
            if top_color == BG:
                c += 1
                continue

            c0 = c
            while c < w and int(g[r, c]) == top_color:
                c += 1
            c1 = c - 1

            lower = g[r + 1, c0 : c1 + 1]
            if lower.size == 0:
                continue
            bottom_color = int(lower[0])
            if bottom_color == BG or bottom_color == top_color:
                continue
            if not np.all(lower == bottom_color):
                continue

            left_clear = c0 == 0 or (int(g[r, c0 - 1]) == BG and int(g[r + 1, c0 - 1]) == BG)
            right_clear = c1 == w - 1 or (int(g[r, c1 + 1]) == BG and int(g[r + 1, c1 + 1]) == BG)
            if not (left_clear and right_clear):
                continue

            gates.append((r, c0, c1, top_color, bottom_color))

    gates.sort(key=lambda gate: gate[0])
    return gates


def _trace_drop(h: int, gates: list[Gate], col: int, color: int) -> tuple[int, int, list[tuple[int, int, int, int, bool]]]:
    """Return (settle_row, final_color, events) for one falling source pixel."""
    current = int(color)
    events: list[tuple[int, int, int, int, bool]] = []
    for idx, (r, c0, c1, top_color, bottom_color) in enumerate(gates):
        if not (c0 <= int(col) <= c1):
            continue
        passed = current == int(top_color)
        events.append((idx, current, int(top_color), int(bottom_color), passed))
        if passed:
            current = int(bottom_color)
        else:
            return int(r - 1), current, events
    return int(h - 1), current, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row source pixels through two-row color gates, recoloring after each matching pass.

    Input structure:
    - The background is black, with isolated colored source pixels on the top row.
    - The middle of the grid contains several horizontal gates. Each gate is exactly
      two rows thick: a monochrome upper row and a same-width aligned monochrome
      lower row of a different color.
    - Gate rows have distinct heights, and each gate may cover only part of the columns.

    Rule:
    - Clear the top-row source pixels, then let each source fall vertically in its own column.
    - When a falling pixel reaches a gate covering that column, compare its current color
      with the gate's upper-row color. If they match, the pixel passes through the gate,
      changes to the lower-row color, and continues falling.
    - If the colors do not match, the pixel stops one cell above that gate. A pixel that
      is never stopped settles on the bottom row.
    - Preserve every gate exactly as drawn and paint each settled source in its final color.

    Key insight:
    - A gate is a conditional recoloring filter: passing one gate can create the exact
      color needed to pass a later gate, so the falling color must be updated in order.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    gates = _find_gates(g)

    out = g.copy()
    source_cols = [int(c) for c in np.flatnonzero(g[0] != BG)]
    for c in source_cols:
        out[0, c] = BG

    for c in source_cols:
        settle_r, final_color, _events = _trace_drop(h, gates, c, int(g[0, c]))
        out[settle_r, c] = int(final_color)

    return out


# --------------- generator ---------------


def _choose_gate_rows(h: int, count: int) -> list[int] | None:
    candidates = list(range(3, h - 3))
    if len(candidates) < count:
        return None
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if all(rows[i + 1] - rows[i] >= 3 for i in range(len(rows) - 1)):
            return rows
    return None


def _sample_interval(
    w: int,
    *,
    must_include: int | None = None,
    avoid: set[int] | None = None,
    min_width: int = 3,
    max_width: int | None = None,
) -> tuple[int, int] | None:
    avoid = set() if avoid is None else set(map(int, avoid))
    if max_width is None:
        max_width = min(8, w - 1)
    intervals: list[tuple[int, int]] = []
    for c0 in range(w):
        for c1 in range(c0 + min_width - 1, min(w, c0 + max_width)):
            if must_include is not None and not (c0 <= int(must_include) <= c1):
                continue
            if any(c0 <= c <= c1 for c in avoid):
                continue
            intervals.append((c0, c1))
    if not intervals:
        return None
    return random.choice(intervals)


def _columns_isolated(cols: list[int]) -> bool:
    return all(abs(a - b) > 1 for i, a in enumerate(cols) for b in cols[i + 1 :])


def _paint_gates(h: int, w: int, gates: list[Gate]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c0, c1, top_color, bottom_color in gates:
        grid[r, c0 : c1 + 1] = int(top_color)
        grid[r + 1, c0 : c1 + 1] = int(bottom_color)
    return grid


def _oracle(grid: np.ndarray, gates: list[Gate], sources: list[tuple[int, int]]) -> np.ndarray:
    h, _w = grid.shape
    out = grid.copy()
    for c, _color in sources:
        out[0, int(c)] = BG
    for c, color in sources:
        r, final_color, _events = _trace_drop(h, gates, int(c), int(color))
        out[r, int(c)] = int(final_color)
    return out


def _source_candidates_for_stop(h: int, gates: list[Gate], used_cols: set[int], w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        if any(abs(c - used) <= 1 for used in used_cols):
            continue
        for color in COLORS:
            settle_r, _final, events = _trace_drop(h, gates, c, color)
            if settle_r == h - 1 or not events:
                continue
            if events[0][4]:
                continue
            candidates.append((c, color))
    return candidates


def _extra_source_candidates(used_cols: set[int], w: int) -> list[int]:
    return [c for c in range(w) if all(abs(c - used) > 1 for used in used_cols)]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(10, 24)
    max_gates = min(5, (h - 7) // 3 + 1)
    if max_gates < 2:
        return None
    gate_count = random.randint(2, max_gates)
    rows = _choose_gate_rows(h, gate_count)
    if rows is None:
        return None

    chain_col = random.randrange(1, w - 1)
    bottom_options = [c for c in range(w) if abs(c - chain_col) > 2]
    if not bottom_options:
        return None
    bottom_col = random.choice(bottom_options)
    chain_idx = random.randint(0, gate_count - 2)

    chain_a, chain_mid, chain_final = random.sample(COLORS, 3)
    gates: list[Gate] = []
    for idx, r in enumerate(rows):
        if idx in (chain_idx, chain_idx + 1):
            interval = _sample_interval(
                w,
                must_include=chain_col,
                avoid={bottom_col},
                min_width=4,
                max_width=min(9, w - 1),
            )
            if interval is None:
                return None
            top_color = chain_a if idx == chain_idx else chain_mid
            bottom_color = chain_mid if idx == chain_idx else chain_final
        else:
            interval = _sample_interval(
                w,
                avoid={chain_col, bottom_col},
                min_width=3,
                max_width=min(8, w - 1),
            )
            if interval is None:
                return None
            top_color, bottom_color = random.sample(COLORS, 2)
        c0, c1 = interval
        gates.append((int(r), int(c0), int(c1), int(top_color), int(bottom_color)))

    grid = _paint_gates(h, w, gates)

    sources: list[tuple[int, int]] = [(chain_col, chain_a), (bottom_col, random.choice(COLORS))]
    used_cols = {chain_col, bottom_col}
    if not _columns_isolated([c for c, _color in sources]):
        return None

    stop_candidates = _source_candidates_for_stop(h, gates, used_cols, w)
    if not stop_candidates:
        return None
    stop_source = random.choice(stop_candidates)
    sources.append(stop_source)
    used_cols.add(stop_source[0])

    extras = _extra_source_candidates(used_cols, w)
    random.shuffle(extras)
    for c in extras[: random.randint(0, min(3, len(extras)))]:
        sources.append((int(c), random.choice(COLORS)))
        used_cols.add(int(c))

    if not _columns_isolated([c for c, _color in sources]):
        return None

    for c, color in sources:
        grid[0, int(c)] = int(color)

    expected = _oracle(grid, gates, sources)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    chain_trace = _trace_drop(h, gates, chain_col, chain_a)
    chain_events = chain_trace[2]
    has_chain_dependent_pass = any(passed and before != chain_a for _idx, before, _top, _bottom, passed in chain_events)
    if not has_chain_dependent_pass or chain_trace[0] != h - 1:
        return None

    bottom_trace = _trace_drop(h, gates, bottom_col, sources[1][1])
    if bottom_trace[0] != h - 1 or bottom_trace[2]:
        return None

    stop_trace = _trace_drop(h, gates, stop_source[0], stop_source[1])
    if stop_trace[0] == h - 1 or not stop_trace[2] or stop_trace[2][0][4]:
        return None

    if len({c1 - c0 + 1 for _r, c0, c1, _a, _b in gates}) < 2 and len(gates) > 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained gate drop input")
