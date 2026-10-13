from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7)
GRID_MIN = 13
GRID_MAX = 17
TOTAL_BAR_RANGE = range(4, 7)
GEN_RETRY_LIMIT = 300

Bar = dict[str, int | bool | str]
BarId = tuple[str, int]


def _marker_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (1, 2), "expected exactly one grey marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _bar_axes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    horizontal_rows = [
        r for r in range(grid.shape[0]) if bool(np.all((grid[r, :] != BACKGROUND) & (grid[r, :] != MARKER)))
    ]
    vertical_cols = [
        c for c in range(grid.shape[1]) if bool(np.all((grid[:, c] != BACKGROUND) & (grid[:, c] != MARKER)))
    ]
    assert len(horizontal_rows) >= 2, "expected at least two horizontal bars"
    assert len(vertical_cols) >= 2, "expected at least two vertical bars"
    return horizontal_rows, vertical_cols


def _bar_colors(grid: np.ndarray, horizontal_rows: list[int], vertical_cols: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    non_vertical_mask = np.ones(grid.shape[1], dtype=bool)
    non_vertical_mask[vertical_cols] = False
    non_horizontal_mask = np.ones(grid.shape[0], dtype=bool)
    non_horizontal_mask[horizontal_rows] = False
    assert bool(np.any(non_vertical_mask)), "need background columns to read horizontal colors"
    assert bool(np.any(non_horizontal_mask)), "need background rows to read vertical colors"

    horizontal_colors: dict[int, int] = {}
    for row in horizontal_rows:
        colors = {int(v) for v in grid[row, non_vertical_mask]}
        assert len(colors) == 1, "horizontal bar color must be unambiguous off crossings"
        color = colors.pop()
        assert color not in (BACKGROUND, MARKER)
        horizontal_colors[int(row)] = color

    vertical_colors: dict[int, int] = {}
    for col in vertical_cols:
        colors = {int(v) for v in grid[non_horizontal_mask, col]}
        assert len(colors) == 1, "vertical bar color must be unambiguous off crossings"
        color = colors.pop()
        assert color not in (BACKGROUND, MARKER)
        vertical_colors[int(col)] = color

    visible_bar_colors = set(horizontal_colors.values()) | set(vertical_colors.values())
    assert len(visible_bar_colors) == len(horizontal_rows) + len(vertical_cols), "bars must have distinct colors"
    return horizontal_colors, vertical_colors


def _candidate_bars(
    horizontal_rows: list[int],
    vertical_cols: list[int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
    marker_row: int,
    marker_col: int,
) -> tuple[list[BarId], dict[BarId, int]]:
    bars: list[BarId] = []
    colors: dict[BarId, int] = {}

    for row in horizontal_rows:
        if row % 2 == marker_row % 2:
            bar_id = ("h", int(row))
            bars.append(bar_id)
            colors[bar_id] = int(horizontal_colors[row])
    for col in vertical_cols:
        if col % 2 == marker_col % 2:
            bar_id = ("v", int(col))
            bars.append(bar_id)
            colors[bar_id] = int(vertical_colors[col])

    assert len(bars) >= 3, "expected at least three parity-selected bars"
    assert any(kind == "h" for kind, _ in bars), "candidate set needs a horizontal bar"
    assert any(kind == "v" for kind, _ in bars), "candidate set needs a vertical bar"
    return bars, colors


def _candidate_edges(
    grid: np.ndarray,
    candidate_colors: dict[BarId, int],
) -> dict[BarId, set[BarId]]:
    horizontal_ids = [bar_id for bar_id in candidate_colors if bar_id[0] == "h"]
    vertical_ids = [bar_id for bar_id in candidate_colors if bar_id[0] == "v"]
    higher: dict[BarId, set[BarId]] = {bar_id: set() for bar_id in candidate_colors}

    for h_id in horizontal_ids:
        row = h_id[1]
        h_color = candidate_colors[h_id]
        for v_id in vertical_ids:
            col = v_id[1]
            v_color = candidate_colors[v_id]
            visible = int(grid[row, col])
            assert visible in (h_color, v_color), "crossing must show one of the two candidate bars"
            if visible == h_color:
                higher[v_id].add(h_id)
            else:
                higher[h_id].add(v_id)

    return higher


def _reachable_higher(higher: dict[BarId, set[BarId]], start: BarId) -> set[BarId]:
    seen: set[BarId] = set()
    stack = list(higher[start])
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(higher[cur] - seen)
    return seen


def _top_candidate_color(grid: np.ndarray) -> int:
    marker_row, marker_col = _marker_coord(grid)
    horizontal_rows, vertical_cols = _bar_axes(grid)
    horizontal_colors, vertical_colors = _bar_colors(grid, horizontal_rows, vertical_cols)
    candidate_ids, candidate_colors = _candidate_bars(
        horizontal_rows, vertical_cols, horizontal_colors, vertical_colors, marker_row, marker_col
    )
    higher = _candidate_edges(grid, candidate_colors)

    tops = []
    for bar_id in candidate_ids:
        assert bar_id not in _reachable_higher(higher, bar_id), "layer constraints must be acyclic"
        if not _reachable_higher(higher, bar_id):
            tops.append(bar_id)

    assert len(tops) == 1, "parity-filtered bars must have a unique top layer"
    return int(candidate_colors[tops[0]])


def main(grid: np.ndarray) -> np.ndarray:
    """Return the top color among the bars selected by the grey marker's row/column parity.

    Input structure:
    - The background is black, with one grey marker pixel on an uncovered background cell.
    - Foreground objects are distinct one-pixel-wide full-width horizontal bars and full-height vertical bars.
    - At each horizontal/vertical crossing, the visible color is the higher-layer bar.

    Rule:
    - Read the marker row parity and select only horizontal bars on rows with the same parity.
    - Read the marker column parity and select only vertical bars on columns with the same parity.
    - Use the visible colors at crossings between the selected horizontal and vertical bars to infer which selected bars are above others, then output the unique selected top color as a 1x1 grid.

    Key insight:
    - The answer is not the global top bar; the grey marker first filters the bars by coordinate parity, and only that filtered substack competes for the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    top_color = _top_candidate_color(g)
    return np.array([[top_color]], dtype=int)


def _sample_grid_size() -> tuple[int, int]:
    return random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)


def _parity_positions(length: int) -> dict[int, list[int]]:
    return {
        0: [idx for idx in range(length) if idx % 2 == 0],
        1: [idx for idx in range(length) if idx % 2 == 1],
    }


def _feasible_configs(h: int, w: int, row_parity: int, col_parity: int) -> list[tuple[int, int, int, int]]:
    row_slots = _parity_positions(h)
    col_slots = _parity_positions(w)
    configs: list[tuple[int, int, int, int]] = []

    for total_bars in TOTAL_BAR_RANGE:
        for h_count in range(2, total_bars - 1):
            v_count = total_bars - h_count
            if v_count < 2:
                continue
            for cand_h in range(1, h_count + 1):
                for cand_v in range(1, v_count + 1):
                    if cand_h + cand_v < 3:
                        continue
                    noncand_h = h_count - cand_h
                    noncand_v = v_count - cand_v
                    if cand_h > len(row_slots[row_parity]) - 1:
                        continue
                    if noncand_h > len(row_slots[1 - row_parity]):
                        continue
                    if cand_v > len(col_slots[col_parity]) - 1:
                        continue
                    if noncand_v > len(col_slots[1 - col_parity]):
                        continue
                    configs.append((h_count, v_count, cand_h, cand_v))

    return configs


def _pick_positions(h: int, w: int) -> tuple[list[int], list[int], int, int]:
    row_slots = _parity_positions(h)
    col_slots = _parity_positions(w)
    row_parity = random.randint(0, 1)
    col_parity = random.randint(0, 1)
    configs = _feasible_configs(h, w, row_parity, col_parity)
    assert configs, "grid size must admit a valid parity-filtered layout"
    h_count, v_count, cand_h, cand_v = random.choice(configs)

    cand_rows = random.sample(row_slots[row_parity], cand_h)
    marker_row = random.choice([r for r in row_slots[row_parity] if r not in cand_rows])
    noncand_rows = random.sample(row_slots[1 - row_parity], h_count - cand_h)

    cand_cols = random.sample(col_slots[col_parity], cand_v)
    marker_col = random.choice([c for c in col_slots[col_parity] if c not in cand_cols])
    noncand_cols = random.sample(col_slots[1 - col_parity], v_count - cand_v)

    rows = sorted(cand_rows + noncand_rows)
    cols = sorted(cand_cols + noncand_cols)
    assert marker_row not in rows and marker_col not in cols
    return rows, cols, marker_row, marker_col


def _make_bars(rows: list[int], cols: list[int], marker_row: int, marker_col: int) -> list[Bar]:
    colors = random.sample(BAR_COLORS, len(rows) + len(cols))
    bars: list[Bar] = []
    for row in rows:
        bars.append(
            {
                "kind": "h",
                "axis": int(row),
                "candidate": bool(row % 2 == marker_row % 2),
                "color": int(colors.pop()),
            }
        )
    for col in cols:
        bars.append(
            {
                "kind": "v",
                "axis": int(col),
                "candidate": bool(col % 2 == marker_col % 2),
                "color": int(colors.pop()),
            }
        )
    return bars


def _ordered_candidate_bars(candidates: list[Bar]) -> tuple[list[Bar], int]:
    by_kind = {
        "h": [bar for bar in candidates if bar["kind"] == "h"],
        "v": [bar for bar in candidates if bar["kind"] == "v"],
    }
    winner_kind = random.choice([kind for kind, bars in by_kind.items() if bars])
    winner = random.choice(by_kind[winner_kind])
    same_lower = [bar for bar in by_kind[winner_kind] if bar is not winner]
    opposite = by_kind["v" if winner_kind == "h" else "h"][:]
    random.shuffle(opposite)

    if same_lower:
        witness_count = random.randint(1, len(opposite))
        upper_opposite = opposite[:witness_count]
        lower_part = opposite[witness_count:] + same_lower
    else:
        upper_opposite = []
        lower_part = opposite

    random.shuffle(lower_part)
    random.shuffle(upper_opposite)
    return lower_part + upper_opposite + [winner], int(winner["color"])


def _merge_preserving_order(first: list[Bar], second: list[Bar]) -> list[Bar]:
    i = 0
    j = 0
    merged: list[Bar] = []
    while i < len(first) or j < len(second):
        if i == len(first):
            merged.append(second[j])
            j += 1
        elif j == len(second):
            merged.append(first[i])
            i += 1
        elif random.random() < 0.5:
            merged.append(first[i])
            i += 1
        else:
            merged.append(second[j])
            j += 1
    return merged


def _paint_bars(h: int, w: int, bars_in_order: list[Bar], marker_row: int, marker_col: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for bar in bars_in_order:
        color = int(bar["color"])
        axis = int(bar["axis"])
        if bar["kind"] == "h":
            grid[axis, :] = color
        else:
            grid[:, axis] = color
    assert int(grid[marker_row, marker_col]) == BACKGROUND, "marker must sit on uncovered background"
    grid[marker_row, marker_col] = MARKER
    return grid


def _build_order_and_expected(bars: list[Bar]) -> tuple[list[Bar], int]:
    candidates = [bar for bar in bars if bar["candidate"]]
    noncandidates = [bar for bar in bars if not bar["candidate"]]
    candidate_order, expected = _ordered_candidate_bars(candidates)

    if noncandidates and random.random() < 0.5:
        global_top = random.choice(noncandidates)
        lower_noncandidates = [bar for bar in noncandidates if bar is not global_top]
        return _merge_preserving_order(candidate_order, lower_noncandidates) + [global_top], expected

    return _merge_preserving_order(candidate_order[:-1], noncandidates) + [candidate_order[-1]], expected


def _attempt_generate() -> np.ndarray | None:
    h, w = _sample_grid_size()
    rows, cols, marker_row, marker_col = _pick_positions(h, w)
    bars = _make_bars(rows, cols, marker_row, marker_col)
    bars_in_order, expected = _build_order_and_expected(bars)
    grid = _paint_bars(h, w, bars_in_order, marker_row, marker_col)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out, np.array([[expected]], dtype=int)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
