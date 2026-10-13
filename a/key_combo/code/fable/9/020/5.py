from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _nonbg_counter(row: np.ndarray, bg: int) -> Counter:
    return Counter(int(v) for v in row if int(v) != int(bg))


def _extract_scatter_rows(grid: np.ndarray, bg: int) -> list[dict]:
    rows: list[dict] = []
    for r in range(grid.shape[0]):
        counts = _nonbg_counter(grid[r], bg)
        if not counts:
            continue
        if len(counts) != 2:
            continue
        badge_colors = [color for color, count in counts.items() if count == 1]
        if len(badge_colors) != 1:
            continue
        badge = badge_colors[0]
        main_items = [(color, count) for color, count in counts.items() if color != badge]
        if len(main_items) != 1:
            continue
        main_color, main_count = main_items[0]
        if int(main_count) < 2:
            continue
        rows.append(
            {
                "row": int(r),
                "main_color": int(main_color),
                "main_count": int(main_count),
                "badge_color": int(badge),
            }
        )
    return rows


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    r = 0
    while r < grid.shape[0]:
        counts = _nonbg_counter(grid[r], bg)
        if len(counts) != 1:
            r += 1
            continue

        color = next(iter(counts))
        r0 = r
        while r < grid.shape[0]:
            row_counts = _nonbg_counter(grid[r], bg)
            if len(row_counts) != 1 or next(iter(row_counts)) != color:
                break
            r += 1
        r1 = r - 1

        block = grid[r0 : r1 + 1]
        coords = np.argwhere(block == color)
        if coords.size == 0:
            continue
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        bbox = block[:, c0 : c1 + 1]
        hole_mask = bbox == bg
        hole_count = int(np.count_nonzero(hole_mask))
        if hole_count <= 0:
            continue
        bars.append(
            {
                "color": int(color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "hole_count": int(hole_count),
            }
        )
    return bars


def _parse_puzzle(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    scatter_rows = _extract_scatter_rows(g, bg)
    bars = _extract_bars(g, bg)

    used_rows = {info["row"] for info in scatter_rows}
    for bar in bars:
        used_rows.update(range(bar["r0"], bar["r1"] + 1))

    for r in range(g.shape[0]):
        if r in used_rows:
            continue
        if np.any(g[r] != bg):
            raise AssertionError("every non-empty row must be either a scatter row or a bar row")

    if not scatter_rows or not bars:
        raise AssertionError("expected both scatter rows and bars")
    return bg, scatter_rows, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scatter row to the bar with the same number of holes, then recolor those holes from the row's odd badge pixel.

    Input structure:
    - The lower part contains several separated monochrome horizontal bars; each bar has one or more background gaps inside its own bounding box.
    - The upper part contains one or more scatter rows; each such row has several same-color main dots plus exactly one different-color badge dot.
    - Each bar has a unique gap count, and each scatter row's main-dot count equals exactly one bar's gap count.

    Rule:
    - For each scatter row, ignore the single odd badge dot and count only the same-color main dots.
    - Pair that row with the unique bar whose gap count matches the main-dot count.
    - Fill every gap cell inside that bar with the badge dot's color, then erase all scatter rows to background.

    Key insight:
    - The majority dots choose the target by cardinality, while the lone odd badge contributes only the fill color; color similarity is a decoy, not the pairing key.
    """
    g = np.asarray(grid, dtype=int)
    bg, scatter_rows, bars = _parse_puzzle(g)

    gap_to_bar = {bar["hole_count"]: bar for bar in bars}
    if len(gap_to_bar) != len(bars):
        raise AssertionError("bar gap counts must be unique")
    if len(scatter_rows) != len(bars):
        raise AssertionError("expected one scatter row per bar")

    out = g.copy()
    matched_gap_counts: set[int] = set()
    for info in scatter_rows:
        gap_count = int(info["main_count"])
        if gap_count not in gap_to_bar:
            raise AssertionError("scatter row does not match any bar gap count")
        if gap_count in matched_gap_counts:
            raise AssertionError("multiple scatter rows matched the same bar")
        matched_gap_counts.add(gap_count)

        bar = gap_to_bar[gap_count]
        window = out[bar["r0"] : bar["r1"] + 1, bar["c0"] : bar["c1"] + 1]
        window[window == bg] = int(info["badge_color"])
        out[info["row"], :] = bg

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
BAR_COUNT_RANGE = (3, 4)
BAR_HEIGHT_CHOICES = (1, 2)
BAR_GAP_CHOICES = (1, 2)
HOLE_COUNT_MIN = 2
TOP_ROW_STRIDE = 2
GEN_RETRIES = 500


def _derangement(n: int, *, avoid: tuple[int, ...] | None = None, non_identity: bool = True) -> tuple[int, ...]:
    while True:
        perm = tuple(random.sample(range(n), n))
        if any(perm[i] == i for i in range(n)):
            continue
        if avoid is not None and any(perm[i] == avoid[i] for i in range(n)):
            continue
        if non_identity and perm == tuple(range(n)):
            continue
        return perm


def _pick_hole_counts(n_bars: int, width: int) -> list[int] | None:
    max_count = min(6, width - 4)
    choices = list(range(HOLE_COUNT_MIN, max_count + 1))
    if len(choices) < n_bars:
        return None
    for _ in range(50):
        counts = sorted(random.sample(choices, n_bars))
        if any(count + 1 not in counts for count in counts):
            return counts
    return None


def _top_row_slots(top_height: int, n_rows: int) -> list[int] | None:
    slots = list(range(1, top_height - 1, TOP_ROW_STRIDE))
    if len(slots) < n_rows:
        return None
    return sorted(random.sample(slots, n_rows))


def _choose_isolated_positions(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    picked: list[int] = []
    for col in cols:
        if all(abs(col - prev) >= 2 for prev in picked):
            picked.append(col)
            if len(picked) == count:
                return sorted(picked)
    return None


def _choose_hole_cells(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height) for c in range(1, width - 1)]
    random.shuffle(candidates)
    holes: list[tuple[int, int]] = []
    for cell in candidates:
        r, c = cell
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in holes):
            continue
        holes.append(cell)
        if len(holes) == hole_count:
            return holes
    return None


def _build_expected(spec: dict) -> np.ndarray:
    grid = np.full((spec["H"], spec["W"]), spec["bg"], dtype=int)
    for bar in spec["bars"]:
        r0, r1 = bar["r0"], bar["r1"]
        c0, c1 = bar["c0"], bar["c1"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(bar["bar_color"])
        for hr, hc in bar["holes"]:
            grid[r0 + hr, c0 + hc] = int(bar["badge_color"])
    return grid


def _build_input(spec: dict) -> np.ndarray:
    grid = _build_expected(spec)
    bg = int(spec["bg"])
    for bar in spec["bars"]:
        for hr, hc in bar["holes"]:
            grid[bar["r0"] + hr, bar["c0"] + hc] = bg
    for row_spec in spec["rows"]:
        r = row_spec["row"]
        for c in row_spec["main_cols"]:
            grid[r, c] = int(row_spec["main_color"])
        grid[r, row_spec["badge_col"]] = int(row_spec["badge_color"])
    return grid


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, scatter_rows, bars = _parse_puzzle(inp)
    except AssertionError:
        return False

    if len(scatter_rows) != len(bars):
        return False
    if len({bar["hole_count"] for bar in bars}) != len(bars):
        return False
    if any(info["main_count"] < 2 for info in scatter_rows):
        return False

    gap_to_bar = {bar["hole_count"]: bar for bar in bars}
    if {info["main_count"] for info in scatter_rows} != set(gap_to_bar):
        return False

    ordered_bars = sorted(bars, key=lambda bar: bar["r0"])
    ordered_rows = sorted(scatter_rows, key=lambda info: info["row"])
    if all(info["main_count"] == bar["hole_count"] for info, bar in zip(ordered_rows, ordered_bars)):
        return False

    bar_colors = {bar["color"] for bar in bars}
    if any(info["badge_color"] not in bar_colors for info in scatter_rows):
        return False
    if any(info["main_color"] not in bar_colors for info in scatter_rows):
        return False

    badge_color_match_same = True
    main_color_match_same = True
    color_to_gap = {bar["color"]: bar["hole_count"] for bar in bars}
    for info in scatter_rows:
        badge_gap = color_to_gap[info["badge_color"]]
        main_gap = color_to_gap[info["main_color"]]
        if badge_gap != info["main_count"]:
            badge_color_match_same = False
        if main_gap != info["main_count"]:
            main_color_match_same = False
    if badge_color_match_same or main_color_match_same:
        return False

    if not any(info["main_count"] + 1 not in gap_to_bar for info in scatter_rows):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(inp, expected):
        return False
    if any(np.array_equal(inp, np.roll(expected, shift, axis=0)) for shift in range(1, expected.shape[0])):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_bars = random.randint(*BAR_COUNT_RANGE)

    colors = list(range(10))
    bg = random.choice(colors)
    non_bg = [color for color in colors if color != bg]
    if len(non_bg) < n_bars:
        return None
    bar_colors = random.sample(non_bg, n_bars)

    hole_counts = _pick_hole_counts(n_bars, W)
    if hole_counts is None:
        return None

    heights = [random.choice(BAR_HEIGHT_CHOICES) for _ in range(n_bars)]
    gaps = [random.choice(BAR_GAP_CHOICES) for _ in range(n_bars - 1)]
    bars_total = sum(heights) + sum(gaps)
    top_height = H - bars_total
    row_slots = _top_row_slots(top_height, n_bars)
    if row_slots is None:
        return None

    badge_perm = _derangement(n_bars)
    main_perm = _derangement(n_bars, avoid=badge_perm)
    row_perm = _derangement(n_bars)

    bars: list[dict] = [None] * n_bars  # type: ignore[assignment]
    r1 = H - 1
    for idx in range(n_bars - 1, -1, -1):
        height = heights[idx]
        hole_count = hole_counts[idx]
        min_width = max(8, hole_count + 5)
        width = random.randint(min_width, W - 2)
        c0 = random.randint(1, W - width - 1)
        holes = _choose_hole_cells(height, width, hole_count)
        if holes is None:
            return None

        r0 = r1 - height + 1
        badge_color = bar_colors[badge_perm[idx]]
        bars[idx] = {
            "r0": int(r0),
            "r1": int(r1),
            "c0": int(c0),
            "c1": int(c0 + width - 1),
            "bar_color": int(bar_colors[idx]),
            "badge_color": int(badge_color),
            "hole_count": int(hole_count),
            "holes": holes,
        }
        if idx > 0:
            r1 = r0 - 1 - gaps[idx - 1]

    rows: list[dict] = []
    for order_idx, row in enumerate(row_slots):
        bar_idx = row_perm[order_idx]
        hole_count = hole_counts[bar_idx]
        positions = _choose_isolated_positions(W, hole_count + 1)
        if positions is None:
            return None
        badge_col = random.choice(positions)
        main_cols = [col for col in positions if col != badge_col]
        rows.append(
            {
                "row": int(row),
                "main_color": int(bar_colors[main_perm[bar_idx]]),
                "badge_color": int(bar_colors[badge_perm[bar_idx]]),
                "main_cols": main_cols,
                "badge_col": int(badge_col),
            }
        )

    spec = {"H": H, "W": W, "bg": bg, "bars": bars, "rows": rows}
    inp = _build_input(spec)
    expected = _build_expected(spec)
    if not _validate_semantics(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
