# Combo variant of cb2d8a2c (index 19)
# combo_spec: D12/1+D15/0
# Name: 链式激活+路径擦除障碍

import itertools
import random

import numpy as np


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


# Design analysis for the fused variant
# a) Concrete behavior:
#    - Original task: every bar is fully visible up front; one global path avoids all keep-out zones,
#      then blue pixels turn red.
#    - Fused task: only the first bar is fully drawn. Each later bar is a dormant template on its own
#      row: wall-side blue radius markers plus a single red tip at the future open end. The solver
#      reconstructs the full bar from that template, but the output reveals it only after the previous
#      bar is bypassed. Every bypass also shortens the current bar by exactly its blue count from the
#      open side, leaving a red stump. Example in canonical top-start orientation:
#      row 4: full left bar `B R B R R R`
#      row 9: dormant right bar `R . B . B . B` (only tip + blue markers are visible)
#      Output: row 4 becomes `R R R . . .`; row 9 becomes `. . R R R R R` after activation and erasure.
# b) Complexity calibration:
#    - Dormant bars keep the same row, wall attachment, encoded blue count, and final extent, so the
#      hidden information is still visually recoverable by humans. There are only 2-4 bars and 1-4
#      blue-count levels, which keeps the rule readable but non-trivial.
# c) Broken original assumptions:
#    - The original solver assumes all obstacles are already active and therefore usable in one global
#      Dijkstra pass. That no longer holds. This solver instead processes bars in sorted main-axis
#      order and makes one rectangular detour per bar.
#    - The original solver assumes obstacle components are contiguous. Dormant bars are sparse, so this
#      version parses one bar per occupied row instead of connected components.
# d) Placement legality guaranteed by the generator:
#    - Bars stay on distinct rows, alternate walls, satisfy the same blue-radius spacing bounds as the
#      base task, and every new bar covers the bypass lane produced by the previous one.
# e/f/g) Visual observability and cleanliness:
#    - Input/output pairs visibly show two coupled effects: BG->RED filling on dormant rows (activation)
#      and colored->BG trimming at the exposed end (erasure). The output uses only the original palette,
#      and each bar remains a single clean horizontal segment, so the chain is readable from a few cases.


def _active_row_colors(length: int, blue_count: int, side: int) -> np.ndarray:
    """Return the full visible bar colors for one row."""
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        return colors
    return colors[::-1]


def _latent_row_colors(length: int, blue_count: int, side: int) -> np.ndarray:
    """Dormant template: keep only blue markers plus the future open-end tip."""
    full = _active_row_colors(length, blue_count, side)
    latent = np.full(length, BG, dtype=int)
    latent[full == BLUE] = BLUE
    tip_idx = length - 1 if side == 0 else 0
    latent[tip_idx] = RED
    return latent


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-start) grids to the requested start edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    """Rotate any valid input so the green start lies on the top edge."""
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, starts[0])
    h, w = grid.shape
    if sr == 0:
        return np.array(grid, copy=True), "top"
    if sr == h - 1:
        return np.flipud(grid).copy(), "bottom"
    if sc == 0:
        return grid.T.copy(), "left"
    if sc == w - 1:
        return np.flipud(grid.T).copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    """Undo `_to_canonical`."""
    return _orient_grid(grid, edge)


def _scan_bars(canonical: np.ndarray) -> list[dict[str, int]]:
    """Parse one bar per occupied row in canonical top-start orientation."""
    h, w = canonical.shape
    bars: list[dict[str, int]] = []
    for row in range(h):
        cols = [c for c in range(w) if int(canonical[row, c]) in (BLUE, RED)]
        if not cols:
            continue
        left_touch = 0 in cols
        right_touch = (w - 1) in cols
        assert left_touch ^ right_touch, "each bar row must touch exactly one side wall"
        side = 0 if left_touch else 1
        length = max(cols) + 1 if side == 0 else w - min(cols)
        blue_count = int(np.count_nonzero(canonical[row] == BLUE))
        assert BLUE_PER_BAR_LO <= blue_count <= BLUE_PER_BAR_HI
        bars.append({"row": row, "side": side, "length": length, "blue_count": blue_count})
    assert BAR_COUNT_LO <= len(bars) <= BAR_COUNT_HI
    return bars


def _bar_span(bar: dict[str, int], width: int) -> tuple[int, int]:
    """Inclusive bar span on the cross axis."""
    if bar["side"] == 0:
        return 0, bar["length"] - 1
    return width - bar["length"], width - 1


def _pass_column(bar: dict[str, int], width: int) -> int:
    """Deterministic bypass lane for one bar in canonical orientation."""
    if bar["side"] == 0:
        return bar["length"] + bar["blue_count"]
    return width - bar["length"] - bar["blue_count"] - 1


def _append_vertical(path: list[tuple[int, int]], to_row: int) -> None:
    """Extend the current path endpoint vertically."""
    r0, c0 = path[-1]
    step = 1 if to_row >= r0 else -1
    for rr in range(r0 + step, to_row + step, step):
        path.append((rr, c0))


def _append_horizontal(path: list[tuple[int, int]], to_col: int) -> None:
    """Extend the current path endpoint horizontally."""
    r0, c0 = path[-1]
    step = 1 if to_col >= c0 else -1
    for cc in range(c0 + step, to_col + step, step):
        path.append((r0, cc))


def _build_canonical_path(
    canonical: np.ndarray,
    bars: list[dict[str, int]],
) -> list[tuple[int, int]]:
    """Build the chain-activation path by making one local detour per bar."""
    h, w = canonical.shape
    starts = np.argwhere(canonical == GREEN)
    sr, sc = map(int, starts[0])
    assert sr == 0, "canonical inputs must start on the top edge"

    path: list[tuple[int, int]] = [(sr, sc)]
    for idx, bar in enumerate(bars):
        left, right = _bar_span(bar, w)
        cur_r, cur_c = path[-1]
        assert left <= cur_c <= right, "the current descent lane must be blocked by the next bar"

        turn_row = bar["row"] - bar["blue_count"] - 1
        exit_row = bar["row"] + bar["blue_count"] + 1
        lane_col = _pass_column(bar, w)

        assert 0 <= lane_col < w
        assert turn_row >= cur_r
        _append_vertical(path, turn_row)
        _append_horizontal(path, lane_col)
        _append_vertical(path, exit_row)

        if idx + 1 < len(bars):
            next_left, next_right = _bar_span(bars[idx + 1], w)
            assert next_left <= lane_col <= next_right

    _append_vertical(path, h - 1)
    return path


def _paint_output_canonical(
    canonical: np.ndarray,
    bars: list[dict[str, int]],
    path: list[tuple[int, int]],
) -> np.ndarray:
    """Reveal dormant bars, shorten them from the open side, then paint the green path."""
    h, w = canonical.shape
    out = np.array(canonical, copy=True)

    for bar in bars:
        row = bar["row"]
        length = bar["length"]
        shortened = length - bar["blue_count"]
        assert shortened >= 1
        if bar["side"] == 0:
            out[row, :length] = BG
            out[row, :shortened] = RED
        else:
            out[row, w - length :] = BG
            out[row, w - shortened :] = RED

    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[canonical == GREEN] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a green path through a bar-by-bar activation chain, where each bypass also erases the bar's exposed tip.

    Input structure:
    - bg = azure (8). One green (3) start pixel sits on a border edge.
    - Obstacle rows use red/blue fragments. The first bar is fully visible; later bars are dormant templates made of
      wall-side blue markers plus one red tip that still encodes the future full bar.

    Rule:
    - Read bars in the inward travel order. A bar's blue-pixel count is still its detour radius.
    - Route the green path around the currently active bar with a rectangular right-angle detour; clearing one bar
      activates the next dormant template.
    - After activation, each bar becomes a solid red segment but loses exactly `blue_count` cells from its open side,
      as if the passing route scraped away the exposed end.

    Key insight: The sparse dormant rows already contain both numbers the solver needs: blue markers say how far to stay
    away, and the far red tip says how long the hidden bar will be once the chain reaches it.
    """
    canonical, edge = _to_canonical(np.array(grid, copy=True))
    bars = _scan_bars(canonical)
    path = _build_canonical_path(canonical, bars)
    out_canonical = _paint_output_canonical(canonical, bars, path)
    return _from_canonical(out_canonical, edge)


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all detours with one blank row after the last bar."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform non-negative composition of `total` into `parts` buckets."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    """Pick blue counts that fit the available main-axis size."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Original physical bar bounds, kept for the fused variant."""
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _choose_lengths(cross_size: int, blues: tuple[int, ...]) -> list[int] | None:
    """Choose lengths so each bar blocks the descent lane coming from the previous one."""
    ranges = [_length_bounds(cross_size, blue) for blue in blues]
    if any(bounds is None for bounds in ranges):
        return None
    typed_ranges = [bounds for bounds in ranges if bounds is not None]

    def backtrack(i: int, chosen: list[int]) -> list[int] | None:
        if i == len(blues):
            return list(chosen)

        lo, hi = typed_ranges[i]
        if i > 0:
            lo = max(lo, cross_size - chosen[-1] - blues[i - 1])
        if i + 1 < len(blues):
            lo = max(lo, cross_size - blues[i] - typed_ranges[i + 1][1])
        if lo > hi:
            return None

        values = list(range(lo, hi + 1))
        random.shuffle(values)
        for length in values:
            chosen.append(length)
            solved = backtrack(i + 1, chosen)
            if solved is not None:
                return solved
            chosen.pop()
        return None

    return backtrack(0, [])


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int, dormant: bool) -> None:
    """Paint one canonical bar row."""
    w = grid.shape[1]
    colors = _latent_row_colors(length, blue_count, side) if dormant else _active_row_colors(length, blue_count, side)
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
    lengths: list[int],
) -> np.ndarray | None:
    """Build canonical input: first bar active, later bars dormant."""
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    bars: list[dict[str, int]] = []
    for i, (row, blue_count, length) in enumerate(zip(rows, blues, lengths)):
        side = (first_side + i) % 2
        _paint_bar(grid, row, side, length, blue_count, dormant=(i != 0))
        bars.append({"row": row, "side": side, "length": length, "blue_count": blue_count})

    first_left, first_right = _bar_span(bars[0], cross_size)
    start_col = random.randint(first_left + 1, first_right - 1)
    grid[0, start_col] = GREEN

    # Validate the chain geometry directly in canonical space.
    _build_canonical_path(grid, bars)
    return grid


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output hits the opposite border."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    lengths = _choose_lengths(cross_size, blues)
    if lengths is None:
        return None

    first_side = random.randrange(2)
    canonical = _canonical_grid(main_size, cross_size, blues, first_side, lengths)
    if canonical is None:
        return None

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cb2d8a2c input")
