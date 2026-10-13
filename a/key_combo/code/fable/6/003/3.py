from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg


_RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 400
BAR_COUNT_MIN = 4
BAR_COUNT_MAX_EXCLUSIVE = 7
GRID_HEIGHT_MIN = 18
GRID_HEIGHT_MAX = 27
STRIDE_OPTIONS = (3, 4)
MARGIN_OPTIONS = (1, 2)
DOT_COLOR_POOL = tuple(range(10))


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    height = int(grid.shape[0])
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)

    bars: list[dict[str, int]] = []
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        area = int(comp["area"])
        color = int(comp["color"])
        if c0 != c1:
            continue
        if r1 != height - 1:
            continue
        if area != r1 - r0 + 1:
            continue
        if not bool(np.all(grid[r0 : r1 + 1, c0] == color)):
            continue
        bars.append({"color": color, "col": c0, "height": area})

    bars.sort(key=lambda bar: bar["col"])
    assert len(bars) >= 3
    assert len({bar["col"] for bar in bars}) == len(bars)
    assert len({bar["color"] for bar in bars}) == len(bars)

    if len(bars) >= 3:
        gaps = np.diff([bar["col"] for bar in bars]).tolist()
        assert len(set(map(int, gaps))) == 1

    return bars


def _instruction_counts(grid: np.ndarray, bars: list[dict[str, int]]) -> list[int]:
    counts: list[int] = []
    for bar in bars:
        color = int(bar["color"])
        count = int(np.sum(grid == color)) - int(bar["height"])
        assert count >= 0
        counts.append(count)
    return counts


def _fuel_counts_by_slot(grid: np.ndarray, bars: list[dict[str, int]], bg: int) -> tuple[list[int], int | None]:
    bar_colors = {int(bar["color"]) for bar in bars}
    non_bar = [c for c in colors_present(grid, exclude={bg}) if int(c) not in bar_colors]
    assert len(non_bar) <= 1
    fuel_color = int(non_bar[0]) if non_bar else None
    if fuel_color is None:
        return [0] * len(bars), None
    return [int(np.sum(grid[:, int(bar["col"])] == fuel_color)) for bar in bars], fuel_color


def _fuel_gain(slot_fuel_counts: list[int], start_idx: int, steps: int, mode: str = "true") -> int:
    n = len(slot_fuel_counts)
    steps = int(steps)
    start_idx = int(start_idx)

    if mode == "shift_only":
        return 0
    if mode == "distance":
        return max(0, steps - 1)
    if mode == "true":
        return sum(int(slot_fuel_counts[(start_idx + t) % n]) for t in range(1, steps))
    if mode == "binary":
        return sum(1 for t in range(1, steps) if int(slot_fuel_counts[(start_idx + t) % n]) > 0)
    if mode == "inclusive":
        if steps == 0:
            return int(slot_fuel_counts[start_idx])
        return sum(int(slot_fuel_counts[(start_idx + t) % n]) for t in range(0, steps + 1))
    raise ValueError(f"unknown fuel mode: {mode}")


def _render_output(
    shape: tuple[int, int],
    bg: int,
    bars: list[dict[str, int]],
    shifts: list[int],
    slot_fuel_counts: list[int],
    *,
    mode: str = "true",
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), int(bg), dtype=int)
    slot_cols = [int(bar["col"]) for bar in bars]
    n = len(bars)

    for i, bar in enumerate(bars):
        steps = int(shifts[i])
        dest_idx = (int(i) + steps) % n
        gain = _fuel_gain(slot_fuel_counts, int(i), steps, mode=mode)
        final_height = int(bar["height"]) + int(gain)
        assert 1 <= final_height <= height
        out[height - final_height : height, slot_cols[dest_idx]] = int(bar["color"])

    return out


def _try_render_output(
    shape: tuple[int, int],
    bg: int,
    bars: list[dict[str, int]],
    shifts: list[int],
    slot_fuel_counts: list[int],
    *,
    mode: str,
) -> np.ndarray | None:
    try:
        return _render_output(shape, bg, bars, shifts, slot_fuel_counts, mode=mode)
    except AssertionError:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each bottom-aligned vertical bar right by its same-color dot count and make it taller by the fuel dots on the swept intermediate slot columns.

    Input structure:
    - A dominant background contains several one-cell-wide, bottom-aligned vertical bars in equally spaced slot columns.
    - Every bar color is unique; extra single-pixel dots of a bar's own color are instruction dots for that bar.
    - A second dot type uses one neutral color not shared by any bar, and every neutral fuel dot lies in some slot column.

    Rule:
    - Read the bars from left to right as ordered slots.
    - For each bar, count its same-color instruction dots and move that whole bar right by exactly that many slots with cyclic wraparound.
    - While traversing from the original slot to the destination slot, count every neutral fuel dot in the intermediate slot columns only; add that count to the bar's height from the top, then clear all dots in the output.

    Key insight:
    - The bar's color identifies which scattered dots encode its shift, while height growth depends on the full swept arc of intermediate slot columns rather than on only the source or destination column.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _extract_bars(g, bg)
    shifts = _instruction_counts(g, bars)
    slot_fuel_counts, _fuel_color = _fuel_counts_by_slot(g, bars, bg)
    return _render_output(g.shape, bg, bars, shifts, slot_fuel_counts, mode="true")


def _sample_palette(n_bars: int) -> tuple[int, int, list[int]]:
    palette = list(DOT_COLOR_POOL)
    bg = int(_RNG.choice(palette))
    palette.remove(bg)
    fuel = int(_RNG.choice(palette))
    palette.remove(fuel)
    bar_colors = [int(c) for c in _RNG.choice(palette, size=n_bars, replace=False)]
    return bg, fuel, bar_colors


def _sample_shifts(n_bars: int) -> list[int] | None:
    idx = np.arange(n_bars)
    for _ in range(500):
        perm = _RNG.permutation(n_bars)
        shifts = ((perm - idx) % n_bars).astype(int)
        if len(set(map(int, shifts.tolist()))) < 3:
            continue
        if not bool(np.any(shifts == 0)):
            continue
        if not bool(np.any(shifts == 1)):
            continue
        if not bool(np.any(shifts >= 2)):
            continue
        if not any(int(i) + int(steps) >= n_bars and int(steps) > 0 for i, steps in enumerate(shifts.tolist())):
            continue
        return shifts.tolist()
    return None


def _sample_slot_fuel_counts(shifts: list[int]) -> list[int] | None:
    n_bars = len(shifts)
    for _ in range(800):
        counts = _RNG.integers(0, 4, size=n_bars).astype(int).tolist()
        if max(counts) < 2:
            continue
        if not any(int(v) == 0 for v in counts):
            continue

        gains = [_fuel_gain(counts, i, int(steps), mode="true") for i, steps in enumerate(shifts)]
        if max(gains) == 0:
            continue
        if not any(int(steps) > 1 and gains[i] > 0 for i, steps in enumerate(shifts)):
            continue
        if not any(
            int(steps) > 1 and any(int(counts[(i + t) % n_bars]) >= 2 for t in range(1, int(steps)))
            for i, steps in enumerate(shifts)
        ):
            continue
        if not any(
            int(steps) > 2 and any(int(counts[(i + t) % n_bars]) == 0 for t in range(1, int(steps)))
            for i, steps in enumerate(shifts)
        ):
            continue

        binary = [_fuel_gain(counts, i, int(steps), mode="binary") for i, steps in enumerate(shifts)]
        inclusive = [_fuel_gain(counts, i, int(steps), mode="inclusive") for i, steps in enumerate(shifts)]
        distance = [_fuel_gain(counts, i, int(steps), mode="distance") for i, steps in enumerate(shifts)]
        if gains == binary or gains == inclusive or gains == distance:
            continue
        return counts
    return None


def _sample_shape(n_bars: int) -> tuple[int, list[int], int] | None:
    for _ in range(200):
        stride = int(_RNG.choice(np.array(STRIDE_OPTIONS)))
        left = int(_RNG.choice(np.array(MARGIN_OPTIONS)))
        right = int(_RNG.choice(np.array(MARGIN_OPTIONS)))
        slot_cols = [left + stride * i for i in range(n_bars)]
        width = int(slot_cols[-1] + right + 1)
        if width <= 30:
            return width, slot_cols, stride
    return None


def _sample_heights(n_bars: int, slot_fuel_counts: list[int], shifts: list[int]) -> tuple[int, list[int]] | None:
    max_gain = max(_fuel_gain(slot_fuel_counts, i, int(steps), mode="true") for i, steps in enumerate(shifts))
    max_fuel_per_slot = max(slot_fuel_counts)

    for _ in range(300):
        height = int(_RNG.integers(GRID_HEIGHT_MIN, GRID_HEIGHT_MAX))
        max_initial = min(
            11,
            height - max_gain - 1,
            height - 2 * max_fuel_per_slot - 3,
        )
        choices = np.arange(2, max_initial + 1)
        if len(choices) < n_bars:
            continue
        heights = [int(v) for v in _RNG.choice(choices, size=n_bars, replace=False)]
        if len(set(heights)) != n_bars:
            continue
        return height, heights
    return None


def _dot_can_fit(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    height, width = map(int, grid.shape)
    if not (0 <= int(r) < height and 0 <= int(c) < width):
        return False
    if int(grid[int(r), int(c)]) != int(bg):
        return False
    r0 = max(0, int(r) - 1)
    r1 = min(height - 1, int(r) + 1)
    c0 = max(0, int(c) - 1)
    c1 = min(width - 1, int(c) + 1)
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)))


def _place_fuel_dots(
    grid: np.ndarray,
    bg: int,
    fuel_color: int,
    slot_cols: list[int],
    heights: list[int],
    slot_fuel_counts: list[int],
) -> bool:
    canvas_height = int(grid.shape[0])
    for col, bar_height, count in zip(slot_cols, heights, slot_fuel_counts, strict=False):
        if int(count) == 0:
            continue
        max_row = canvas_height - int(bar_height) - 2
        if max_row < 0:
            return False

        placed = False
        for _ in range(80):
            parity = int(_RNG.integers(0, 2))
            rows = [r for r in range(parity, max_row + 1, 2) if _dot_can_fit(grid, bg, r, int(col))]
            if len(rows) < int(count):
                continue
            chosen = [int(r) for r in _RNG.choice(np.array(rows), size=int(count), replace=False)]
            for r in chosen:
                grid[int(r), int(col)] = int(fuel_color)
            placed = True
            break
        if not placed:
            return False
    return True


def _list_dot_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(height - 1):
        for c in range(width):
            if _dot_can_fit(grid, bg, r, c):
                candidates.append((int(r), int(c)))
    return candidates


def _place_instruction_dots(grid: np.ndarray, bg: int, bar_colors: list[int], shifts: list[int]) -> bool:
    for color, count in zip(bar_colors, shifts, strict=False):
        for _ in range(int(count)):
            candidates = _list_dot_candidates(grid, bg)
            if not candidates:
                return False
            idx = int(_RNG.integers(len(candidates)))
            r, c = candidates[idx]
            grid[int(r), int(c)] = int(color)
    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    bars = _extract_bars(grid, bg)
    shifts = _instruction_counts(grid, bars)
    slot_fuel_counts, fuel_color = _fuel_counts_by_slot(grid, bars, bg)
    if fuel_color is None:
        return False

    n_bars = len(bars)
    if len(set(shifts)) < 3:
        return False
    if not any(int(v) == 0 for v in shifts):
        return False
    if not any(int(v) == 1 for v in shifts):
        return False
    if not any(int(v) >= 2 for v in shifts):
        return False
    if not any(int(i) + int(v) >= n_bars and int(v) > 0 for i, v in enumerate(shifts)):
        return False
    if max(slot_fuel_counts) < 2:
        return False
    if not any(int(v) == 0 for v in slot_fuel_counts):
        return False

    true_out = _render_output(grid.shape, bg, bars, shifts, slot_fuel_counts, mode="true")
    shift_only = _try_render_output(grid.shape, bg, bars, shifts, slot_fuel_counts, mode="shift_only")
    binary = _try_render_output(grid.shape, bg, bars, shifts, slot_fuel_counts, mode="binary")
    inclusive = _try_render_output(grid.shape, bg, bars, shifts, slot_fuel_counts, mode="inclusive")
    distance = _try_render_output(grid.shape, bg, bars, shifts, slot_fuel_counts, mode="distance")

    if np.array_equal(true_out, grid):
        return False
    if shift_only is not None and np.array_equal(true_out, shift_only):
        return False
    if binary is not None and np.array_equal(true_out, binary):
        return False
    if inclusive is not None and np.array_equal(true_out, inclusive):
        return False
    if distance is not None and np.array_equal(true_out, distance):
        return False
    if not np.array_equal(main(grid), true_out):
        return False

    gains = [_fuel_gain(slot_fuel_counts, i, int(steps), mode="true") for i, steps in enumerate(shifts)]
    if max(gains) == 0:
        return False
    if not any(int(steps) > 1 and gains[i] > 0 for i, steps in enumerate(shifts)):
        return False
    if not any(
        int(steps) > 1 and any(int(slot_fuel_counts[(i + t) % n_bars]) >= 2 for t in range(1, int(steps)))
        for i, steps in enumerate(shifts)
    ):
        return False
    if not any(
        int(steps) > 2 and any(int(slot_fuel_counts[(i + t) % n_bars]) == 0 for t in range(1, int(steps)))
        for i, steps in enumerate(shifts)
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_bars = int(_RNG.integers(BAR_COUNT_MIN, BAR_COUNT_MAX_EXCLUSIVE))
    shifts = _sample_shifts(n_bars)
    if shifts is None:
        return None

    slot_fuel_counts = _sample_slot_fuel_counts(shifts)
    if slot_fuel_counts is None:
        return None

    shape = _sample_shape(n_bars)
    if shape is None:
        return None
    width, slot_cols, _stride = shape

    sampled_heights = _sample_heights(n_bars, slot_fuel_counts, shifts)
    if sampled_heights is None:
        return None
    height, bar_heights = sampled_heights

    bg, fuel_color, bar_colors = _sample_palette(n_bars)
    grid = np.full((height, width), int(bg), dtype=int)

    for col, color, bar_height in zip(slot_cols, bar_colors, bar_heights, strict=False):
        grid[height - int(bar_height) : height, int(col)] = int(color)

    if not _place_fuel_dots(grid, bg, fuel_color, slot_cols, bar_heights, slot_fuel_counts):
        return None
    if not _place_instruction_dots(grid, bg, bar_colors, shifts):
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dust-sweep input")
