from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, randomly_spaced_indices


def _find_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    height, width = grid.shape
    for r in range(height):
        color = int(grid[r, 0])
        if color == bg or not np.all(grid[r] == color):
            continue
        has_neighbor_bar = False
        if r > 0 and np.any((grid[r - 1] != bg) & (grid[r - 1] != color)):
            has_neighbor_bar = True
        if r + 1 < height and np.any((grid[r + 1] != bg) & (grid[r + 1] != color)):
            has_neighbor_bar = True
        if has_neighbor_bar:
            candidates.append((r, color))
    assert len(candidates) == 1, "expected a unique full-width baseline"
    base_r, base_color = candidates[0]
    assert grid.shape[1] == width
    return int(base_r), int(base_color)


def _read_legend(grid: np.ndarray, bg: int, base_r: int, base_color: int) -> dict[int, int]:
    height, width = grid.shape
    shift_by_color: dict[int, int] = {}
    for c in range(width):
        if base_r > 0 and int(grid[base_r - 1, c]) != bg:
            color = int(grid[base_r - 1, c])
            assert color != base_color
            r = base_r - 1
            bar_h = 0
            while r >= 0 and int(grid[r, c]) == color:
                bar_h += 1
                r -= 1
            assert color not in shift_by_color, "bar colors must be unique"
            shift_by_color[color] = int(bar_h)

        if base_r + 1 < height and int(grid[base_r + 1, c]) != bg:
            color = int(grid[base_r + 1, c])
            assert color != base_color
            r = base_r + 1
            bar_h = 0
            while r < height and int(grid[r, c]) == color:
                bar_h += 1
                r += 1
            assert color not in shift_by_color, "bar colors must be unique"
            shift_by_color[color] = -int(bar_h)

    assert shift_by_color, "expected at least one legend bar"
    return shift_by_color


def _pattern_height(grid: np.ndarray, bg: int, base_r: int) -> int:
    blank_rows = [r for r in range(base_r) if np.all(grid[r] == bg)]
    assert blank_rows, "expected a blank separator between pattern and bars"
    pattern_h = int(blank_rows[0])
    assert pattern_h > 0, "pattern area must be non-empty"
    return pattern_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each pattern row by the height and side of its same-colored legend bar.

    Input structure:
    - A single-color baseline spans the full grid width near the middle.
    - Distinct-colored one-column bars touch the baseline: bars above it grow upward,
      and bars below it hang downward.
    - The top pattern area is separated from the bars by at least one blank row, and
      each pattern row contains at most one non-background color in any arrangement.

    Rule:
    - For each pattern row, read its unique foreground color, if any.
    - Find the bar of the same color: an above-baseline bar means roll that row to the
      right by the bar height, while a below-baseline bar means roll it to the left by
      the bar height.
    - Rows with no same-colored bar stay unchanged, and bars with no same-colored row
      are legend only. Return only the shifted pattern area.

    Key insight:
    - The bars are a color-keyed control table: color chooses the row, height chooses
      the shift amount, and the side of the baseline chooses the direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    base_r, base_color = _find_baseline(g, bg)
    shift_by_color = _read_legend(g, bg, base_r, base_color)
    pattern_h = _pattern_height(g, bg, base_r)

    out = g[:pattern_h].copy()
    for r in range(pattern_h):
        row = out[r]
        colors = sorted(set(map(int, row)) - {bg})
        assert len(colors) <= 1, "each pattern row may use at most one foreground color"
        if not colors:
            continue
        color = colors[0]
        if color in shift_by_color:
            out[r] = np.roll(row, int(shift_by_color[color]))
    return out


_GEN_BG = 0
_GEN_MAX_ATTEMPTS = 300
_GEN_WIDTH_CHOICES = tuple(range(11, 20, 2))
_GEN_SEPARATOR_CHOICES = (1, 2)
_GEN_BOTTOM_PAD_CHOICES = (1, 2, 3)
_GEN_SHARED_HEIGHTS = (1, 2, 3, 4)
_GEN_EXTRA_HEIGHTS = (1, 2, 3, 4, 5)
_GEN_FOREGROUND_COLORS = tuple(range(1, 10))


def _sample_scene_spec() -> tuple[int, int, list[dict], list[dict]]:
    width = random.choice(_GEN_WIDTH_CHOICES)
    unmatched_rows = random.choice((1, 1, 2))
    unused_bars = random.choice((1, 1, 2))
    total_unique = 3 + unmatched_rows + unused_bars
    colors = random.sample(_GEN_FOREGROUND_COLORS, total_unique + 1)
    baseline_color = int(colors[0])
    matched_colors = [int(v) for v in colors[1:4]]
    unmatched_colors = [int(v) for v in colors[4 : 4 + unmatched_rows]]
    unused_colors = [int(v) for v in colors[4 + unmatched_rows : 4 + unmatched_rows + unused_bars]]

    if random.random() < 0.5:
        matched_sides = ["up", "up", "down"]
    else:
        matched_sides = ["up", "down", "down"]

    shared_height = random.choice(_GEN_SHARED_HEIGHTS)
    extra_height_choices = [h for h in _GEN_EXTRA_HEIGHTS if h != shared_height]
    extra_height = random.choice(extra_height_choices)
    up_assigned = False
    down_assigned = False
    matched_bar_specs: list[dict] = []
    for color, side in zip(matched_colors, matched_sides):
        if side == "up" and not up_assigned:
            height = shared_height
            up_assigned = True
        elif side == "down" and not down_assigned:
            height = shared_height
            down_assigned = True
        else:
            height = extra_height
        matched_bar_specs.append({"color": color, "side": side, "height": int(height)})

    all_bar_specs = matched_bar_specs[:]
    for color in unused_colors:
        side = random.choice(("up", "down"))
        height = random.choice(_GEN_EXTRA_HEIGHTS)
        all_bar_specs.append({"color": int(color), "side": side, "height": int(height)})

    row_specs = [{"color": color, "matched": True} for color in matched_colors]
    row_specs.extend({"color": color, "matched": False} for color in unmatched_colors)
    random.shuffle(row_specs)

    return width, baseline_color, row_specs, all_bar_specs


def _bar_shifts(bar_specs: list[dict]) -> dict[int, int]:
    shifts: dict[int, int] = {}
    for spec in bar_specs:
        sign = 1 if spec["side"] == "up" else -1
        shifts[int(spec["color"])] = sign * int(spec["height"])
    return shifts


def _random_mask(width: int) -> np.ndarray:
    mask = np.zeros(width, dtype=bool)
    style = random.choice(("runs", "runs", "scatter"))
    if style == "scatter":
        count = random.randint(2, min(6, width - 2))
        cols = random.sample(range(width), count)
        mask[cols] = True
        return mask

    run_count = random.choice((1, 2, 2, 3))
    starts = sorted(random.sample(range(width), run_count))
    for start in starts:
        max_len = min(3, width - start)
        length = random.randint(1, max_len)
        mask[start : start + length] = True
    if mask.sum() >= width:
        mask[random.randrange(width)] = False
    if mask.sum() < 2:
        extra = [c for c in range(width) if not mask[c]]
        mask[random.choice(extra)] = True
    return mask


def _row_is_distinguishable(mask: np.ndarray, target_shift: int, bar_shift_values: list[int]) -> bool:
    target = np.roll(mask, target_shift)
    alt_shifts = {0, *bar_shift_values}
    for alt in alt_shifts:
        if int(alt) == int(target_shift):
            continue
        if np.array_equal(target, np.roll(mask, int(alt))):
            return False
    return True


def _build_row_masks(width: int, row_specs: list[dict], bar_specs: list[dict]) -> dict[int, np.ndarray] | None:
    shift_by_color = _bar_shifts(bar_specs)
    bar_shift_values = list(shift_by_color.values())
    masks: dict[int, np.ndarray] = {}
    seen_patterns: set[bytes] = set()

    for row in row_specs:
        color = int(row["color"])
        target_shift = shift_by_color.get(color, 0)
        for _ in range(200):
            mask = _random_mask(width)
            if mask.all() or not mask.any():
                continue
            if mask.tobytes() in seen_patterns:
                continue
            if not _row_is_distinguishable(mask, int(target_shift), bar_shift_values):
                continue
            masks[color] = mask
            seen_patterns.add(mask.tobytes())
            break
        else:
            return None
    return masks


def _matched_row_order(row_specs: list[dict], matched_colors: set[int]) -> list[int]:
    return [int(row["color"]) for row in row_specs if int(row["color"]) in matched_colors]


def _matched_bar_order(bar_specs: list[dict], columns: np.ndarray, matched_colors: set[int]) -> list[int]:
    ordered = sorted(
        [(int(col), int(spec["color"])) for col, spec in zip(columns, bar_specs)],
        key=lambda item: item[0],
    )
    return [color for _col, color in ordered if color in matched_colors]


def _scene_orders_are_tricky(row_specs: list[dict], bar_specs: list[dict], columns: np.ndarray) -> bool:
    matched_colors = {int(row["color"]) for row in row_specs if row["matched"]}
    return _matched_row_order(row_specs, matched_colors) != _matched_bar_order(bar_specs, columns, matched_colors)


def _render_scene(
    width: int,
    baseline_color: int,
    row_specs: list[dict],
    bar_specs: list[dict],
    row_masks: dict[int, np.ndarray],
    columns: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    shift_by_color = _bar_shifts(bar_specs)
    max_up = max(int(spec["height"]) for spec in bar_specs if spec["side"] == "up")
    max_down = max(int(spec["height"]) for spec in bar_specs if spec["side"] == "down")
    separator_h = random.choice(_GEN_SEPARATOR_CHOICES)
    bottom_pad = random.choice(_GEN_BOTTOM_PAD_CHOICES)
    pattern_h = len(row_specs)
    baseline_r = pattern_h + separator_h + max_up
    height = baseline_r + 1 + max_down + bottom_pad
    grid = np.full((height, width), _GEN_BG, dtype=int)

    for r, row in enumerate(row_specs):
        color = int(row["color"])
        grid[r, row_masks[color]] = color

    grid[baseline_r, :] = baseline_color
    for c, spec in zip(columns, bar_specs):
        color = int(spec["color"])
        bar_h = int(spec["height"])
        if spec["side"] == "up":
            grid[baseline_r - bar_h : baseline_r, int(c)] = color
        else:
            grid[baseline_r + 1 : baseline_r + 1 + bar_h, int(c)] = color

    return grid, shift_by_color


def _expected_output(width: int, row_specs: list[dict], row_masks: dict[int, np.ndarray], shift_by_color: dict[int, int]) -> np.ndarray:
    out = np.full((len(row_specs), width), _GEN_BG, dtype=int)
    for r, row in enumerate(row_specs):
        color = int(row["color"])
        out[r, row_masks[color]] = color
        if color in shift_by_color:
            out[r] = np.roll(out[r], int(shift_by_color[color]))
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    row_specs: list[dict],
    row_masks: dict[int, np.ndarray],
    shift_by_color: dict[int, int],
    bar_specs: list[dict],
    columns: np.ndarray,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if not _scene_orders_are_tricky(row_specs, bar_specs, columns):
        return False

    changed_right = False
    changed_left = False
    all_shifts = list(shift_by_color.values())
    expected_by_color = {
        int(row["color"]): expected[r].copy()
        for r, row in enumerate(row_specs)
    }
    for row in row_specs:
        color = int(row["color"])
        base = np.full(grid.shape[1], _GEN_BG, dtype=int)
        base[row_masks[color]] = color
        target = expected_by_color[color]
        if color in shift_by_color:
            shift = int(shift_by_color[color])
            if shift > 0:
                changed_right = True
            if shift < 0:
                changed_left = True
            if np.array_equal(base, target):
                return False
            for alt in {0, *all_shifts}:
                if int(alt) == shift:
                    continue
                if np.array_equal(target, np.roll(base, int(alt))):
                    return False
        else:
            if not np.array_equal(base, target):
                return False
            if any(np.array_equal(base, np.roll(base, int(alt))) for alt in all_shifts):
                return False

    return changed_right and changed_left


def _gen_attempt() -> np.ndarray | None:
    width, baseline_color, row_specs, bar_specs = _sample_scene_spec()

    try:
        columns = randomly_spaced_indices(width, len(bar_specs), border_size=1, padding=1)
    except ValueError:
        return None
    bar_specs = bar_specs[:]
    random.shuffle(bar_specs)

    row_masks = _build_row_masks(width, row_specs, bar_specs)
    if row_masks is None:
        return None

    grid, shift_by_color = _render_scene(width, baseline_color, row_specs, bar_specs, row_masks, columns)
    expected = _expected_output(width, row_specs, row_masks, shift_by_color)
    if not _semantic_checks(grid, expected, row_specs, row_masks, shift_by_color, bar_specs, columns):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
