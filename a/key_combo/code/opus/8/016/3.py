from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_ROOT = Path("/home/lixinyu/snap/new")
if str(_UTILS_ROOT) not in sys.path:
    sys.path.append(str(_UTILS_ROOT))

from solvers.arc_utils import Color, crop_bbox, find_components, get_bg, randomly_spaced_indices


BG = Color.BLACK
FRAME = Color.GREY
BAR_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
INNER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color not in BAR_COLORS:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if c0 != c1:
            continue
        height = int(r1 - r0 + 1)
        if int(comp["area"]) != height:
            continue
        bars.append(
            {
                "color": color,
                "col": c0,
                "top": r0,
                "bottom": r1,
                "height": height,
            }
        )
    bars.sort(key=lambda bar: int(bar["col"]))
    assert len(bars) >= 2, "expected at least two vertical bars"
    bottoms = {int(bar["bottom"]) for bar in bars}
    assert len(bottoms) == 1, "bars must share one baseline"
    return bars


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    frames: list[dict] = []
    for r0 in range(H - 3):
        for c0 in range(W - 3):
            sub = crop_bbox(grid, (r0, c0, r0 + 3, c0 + 3))
            inner = sub[1:3, 1:3]
            inner_vals = {int(v) for v in inner.flatten()}
            if len(inner_vals) != 1:
                continue
            inner_color = int(next(iter(inner_vals)))
            if inner_color in {bg, FRAME}:
                continue

            top = sub[0, :]
            gap_cols = np.flatnonzero(top == int(bg))
            if gap_cols.size != 1:
                continue
            gap_offset = int(gap_cols[0])
            if not np.all(top[np.arange(4) != gap_offset] == int(FRAME)):
                continue

            border_ok = True
            for rr in range(4):
                for cc in range(4):
                    is_border = rr in {0, 3} or cc in {0, 3}
                    if not is_border or (rr == 0 and cc == gap_offset):
                        continue
                    if int(sub[rr, cc]) != int(FRAME):
                        border_ok = False
                        break
                if not border_ok:
                    break
            if not border_ok:
                continue

            frames.append(
                {
                    "top": r0,
                    "left": c0,
                    "gap_col": c0 + gap_offset,
                    "inner_color": inner_color,
                }
            )
    frames.sort(key=lambda frame: (int(frame["top"]), int(frame["left"])))
    return frames


def _apply_rays(grid: np.ndarray, frames: list[dict], bg: int) -> tuple[np.ndarray, set[int], dict[int, int]]:
    out = np.array(grid, dtype=int, copy=True)
    locked_cols: set[int] = set()
    hit_rows: dict[int, int] = {}
    for frame in frames:
        top = int(frame["top"])
        col = int(frame["gap_col"])
        color = int(frame["inner_color"])

        hit_row = None
        for rr in range(top - 1, -1, -1):
            if int(grid[rr, col]) != int(bg):
                hit_row = int(rr)
                break

        fill_r0 = 0 if hit_row is None else hit_row + 1
        out[fill_r0 : top + 1, col] = color
        if hit_row is not None:
            locked_cols.add(col)
            hit_rows[col] = hit_row
    return out, locked_cols, hit_rows


def _transformed_specs(
    bars: list[dict],
    locked_cols: set[int],
    *,
    rotate_colors: bool,
    rotate_heights: bool,
) -> dict[int, tuple[int, int]]:
    specs = {int(bar["col"]): (int(bar["color"]), int(bar["height"])) for bar in bars}
    unlocked = [bar for bar in bars if int(bar["col"]) not in locked_cols]
    n = len(unlocked)
    if n == 0:
        return specs

    for i, bar in enumerate(unlocked):
        new_color = int(bar["color"])
        new_height = int(bar["height"])
        if n > 1 and rotate_colors:
            new_color = int(unlocked[(i - 1) % n]["color"])
        if n > 1 and rotate_heights:
            new_height = int(unlocked[(i + 1) % n]["height"])
        specs[int(bar["col"])] = (new_color, new_height)
    return specs


def _render_with_locked(
    grid: np.ndarray,
    bars: list[dict],
    frames: list[dict],
    bg: int,
    locked_cols: set[int],
    *,
    rotate_colors: bool = True,
    rotate_heights: bool = True,
) -> np.ndarray:
    out, _locked_from_rays, _hit_rows = _apply_rays(grid, frames, bg)
    baseline = int(bars[0]["bottom"])
    for bar in bars:
        out[int(bar["top"]) : baseline + 1, int(bar["col"])] = int(bg)

    specs = _transformed_specs(bars, locked_cols, rotate_colors=rotate_colors, rotate_heights=rotate_heights)
    for bar in bars:
        col = int(bar["col"])
        color, height = specs[col]
        out[baseline - height + 1 : baseline + 1, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Lock ray-hit vertical bars, then cyclically rotate color and height across the remaining bars.

    Input structure:
    - A black background contains several one-cell-wide vertical bars in the upper half, all bottom-aligned on one baseline row.
    - The lower half contains one or more 4x4 gray frames with a single top-edge gap and a solid 2x2 inner block.
    - Each frame gap shares its column with exactly one bar above it.

    Rule:
    - From every frame gap, fire a vertical ray upward in the inner-block color, filling only background cells until the first non-background cell.
    - The bar whose bottom is immediately above that stop point is locked and keeps its original color and height.
    - Order the remaining unlocked bars from left to right; at slot i they receive the previous unlocked bar's color and the next unlocked bar's height cyclically, then redraw all bars on the original baseline.

    Key insight:
    - The frame gaps do not just add colored rays; they remove specific bars from the rotation pool, so the left-to-right cyclic mapping must be computed after identifying the ray-locked subset.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _extract_bars(g, bg)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one framed block"

    bar_cols = {int(bar["col"]) for bar in bars}
    assert len({int(frame["gap_col"]) for frame in frames}) == len(frames), "frame gaps must be distinct"

    _rayed, locked_cols, hit_rows = _apply_rays(g, frames, bg)
    assert locked_cols, "at least one bar must be locked"
    assert locked_cols <= bar_cols, "every ray must hit a bar column"

    baseline = int(bars[0]["bottom"])
    for col in locked_cols:
        assert hit_rows[col] == baseline, "ray must stop on the aligned bar's bottom cell"

    return _render_with_locked(g, bars, frames, bg, locked_cols)


def _frame_patch(gap_offset: int, inner_color: int) -> np.ndarray:
    patch = np.full((4, 4), int(FRAME), dtype=int)
    patch[1:3, 1:3] = int(inner_color)
    patch[0, int(gap_offset)] = int(BG)
    return patch


def _boxes_separated(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, candidate)
    for br0, bc0, br1, bc1 in boxes:
        if not (r1 + gap < br0 or br1 + gap < r0 or c1 + gap < bc0 or bc1 + gap < c0):
            return False
    return True


def _sample_bar_columns(width: int, count: int) -> np.ndarray | None:
    span = width - 2
    if span <= 0:
        return None
    try:
        cols = randomly_spaced_indices(span, count, border_size=0, padding=1) + 1
    except ValueError:
        return None
    cols = np.array(sorted(int(c) for c in cols), dtype=int)
    return cols


def _build_candidate_grid() -> np.ndarray | None:
    bar_count = int(_RNG.integers(5, 8))
    frame_count = int(_RNG.integers(1, min(3, bar_count - 2) + 1))

    width = int(_RNG.integers(max(20, 2 * bar_count + 8), 29))
    height = int(_RNG.integers(14, 19))
    baseline = int(_RNG.integers(7, min(10, height - 5)))

    bar_cols = _sample_bar_columns(width, bar_count)
    if bar_cols is None:
        return None

    bar_colors = list(map(int, _RNG.choice(np.array(BAR_COLORS, dtype=int), size=bar_count, replace=False)))
    height_hi = min(8, baseline + 1)
    if height_hi < 4:
        return None
    bar_heights = list(map(int, _RNG.integers(3, height_hi + 1, size=bar_count)))

    locked_indices = sorted(map(int, _RNG.choice(np.arange(bar_count), size=frame_count, replace=False)))
    if len(set(bar_heights[i] for i in range(bar_count) if i not in locked_indices)) == 1:
        return None

    inner_colors = list(map(int, _RNG.choice(np.array(INNER_COLORS, dtype=int), size=frame_count, replace=False)))

    grid = np.full((height, width), int(BG), dtype=int)
    for col, color, bar_height in zip(bar_cols, bar_colors, bar_heights):
        grid[baseline - bar_height + 1 : baseline + 1, int(col)] = int(color)

    boxes: list[tuple[int, int, int, int]] = []
    long_ray_exists = False
    for order_idx, bar_idx in enumerate(locked_indices):
        col = int(bar_cols[bar_idx])
        inner_color = int(inner_colors[order_idx])
        candidates: list[tuple[int, int, int]] = []
        top_min = baseline + 2
        top_max = height - 4
        if top_min > top_max:
            return None

        for top in range(top_min, top_max + 1):
            for gap_offset in _RNG.permutation(4):
                left = col - int(gap_offset)
                if not (0 <= left <= width - 4):
                    continue
                box = (top, left, top + 3, left + 3)
                if _boxes_separated(boxes, box, gap=1):
                    candidates.append((top, left, int(gap_offset)))
        if not candidates:
            return None

        top, left, gap_offset = candidates[int(_RNG.integers(len(candidates)))]
        patch = _frame_patch(gap_offset, inner_color)
        grid[top : top + 4, left : left + 4] = patch
        boxes.append((top, left, top + 3, left + 3))
        if top - baseline >= 2:
            long_ray_exists = True

    if not long_ray_exists:
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        bg = int(get_bg(grid))
        bars = _extract_bars(grid, bg)
        frames = _find_frames(grid, bg)
    except AssertionError:
        return False

    if bg != int(BG):
        return False
    if not (5 <= len(bars) <= 7):
        return False
    if not (1 <= len(frames) <= 3):
        return False

    bar_cols = [int(bar["col"]) for bar in bars]
    gap_cols = [int(frame["gap_col"]) for frame in frames]
    if len(set(gap_cols)) != len(gap_cols):
        return False
    if not set(gap_cols) <= set(bar_cols):
        return False

    rayed, locked_cols, hit_rows = _apply_rays(grid, frames, bg)
    baseline = int(bars[0]["bottom"])
    if locked_cols != set(gap_cols):
        return False
    if len(bars) - len(locked_cols) < 2:
        return False
    if any(hit_rows[col] != baseline for col in locked_cols):
        return False

    ray_lengths = [int(frame["top"]) - hit_rows[int(frame["gap_col"])] for frame in frames]
    if max(ray_lengths) < 2:
        return False

    actual = _render_with_locked(grid, bars, frames, bg, locked_cols)
    if np.array_equal(actual, grid):
        return False

    alt_no_lock = _render_with_locked(grid, bars, frames, bg, set())
    if np.array_equal(actual, alt_no_lock):
        return False

    alt_keep_colors = _render_with_locked(grid, bars, frames, bg, locked_cols, rotate_colors=False, rotate_heights=True)
    if np.array_equal(actual, alt_keep_colors):
        return False

    alt_keep_heights = _render_with_locked(grid, bars, frames, bg, locked_cols, rotate_colors=True, rotate_heights=False)
    if np.array_equal(actual, alt_keep_heights):
        return False

    cols = [int(bar["col"]) for bar in bars]
    for mask in range(1 << len(cols)):
        other_locked = {cols[i] for i in range(len(cols)) if (mask >> i) & 1}
        if other_locked == locked_cols or len(cols) - len(other_locked) < 2:
            continue
        other = _render_with_locked(grid, bars, frames, bg, other_locked)
        if np.array_equal(actual, other):
            return False

    if not np.any(rayed != grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-lock cyclic-shift instance")
