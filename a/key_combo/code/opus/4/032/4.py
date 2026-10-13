from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils_4_032", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg


HEIGHT = 10
WIDTH = 10
TOP_HEIGHTS = (4, 5)
PERIOD_CHOICES = (1, 2)
SHIFT_CHOICES = (1, 2, 3)
GEN_ATTEMPTS = 500


def _cyclic_right(mask: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(mask.astype(bool), int(shift)).astype(bool)


def _shift_right(mask: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift)
    out = np.zeros(WIDTH, dtype=bool)
    if shift == 0:
        out[:] = mask.astype(bool)
    elif shift < WIDTH:
        out[shift:] = mask[:-shift]
    return out


def _blue_prefix_height(grid: np.ndarray) -> int:
    top_h = 0
    for r in range(grid.shape[0]):
        if np.any(grid[r] == Color.BLUE):
            top_h += 1
        else:
            break
    return top_h


def _infer_period_shift(top_masks: np.ndarray) -> tuple[int, int]:
    h, w = top_masks.shape
    candidates: list[tuple[int, int]] = []
    for period in range(1, h):
        valid_shifts = []
        for shift in range(w):
            if all(np.array_equal(top_masks[r + period], _cyclic_right(top_masks[r], shift)) for r in range(h - period)):
                valid_shifts.append(shift)
        if valid_shifts:
            min_period = period
            candidates.extend((period, shift) for shift in valid_shifts)
            break
    assert candidates, "expected at least one valid period/shift pair"
    assert len(candidates) == 1, "minimal period must determine a unique horizontal shift"
    return candidates[0]


def _compute_full_masks(top_masks: np.ndarray, period: int, shift: int) -> list[np.ndarray]:
    full = [np.zeros(WIDTH, dtype=bool) for _ in range(HEIGHT)]
    for r in range(top_masks.shape[0]):
        full[r] = top_masks[r].copy()
    for r in range(top_masks.shape[0], HEIGHT):
        full[r] = _shift_right(full[r - period], shift)
    return full


def _interval_from_walls(walls: list[int], mark_col: int) -> tuple[int, int]:
    left = max((c for c in walls if c < mark_col), default=None)
    right = min((c for c in walls if c >= mark_col), default=None)
    assert left is not None or right is not None
    if left is None:
        return 0, right - 1
    if right is None:
        return left + 1, WIDTH - 1
    return left + 1, right - 1


def _crop_mask(mask: np.ndarray, lo: int, hi: int) -> np.ndarray:
    out = np.zeros(WIDTH, dtype=bool)
    if lo <= hi:
        out[lo : hi + 1] = mask[lo : hi + 1]
    return out


def _base_mask() -> np.ndarray | None:
    for _ in range(200):
        count = random.randint(3, 5)
        cols = sorted(random.sample(range(WIDTH), count))
        if cols[-1] - cols[0] < 3:
            continue
        mask = np.zeros(WIDTH, dtype=bool)
        mask[cols] = True
        if any(np.array_equal(mask, _cyclic_right(mask, shift)) for shift in range(1, WIDTH)):
            continue
        return mask
    return None


def _visible_top_masks(top_h: int, period: int, shift: int) -> np.ndarray | None:
    residues = []
    for _ in range(period):
        mask = _base_mask()
        if mask is None:
            return None
        residues.append(mask)
    if period == 2 and any(np.array_equal(residues[1], _cyclic_right(residues[0], shift2)) for shift2 in range(WIDTH)):
        return None

    top_masks = np.zeros((top_h, WIDTH), dtype=bool)
    for r in range(top_h):
        top_masks[r] = _cyclic_right(residues[r % period], (r // period) * shift)
    inferred = _infer_period_shift(top_masks)
    if inferred != (period, shift):
        return None
    return top_masks


def _wall_options(mask: np.ndarray, mark_col: int, row: int, period: int, shift: int) -> list[dict]:
    cols = [int(c) for c in np.flatnonzero(mask)]
    if len(cols) < 2:
        return []

    options: list[dict] = []

    for left in range(mark_col):
        kept = [c for c in cols if c > left]
        removed = [c for c in cols if c <= left]
        if kept and removed:
            options.append(
                {
                    "type": "left",
                    "walls": (left,),
                    "lo": left + 1,
                    "hi": WIDTH - 1,
                    "removed": tuple(removed),
                    "kept": tuple(kept),
                    "propagates": row + period < HEIGHT and any(c + shift < WIDTH for c in removed),
                }
            )

    for right in range(mark_col, WIDTH):
        kept = [c for c in cols if c < right]
        removed = [c for c in cols if c >= right]
        if kept and removed:
            options.append(
                {
                    "type": "right",
                    "walls": (right,),
                    "lo": 0,
                    "hi": right - 1,
                    "removed": tuple(removed),
                    "kept": tuple(kept),
                    "propagates": row + period < HEIGHT and any(c + shift < WIDTH for c in removed),
                }
            )

    for left in range(mark_col):
        for right in range(mark_col, WIDTH):
            if left >= right:
                continue
            kept = [c for c in cols if left < c < right]
            removed = [c for c in cols if c <= left or c >= right]
            if kept and removed:
                options.append(
                    {
                        "type": "both",
                        "walls": (left, right),
                        "lo": left + 1,
                        "hi": right - 1,
                        "removed": tuple(removed),
                        "kept": tuple(kept),
                        "propagates": row + period < HEIGHT and any(c + shift < WIDTH for c in removed),
                    }
                )

    random.shuffle(options)
    options.sort(key=lambda opt: (-len(opt["removed"]), -len(opt["kept"]), opt["hi"] - opt["lo"]))
    trimmed: list[dict] = []
    for branch in ("left", "right", "both"):
        branch_opts = [opt for opt in options if opt["type"] == branch]
        trimmed.extend(branch_opts[:3])
    random.shuffle(trimmed)
    return trimmed


def _choose_marker_and_walls(full_masks: list[np.ndarray], top_h: int, period: int, shift: int) -> tuple[tuple[int, int], list[tuple[int, tuple[int, ...]]]] | None:
    marker_candidates = []
    for row in range(top_h, HEIGHT):
        cols = [int(c) for c in np.flatnonzero(full_masks[row])]
        if len(cols) >= 2:
            for col in cols:
                marker_candidates.append((row, col))
    random.shuffle(marker_candidates)

    for mark_row, mark_col in marker_candidates:
        row_options: dict[int, list[dict]] = {}
        for row in range(top_h, HEIGHT):
            if row == mark_row:
                continue
            opts = _wall_options(full_masks[row], mark_col, row, period, shift)
            if opts:
                row_options[row] = opts
        rows = list(row_options)
        random.shuffle(rows)

        for wall_count in random.sample([2, 3], 2):
            if len(rows) < wall_count:
                continue
            combos = list(itertools.combinations(rows, wall_count))
            random.shuffle(combos)
            for combo in combos:
                filtered = [row_options[row] for row in combo]
                for picked in itertools.product(*filtered):
                    branch_count = len({opt["type"] for opt in picked})
                    if wall_count >= 2 and branch_count < 2:
                        continue
                    if not any(opt["propagates"] for opt in picked):
                        continue
                    return (mark_row, mark_col), [(row, tuple(opt["walls"])) for row, opt in zip(combo, picked)]
    return None


def _render_input(top_masks: np.ndarray, marker: tuple[int, int], walls: list[tuple[int, tuple[int, ...]]]) -> np.ndarray:
    grid = np.full((HEIGHT, WIDTH), Color.BLACK, dtype=int)
    for r in range(top_masks.shape[0]):
        grid[r, top_masks[r]] = Color.BLUE
    for row, wall_cols in walls:
        for col in wall_cols:
            grid[row, col] = Color.GREY
    grid[marker[0], marker[1]] = Color.RED
    return grid


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (HEIGHT, WIDTH), "expected a 10x10 ARC grid"
    assert g.ndim == 2
    assert get_bg(g) == Color.BLACK

    top_h = _blue_prefix_height(g)
    assert top_h in TOP_HEIGHTS, "expected 4 or 5 visible top pattern rows"
    assert np.all(np.isin(g[:top_h], [Color.BLACK, Color.BLUE]))
    assert not np.any(g[top_h:, :] == Color.BLUE), "only the top rows may contain visible blue input pixels"

    red_coords = np.argwhere(g == Color.RED)
    assert red_coords.shape == (1, 2), "expected exactly one red marker"
    mark_row, mark_col = map(int, red_coords[0])
    assert mark_row >= top_h, "marker must sit below the visible top pattern"

    wall_rows = [r for r in range(top_h, HEIGHT) if np.any(g[r] == Color.GREY)]
    assert len(wall_rows) in (2, 3), "expected 2 or 3 wall rows"
    assert mark_row not in wall_rows, "marker row must not also be a wall row"

    for row in wall_rows:
        wall_cols = [int(c) for c in np.flatnonzero(g[row] == Color.GREY)]
        assert 1 <= len(wall_cols) <= 2
        assert g[row, mark_col] != Color.RED

    top_masks = (g[:top_h] == Color.BLUE)
    period, shift = _infer_period_shift(top_masks)
    full_masks = _compute_full_masks(top_masks, period, shift)

    out = np.full_like(g, Color.BLACK)
    out[:top_h] = g[:top_h]
    intervals: dict[int, tuple[int, int]] = {}

    for row in range(top_h, HEIGHT):
        row_mask = full_masks[row].copy()
        if row in wall_rows:
            wall_cols = [int(c) for c in np.flatnonzero(g[row] == Color.GREY)]
            lo, hi = _interval_from_walls(wall_cols, mark_col)
            intervals[row] = (lo, hi)
            row_mask = _crop_mask(row_mask, lo, hi)
        out[row, row_mask] = Color.BLUE

    out[g == Color.GREY] = Color.GREY
    out[mark_row, mark_col] = Color.RED

    return {
        "grid": g,
        "output": out,
        "top_h": top_h,
        "period": period,
        "shift": shift,
        "full_masks": full_masks,
        "wall_rows": wall_rows,
        "intervals": intervals,
        "marker": (mark_row, mark_col),
    }


def _variant_without_cropping(details: dict) -> np.ndarray:
    out = np.full((HEIGHT, WIDTH), Color.BLACK, dtype=int)
    g = details["grid"]
    top_h = details["top_h"]
    out[:top_h] = g[:top_h]
    for row in range(top_h, HEIGHT):
        out[row, details["full_masks"][row]] = Color.BLUE
    out[g == Color.GREY] = Color.GREY
    mark_row, mark_col = details["marker"]
    out[mark_row, mark_col] = Color.RED
    return out


def _variant_sequential_from_cropped(details: dict) -> np.ndarray:
    g = details["grid"]
    top_h = details["top_h"]
    period = details["period"]
    shift = details["shift"]
    mark_row, mark_col = details["marker"]
    wall_rows = set(details["wall_rows"])

    blue_rows = [np.zeros(WIDTH, dtype=bool) for _ in range(HEIGHT)]
    for row in range(top_h):
        blue_rows[row] = g[row] == Color.BLUE
    for row in range(top_h, HEIGHT):
        row_mask = _shift_right(blue_rows[row - period], shift)
        if row in wall_rows:
            lo, hi = details["intervals"][row]
            row_mask = _crop_mask(row_mask, lo, hi)
        row_mask[mark_col] = False if row == mark_row else row_mask[mark_col]
        blue_rows[row] = row_mask

    out = np.full((HEIGHT, WIDTH), Color.BLACK, dtype=int)
    out[:top_h] = g[:top_h]
    for row in range(top_h, HEIGHT):
        out[row, blue_rows[row]] = Color.BLUE
    out[g == Color.GREY] = Color.GREY
    out[mark_row, mark_col] = Color.RED
    return out


def _variant_skip_marker_row(details: dict) -> np.ndarray:
    out = details["output"].copy()
    mark_row, mark_col = details["marker"]
    if mark_row >= details["top_h"]:
        out[mark_row] = Color.BLACK
        if mark_row in details["wall_rows"]:
            wall_cols = np.flatnonzero(details["grid"][mark_row] == Color.GREY)
            out[mark_row, wall_cols] = Color.GREY
        out[mark_row, mark_col] = Color.RED
    return out


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        details = _solve_details(grid)
    except AssertionError:
        return False

    out = details["output"]
    mark_row, mark_col = details["marker"]
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _variant_without_cropping(details)):
        return False
    if np.array_equal(out, _variant_sequential_from_cropped(details)):
        return False
    if np.array_equal(out, _variant_skip_marker_row(details)):
        return False
    if np.count_nonzero(details["full_masks"][mark_row]) < 2:
        return False
    if not details["full_masks"][mark_row][mark_col]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = random.choice(TOP_HEIGHTS)
    period = random.choice(PERIOD_CHOICES)
    if period >= top_h:
        return None
    shift = random.choice(SHIFT_CHOICES)

    top_masks = _visible_top_masks(top_h, period, shift)
    if top_masks is None:
        return None

    full_masks = _compute_full_masks(top_masks, period, shift)
    choice = _choose_marker_and_walls(full_masks, top_h, period, shift)
    if choice is None:
        return None
    marker, walls = choice

    grid = _render_input(top_masks, marker, walls)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-wall-cropping ARC input")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the visible top-row blue pattern downward and crop only the wall rows by the wall interval anchored to the red marker column.

    Input structure:
    - A 10x10 black grid has 4 or 5 blue pattern rows at the top; rows separated by the same vertical period are cyclic right-shifts of each other.
    - Below them, 2 or 3 rows contain 1 or 2 gray wall pixels, and one different lower row contains a single red marker pixel.
    - No input blue appears outside the top pattern rows; the marker never shares a row with walls.

    Rule:
    - Infer the smallest vertical period P and the unique cyclic right-shift D such that top row r+P matches top row r shifted by D.
    - Use those top rows as the source sequence, extend the blue masks downward by repeatedly applying a non-cyclic right shift of D, and keep the original top rows unchanged.
    - On each wall row, keep only the candidate blue pixels strictly between the nearest wall left of the marker column and the nearest wall at or right of it; if a wall exists on only one side, crop to the corresponding edge instead.
    - Gray walls and the red marker stay visible and overwrite any blue candidate that would land on them.

    Key insight:
    - The downward recurrence creates a full candidate pattern for every lower row, but walls do not generate the pattern; they only clip the candidate pixels on their own row using an interval measured from the marker column.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]
