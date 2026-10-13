from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import bbox, colors_present, find_components, get_bg  # noqa: E402


BG_COLOR = 0
FG_COLOR = 1
CTRL_COLOR = 2
ORTHO_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 500


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ORTHO_STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _detect_axis(grid: np.ndarray) -> tuple[int, int, int, str, int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    border = np.zeros(g.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    border_non_bg = np.argwhere(border & (g != bg))
    assert len(border_non_bg) == 1, "expected exactly one non-background border control pixel"
    ctrl_r, ctrl_c = map(int, border_non_bg[0])
    h, w = g.shape
    assert (ctrl_r, ctrl_c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, (
        "control pixel must not be on a corner"
    )

    ctrl_color = int(g[ctrl_r, ctrl_c])
    fg_colors = set(colors_present(g, exclude={bg, ctrl_color}))
    assert len(fg_colors) == 1, "expected exactly one foreground color besides the control pixel"
    fg_color = int(next(iter(fg_colors)))

    axis_kind = "h" if ctrl_c in {0, w - 1} else "v"
    axis_index = ctrl_r if axis_kind == "h" else ctrl_c
    return bg, fg_color, ctrl_color, axis_kind, axis_index, (ctrl_r, ctrl_c)


def _fold_masks(
    fg_mask: np.ndarray, axis_kind: str, axis_index: int
) -> tuple[np.ndarray, np.ndarray, str]:
    if axis_kind == "h":
        top = fg_mask[:axis_index, :]
        bottom = fg_mask[axis_index + 1 :, :]
        assert min(top.shape[0], bottom.shape[0]) > 0 and top.shape[0] != bottom.shape[0]
        if top.shape[0] > bottom.shape[0]:
            large = top.copy()
            folded = np.zeros_like(large, dtype=bool)
            folded[-bottom.shape[0] :, :] = np.flipud(bottom)
            return large, folded, "top"
        large = bottom.copy()
        folded = np.zeros_like(large, dtype=bool)
        folded[: top.shape[0], :] = np.flipud(top)
        return large, folded, "bottom"

    left = fg_mask[:, :axis_index]
    right = fg_mask[:, axis_index + 1 :]
    assert min(left.shape[1], right.shape[1]) > 0 and left.shape[1] != right.shape[1]
    if left.shape[1] > right.shape[1]:
        large = left.copy()
        folded = np.zeros_like(large, dtype=bool)
        folded[:, -right.shape[1] :] = np.fliplr(right)
        return large, folded, "left"
    large = right.copy()
    folded = np.zeros_like(large, dtype=bool)
    folded[:, : left.shape[1]] = np.fliplr(left)
    return large, folded, "right"


def _build_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg, fg_color, ctrl_color, axis_kind, axis_index, ctrl = _detect_axis(g)
    fg_mask = g == fg_color
    large, folded, large_side = _fold_masks(fg_mask, axis_kind, axis_index)

    out = np.full(large.shape, bg, dtype=int)
    out[large | folded] = fg_color
    info = {
        "bg": bg,
        "fg_color": fg_color,
        "ctrl_color": ctrl_color,
        "axis_kind": axis_kind,
        "axis_index": axis_index,
        "ctrl": ctrl,
        "large": large,
        "folded": folded,
        "large_side": large_side,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """Use the border control pixel's ray as the fold axis and overlay the smaller side onto the larger side.

    Input structure:
    - The grid has a dominant black background, one irregular connected foreground object, and exactly one red control pixel on a non-corner border cell.
    - A control pixel on the left or right border marks a horizontal ray through its row; a control pixel on the top or bottom border marks a vertical ray through its column.
    - The foreground object occupies cells on both sides of that ray and may also touch the ray itself.

    Rule:
    - Treat the full row or column indicated by the control pixel as a fold axis and discard that axis from the output.
    - Compare the two side regions separated by the axis, mirror the smaller side across the axis, and align it onto the larger side's coordinate system.
    - Output only the larger side's canvas, painting the foreground color wherever either side contributes a foreground cell after folding; erase the control pixel and any foreground cells on the axis itself.

    Key insight:
    - The border control pixel does not add paint by itself; it uniquely determines the row or column that becomes the folding axis, and that axis controls how the object's two unequal halves overlap.
    """
    out, _ = _build_output(grid)
    return out


def _grow_cells(
    shape: tuple[int, int], seeds: set[tuple[int, int]], target_size: int, allowed: np.ndarray
) -> set[tuple[int, int]] | None:
    if target_size < len(seeds):
        return None
    if any(not allowed[r, c] for r, c in seeds):
        return None

    cells = set(seeds)
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in _neighbors4(shape, r, c):
            if allowed[nr, nc] and (nr, nc) not in cells:
                frontier.add((nr, nc))

    for r, c in cells:
        add_frontier(r, c)

    while len(cells) < target_size and frontier:
        options = list(frontier)
        weights = []
        for r, c in options:
            touch = sum((nr, nc) in cells for nr, nc in _neighbors4(shape, r, c))
            upward_bias = shape[0] - r
            weights.append(1 + 2 * touch + upward_bias)
        pick = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    return cells if len(cells) == target_size else None


def _cells_to_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _canonical_attempt() -> np.ndarray | None:
    top_rows = random.randint(5, 8)
    bottom_rows = random.randint(2, min(5, top_rows - 1))
    width = random.randint(10, 14)
    height = top_rows + 1 + bottom_rows

    band_width = random.randint(4, min(7, width - 4))
    band_left = random.randint(2, width - band_width - 2)

    axis_len = random.randint(1, min(3, band_width - 2))
    axis_start = random.randint(1, band_width - axis_len - 1)
    axis_cols = tuple(range(axis_start, axis_start + axis_len))

    allow_top = np.zeros((top_rows, band_width), dtype=bool)
    allow_bottom = np.zeros((bottom_rows, band_width), dtype=bool)
    allow_top[1:, :] = True
    allow_bottom[1:, :] = True

    top_cap = int(allow_top.sum())
    bottom_cap = int(allow_bottom.sum())
    bottom_size_min = axis_len + 1
    bottom_size_max = min(6, bottom_cap)
    if bottom_size_min > bottom_size_max:
        return None
    bottom_size = random.randint(bottom_size_min, bottom_size_max)

    top_size_min = max(axis_len + 2, bottom_size + 1, 10 - axis_len - bottom_size)
    top_size_max = min(10, top_cap, 18 - axis_len - bottom_size)
    if top_size_min > top_size_max:
        return None
    top_size = random.randint(top_size_min, top_size_max)

    top_seeds = {(top_rows - 1, c) for c in axis_cols}
    bottom_seeds = {(bottom_rows - 1, c) for c in axis_cols}
    top_cells = _grow_cells((top_rows, band_width), top_seeds, top_size, allow_top)
    bottom_cells = _grow_cells((bottom_rows, band_width), bottom_seeds, bottom_size, allow_bottom)
    if top_cells is None or bottom_cells is None:
        return None

    top_mask = _cells_to_mask((top_rows, band_width), top_cells)
    bottom_folded = _cells_to_mask((bottom_rows, band_width), bottom_cells)
    bottom_aligned = np.zeros_like(top_mask)
    bottom_aligned[-bottom_rows:, :] = bottom_folded

    overlap = top_mask & bottom_aligned
    small_only = bottom_aligned & ~top_mask
    large_only = top_mask & ~bottom_aligned
    if not overlap.any() or not small_only.any() or not large_only.any():
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for r, c in top_cells:
        grid[r, band_left + c] = FG_COLOR
    for c in axis_cols:
        grid[top_rows, band_left + c] = FG_COLOR
    for r, c in bottom_cells:
        grid[top_rows + 1 + (bottom_rows - 1 - r), band_left + c] = FG_COLOR
    grid[top_rows, 0] = CTRL_COLOR

    return grid


def _apply_variant(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        bg, fg_color, ctrl_color, axis_kind, axis_index, _, = _detect_axis(g)
        assert bg == BG_COLOR and fg_color == FG_COLOR and ctrl_color == CTRL_COLOR
        fg_mask = g == fg_color

        components = find_components(
            g, bg=bg, connectivity=4, monochromatic=True, colors=[fg_color]
        )
        assert len(components) == 1

        bb = bbox(fg_mask)
        assert bb is not None
        r0, c0, r1, c1 = bb
        assert (r1 - r0 + 1) * (c1 - c0 + 1) > int(fg_mask.sum())

        if axis_kind == "h":
            on_axis = fg_mask[axis_index, :]
            part_a = int(fg_mask[:axis_index, :].sum())
            part_b = int(fg_mask[axis_index + 1 :, :].sum())
        else:
            on_axis = fg_mask[:, axis_index]
            part_a = int(fg_mask[:, :axis_index].sum())
            part_b = int(fg_mask[:, axis_index + 1 :].sum())

        assert on_axis.any()
        assert part_a > 0 and part_b > 0 and part_a != part_b

        out, info = _build_output(g)
        large = info["large"]
        folded = info["folded"]
        overlap = large & folded
        small_only = folded & ~large
        large_only = large & ~folded
        assert overlap.any() and small_only.any() and large_only.any()
        assert not np.any(out == ctrl_color)
        assert np.array_equal(out == fg_color, large | folded)

        large_only_out = np.full_like(out, bg)
        large_only_out[large] = fg_color
        folded_only_out = np.full_like(out, bg)
        folded_only_out[folded] = fg_color
        assert not np.array_equal(out, large_only_out)
        assert not np.array_equal(out, folded_only_out)

        axis_limit = g.shape[0] if axis_kind == "h" else g.shape[1]
        alt_changed = False
        for delta in (-1, 1):
            alt_axis = axis_index + delta
            if not (1 <= alt_axis <= axis_limit - 2):
                continue
            side1 = alt_axis
            side2 = axis_limit - alt_axis - 1
            if side1 == side2:
                continue
            alt_large, alt_folded, _ = _fold_masks(fg_mask, axis_kind, alt_axis)
            alt_union = alt_large | alt_folded
            if alt_union.shape != out.shape or not np.array_equal(alt_union, out == fg_color):
                alt_changed = True
                break
        assert alt_changed
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        base = _canonical_attempt()
        if base is None:
            continue
        grid = _apply_variant(base)
        if _semantic_ok(grid) and np.array_equal(main(grid), _build_output(grid)[0]):
            return grid
    raise RuntimeError("failed to generate a valid control-ray fold instance")
