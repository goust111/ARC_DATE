from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_w, find_components, get_bg, paste, random_sprite


BG = 0
OBJECT_COLOR = 5
TOP_MARKER_COLOR = 7
ROW_MARKER_COLORS = (1, 2, 3)

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
OBJECT_COUNT_RANGE = (4, 6)
MAX_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the self-shifted grey objects whose shifted bbox anchor lands on the edge-marker Cartesian grid.

    Input structure:
    - The background is black.
    - Orange markers on the top edge select legal columns, and colored markers on the
      right edge select legal rows plus the recolor for that row.
    - The interior contains separated grey 8-connected objects of varying shapes.

    Rule:
    - Read the top-edge orange marker columns and the right-edge colored marker rows.
    - For each grey object, shift the whole object right by its own bounding-box width.
    - If the shifted object's bounding-box top-left corner lands on a selected
      row/column intersection, keep the shifted object and recolor it to that row's
      right-edge marker color; otherwise erase it.
    - Keep the edge markers unchanged.

    Key insight:
    - Survival depends on a resonance between each object's local width and the global
      Cartesian grid defined by the edge markers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    legal_cols = {int(c) for c in np.flatnonzero(g[0, : w - 1] == TOP_MARKER_COLOR)}
    row_colors = {
        int(r): int(g[r, w - 1])
        for r in range(1, h)
        if int(g[r, w - 1]) in ROW_MARKER_COLORS
    }

    objects = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[OBJECT_COLOR],
        return_format="dict",
    )

    out = g.copy()
    out[g == OBJECT_COLOR] = bg
    occupied = np.zeros(g.shape, dtype=bool)

    for obj in objects:
        coords = obj["coords"]
        r0, c0, _r1, _c1 = map(int, obj["bbox"])
        shift = int(bbox_w(obj["bbox"]))
        dst = coords + np.array([0, shift], dtype=int)
        dst_anchor = (int(r0), int(c0 + shift))

        if dst_anchor[0] not in row_colors or dst_anchor[1] not in legal_cols:
            continue

        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(occupied[dst[:, 0], dst[:, 1]].any())
        assert bool(np.all(out[dst[:, 0], dst[:, 1]] == bg))

        color = int(row_colors[dst_anchor[0]])
        out[dst[:, 0], dst[:, 1]] = color
        occupied[dst[:, 0], dst[:, 1]] = True

    return out


def _patch_area(patch: np.ndarray) -> int:
    return int(np.count_nonzero(patch == OBJECT_COLOR))


def _sample_object_patch(width: int) -> np.ndarray | None:
    height_options = {
        1: [3, 4],
        2: [2, 3, 4],
        3: [1, 2, 3],
        4: [1, 2],
        5: [1, 2],
    }[int(width)]
    for _ in range(200):
        height = random.choice(height_options)
        patch = random_sprite(
            height,
            width,
            density=random.uniform(0.45, 0.9),
            symmetry="not_symmetric",
            color_palette=[OBJECT_COLOR],
            connectivity=8,
            background=BG,
        )
        area = _patch_area(patch)
        if 3 <= area <= 9:
            return patch
    return None


def _assign_rows(heights: list[int], grid_h: int) -> list[int] | None:
    placed: dict[int, int] = {}
    intervals: list[tuple[int, int]] = []
    order = sorted(range(len(heights)), key=lambda i: heights[i], reverse=True)

    for idx in order:
        height = int(heights[idx])
        candidates = []
        for r0 in range(2, grid_h - height - 1):
            r1 = r0 + height - 1
            if all(r1 < a0 - 1 or r0 > a1 + 1 for a0, a1 in intervals):
                candidates.append(r0)
        if not candidates:
            return None
        top = random.choice(candidates)
        placed[idx] = int(top)
        intervals.append((int(top), int(top + height - 1)))

    return [placed[i] for i in range(len(heights))]


def _choose_row_colors(rows: list[int]) -> dict[int, int]:
    while True:
        colors = {int(r): int(random.choice(ROW_MARKER_COLORS)) for r in rows}
        if len(set(colors.values())) >= min(2, len(rows)):
            return colors


def _mode_plan(obj_count: int) -> tuple[set[int], dict[int, str]]:
    survivor_count = random.randint(2, obj_count // 2)
    indices = list(range(obj_count))
    random.shuffle(indices)
    survivors = set(indices[:survivor_count])
    failures = indices[survivor_count:]
    modes: dict[int, str] = {}

    for idx in survivors:
        modes[idx] = "survive"
    if failures:
        modes[failures[0]] = "row_hit_col_miss"
    if len(failures) >= 2:
        modes[failures[1]] = "col_hit_row_miss"
    for idx in failures[2:]:
        modes[idx] = random.choice(
            ["row_hit_col_miss", "col_hit_row_miss", "neither"]
        )
    return survivors, modes


def _compatible_cols(grid_w: int, width: int) -> list[int]:
    lo = int(width) + 1
    hi = grid_w - int(width) - 2
    return list(range(lo, hi + 1))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    safe_top_cols = list(range(6, w - 6))
    if len(safe_top_cols) < 3:
        return None

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    widths = [random.randint(1, 5) for _ in range(obj_count)]
    patches = [_sample_object_patch(width) for width in widths]
    if any(patch is None for patch in patches):
        return None
    patches = [patch for patch in patches if patch is not None]
    heights = [int(patch.shape[0]) for patch in patches]

    row_tops = _assign_rows(heights, h)
    if row_tops is None:
        return None

    survivors, modes = _mode_plan(obj_count)

    max_top_markers = min(4, len(safe_top_cols) - 1)
    if max_top_markers < 2:
        return None
    top_marker_count = random.randint(2, max_top_markers)
    top_cols = sorted(random.sample(safe_top_cols, top_marker_count))
    top_col_set = set(top_cols)

    marked_object_rows = sorted(
        {row_tops[i] for i in range(obj_count) if modes[i] in {"survive", "row_hit_col_miss"}}
    )
    extra_row_pool = [
        r for r in range(2, h - 2) if r not in set(row_tops)
    ]
    max_row_markers = min(5, len(marked_object_rows) + len(extra_row_pool))
    if max_row_markers < max(2, len(marked_object_rows)):
        return None
    extra_count = random.randint(
        max(0, 2 - len(marked_object_rows)),
        max(0, min(2, max_row_markers - len(marked_object_rows))),
    )
    extra_rows = random.sample(extra_row_pool, extra_count) if extra_count else []
    row_marker_rows = sorted(marked_object_rows + extra_rows)
    row_colors = _choose_row_colors(row_marker_rows)

    final_cols: list[int] = []
    for idx, width in enumerate(widths):
        compatible = _compatible_cols(w, width)
        legal = [c for c in compatible if c in top_col_set]
        illegal = [c for c in compatible if c not in top_col_set]
        if not legal or not illegal:
            return None

        mode = modes[idx]
        if mode in {"survive", "col_hit_row_miss"}:
            final_cols.append(int(random.choice(legal)))
        else:
            final_cols.append(int(random.choice(illegal)))

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    inp[0, top_cols] = TOP_MARKER_COLOR
    expected[0, top_cols] = TOP_MARKER_COLOR
    for row, color in row_colors.items():
        inp[int(row), w - 1] = int(color)
        expected[int(row), w - 1] = int(color)

    for idx, patch in enumerate(patches):
        row_top = int(row_tops[idx])
        width = int(widths[idx])
        input_left = int(final_cols[idx] - width)
        if input_left < 1:
            return None

        paste(inp, patch, row_top, input_left, transparent=BG)

        if modes[idx] == "survive":
            out_patch = np.where(patch == OBJECT_COLOR, row_colors[row_top], BG)
            paste(expected, out_patch, row_top, int(final_cols[idx]), transparent=BG)

    objects = find_components(
        inp,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=[OBJECT_COLOR],
        return_format="dict",
    )
    if len(objects) != obj_count:
        return None
    if len(survivors) < 2 or len(survivors) > obj_count // 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
