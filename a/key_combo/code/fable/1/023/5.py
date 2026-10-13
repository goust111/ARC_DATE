from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask, get_bg


def _shadow_runs(values: np.ndarray, shadow_color: int, *, start: int = 1) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = int(start)
    while i < len(values):
        if int(values[i]) != int(shadow_color):
            i += 1
            continue
        j = i
        while j + 1 < len(values) and int(values[j + 1]) == int(shadow_color):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _visible_squares(grid: np.ndarray, wall_color: int) -> list[dict]:
    mask = grid == int(wall_color)
    mask = mask.copy()
    mask[0, :] = False
    mask[:, 0] = False

    squares: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        box = bbox_coords(coords)
        h = bbox_h(box)
        w = bbox_w(box)
        assert h == w, "visible occluders must be square"
        assert int(coords.shape[0]) == h * w, "visible occluders must be solid"
        r0, c0, r1, c1 = box
        squares.append(
            {
                "bbox": box,
                "size": h,
                "row_seg": (r0, r1),
                "col_seg": (c0, c1),
            }
        )
    return sorted(squares, key=lambda item: (item["row_seg"][0], item["col_seg"][0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild missing wall projections and any erased square by intersecting top and left shadow segments.

    Input structure:
    - The top row and the left column are bright wall lines sharing the corner color.
    - Inside the walls there may be several bright filled squares separated by background.
    - Dark shadow segments can appear on the top wall over square column spans and on the left wall over square row spans.

    Rule:
    - For every visible filled square, its column span and row span determine the top-wall and left-wall shadow intervals it should own.
    - If one of those two intervals is still bright instead of dark, recolor that whole interval to the dark shadow color.
    - Any remaining top-wall and left-wall shadow segments without a visible owner are paired by equal length; the row interval from the left wall and the column interval from the top wall intersect to restore an erased bright filled square.

    Key insight:
    - The same square can be recovered in two opposite ways: from its surviving body to its missing wall trace, or from two orthogonal wall traces back to a completely erased body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D ARC grid with walls"

    wall_color = int(g[0, 0])
    bg = int(get_bg(g))

    wall_palette = set(map(int, g[0, :])) | set(map(int, g[:, 0]))
    assert wall_color in wall_palette
    shadow_candidates = sorted(wall_palette - {wall_color})
    assert len(shadow_candidates) == 1, "expected exactly one shadow color on the walls"
    shadow_color = int(shadow_candidates[0])
    assert shadow_color != bg, "shadow color must differ from background"

    top_segments = _shadow_runs(g[0], shadow_color, start=1)
    left_segments = _shadow_runs(g[:, 0], shadow_color, start=1)
    top_segment_set = set(top_segments)
    left_segment_set = set(left_segments)

    out = g.copy()
    owned_top: set[tuple[int, int]] = set()
    owned_left: set[tuple[int, int]] = set()

    for square in _visible_squares(g, wall_color):
        row_seg = square["row_seg"]
        col_seg = square["col_seg"]

        if col_seg in top_segment_set:
            owned_top.add(col_seg)
        else:
            assert np.all(g[0, col_seg[0] : col_seg[1] + 1] == wall_color)
            out[0, col_seg[0] : col_seg[1] + 1] = shadow_color

        if row_seg in left_segment_set:
            owned_left.add(row_seg)
        else:
            assert np.all(g[row_seg[0] : row_seg[1] + 1, 0] == wall_color)
            out[row_seg[0] : row_seg[1] + 1, 0] = shadow_color

    orphan_top = [seg for seg in top_segments if seg not in owned_top]
    orphan_left = [seg for seg in left_segments if seg not in owned_left]

    top_by_size: dict[int, list[tuple[int, int]]] = defaultdict(list)
    left_by_size: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seg in orphan_top:
        top_by_size[seg[1] - seg[0] + 1].append(seg)
    for seg in orphan_left:
        left_by_size[seg[1] - seg[0] + 1].append(seg)

    assert set(top_by_size) == set(left_by_size), "orphan wall segments must pair by side length"
    for size in sorted(top_by_size):
        top_list = sorted(top_by_size[size])
        left_list = sorted(left_by_size[size])
        assert len(top_list) == len(left_list), "equal-length orphan segments must appear on both walls"
        for col_seg, row_seg in zip(top_list, left_list):
            r0, r1 = row_seg
            c0, c1 = col_seg
            assert (r1 - r0 + 1) == size and (c1 - c0 + 1) == size
            region = out[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(region == bg), "erased square region should be empty before reconstruction"
            out[r0 : r1 + 1, c0 : c1 + 1] = wall_color

    return out


_BG = Color.BLACK
_WALL = Color.YELLOW
_SHADOW = Color.MAROON
_SIZE_POOL = (2, 3, 4, 5)
_MODE_CYCLE = ("mixed", "visible_only", "erased_only")
_MAX_GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _layout_intervals(sizes: list[int]) -> tuple[int, dict[int, tuple[int, int]]]:
    order = random.sample(sizes, len(sizes))
    base_dim = 1 + sum(order) + len(order) + 1
    min_dim = max(12, base_dim)
    max_dim = min(30, base_dim + 7)
    dim = random.randint(min_dim, max_dim)

    gaps = [1] * (len(order) + 1)
    for _ in range(dim - base_dim):
        gaps[random.randrange(len(gaps))] += 1

    intervals: dict[int, tuple[int, int]] = {}
    cursor = 1 + gaps[0]
    for idx, size in enumerate(order):
        intervals[size] = (cursor, cursor + size - 1)
        cursor += size + gaps[idx + 1]
    return dim, intervals


def _choose_roles(sizes: list[int], mode: str, cycle_index: int) -> tuple[set[int], dict[int, str]]:
    if mode == "mixed":
        erased_count = random.randint(1, min(2, len(sizes) - 1))
        missing_count = random.randint(1, len(sizes) - erased_count)
    elif mode == "visible_only":
        erased_count = 0
        missing_count = random.randint(1, len(sizes))
    else:
        erased_count = random.randint(1, min(2, len(sizes) - 1))
        missing_count = 0

    erased_sizes = set(random.sample(sizes, erased_count))
    visible_sizes = [size for size in sizes if size not in erased_sizes]
    missing_sizes = random.sample(visible_sizes, missing_count)

    missing_side: dict[int, str] = {}
    if len(missing_sizes) == 1:
        missing_side[missing_sizes[0]] = "top" if cycle_index % 2 == 0 else "left"
    elif len(missing_sizes) >= 2:
        sides = ["top", "left"]
        while len(sides) < len(missing_sizes):
            sides.append(random.choice(("top", "left")))
        random.shuffle(sides)
        for size, side in zip(random.sample(missing_sizes, len(missing_sizes)), sides):
            missing_side[size] = side

    return erased_sizes, missing_side


def _render_complete(
    row_intervals: dict[int, tuple[int, int]],
    col_intervals: dict[int, tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    out[0, :] = _WALL
    out[:, 0] = _WALL

    for size in row_intervals:
        r0, r1 = row_intervals[size]
        c0, c1 = col_intervals[size]
        out[r0 : r1 + 1, c0 : c1 + 1] = _WALL
        out[0, c0 : c1 + 1] = _SHADOW
        out[r0 : r1 + 1, 0] = _SHADOW

    return out


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    sizes: list[int],
    row_intervals: dict[int, tuple[int, int]],
    col_intervals: dict[int, tuple[int, int]],
    erased_sizes: set[int],
    missing_side: dict[int, str],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    visible_sizes = sorted(size for size in sizes if size not in erased_sizes)
    seen_sizes = sorted(square["size"] for square in _visible_squares(inp, _WALL))
    if seen_sizes != visible_sizes:
        return False

    expected_top = sorted(
        col_intervals[size]
        for size in sizes
        if size in erased_sizes or missing_side.get(size) != "top"
    )
    expected_left = sorted(
        row_intervals[size]
        for size in sizes
        if size in erased_sizes or missing_side.get(size) != "left"
    )
    if _shadow_runs(inp[0], _SHADOW, start=1) != expected_top:
        return False
    if _shadow_runs(inp[:, 0], _SHADOW, start=1) != expected_left:
        return False

    for size in sizes:
        r0, r1 = row_intervals[size]
        c0, c1 = col_intervals[size]
        inp_region = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_region = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(out_region == _WALL):
            return False
        if size in erased_sizes:
            if not np.all(inp_region == _BG):
                return False
            if not np.all(inp[0, c0 : c1 + 1] == _SHADOW):
                return False
            if not np.all(inp[r0 : r1 + 1, 0] == _SHADOW):
                return False
        else:
            if not np.all(inp_region == _WALL):
                return False
            if missing_side.get(size) == "top":
                if not np.all(inp[0, c0 : c1 + 1] == _WALL):
                    return False
                if not np.all(inp[r0 : r1 + 1, 0] == _SHADOW):
                    return False
            elif missing_side.get(size) == "left":
                if not np.all(inp[0, c0 : c1 + 1] == _SHADOW):
                    return False
                if not np.all(inp[r0 : r1 + 1, 0] == _WALL):
                    return False
            else:
                if not np.all(inp[0, c0 : c1 + 1] == _SHADOW):
                    return False
                if not np.all(inp[r0 : r1 + 1, 0] == _SHADOW):
                    return False

    return True


def _attempt_generate(mode: str, cycle_index: int) -> np.ndarray | None:
    total_objects = random.randint(2, 4)
    sizes = sorted(random.sample(_SIZE_POOL, total_objects))
    row_dim, row_intervals = _layout_intervals(sizes)
    col_dim, col_intervals = _layout_intervals(sizes)

    out = _render_complete(row_intervals, col_intervals, (row_dim, col_dim))
    inp = out.copy()

    erased_sizes, missing_side = _choose_roles(sizes, mode, cycle_index)
    for size in erased_sizes:
        r0, r1 = row_intervals[size]
        c0, c1 = col_intervals[size]
        inp[r0 : r1 + 1, c0 : c1 + 1] = _BG

    for size, side in missing_side.items():
        r0, r1 = row_intervals[size]
        c0, c1 = col_intervals[size]
        if side == "top":
            inp[0, c0 : c1 + 1] = _WALL
        else:
            inp[r0 : r1 + 1, 0] = _WALL

    if not _validate_case(inp, out, sizes, row_intervals, col_intervals, erased_sizes, missing_side):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    cycle_index = _GEN_CALL_COUNT
    mode = _MODE_CYCLE[cycle_index % len(_MODE_CYCLE)]
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(mode, cycle_index)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate valid dual-wall projection sample for mode={mode!r}")
