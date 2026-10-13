from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


RANK_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
RECT_DIMENSIONS = tuple(
    (h, w) for h in range(2, 6) for w in range(2, 6) if h != w
)
NON_IDENTITY_PERMUTATIONS = tuple(
    perm for perm in itertools.permutations(range(3)) if perm != (0, 1, 2)
)
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 22
INPUT_GAP_MIN = 1
INPUT_GAP_MAX = 3
MAX_GENERATION_ATTEMPTS = 400


def _rotate_by_rank(patch: np.ndarray, rank_index: int) -> np.ndarray:
    if rank_index == 0:
        return patch.copy()
    if rank_index == 1:
        return np.rot90(patch, -1).copy()
    if rank_index == 2:
        return np.rot90(patch, 2).copy()
    raise ValueError(f"unsupported rank index: {rank_index}")


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = map(int, g.shape)
    bg = int(get_bg(g))

    present = set(map(int, np.unique(g)))
    assert present <= {bg, Color.GREY}, "expected only background and grey rectangles"

    objects = find_components(
        g,
        bg=bg,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
    )
    assert len(objects) == 3, "expected exactly three grey rectangles"

    rectangles: list[dict[str, object]] = []
    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        assert r1 == height - 1, "each rectangle must touch the bottom row"
        patch = crop_bbox(g, (r0, c0, r1, c1))
        assert np.all(patch == Color.GREY), "each object must be a solid grey rectangle"

        h, w = map(int, patch.shape)
        assert 2 <= h <= 5 and 2 <= w <= 5, "rectangle sides must be between 2 and 5"
        assert h != w, "rectangles must be non-square"
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "left": c0,
                "area": h * w,
                "patch": patch,
            }
        )

    left_edges = [int(rect["left"]) for rect in rectangles]
    areas = [int(rect["area"]) for rect in rectangles]
    assert len(set(left_edges)) == 3, "left edges must be distinct"
    assert len(set(areas)) == 3, "areas must be distinct"

    for prev, cur in zip(
        sorted(rectangles, key=lambda rect: int(rect["left"])),
        sorted(rectangles, key=lambda rect: int(rect["left"]))[1:],
    ):
        prev_c1 = int(prev["bbox"][3])
        cur_c0 = int(cur["bbox"][1])
        assert cur_c0 > prev_c1 + 1, "rectangles must be separated by background columns"

    return bg, rectangles


def _chain_layout(
    rectangles: list[dict[str, object]],
    *,
    order_key: str,
    rotate: bool,
    offset_from_rotated: bool,
) -> tuple[list[tuple[int, int, np.ndarray]], int, int]:
    ordered = sorted(rectangles, key=lambda rect: int(rect[order_key]))
    placements: list[tuple[int, int, np.ndarray]] = []
    row = 0
    col = 0
    max_row = 0
    max_col = 0

    for rank_index, rect in enumerate(ordered):
        original = np.asarray(rect["patch"], dtype=int)
        placed = _rotate_by_rank(original, rank_index) if rotate else original.copy()
        placements.append((row, col, placed))

        h, w = map(int, placed.shape)
        max_row = max(max_row, row + h)
        max_col = max(max_col, col + w)

        if offset_from_rotated:
            step_h, step_w = h, w
        else:
            step_h, step_w = map(int, original.shape)
        row += step_h - 1
        col += step_w - 1

    return placements, max_row, max_col


def _build_output(
    grid_shape: tuple[int, int],
    bg: int,
    rectangles: list[dict[str, object]],
    *,
    order_key: str,
    rotate: bool,
    offset_from_rotated: bool,
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    placements, max_row, max_col = _chain_layout(
        rectangles,
        order_key=order_key,
        rotate=rotate,
        offset_from_rotated=offset_from_rotated,
    )
    assert max_row <= grid_shape[0] and max_col <= grid_shape[1], "chain must fit on the canvas"

    for rank_index, (row, col, placed) in enumerate(placements):
        colored = np.full(placed.shape, RANK_COLORS[rank_index], dtype=int)
        paste(out, colored, row, col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank bottom grey rectangles by area, then recolor, rotate, and repack them into a diagonal chain.

    Input structure:
    - The background is black.
    - Exactly three grey solid rectangles touch the bottom row.
    - The rectangles are pairwise separated by background columns, have distinct left edges, distinct areas, and non-square side lengths between 2 and 5.

    Rule:
    - Rank the three grey rectangles by pixel area from smallest to largest.
    - Recolor them by rank: smallest blue, middle red, largest green.
    - Rotate the rank-1 rectangle by 0 degrees, the rank-2 rectangle by 90 degrees clockwise, and the rank-3 rectangle by 180 degrees.
    - Clear the canvas to the background, place the rank-1 rectangle at the upper-left corner, then place each next rotated rectangle at the previous top-left plus (h-1, w-1), where h and w are the previous rotated height and width.
    - Later placements overwrite the shared corner pixel.

    Key insight:
    - Area rank is the single decision variable, and it simultaneously determines color, rotation, and placement order; the middle rectangle's 90-degree rotation changes the offset that places the final rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg, rectangles = _extract_rectangles(g)
    return _build_output(
        g.shape,
        bg,
        rectangles,
        order_key="area",
        rotate=True,
        offset_from_rotated=True,
    )


def _sample_dimensions() -> list[tuple[int, int]] | None:
    choice = np.random.choice(len(RECT_DIMENSIONS), size=3, replace=False)
    dims = [RECT_DIMENSIONS[int(idx)] for idx in choice]
    if len({h * w for h, w in dims}) != 3:
        return None
    return dims


def _random_canvas_size(min_side: int) -> int:
    low = max(GRID_SIDE_MIN, min_side + 1)
    high = min(GRID_SIDE_MAX, min_side + 6)
    if low > high:
        raise ValueError("no valid canvas size range")
    return int(np.random.randint(low, high + 1))


def _distribute_extra_width(extra: int) -> tuple[int, int, int]:
    left_margin = int(np.random.randint(extra + 1))
    remaining = extra - left_margin
    extra_gap0 = int(np.random.randint(remaining + 1))
    remaining -= extra_gap0
    extra_gap1 = int(np.random.randint(remaining + 1))
    return left_margin, extra_gap0, extra_gap1


def _gen_attempt() -> np.ndarray | None:
    dims = _sample_dimensions()
    if dims is None:
        return None

    ranked_dims = sorted(dims, key=lambda hw: hw[0] * hw[1])
    perm = NON_IDENTITY_PERMUTATIONS[
        int(np.random.randint(len(NON_IDENTITY_PERMUTATIONS)))
    ]
    input_dims = [ranked_dims[idx] for idx in perm]

    chain_rectangles = [
        {"patch": np.full(hw, Color.GREY, dtype=int), "area": hw[0] * hw[1], "left": i}
        for i, hw in enumerate(ranked_dims)
    ]
    _placements, chain_h, chain_w = _chain_layout(
        chain_rectangles,
        order_key="area",
        rotate=True,
        offset_from_rotated=True,
    )

    gaps = [
        int(np.random.randint(INPUT_GAP_MIN, INPUT_GAP_MAX + 1)),
        int(np.random.randint(INPUT_GAP_MIN, INPUT_GAP_MAX + 1)),
    ]
    input_h = max(h for h, _w in input_dims)
    input_w = sum(w for _h, w in input_dims) + sum(gaps)

    min_h = max(input_h, chain_h)
    min_w = max(input_w, chain_w)
    try:
        height = _random_canvas_size(min_h)
        width = _random_canvas_size(min_w)
    except ValueError:
        return None

    extra_width = width - input_w
    left_margin, extra_gap0, extra_gap1 = _distribute_extra_width(extra_width)
    gaps[0] += extra_gap0
    gaps[1] += extra_gap1

    inp = np.full((height, width), Color.BLACK, dtype=int)
    rectangles: list[dict[str, object]] = []
    left = left_margin
    for index, (h, w) in enumerate(input_dims):
        patch = np.full((h, w), Color.GREY, dtype=int)
        paste(inp, patch, height - h, left)
        rectangles.append(
            {
                "bbox": (height - h, left, height - 1, left + w - 1),
                "left": left,
                "area": h * w,
                "patch": patch,
            }
        )
        left += w
        if index < 2:
            left += gaps[index]

    correct = _build_output(
        inp.shape,
        Color.BLACK,
        rectangles,
        order_key="area",
        rotate=True,
        offset_from_rotated=True,
    )
    by_left_edge = _build_output(
        inp.shape,
        Color.BLACK,
        rectangles,
        order_key="left",
        rotate=True,
        offset_from_rotated=True,
    )
    without_rotation = _build_output(
        inp.shape,
        Color.BLACK,
        rectangles,
        order_key="area",
        rotate=False,
        offset_from_rotated=False,
    )
    wrong_offset = _build_output(
        inp.shape,
        Color.BLACK,
        rectangles,
        order_key="area",
        rotate=True,
        offset_from_rotated=False,
    )

    if np.array_equal(correct, by_left_edge):
        return None
    if np.array_equal(correct, without_rotation):
        return None
    if np.array_equal(correct, wrong_offset):
        return None
    if not np.array_equal(main(inp), correct):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
