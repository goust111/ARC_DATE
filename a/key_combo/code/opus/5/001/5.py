from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
TRACK = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIZE_OPTIONS = (3, 4, 5, 6)
GEN_ATTEMPTS = 400


def _make_bracket(size: int, color: int) -> np.ndarray:
    patch = np.zeros((size, 3), dtype=int)
    patch[:, 0] = color
    patch[0, :] = color
    patch[-1, :] = color
    return patch


def _make_square(size: int, border: int, fill: int) -> np.ndarray:
    patch = np.full((size, size), fill, dtype=int)
    patch[0, :] = border
    patch[-1, :] = border
    patch[:, 0] = border
    patch[:, -1] = border
    return patch


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rectangles_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return not (
        ar + asz <= br
        or br + bsz <= ar
        or ac + asz <= bc
        or bc + bsz <= ac
    )


def _is_bracket_component(comp: dict, grid: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    color = int(comp["color"])
    if patch.shape[1] != 3 or patch.shape[0] < 3:
        return None
    expected = np.zeros_like(patch)
    expected[:, 0] = color
    expected[0, :] = color
    expected[-1, :] = color
    if not np.array_equal(patch, expected):
        return None
    return {
        "color": color,
        "size": int(patch.shape[0]),
        "r": r0,
        "c": c0,
        "target_c": c1 + 1,
        "patch": patch,
    }


def _is_square_component(comp: dict, grid: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    h, w = patch.shape
    if h != w or h < 3:
        return None
    if int(comp["area"]) != h * w:
        return None
    border = patch[0, 0]
    if border == bg:
        return None
    border_mask = np.zeros_like(patch, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    if not np.all(patch[border_mask] == border):
        return None
    fill_values = np.unique(patch[1:-1, 1:-1])
    if fill_values.size != 1:
        return None
    fill = int(fill_values[0])
    if fill in (bg, int(border)):
        return None
    return {
        "color": int(border),
        "fill": fill,
        "size": h,
        "r": r0,
        "c": c0,
        "patch": patch,
    }


def _parse_scene(grid: np.ndarray) -> tuple[dict[int, dict], list[dict]]:
    bg = int(get_bg(grid))
    mono = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    brackets: dict[int, dict] = {}
    for comp in mono:
        info = _is_bracket_component(comp, grid, bg)
        if info is None:
            continue
        assert info["color"] not in brackets, "each bracket color must be unique"
        brackets[info["color"]] = info

    mixed = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    squares: list[dict] = []
    for comp in mixed:
        info = _is_square_component(comp, grid, bg)
        if info is not None:
            squares.append(info)

    assert 3 <= len(squares) <= 4, "expected 3-4 framed squares"
    assert len(brackets) == len(squares), "every square needs one matching bracket"
    colors = {sq["color"] for sq in squares}
    assert colors == set(brackets), "brackets and squares must match by border color"
    assert len({sq["size"] * sq["size"] for sq in squares}) == len(squares), "square areas must be distinct"
    assert len({sq["fill"] for sq in squares}) == len(squares), "fill colors must be unique"
    return brackets, squares


def _paint_track(out: np.ndarray, r: int, src_c: int, dst_c: int, size: int) -> None:
    if src_c == dst_c:
        return
    if dst_c < src_c:
        start = dst_c + 1
        end = src_c + size - 1
    else:
        start = src_c
        end = dst_c + size - 2
    if start <= end:
        out[r : r + size, start : end + 1] = TRACK


def _solve_scene(
    shape: tuple[int, int],
    brackets: dict[int, dict],
    squares: list[dict],
    *,
    order: list[int] | None = None,
    tracks_block: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    out = np.full(shape, BG, dtype=int)
    occupancy = np.full(shape, BG, dtype=int)
    for bracket in brackets.values():
        paste(out, bracket["patch"], bracket["r"], bracket["c"], transparent=BG)
        paste(occupancy, bracket["patch"], bracket["r"], bracket["c"], transparent=BG)

    if order is None:
        order = sorted(range(len(squares)), key=lambda i: squares[i]["size"] * squares[i]["size"], reverse=True)

    placements: list[dict] = []
    for idx in order:
        square = squares[idx]
        size = int(square["size"])
        row = int(square["r"])
        src_c = int(square["c"])
        col = int(brackets[square["color"]]["target_c"])
        shifts = 0
        while True:
            assert row + size <= shape[0], "square row out of bounds"
            assert col + size <= shape[1], "square column out of bounds"
            region = occupancy[row : row + size, col : col + size]
            if np.all(region == BG):
                break
            col += size
            shifts += 1

        _paint_track(out, row, src_c, col, size)
        if tracks_block:
            _paint_track(occupancy, row, src_c, col, size)
        paste(out, square["patch"], row, col, transparent=BG)
        paste(occupancy, square["patch"], row, col, transparent=BG)
        placements.append({"index": idx, "final_c": col, "shifts": shifts})
    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide border squares left toward their same-color brackets in descending area order, leaving gray tracks that block later docking.

    Input structure:
    - The background is black.
    - Several colored C-shaped brackets sit on the left, one per border color.
    - The right side contains the same number of filled square frames whose border colors match the brackets and whose interior colors are unique.

    Rule:
    - Match each square to the bracket with the same border color.
    - Sort the squares by area from largest to smallest.
    - Process them in that order: keep the square's row fixed, try to place it with its left edge on the first column immediately right of the matching bracket, and if any target cell is already non-black then shift the target right by exactly one square width until the whole square fits on black cells.
    - Draw the moved square at its final location, erase its source location, and paint every swept horizontal column in its row span gray except the final resting position itself.
    - The brackets stay unchanged, and earlier tracks or earlier placed squares act as blockers for later squares.

    Key insight:
    - Color identifies the bracket, but area determines who moves first; the early squares leave wide gray obstacle bands, so the later squares can be forced farther right than their own bracket-aligned targets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    brackets, squares = _parse_scene(g)
    out, _placements = _solve_scene(g.shape, brackets, squares)
    return out


def _sample_source_rows(height: int, sizes: list[int]) -> list[int] | None:
    for _ in range(200):
        rows = [random.randint(0, height - size) for size in sizes]
        overlaps = [
            (i, j)
            for i in range(len(sizes))
            for j in range(i + 1, len(sizes))
            if _intervals_overlap(rows[i], rows[i] + sizes[i] - 1, rows[j], rows[j] + sizes[j] - 1)
        ]
        if not overlaps:
            continue
        largest = max(range(len(sizes)), key=lambda i: sizes[i])
        if not any(largest in pair for pair in overlaps):
            continue
        return rows
    return None


def _sample_source_cols(width: int, sizes: list[int], rows: list[int], min_col: int) -> list[int] | None:
    for _ in range(400):
        cols = [random.randint(min_col, width - size) for size in sizes]
        rects = [(rows[i], cols[i], sizes[i]) for i in range(len(sizes))]
        if any(_rectangles_overlap(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects))):
            continue
        return cols
    return None


def _sample_bracket_rows(height: int, sizes: list[int]) -> list[int] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    rows = [0] * len(sizes)
    cursor = random.randint(0, 2)
    for pos, idx in enumerate(order):
        remaining = sum(sizes[j] for j in order[pos + 1 :]) + 2 * (len(order) - pos - 1)
        top_max = height - remaining - sizes[idx]
        if cursor > top_max:
            return None
        top = random.randint(cursor, top_max)
        rows[idx] = top
        cursor = top + sizes[idx] + 2
    return rows


def _semantic_checks(
    inp: np.ndarray,
    brackets: dict[int, dict],
    squares: list[dict],
    expected: np.ndarray,
    placements: list[dict],
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    desc_order = sorted(range(len(squares)), key=lambda i: squares[i]["size"], reverse=True)
    asc_order = list(reversed(desc_order))
    try:
        asc_out, _ = _solve_scene(inp.shape, brackets, squares, order=asc_order, tracks_block=True)
    except AssertionError:
        asc_out = None
    if asc_out is not None and np.array_equal(expected, asc_out):
        return False

    track_sensitive = True
    try:
        no_track_out, no_track_places = _solve_scene(inp.shape, brackets, squares, order=desc_order, tracks_block=False)
    except AssertionError:
        no_track_out = None
        no_track_places = None
    else:
        if np.array_equal(expected, no_track_out):
            return False
        track_sensitive = any(a["final_c"] != b["final_c"] for a, b in zip(placements, no_track_places))

    if not any(item["shifts"] > 0 for item in placements):
        return False
    if not track_sensitive:
        return False

    row_overlaps = 0
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if _intervals_overlap(
                squares[i]["r"],
                squares[i]["r"] + squares[i]["size"] - 1,
                squares[j]["r"],
                squares[j]["r"] + squares[j]["size"] - 1,
            ):
                row_overlaps += 1
    return row_overlaps >= 1


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((3, 4))
    sizes = random.sample(SIZE_OPTIONS, count)
    border_colors = random.sample(PALETTE, count)
    fill_pool = [color for color in PALETTE if color not in border_colors]
    fill_colors = random.sample(fill_pool, count)

    min_height = sum(sizes) + 2 * (count - 1)
    height = random.randint(max(20, min_height), 29)
    width = random.randint(24, 29)
    bracket_col = random.choice((0, 1))

    bracket_rows = _sample_bracket_rows(height, sizes)
    if bracket_rows is None:
        return None
    source_rows = _sample_source_rows(height, sizes)
    if source_rows is None:
        return None
    source_cols = _sample_source_cols(width, sizes, source_rows, min_col=width // 2)
    if source_cols is None:
        return None

    brackets: dict[int, dict] = {}
    squares: list[dict] = []
    for size, border, fill, br, sr, sc in zip(sizes, border_colors, fill_colors, bracket_rows, source_rows, source_cols):
        bracket_patch = _make_bracket(size, border)
        square_patch = _make_square(size, border, fill)
        if br + size > height or sr + size > height or sc + size > width:
            return None
        brackets[border] = {
            "color": border,
            "size": size,
            "r": br,
            "c": bracket_col,
            "target_c": bracket_col + bracket_patch.shape[1],
            "patch": bracket_patch,
        }
        squares.append(
            {
                "color": border,
                "fill": fill,
                "size": size,
                "r": sr,
                "c": sc,
                "patch": square_patch,
            }
        )

    inp = np.zeros((height, width), dtype=int)
    for bracket in brackets.values():
        paste(inp, bracket["patch"], bracket["r"], bracket["c"], transparent=BG)
    for square in squares:
        paste(inp, square["patch"], square["r"], square["c"], transparent=BG)

    try:
        expected, placements = _solve_scene(inp.shape, brackets, squares)
    except AssertionError:
        return None
    if not _semantic_checks(inp, brackets, squares, expected, placements):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid area-sorted sliding-track puzzle")
