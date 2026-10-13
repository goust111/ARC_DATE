from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
RECT_COLORS = (GREEN, GREY)

_GEN_CALL_INDEX = 0
_GEN_RETRY_LIMIT = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _edge_bar(row: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(row == BLUE)
    assert cols.size > 0, "expected a blue edge bar"
    start = int(cols[0])
    end = int(cols[-1])
    assert np.array_equal(cols, np.arange(start, end + 1)), "edge bar must be contiguous"
    row_copy = row.copy()
    row_copy[start : end + 1] = BLACK
    assert not np.any(row_copy != BLACK), "edge-bar rows may only contain blue and black"
    return start, end


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in RECT_COLORS:
        for coords in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h >= 4 and w >= 4, "rectangles must be at least 4x4"
            assert int(coords.shape[0]) == h * w, "expected solid rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochrome"
            rectangles.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    rectangles.sort(key=lambda rect: rect["bbox"])
    assert 2 <= len(rectangles) <= 4, "expected 1-2 rectangles in each half"
    return rectangles


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, RED, GREEN, GREY}, "unexpected input color"

    separator_rows = [r for r in range(1, h - 1) if np.all(g[r] == RED)]
    assert len(separator_rows) == 1, "expected exactly one red separator row"
    separator_r = separator_rows[0]

    top_c0, top_c1 = _edge_bar(g[0])
    bottom_c0, bottom_c1 = _edge_bar(g[-1])
    overlap_c0 = max(top_c0, bottom_c0)
    overlap_c1 = min(top_c1, bottom_c1)
    assert overlap_c0 <= overlap_c1, "edge bars must overlap"

    top_len = top_c1 - top_c0 + 1
    bottom_len = bottom_c1 - bottom_c0 + 1
    overlap_len = overlap_c1 - overlap_c0 + 1
    assert top_len != bottom_len, "one edge bar must be longer"
    assert overlap_len < min(top_len, bottom_len), "edge-bar overlap must be partial"

    rectangles = _extract_rectangles(g)
    top_rects: list[dict] = []
    bottom_rects: list[dict] = []
    border_mask = np.zeros_like(g, dtype=bool)
    interior_mask = np.zeros_like(g, dtype=bool)

    for i, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        assert r0 > 0 and r1 < h - 1, "rectangles must avoid edge-bar rows"
        assert not (r0 <= separator_r <= r1), "rectangles must avoid the separator"
        if r1 < separator_r:
            top_rects.append(rect)
        elif r0 > separator_r:
            bottom_rects.append(rect)
        else:
            raise AssertionError("rectangle must lie fully above or below the separator")

        border_mask[r0 : r1 + 1, c0] = True
        border_mask[r0 : r1 + 1, c1] = True
        border_mask[r0, c0 : c1 + 1] = True
        border_mask[r1, c0 : c1 + 1] = True
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

        for other in rectangles[:i]:
            assert not _boxes_touch(rect["bbox"], other["bbox"]), "rectangles must be separated"

    assert 1 <= len(top_rects) <= 2, "expected 1-2 rectangles above the separator"
    assert 1 <= len(bottom_rects) <= 2, "expected 1-2 rectangles below the separator"

    project_from_top = top_len > bottom_len
    fill_slice = slice(1, separator_r) if project_from_top else slice(separator_r + 1, h - 1)
    projection_mask = np.zeros_like(g, dtype=bool)
    projection_mask[fill_slice, overlap_c0 : overlap_c1 + 1] = True
    projection_fill_mask = projection_mask & ~border_mask

    return {
        "grid": g,
        "shape": (h, w),
        "separator_r": separator_r,
        "top_bar": (top_c0, top_c1),
        "bottom_bar": (bottom_c0, bottom_c1),
        "overlap": (overlap_c0, overlap_c1),
        "project_from_top": project_from_top,
        "fill_slice": fill_slice,
        "projection_mask": projection_mask,
        "projection_fill_mask": projection_fill_mask,
        "rectangles": rectangles,
        "top_rects": top_rects,
        "bottom_rects": bottom_rects,
        "border_mask": border_mask,
        "interior_mask": interior_mask,
    }


def _checkerboard_only(grid: np.ndarray, rectangles: Iterable[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        rows, cols = np.indices(interior.shape)
        interior[(rows + cols) % 2 == 0] = BLACK
    return out


def _apply_rules(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    out = _checkerboard_only(info["grid"], info["rectangles"])
    out[info["projection_fill_mask"]] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Carve checkerboard interiors into each solid rectangle, then overwrite the longer-side overlap projection in yellow.

    Input structure:
    - The background is black.
    - One internal full-width red row separates the grid into an upper and a lower half.
    - The first and last rows each contain one contiguous blue horizontal edge bar of different lengths with a non-empty column overlap.
    - Each half contains one or two solid green or grey rectangles, each at least 4x4 and separated from the others.

    Rule:
    - For every rectangle, keep its one-cell border unchanged and checkerboard-carve the interior using the interior top-left cell as the parity anchor, turning even-parity interior cells black.
    - Compute the overlapping columns of the top and bottom blue bars.
    - If the top bar is longer, fill yellow from row 1 down to the row just above the separator in those overlap columns; otherwise fill yellow from the row just below the separator down to the row just above the bottom edge bar.
    - Projection never changes the red separator, the blue edge bars, or any rectangle border, but it overwrites both black checkerboard holes and preserved rectangle-color interior cells everywhere else in its slab.

    Key insight:
    - Two independently valid transformations coexist: local checkerboard carving inside rectangles and a global overlap projection chosen by the longer edge bar; the output is explained only by giving the yellow projection unconditional priority over rectangle interiors.
    """
    return _apply_rules(np.asarray(grid, dtype=int))


def _sample_bars(width: int, project_from_top: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    max_len = min(8, width - 2)
    for _ in range(300):
        long_len = random.randint(5, max_len)
        short_len = random.randint(3, long_len - 1)
        if project_from_top:
            top_len, bottom_len = long_len, short_len
        else:
            top_len, bottom_len = short_len, long_len

        top_c0 = random.randint(0, width - top_len)
        bottom_c0 = random.randint(0, width - bottom_len)
        top_c1 = top_c0 + top_len - 1
        bottom_c1 = bottom_c0 + bottom_len - 1
        overlap_c0 = max(top_c0, bottom_c0)
        overlap_c1 = min(top_c1, bottom_c1)
        overlap_len = overlap_c1 - overlap_c0 + 1
        if overlap_len < 2:
            continue
        if overlap_len >= min(top_len, bottom_len):
            continue
        if overlap_c0 < 1 or overlap_c1 > width - 2:
            continue
        return (top_c0, top_c1), (bottom_c0, bottom_c1)
    raise RuntimeError("failed to sample valid edge bars")


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _place_rectangle(
    blocked: np.ndarray,
    row_lo: int,
    row_hi: int,
    width: int,
    height: int,
    forced_c0: int | None = None,
) -> tuple[int, int, int, int] | None:
    assert row_lo <= row_hi
    h, w = blocked.shape
    if height > row_hi - row_lo + 1:
        return None

    if forced_c0 is None:
        col_lo = 1
        col_hi = w - width - 1
        if col_lo > col_hi:
            return None
        c0_choices = list(range(col_lo, col_hi + 1))
        random.shuffle(c0_choices)
    else:
        if forced_c0 < 0 or forced_c0 + width > w:
            return None
        c0_choices = [forced_c0]

    r0_choices = list(range(row_lo, row_hi - height + 2))
    random.shuffle(r0_choices)

    for c0 in c0_choices:
        for r0 in r0_choices:
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if np.any(blocked[r0 : r1 + 1, c0 : c1 + 1]):
                continue
            return int(r0), int(c0), int(r1), int(c1)
    return None


def _place_conflict_rectangle(
    blocked: np.ndarray,
    row_lo: int,
    row_hi: int,
    overlap: tuple[int, int],
    align_mode: int,
) -> tuple[int, int, int, int] | None:
    overlap_c0, overlap_c1 = overlap
    widths = list(range(4, min(7, blocked.shape[1] - 1) + 1))
    heights = list(range(4, min(6, row_hi - row_lo + 1) + 1))
    random.shuffle(widths)
    random.shuffle(heights)

    for width in widths:
        forced: list[int] = []
        left_c0 = overlap_c0
        right_c0 = overlap_c1 - width + 1
        if overlap_c0 + 1 <= overlap_c1:
            if align_mode % 2 == 0:
                forced.extend([left_c0, right_c0])
            else:
                forced.extend([right_c0, left_c0])
        seen: set[int] = set()
        forced = [c0 for c0 in forced if not (c0 in seen or seen.add(c0))]
        for height in heights:
            for c0 in forced:
                bbox = _place_rectangle(blocked, row_lo, row_hi, width, height, forced_c0=c0)
                if bbox is not None:
                    r0, _, r1, c1 = bbox
                    rect_overlap_c0 = max(c0, overlap_c0)
                    rect_overlap_c1 = min(c1, overlap_c1)
                    if rect_overlap_c0 <= rect_overlap_c1 and (rect_overlap_c0 < c0 + 1 <= rect_overlap_c1 or rect_overlap_c0 <= c1 - 1 < rect_overlap_c1 + 1):
                        return bbox
    return None


def _place_random_rectangles(
    blocked: np.ndarray,
    row_lo: int,
    row_hi: int,
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    h_max = min(6, row_hi - row_lo + 1)
    if h_max < 4:
        return None
    for _ in range(count):
        placed = None
        for _ in range(200):
            height = random.randint(4, h_max)
            width = random.randint(4, min(7, blocked.shape[1] - 2))
            placed = _place_rectangle(blocked, row_lo, row_hi, width, height)
            if placed is not None:
                _mark_blocked(blocked, placed)
                boxes.append(placed)
                break
        if placed is None:
            return None
    return boxes


def _paint_rectangles(grid: np.ndarray, boxes: Iterable[tuple[int, int, int, int]]) -> list[dict]:
    rectangles: list[dict] = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        color = random.choice(RECT_COLORS)
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        rectangles.append({"bbox": bbox, "color": color})
    return rectangles


def _build_candidate(project_from_top: bool, call_index: int) -> np.ndarray | None:
    width = random.randint(14, 20)
    top_zone = random.randint(6, 8)
    bottom_zone = random.randint(6, 8)
    height = top_zone + bottom_zone + 3
    separator_r = top_zone + 1

    grid = np.full((height, width), BLACK, dtype=int)
    grid[separator_r, :] = RED

    top_bar, bottom_bar = _sample_bars(width, project_from_top)
    grid[0, top_bar[0] : top_bar[1] + 1] = BLUE
    grid[-1, bottom_bar[0] : bottom_bar[1] + 1] = BLUE
    overlap = (max(top_bar[0], bottom_bar[0]), min(top_bar[1], bottom_bar[1]))

    top_row_lo, top_row_hi = 2, separator_r - 2
    bottom_row_lo, bottom_row_hi = separator_r + 2, height - 3

    top_count = 1 + int(random.random() < 0.45)
    bottom_count = 1 + int(random.random() < 0.45)

    blocked = np.zeros((height, width), dtype=bool)
    all_boxes: list[tuple[int, int, int, int]] = []

    if project_from_top:
        conflict_box = _place_conflict_rectangle(blocked, top_row_lo, top_row_hi, overlap, call_index)
        if conflict_box is None:
            return None
        _mark_blocked(blocked, conflict_box)
        all_boxes.append(conflict_box)
        extra_top = _place_random_rectangles(blocked, top_row_lo, top_row_hi, top_count - 1)
        bottom_boxes = _place_random_rectangles(blocked, bottom_row_lo, bottom_row_hi, bottom_count)
        if extra_top is None or bottom_boxes is None:
            return None
        all_boxes.extend(extra_top)
        all_boxes.extend(bottom_boxes)
    else:
        conflict_box = _place_conflict_rectangle(blocked, bottom_row_lo, bottom_row_hi, overlap, call_index)
        if conflict_box is None:
            return None
        _mark_blocked(blocked, conflict_box)
        top_boxes = _place_random_rectangles(blocked, top_row_lo, top_row_hi, top_count)
        extra_bottom = _place_random_rectangles(blocked, bottom_row_lo, bottom_row_hi, bottom_count - 1)
        if top_boxes is None or extra_bottom is None:
            return None
        all_boxes.extend(top_boxes)
        all_boxes.append(conflict_box)
        all_boxes.extend(extra_bottom)

    _paint_rectangles(grid, all_boxes)
    return grid


def _validate_example(grid: np.ndarray, out: np.ndarray) -> bool:
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return False

    checker = _checkerboard_only(info["grid"], info["rectangles"])
    if not np.array_equal(out, _apply_rules(grid)):
        return False

    if not np.array_equal(out[info["border_mask"]], info["grid"][info["border_mask"]]):
        return False

    if not np.all(out[info["projection_fill_mask"]] == YELLOW):
        return False

    non_projected_interiors = info["interior_mask"] & ~info["projection_fill_mask"]
    if not np.any(non_projected_interiors):
        return False
    if not np.array_equal(out[non_projected_interiors], checker[non_projected_interiors]):
        return False

    projected_rectangles = info["top_rects"] if info["project_from_top"] else info["bottom_rects"]
    conflict_cells = np.zeros_like(grid, dtype=bool)
    border_demo = False
    for rect in projected_rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        conflict_cells[r0 + 1 : r1, c0 + 1 : c1] = True
        if info["overlap"][0] <= c0 <= info["overlap"][1] or info["overlap"][0] <= c1 <= info["overlap"][1]:
            border_demo = True
    conflict_cells &= info["projection_fill_mask"]
    if not np.any(conflict_cells):
        return False

    projected_before = checker[conflict_cells]
    if not np.any(projected_before == BLACK):
        return False
    if not np.any(projected_before != BLACK):
        return False
    if not border_demo:
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    call_index = _GEN_CALL_INDEX
    _GEN_CALL_INDEX += 1
    forced_branch = (call_index % 2 == 0)

    for _ in range(_GEN_RETRY_LIMIT):
        grid = _build_candidate(project_from_top=forced_branch, call_index=call_index)
        if grid is None:
            continue
        out = main(grid)
        if _validate_example(grid, out):
            return grid
    raise RuntimeError("failed to generate a valid input")
