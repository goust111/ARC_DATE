from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_separated(a: BBox, b: BBox, gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _tile_grid(shape: tuple[int, int], tile: np.ndarray, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    h, w = tile.shape
    dr, dc = shift
    rows = (np.arange(shape[0])[:, None] + int(dr)) % h
    cols = (np.arange(shape[1])[None, :] + int(dc)) % w
    return tile[rows, cols].astype(int)


def _tile_crop(tile: np.ndarray, box: BBox, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    h, w = tile.shape
    r0, c0, r1, c1 = box
    dr, dc = shift
    rows = (np.arange(r0, r1 + 1)[:, None] + int(dr)) % h
    cols = (np.arange(c0, c1 + 1)[None, :] + int(dc)) % w
    return tile[rows, cols].astype(int)


def _swap_values(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = arr.copy()
    ma = arr == int(a)
    mb = arr == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _majority_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.empty((row_period, col_period), dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            values = [int(v) for v in grid[rr::row_period, cc::col_period].ravel()]
            counts = Counter(values).most_common()
            if not counts:
                return None
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return None
            if counts[0][1] * 2 <= len(values):
                return None
            tile[rr, cc] = counts[0][0]
    return tile


def _period_shape(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    options: list[tuple[int, int, int]] = []
    for rh in range(1, h + 1):
        for cw in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % rh, c % cw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                options.append((rh * cw, rh, cw))
    area, rh, cw = min(options)
    same = [(r, c) for a, r, c in options if a == area]
    assert len(same) == 1
    return int(rh), int(cw)


def _shift_tile(tile: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return _tile_grid(tile.shape, tile, shift)


def _is_swap_version(base: np.ndarray, other: np.ndarray) -> bool:
    if base.shape != other.shape:
        return False
    colors = sorted(set(map(int, base.ravel())) | set(map(int, other.ravel())))
    for a, b in itertools.combinations(colors, 2):
        if np.array_equal(_swap_values(base, a, b), other):
            return True
    return False


def _tile_has_safe_phase_and_color_structure(tile: np.ndarray) -> bool:
    if len(set(map(int, tile.ravel()))) < 3:
        return False
    if _period_shape(tile) != tile.shape:
        return False
    h, w = tile.shape
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            shifted = _shift_tile(tile, (dr, dc))
            if np.array_equal(shifted, tile):
                return False
            if _is_swap_version(tile, shifted):
                return False
    return True


def _cluster_mismatch_boxes(mask: np.ndarray, row_period: int, col_period: int) -> list[BBox]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return []

    unseen = set(range(len(coords)))
    boxes: list[BBox] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = [start]
        while stack:
            i = stack.pop()
            r, c = coords[i]
            close = [
                j
                for j in list(unseen)
                if abs(coords[j][0] - r) <= row_period and abs(coords[j][1] - c) <= col_period
            ]
            for j in close:
                unseen.remove(j)
                stack.append(j)
                comp.append(j)

        rows = [coords[i][0] for i in comp]
        cols = [coords[i][1] for i in comp]
        boxes.append((min(rows), min(cols), max(rows), max(cols)))

    boxes.sort()
    return boxes


def _recolor_pairs(actual: np.ndarray, expected: np.ndarray, box: BBox) -> list[tuple[int, int]]:
    src = _crop(expected, box)
    dst = _crop(actual, box)
    if np.array_equal(src, dst):
        return []

    colors = sorted(set(map(int, src.ravel())) | set(map(int, dst.ravel())))
    matches: list[tuple[int, int]] = []
    for a, b in itertools.combinations(colors, 2):
        if int(np.sum(src == a)) == 0 or int(np.sum(src == b)) == 0:
            continue
        if int(np.sum(src == a)) != int(np.sum(src == b)):
            continue
        if np.array_equal(_swap_values(src, a, b), dst):
            matches.append((int(a), int(b)))
    return matches


def _phase_shifts(actual: np.ndarray, tile: np.ndarray, box: BBox) -> list[tuple[int, int]]:
    dst = _crop(actual, box)
    matches: list[tuple[int, int]] = []
    h, w = tile.shape
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(_tile_crop(tile, box, (dr, dc)), dst):
                matches.append((int(dr), int(dc)))
    return matches


def _solve_with_period(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = _majority_tile(grid, row_period, col_period)
    if tile is None or not _tile_has_safe_phase_and_color_structure(tile):
        return None

    expected = _tile_grid(grid.shape, tile)
    mismatch = grid != expected
    mismatch_count = int(np.sum(mismatch))
    if mismatch_count == 0 or mismatch_count * 2 >= grid.size:
        return None

    boxes = _cluster_mismatch_boxes(mismatch, row_period, col_period)
    if len(boxes) != 2 or _boxes_overlap(boxes[0], boxes[1]):
        return None

    classified: list[tuple[str, BBox]] = []
    for box in boxes:
        if _box_h(box) < row_period or _box_w(box) < col_period:
            return None
        recolors = _recolor_pairs(grid, expected, box)
        phases = _phase_shifts(grid, tile, box)
        if len(recolors) == 1 and not phases:
            classified.append(("recolor", box))
        elif len(phases) == 1 and not recolors:
            classified.append(("phase", box))
        else:
            return None

    phase_boxes = [box for kind, box in classified if kind == "phase"]
    recolor_boxes = [box for kind, box in classified if kind == "recolor"]
    if len(phase_boxes) != 1 or len(recolor_boxes) != 1:
        return None
    return _crop(grid, phase_boxes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the phase-shifted rectangular anomaly from a corrupted periodic tiling.

    Input structure:
    - The whole grid is mostly a repeat of one smallest rectangular tile using at least three colors.
    - Exactly two separated rectangular patches violate that tiling; each patch is at least one tile tall and one tile wide.
    - One patch is an equal-count swap of two tile colors at the original phase, while the other is the same tile pattern at a nonzero phase shift.

    Rule:
    - Infer the smallest global tile and phase from the majority of uncorrupted cells.
    - Find the two rectangular regions where the inferred tiling fails.
    - Classify each region by exact comparison with either an equal two-color swap or a nonzero tile phase shift, then return the phase-shifted region as it appears in the input.

    Key insight:
    - The target is not drawn with a marker; it is the rectangle whose periodic texture is out of phase, not merely recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    h, w = g.shape
    periods = sorted((rh * cw, rh, cw) for rh in range(1, h + 1) for cw in range(1, w + 1))
    solutions: list[np.ndarray] = []
    found_area: int | None = None
    for area, rh, cw in periods:
        if found_area is not None and area > found_area:
            break
        solved = _solve_with_period(g, rh, cw)
        if solved is not None:
            found_area = int(area)
            solutions.append(solved)

    assert len(solutions) == 1, "expected one uniquely determined phase patch"
    return solutions[0]


# --------------- generator ---------------


_MAX_ATTEMPTS = 600
_PALETTE = tuple(range(10))
_TILE_SHAPES = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))


def _pair_covers_rows_cols(tile: np.ndarray, a: int, b: int) -> bool:
    mask = (tile == int(a)) | (tile == int(b))
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _mismatch_covers_rows_cols(tile: np.ndarray, shift: tuple[int, int]) -> bool:
    diff = _shift_tile(tile, shift) != tile
    return bool(np.all(diff.any(axis=0)) and np.all(diff.any(axis=1)))


def _make_tile() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    th, tw = random.choice(_TILE_SHAPES)
    area = th * tw
    color_count = random.randint(3, min(5, area))
    colors = random.sample(_PALETTE, color_count)
    a, b = colors[0], colors[1]

    min_each = (max(th, tw) + 1) // 2
    max_each = (area - (color_count - 2)) // 2
    if min_each > max_each:
        return None

    for _ in range(250):
        each = random.randint(min_each, max_each)
        values = [a] * each + [b] * each
        remaining = area - len(values)
        fillers = colors[2:]
        values.extend(fillers)
        remaining -= len(fillers)
        values.extend(random.choice(fillers) for _ in range(remaining))
        random.shuffle(values)
        tile = np.array(values, dtype=int).reshape(th, tw)

        if not _pair_covers_rows_cols(tile, a, b):
            continue
        if not _tile_has_safe_phase_and_color_structure(tile):
            continue

        phase_options = [
            (dr, dc)
            for dr in range(th)
            for dc in range(tw)
            if (dr, dc) != (0, 0) and _mismatch_covers_rows_cols(tile, (dr, dc))
        ]
        if not phase_options:
            continue
        return tile, (int(a), int(b)), random.choice(phase_options)
    return None


def _choose_box(height: int, width: int, box_h: int, box_w: int) -> BBox:
    edge_bias = random.random()
    if edge_bias < 0.18:
        r0 = random.choice([0, height - box_h])
    else:
        r0 = random.randint(0, height - box_h)
    if edge_bias > 0.82:
        c0 = random.choice([0, width - box_w])
    else:
        c0 = random.randint(0, width - box_w)
    return int(r0), int(c0), int(r0 + box_h - 1), int(c0 + box_w - 1)


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    made = _make_tile()
    if made is None:
        return None
    tile, swap_pair, phase_shift = made
    th, tw = tile.shape

    height = random.randint(max(14, 4 * th), 30)
    width = random.randint(max(14, 4 * tw), 30)
    base = _tile_grid((height, width), tile)

    max_kh = max(1, min(3, height // th))
    max_kw = max(1, min(3, width // tw))
    dims: list[tuple[int, int]] = []
    for kh in range(1, max_kh + 1):
        for kw in range(1, max_kw + 1):
            bh, bw = kh * th, kw * tw
            if bh <= height and bw <= width and bh * bw <= int(0.22 * height * width):
                dims.append((bh, bw))
    if not dims:
        return None

    for _ in range(250):
        rec_h, rec_w = random.choice(dims)
        phase_h, phase_w = random.choice(dims)
        rec_box = _choose_box(height, width, rec_h, rec_w)
        phase_box = _choose_box(height, width, phase_h, phase_w)
        if not _boxes_separated(rec_box, phase_box, max(th, tw)):
            continue

        inp = base.copy()
        a, b = swap_pair
        r0, c0, r1, c1 = rec_box
        inp[r0 : r1 + 1, c0 : c1 + 1] = _swap_values(base[r0 : r1 + 1, c0 : c1 + 1], a, b)

        r0, c0, r1, c1 = phase_box
        inp[r0 : r1 + 1, c0 : c1 + 1] = _tile_crop(tile, phase_box, phase_shift)
        expected = _crop(inp, phase_box)

        mismatch = inp != base
        if sorted(_cluster_mismatch_boxes(mismatch, th, tw)) != sorted([rec_box, phase_box]):
            continue
        if len(_recolor_pairs(inp, base, rec_box)) != 1 or _phase_shifts(inp, tile, rec_box):
            continue
        if len(_phase_shifts(inp, tile, phase_box)) != 1 or _recolor_pairs(inp, base, phase_box):
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        rendered = _render_attempt()
        if rendered is not None:
            return rendered[0]
    raise RuntimeError("failed to generate a valid phase-patch input")
