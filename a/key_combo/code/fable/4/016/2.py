from __future__ import annotations

import random
from collections import Counter

import numpy as np


ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
GEN_PERIOD_RANGE = (2, 5)
GEN_RECT_GAP = 1
GEN_RECT_ATTEMPTS = 200


BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_shape(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_area(box: BBox) -> int:
    h, w = _bbox_shape(box)
    return int(h * w)


def _mask_from_bbox(shape: tuple[int, int], box: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _expand_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0]) % tile.shape[0]
    cols = np.arange(shape[1]) % tile.shape[1]
    return tile[rows[:, None], cols[None, :]]


def _majority_color(counter: Counter[int]) -> int:
    best_count = max(counter.values())
    return min(color for color, count in counter.items() if count == best_count)


def _axis_period(tile: np.ndarray, axis: int) -> int:
    length = tile.shape[axis]
    idx = np.arange(length)
    for period in range(1, length + 1):
        if axis == 0:
            if np.array_equal(tile, tile[idx % period, :]):
                return period
        else:
            if np.array_equal(tile, tile[:, idx % period]):
                return period
    return length


def _solid_rectangle_candidates(grid: np.ndarray) -> list[tuple[int, BBox]]:
    candidates: list[tuple[int, BBox]] = []
    for color in sorted(map(int, np.unique(grid))):
        coords = np.argwhere(grid == color)
        box = _bbox_from_coords(coords)
        if coords.shape[0] != _bbox_area(box):
            continue
        r0, c0, r1, c1 = box
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((color, box))
    return candidates


def _fit_periodic_model(
    grid: np.ndarray,
    ignore_mask: np.ndarray,
    row_period: int,
    col_period: int,
    *,
    min_support: int,
) -> dict | None:
    counts = [[Counter() for _ in range(col_period)] for _ in range(row_period)]
    support = np.zeros((row_period, col_period), dtype=int)
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if ignore_mask[r, c]:
                continue
            rr = r % row_period
            cc = c % col_period
            counts[rr][cc][int(grid[r, c])] += 1
            support[rr, cc] += 1

    if np.any(support < min_support):
        return None

    tile = np.zeros((row_period, col_period), dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            if not counts[rr][cc]:
                return None
            tile[rr, cc] = _majority_color(counts[rr][cc])

    expected = _expand_tile(tile, grid.shape)
    mismatch = int(np.sum((grid != expected) & ~ignore_mask))
    return {
        "tile": tile,
        "expected": expected,
        "mismatch": mismatch,
        "row_period": int(row_period),
        "col_period": int(col_period),
        "support_min": int(support.min()),
    }


def _best_periodic_model(grid: np.ndarray, ignore_mask: np.ndarray) -> dict:
    max_row_period = max(1, grid.shape[0] // 2)
    max_col_period = max(1, grid.shape[1] // 2)

    for min_support in (2, 1):
        best: dict | None = None
        best_score: tuple[int, int, int, int, int] | None = None
        for row_period in range(1, max_row_period + 1):
            for col_period in range(1, max_col_period + 1):
                model = _fit_periodic_model(
                    grid,
                    ignore_mask,
                    row_period,
                    col_period,
                    min_support=min_support,
                )
                if model is None:
                    continue
                score = (
                    int(model["mismatch"]),
                    int(row_period * col_period),
                    -int(model["support_min"]),
                    int(row_period),
                    int(col_period),
                )
                if best_score is None or score < best_score:
                    best = model
                    best_score = score
        if best is not None:
            best["score"] = best_score
            return best

    raise AssertionError("failed to infer a periodic model")


def _select_patch_and_model(grid: np.ndarray) -> tuple[int, BBox, dict]:
    candidates = _solid_rectangle_candidates(grid)
    assert candidates, "expected at least one solid rectangle candidate"

    best: tuple[tuple[int, int, int, int, int, int], int, BBox, dict] | None = None
    for color, box in candidates:
        ignore_mask = _mask_from_bbox(grid.shape, box)
        model = _best_periodic_model(grid, ignore_mask)
        tile_colors = set(map(int, model["tile"].ravel()))
        score = (
            int(model["mismatch"]),
            0 if color not in tile_colors else 1,
            int(model["row_period"] * model["col_period"]),
            -int(model["support_min"]),
            int(_bbox_area(box)),
            int(color),
        )
        entry = (score, int(color), box, model)
        if best is None or score < best[0]:
            best = entry

    assert best is not None
    _, color, box, model = best
    return color, box, model


def _best_window_for_mask(mask: np.ndarray, size: tuple[int, int], forbidden: BBox) -> BBox:
    h, w = mask.shape
    box_h, box_w = size
    raw_box = _bbox_from_coords(np.argwhere(mask))
    total = int(mask.sum())
    fr0, fc0, fr1, fc1 = forbidden
    best_box: BBox | None = None
    best_score: tuple[int, int, int, int, int] | None = None

    for r0 in range(h - box_h + 1):
        r1 = r0 + box_h - 1
        for c0 in range(w - box_w + 1):
            c1 = c0 + box_w - 1
            if not (r1 < fr0 or fr1 < r0 or c1 < fc0 or fc1 < c0):
                continue
            count = int(mask[r0 : r1 + 1, c0 : c1 + 1].sum())
            if count == 0:
                continue
            score = (
                0 if count == total else 1,
                -(count),
                abs(r0 - raw_box[0]) + abs(c0 - raw_box[1]),
                int(r0),
                int(c0),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_box = (int(r0), int(c0), int(r1), int(c1))

    assert best_box is not None, "failed to locate anomaly window"
    return best_box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the periodic wallpaper under the visible patch, then move that patch onto the real damaged block.

    Input structure:
    - The full grid is a two-dimensional translation pattern repeated from a small hidden tile.
    - One color forms a single solid rectangle and appears nowhere else; this rectangle is not part of the pattern palette.
    - A second region of the same size contains only pattern colors, but those cells violate the dominant row/column periods.

    Rule:
    - Identify the unique solid off-pattern rectangle and treat it as the movable patch.
    - Infer the smallest repeating row and column periods by majority vote over each residue class, ignoring the patch itself.
    - Reconstruct the periodic pattern everywhere, recover the old patch location, find the mismatching cells outside it, take their damaged bbox, and repaint that bbox with the patch color.

    Key insight:
    - The unique filled rectangle marks the patch, while the wallpaper's translational symmetry reveals both the hidden cells beneath it and the true broken rectangle that the patch must cover.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    patch_color, patch_box, model = _select_patch_and_model(g)
    expected = model["expected"]
    patch_h, patch_w = _bbox_shape(patch_box)
    patch_mask = _mask_from_bbox(g.shape, patch_box)

    diff_mask = (g != expected) & ~patch_mask
    assert np.any(diff_mask), "expected a damaged region outside the patch"
    raw_damage_box = _bbox_from_coords(np.argwhere(diff_mask))
    damage_box = (
        raw_damage_box
        if _bbox_shape(raw_damage_box) == (patch_h, patch_w)
        else _best_window_for_mask(diff_mask, (patch_h, patch_w), patch_box)
    )

    out = expected.copy()
    r0, c0, r1, c1 = damage_box
    out[r0 : r1 + 1, c0 : c1 + 1] = patch_color
    return out


def _sample_dimension(period: int) -> int:
    max_repeats = 30 // period
    repeats = random.randint(3, min(6, max_repeats))
    low = period * repeats
    high = min(30, low + period - 1)
    return random.randint(low, high)


def _random_tile(row_period: int, col_period: int) -> np.ndarray:
    for _ in range(200):
        palette_size = random.randint(2, min(4, row_period * col_period, len(ARC_COLORS) - 1))
        palette = random.sample(list(ARC_COLORS), palette_size)
        tile = np.array(
            [
                [random.choice(palette) for _ in range(col_period)]
                for _ in range(row_period)
            ],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _axis_period(tile, 0) != row_period:
            continue
        if _axis_period(tile, 1) != col_period:
            continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate periodic tile")


def _random_bbox(h: int, w: int, box_h: int, box_w: int) -> BBox:
    r0 = random.randint(0, h - box_h)
    c0 = random.randint(0, w - box_w)
    return int(r0), int(c0), int(r0 + box_h - 1), int(c0 + box_w - 1)


def _boxes_separated(a: BBox, b: BBox, pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_patch_and_damage_boxes(h: int, w: int, box_h: int, box_w: int) -> tuple[BBox, BBox] | None:
    for _ in range(GEN_RECT_ATTEMPTS):
        first = _random_bbox(h, w, box_h, box_w)
        second = _random_bbox(h, w, box_h, box_w)
        if _boxes_separated(first, second, GEN_RECT_GAP):
            return first, second
    return None


def _min_support_after_mask(shape: tuple[int, int], blocked: np.ndarray, row_period: int, col_period: int) -> int:
    support = np.zeros((row_period, col_period), dtype=int)
    h, w = shape
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            support[r % row_period, c % col_period] += 1
    return int(support.min())


def _gen_attempt() -> np.ndarray | None:
    row_period = random.randint(*GEN_PERIOD_RANGE)
    col_period = random.randint(*GEN_PERIOD_RANGE)
    h = _sample_dimension(row_period)
    w = _sample_dimension(col_period)

    tile = _random_tile(row_period, col_period)
    base = _expand_tile(tile, (h, w))
    palette = sorted(set(map(int, tile.ravel())))
    patch_color = random.choice([color for color in ARC_COLORS if color not in palette])

    max_box_h = min(6, h - 2)
    max_box_w = min(6, w - 2)
    if max_box_h < 2 or max_box_w < 2:
        return None
    box_h = random.randint(2, max_box_h)
    box_w = random.randint(2, max_box_w)

    boxes = _sample_patch_and_damage_boxes(h, w, box_h, box_w)
    if boxes is None:
        return None
    patch_box, damage_box = boxes

    patch_mask = _mask_from_bbox((h, w), patch_box)
    damage_mask = _mask_from_bbox((h, w), damage_box)
    blocked = patch_mask | damage_mask
    if _min_support_after_mask((h, w), blocked, row_period, col_period) < 2:
        return None

    grid = base.copy()
    dr0, dc0, dr1, dc1 = damage_box
    for r in range(dr0, dr1 + 1):
        for c in range(dc0, dc1 + 1):
            choices = [color for color in palette if color != int(base[r, c])]
            grid[r, c] = random.choice(choices)

    pr0, pc0, pr1, pc1 = patch_box
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = patch_color

    candidates = _solid_rectangle_candidates(grid)
    if candidates != [(patch_color, patch_box)]:
        return None

    model = _fit_periodic_model(grid, patch_mask, row_period, col_period, min_support=2)
    if model is None or not np.array_equal(model["expected"], base):
        return None

    diff_mask = (grid != base) & ~patch_mask
    if _bbox_from_coords(np.argwhere(diff_mask)) != damage_box:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid patch-relocation input")
