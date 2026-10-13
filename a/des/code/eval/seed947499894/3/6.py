from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

MAX_GEN_ATTEMPTS = 400
PAIR_COUNT_RANGE = (1, 4)
GRID_SIDE_RANGE = (18, 31)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 9)
ANCHOR_STRIP = 2
PLACEMENT_TRIES = 250
PADDING = 1


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            pts.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _flat_sides(mask: np.ndarray) -> set[str]:
    sides: set[str] = set()
    if bool(mask[:, 0].all()):
        sides.add("left")
    if bool(mask[:, -1].all()):
        sides.add("right")
    if bool(mask[0, :].all()):
        sides.add("top")
    if bool(mask[-1, :].all()):
        sides.add("bottom")
    return sides


def _rows_match_side(mask: np.ndarray, side: str) -> bool:
    width = int(mask.shape[1])
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size == 0:
            return False
        if int(cols[-1] - cols[0] + 1) != int(cols.size):
            return False
        if side == "left" and int(cols[0]) != 0:
            return False
        if side == "right" and int(cols[-1]) != width - 1:
            return False
    return True


def _cols_match_side(mask: np.ndarray, side: str) -> bool:
    height = int(mask.shape[0])
    for col in mask.T:
        rows = np.flatnonzero(col)
        if rows.size == 0:
            return False
        if int(rows[-1] - rows[0] + 1) != int(rows.size):
            return False
        if side == "top" and int(rows[0]) != 0:
            return False
        if side == "bottom" and int(rows[-1]) != height - 1:
            return False
    return True


def _infer_rectangle_size(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    sides_a = _flat_sides(mask_a)
    sides_b = _flat_sides(mask_b)
    candidates: set[tuple[int, int]] = set()

    for side_a in sides_a:
        for side_b in sides_b:
            if {side_a, side_b} == {"left", "right"}:
                left = mask_a if side_a == "left" else mask_b
                right = mask_b if side_a == "left" else mask_a
                if left.shape[0] != right.shape[0]:
                    continue
                if not _rows_match_side(left, "left"):
                    continue
                if not _rows_match_side(right, "right"):
                    continue
                row_totals = left.sum(axis=1) + right.sum(axis=1)
                if bool(np.all(row_totals == row_totals[0])) and int(row_totals[0]) > 1:
                    candidates.add((int(left.shape[0]), int(row_totals[0])))
            if {side_a, side_b} == {"top", "bottom"}:
                top = mask_a if side_a == "top" else mask_b
                bottom = mask_b if side_a == "top" else mask_a
                if top.shape[1] != bottom.shape[1]:
                    continue
                if not _cols_match_side(top, "top"):
                    continue
                if not _cols_match_side(bottom, "bottom"):
                    continue
                col_totals = top.sum(axis=0) + bottom.sum(axis=0)
                if bool(np.all(col_totals == col_totals[0])) and int(col_totals[0]) > 1:
                    candidates.add((int(col_totals[0]), int(top.shape[1])))

    assert len(candidates) == 1, "each color must define one complementary rectangle size"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, recover each completed rectangle from the two
    complementary flat-sided pieces, and place the solid rectangle at the
    same-color anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = np.full_like(g, BG)

    for color in (int(x) for x in np.unique(g) if int(x) != BG):
        comps = _components_of_color(g, color)
        assert len(comps) == 3, "each active color must form exactly two fragments and one anchor"
        anchors = [comp for comp in comps if int(comp.shape[0]) == 1]
        pieces = [comp for comp in comps if int(comp.shape[0]) > 1]
        assert len(anchors) == 1 and len(pieces) == 2

        anchor_r, anchor_c = map(int, anchors[0][0])
        rect_h, rect_w = _infer_rectangle_size(_component_mask(pieces[0]), _component_mask(pieces[1]))
        assert 0 <= anchor_r <= h - rect_h and 0 <= anchor_c <= w - rect_w
        assert not bool((out[anchor_r : anchor_r + rect_h, anchor_c : anchor_c + rect_w] != BG).any())
        out[anchor_r : anchor_r + rect_h, anchor_c : anchor_c + rect_w] = color

    return out


def _build_horizontal_pair(rng: random.Random, color: int) -> dict | None:
    rect_h = rng.randint(RECT_H_RANGE[0], RECT_H_RANGE[1] - 1)
    rect_w = rng.randint(max(5, RECT_W_RANGE[0]), RECT_W_RANGE[1] - 1)
    for _ in range(100):
        widths = [rng.randint(1, rect_w - 1) for _ in range(rect_h)]
        lo, hi = min(widths), max(widths)
        if lo == hi:
            continue
        if widths[0] in {lo, hi} or widths[-1] in {lo, hi}:
            continue
        if lo not in widths[1:-1] or hi not in widths[1:-1]:
            continue

        left_w = int(max(widths))
        right_w = int(max(rect_w - val for val in widths))
        left = np.zeros((rect_h, left_w), dtype=bool)
        right = np.zeros((rect_h, right_w), dtype=bool)
        for r, width in enumerate(widths):
            left[r, :width] = True
            comp_w = rect_w - width
            right[r, right_w - comp_w :] = True

        if _flat_sides(left) != {"left"}:
            continue
        if _flat_sides(right) != {"right"}:
            continue
        if _infer_rectangle_size(left, right) != (rect_h, rect_w):
            continue
        return {
            "color": color,
            "rect_h": rect_h,
            "rect_w": rect_w,
            "fragments": [left, right],
        }
    return None


def _build_vertical_pair(rng: random.Random, color: int) -> dict | None:
    rect_h = rng.randint(max(5, RECT_H_RANGE[0]), RECT_H_RANGE[1] - 1)
    rect_w = rng.randint(RECT_W_RANGE[0], RECT_W_RANGE[1] - 1)
    for _ in range(100):
        heights = [rng.randint(1, rect_h - 1) for _ in range(rect_w)]
        lo, hi = min(heights), max(heights)
        if lo == hi:
            continue
        if heights[0] in {lo, hi} or heights[-1] in {lo, hi}:
            continue
        if lo not in heights[1:-1] or hi not in heights[1:-1]:
            continue

        top_h = int(max(heights))
        bottom_h = int(max(rect_h - val for val in heights))
        top = np.zeros((top_h, rect_w), dtype=bool)
        bottom = np.zeros((bottom_h, rect_w), dtype=bool)
        for c, height in enumerate(heights):
            top[:height, c] = True
            comp_h = rect_h - height
            bottom[bottom_h - comp_h :, c] = True

        if _flat_sides(top) != {"top"}:
            continue
        if _flat_sides(bottom) != {"bottom"}:
            continue
        if _infer_rectangle_size(top, bottom) != (rect_h, rect_w):
            continue
        return {
            "color": color,
            "rect_h": rect_h,
            "rect_w": rect_w,
            "fragments": [top, bottom],
        }
    return None


def _build_pair_spec(rng: random.Random, color: int) -> dict | None:
    if rng.random() < 0.5:
        return _build_horizontal_pair(rng, color)
    return _build_vertical_pair(rng, color)


def _free_with_padding(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = PADDING) -> bool:
    hh, ww = occ.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(hh, r + h + pad)
    c1 = min(ww, c + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _place_anchor(rng: random.Random, occ_out: np.ndarray, spec: dict) -> tuple[int, int] | None:
    h, w = occ_out.shape
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    strips = ["top", "left", "bottom", "right"]
    rng.shuffle(strips)

    for strip in strips:
        if strip == "top":
            r_lo, r_hi = 0, min(ANCHOR_STRIP, h - rect_h)
            c_lo, c_hi = 0, w - rect_w
        elif strip == "left":
            r_lo, r_hi = 0, h - rect_h
            c_lo, c_hi = 0, min(ANCHOR_STRIP, w - rect_w)
        elif strip == "bottom":
            r_lo, r_hi = max(0, h - rect_h - ANCHOR_STRIP), h - rect_h
            c_lo, c_hi = 0, w - rect_w
        else:
            r_lo, r_hi = 0, h - rect_h
            c_lo, c_hi = max(0, w - rect_w - ANCHOR_STRIP), w - rect_w
        if r_lo > r_hi or c_lo > c_hi:
            continue

        for _ in range(PLACEMENT_TRIES):
            r = rng.randint(r_lo, r_hi)
            c = rng.randint(c_lo, c_hi)
            if _free_with_padding(occ_out, r, c, rect_h, rect_w):
                occ_out[r : r + rect_h, c : c + rect_w] = True
                return r, c
    return None


def _place_fragment(
    rng: random.Random,
    grid: np.ndarray,
    occ_in: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> bool:
    h, w = mask.shape
    grid_h, grid_w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        r = rng.randint(0, grid_h - h)
        c = rng.randint(0, grid_w - w)
        if not _free_with_padding(occ_in, r, c, h, w):
            continue
        _paint_mask(grid, mask, color, r, c)
        occ_in[r : r + h, c : c + w] = True
        return True
    return False


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        r, c = spec["anchor"]
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        out[r : r + rect_h, c : c + rect_w] = int(spec["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] - 1)
    grid_w = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] - 1)
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    colors = rng.sample(PALETTE, pair_count)

    specs: list[dict] = []
    for color in colors:
        spec = _build_pair_spec(rng, color)
        if spec is None:
            return None
        specs.append(spec)

    occ_out = np.zeros((grid_h, grid_w), dtype=bool)
    specs.sort(key=lambda spec: int(spec["rect_h"]) * int(spec["rect_w"]), reverse=True)
    for spec in specs:
        anchor = _place_anchor(rng, occ_out, spec)
        if anchor is None:
            return None
        spec["anchor"] = anchor

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ_in = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in specs:
        ar, ac = spec["anchor"]
        grid[ar, ac] = int(spec["color"])
        occ_in[ar, ac] = True

    fragments: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        for mask in spec["fragments"]:
            fragments.append((int(spec["color"]), mask))
    rng.shuffle(fragments)
    fragments.sort(key=lambda item: int(item[1].shape[0]) * int(item[1].shape[1]), reverse=True)

    for color, mask in fragments:
        if not _place_fragment(rng, grid, occ_in, mask, color):
            return None

    expected = _expected_output((grid_h, grid_w), specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flat-edge fragment puzzle")
