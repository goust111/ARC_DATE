from __future__ import annotations

import random
from itertools import product

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_TRIES = 120
EDGE_MARGIN = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[int(r) - top, int(c) - left] = True
    return mask


def _search_horizontal_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    h_a, w_a = mask_a.shape
    h_b, w_b = mask_b.shape
    assert h_a == h_b
    candidates: list[tuple[int, int]] = []
    for offset in range(w_a + w_b + 1):
        width = max(w_a, offset + w_b)
        canvas = np.zeros((h_a, width), dtype=bool)
        canvas[:, :w_a] |= mask_a
        overlap = canvas[:, offset : offset + w_b] & mask_b
        if bool(np.any(overlap)):
            continue
        canvas[:, offset : offset + w_b] |= mask_b
        if int(canvas.sum()) == h_a * width:
            candidates.append((int(offset), int(width)))
    return candidates


def _search_vertical_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    h_a, w_a = mask_a.shape
    h_b, w_b = mask_b.shape
    assert w_a == w_b
    candidates: list[tuple[int, int]] = []
    for offset in range(h_a + h_b + 1):
        height = max(h_a, offset + h_b)
        canvas = np.zeros((height, w_a), dtype=bool)
        canvas[:h_a, :] |= mask_a
        overlap = canvas[offset : offset + h_b, :] & mask_b
        if bool(np.any(overlap)):
            continue
        canvas[offset : offset + h_b, :] |= mask_b
        if int(canvas.sum()) == height * w_a:
            candidates.append((int(offset), int(height)))
    return candidates


def _horizontal_templates() -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    seen: set[tuple[bytes, bytes, int]] = set()
    for height in range(2, 7):
        for width in range(3, 7):
            if height * width > 12:
                continue
            for splits in product(range(1, width), repeat=height):
                if len(set(splits)) == 1:
                    continue
                left_area = sum(splits)
                right_area = height * width - left_area
                if not (2 <= left_area <= 6 and 2 <= right_area <= 6):
                    continue

                left_width = max(splits)
                right_shift = min(splits)
                right_width = width - right_shift

                left = np.zeros((height, left_width), dtype=bool)
                right = np.zeros((height, right_width), dtype=bool)
                for r, seam in enumerate(splits):
                    left[r, :seam] = True
                    right[r, seam - right_shift :] = True

                candidates = _search_horizontal_candidates(left, right)
                if candidates != [(right_shift, width)]:
                    continue

                signature = (left.tobytes(), right.tobytes(), int(right_shift))
                if signature in seen:
                    continue
                seen.add(signature)
                templates.append(
                    {
                        "orientation": "h",
                        "a": left,
                        "b": right,
                        "offset": int(right_shift),
                        "final_h": int(height),
                        "final_w": int(width),
                    }
                )
    assert templates
    return templates


def _vertical_templates() -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    seen: set[tuple[bytes, bytes, int]] = set()
    for height in range(3, 7):
        for width in range(2, 7):
            if height * width > 12:
                continue
            for splits in product(range(1, height), repeat=width):
                if len(set(splits)) == 1:
                    continue
                top_area = sum(splits)
                bottom_area = height * width - top_area
                if not (2 <= top_area <= 6 and 2 <= bottom_area <= 6):
                    continue

                top_height = max(splits)
                bottom_shift = min(splits)
                bottom_height = height - bottom_shift

                top = np.zeros((top_height, width), dtype=bool)
                bottom = np.zeros((bottom_height, width), dtype=bool)
                for c, seam in enumerate(splits):
                    top[:seam, c] = True
                    bottom[seam - bottom_shift :, c] = True

                candidates = _search_vertical_candidates(top, bottom)
                if candidates != [(bottom_shift, height)]:
                    continue

                signature = (top.tobytes(), bottom.tobytes(), int(bottom_shift))
                if signature in seen:
                    continue
                seen.add(signature)
                templates.append(
                    {
                        "orientation": "v",
                        "a": top,
                        "b": bottom,
                        "offset": int(bottom_shift),
                        "final_h": int(height),
                        "final_w": int(width),
                    }
                )
    assert templates
    return templates


H_TEMPLATES = _horizontal_templates()
V_TEMPLATES = _vertical_templates()


def _solve_horizontal(left_coords: np.ndarray, right_coords: np.ndarray) -> tuple[int, int, int, int]:
    lr0, lc0, lr1, lc1 = _bbox(left_coords)
    rr0, rc0, rr1, rc1 = _bbox(right_coords)
    assert (lr0, lr1) == (rr0, rr1), "horizontal fragments must share a row span"
    assert lc0 < rc0, "expected left fragment before right fragment"

    height = lr1 - lr0 + 1
    left_mask = _coords_to_mask(left_coords, lr0, lc0, height, lc1 - lc0 + 1)
    right_mask = _coords_to_mask(right_coords, rr0, rc0, height, rc1 - rc0 + 1)

    candidates = _search_horizontal_candidates(left_mask, right_mask)
    assert len(candidates) == 1, "expected one horizontal rectangle completion"
    offset, width = candidates[0]

    base_num = lc0 + rc0 - offset
    assert base_num % 2 == 0, "horizontal midpoint must land on an integer column"
    final_left = base_num // 2
    return int(lr0), int(final_left), int(height), int(width)


def _solve_vertical(top_coords: np.ndarray, bottom_coords: np.ndarray) -> tuple[int, int, int, int]:
    tr0, tc0, tr1, tc1 = _bbox(top_coords)
    br0, bc0, br1, bc1 = _bbox(bottom_coords)
    assert (tc0, tc1) == (bc0, bc1), "vertical fragments must share a column span"
    assert tr0 < br0, "expected top fragment before bottom fragment"

    width = tc1 - tc0 + 1
    top_mask = _coords_to_mask(top_coords, tr0, tc0, tr1 - tr0 + 1, width)
    bottom_mask = _coords_to_mask(bottom_coords, br0, bc0, br1 - br0 + 1, width)

    candidates = _search_vertical_candidates(top_mask, bottom_mask)
    assert len(candidates) == 1, "expected one vertical rectangle completion"
    offset, height = candidates[0]

    base_num = tr0 + br0 - offset
    assert base_num % 2 == 0, "vertical midpoint must land on an integer row"
    final_top = base_num // 2
    return int(final_top), int(tc0), int(height), int(width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color fragment pair together until the pair becomes one solid rectangle.

    Input structure:
    - The background is black.
    - Every non-black color appears in exactly two 4-connected fragments.
    - Each same-color pair is aligned either horizontally with a shared row span or vertically
      with a shared column span.
    - Within each pair, the two fragment shapes are complementary: after sliding toward each
      other along their alignment axis, they fit together without overlap as one filled rectangle.

    Rule:
    - Group fragments by color and find the two fragments of each color.
    - Determine whether the pair is horizontal or vertical from the shared bbox span.
    - Along that axis, find the unique relative offset where the translated shapes tile a solid
      axis-aligned rectangle with no gaps.
    - Remove the original fragments and draw that completed rectangle at the midpoint position
      implied by the two original fragment placements.

    Key insight:
    - The pieces are not moved by a fixed distance. Their irregular outlines encode a unique
      jigsaw-style completion offset, and the original symmetric separation determines where the
      finished rectangle must land.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BG)
    for color in (int(x) for x in np.unique(g) if int(x) != BG):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly two fragments"
        first, second = comps
        b1 = _bbox(first)
        b2 = _bbox(second)

        if (b1[0], b1[2]) == (b2[0], b2[2]):
            left, right = (first, second) if b1[1] < b2[1] else (second, first)
            r0, c0, height, width = _solve_horizontal(left, right)
        else:
            assert (b1[1], b1[3]) == (b2[1], b2[3]), "fragments must align by row span or column span"
            top, bottom = (first, second) if b1[0] < b2[0] else (second, first)
            r0, c0, height, width = _solve_vertical(top, bottom)

        assert 0 <= r0 and 0 <= c0 and r0 + height <= g.shape[0] and c0 + width <= g.shape[1]
        assert np.all(out[r0 : r0 + height, c0 : c0 + width] == BG), "completed rectangles must not overlap"
        out[r0 : r0 + height, c0 : c0 + width] = color

    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    if bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])):
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _sample_specs(pair_count: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for _ in range(pair_count):
        orientation = random.choice(("h", "v"))
        template = random.choice(H_TEMPLATES if orientation == "h" else V_TEMPLATES).copy()
        specs.append(template)

    def footprint_key(spec: dict[str, object]) -> int:
        mask_a = np.asarray(spec["a"], dtype=bool)
        offset = int(spec["offset"])
        if spec["orientation"] == "h":
            d_min = max(1, (mask_a.shape[1] - offset + 2) // 2)
            return int(spec["final_h"]) * (int(spec["final_w"]) + 2 * d_min)
        d_min = max(1, (mask_a.shape[0] - offset + 2) // 2)
        return (int(spec["final_h"]) + 2 * d_min) * int(spec["final_w"])

    specs.sort(key=footprint_key, reverse=True)
    return specs


def _place_horizontal(
    grid: np.ndarray,
    occupied: np.ndarray,
    expected: np.ndarray,
    color: int,
    spec: dict[str, object],
) -> bool:
    left_mask = np.asarray(spec["a"], dtype=bool)
    right_mask = np.asarray(spec["b"], dtype=bool)
    offset = int(spec["offset"])
    final_h = int(spec["final_h"])
    final_w = int(spec["final_w"])
    h, w_left = left_mask.shape
    _h2, w_right = right_mask.shape
    assert h == final_h

    d_min = max(1, (w_left - offset + 2) // 2)
    d_max = (grid.shape[1] - 2 * EDGE_MARGIN - offset - w_right) // 2
    if d_max < d_min or grid.shape[0] < final_h + 2 * EDGE_MARGIN:
        return False

    for _ in range(PLACEMENT_TRIES):
        d = random.randint(d_min, d_max)
        row_top = random.randint(EDGE_MARGIN, grid.shape[0] - EDGE_MARGIN - final_h)
        left_lo = EDGE_MARGIN + d
        left_hi = grid.shape[1] - EDGE_MARGIN - (offset + d + w_right)
        if left_lo > left_hi:
            continue
        col_left = random.randint(left_lo, left_hi)

        corridor_r0 = row_top
        corridor_c0 = col_left - d
        corridor_r1 = row_top + final_h - 1
        corridor_c1 = col_left + offset + d + w_right - 1
        if not _reserve(occupied, corridor_r0, corridor_c0, corridor_r1, corridor_c1):
            continue

        _paint_mask(grid, left_mask, row_top, col_left - d, color)
        _paint_mask(grid, right_mask, row_top, col_left + offset + d, color)
        expected[row_top : row_top + final_h, col_left : col_left + final_w] = color
        return True
    return False


def _place_vertical(
    grid: np.ndarray,
    occupied: np.ndarray,
    expected: np.ndarray,
    color: int,
    spec: dict[str, object],
) -> bool:
    top_mask = np.asarray(spec["a"], dtype=bool)
    bottom_mask = np.asarray(spec["b"], dtype=bool)
    offset = int(spec["offset"])
    final_h = int(spec["final_h"])
    final_w = int(spec["final_w"])
    h_top, w = top_mask.shape
    h_bottom, _w2 = bottom_mask.shape
    assert w == final_w

    d_min = max(1, (h_top - offset + 2) // 2)
    d_max = (grid.shape[0] - 2 * EDGE_MARGIN - offset - h_bottom) // 2
    if d_max < d_min or grid.shape[1] < final_w + 2 * EDGE_MARGIN:
        return False

    for _ in range(PLACEMENT_TRIES):
        d = random.randint(d_min, d_max)
        col_left = random.randint(EDGE_MARGIN, grid.shape[1] - EDGE_MARGIN - final_w)
        top_lo = EDGE_MARGIN + d
        top_hi = grid.shape[0] - EDGE_MARGIN - (offset + d + h_bottom)
        if top_lo > top_hi:
            continue
        row_top = random.randint(top_lo, top_hi)

        corridor_r0 = row_top - d
        corridor_c0 = col_left
        corridor_r1 = row_top + offset + d + h_bottom - 1
        corridor_c1 = col_left + final_w - 1
        if not _reserve(occupied, corridor_r0, corridor_c0, corridor_r1, corridor_c1):
            continue

        _paint_mask(grid, top_mask, row_top - d, col_left, color)
        _paint_mask(grid, bottom_mask, row_top + offset + d, col_left, color)
        expected[row_top : row_top + final_h, col_left : col_left + final_w] = color
        return True
    return False


def _attempt_generate() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    min_side = min(30, 12 + 3 * pair_count)
    if min_side > 30:
        return None
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    colors = random.sample(PALETTE, pair_count)
    specs = _sample_specs(pair_count)

    for color, spec in zip(colors, specs):
        ok = (
            _place_horizontal(grid, occupied, expected, color, spec)
            if spec["orientation"] == "h"
            else _place_vertical(grid, occupied, expected, color, spec)
        )
        if not ok:
            return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
