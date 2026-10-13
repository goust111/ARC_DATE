from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT = 2
GEN_ATTEMPTS = 300
MASK_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 200

EQUAL_DIMS = [(h, w) for h in range(5, 9) for w in (6, 8)]
UNEQUAL_DIMS = [(h, w) for h in range(5, 9) for w in range(5, 9)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    while stack:
        r, c = stack.pop()
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                pt = (nr, nc)
                if pt not in seen:
                    seen.add(pt)
                    stack.append(pt)
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    h, w = grid.shape
    border = np.ones((3, 4), dtype=bool)
    border[1, 1:3] = False
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 3):
            block = grid[r0 : r0 + 3, c0 : c0 + 4]
            if not np.all(block[border] == GREY):
                continue
            left = int(block[1, 1])
            right = int(block[1, 2])
            if left in (BG, GREY) or right in (BG, GREY) or left == right:
                continue
            candidates.append((r0, c0))
    assert candidates, "legend not found"
    r0, c0 = sorted(candidates, key=lambda rc: (rc[0], -rc[1]))[0]
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + 3, c0 : c0 + 4] = True
    return (r0, c0), (int(grid[r0 + 1, c0 + 1]), int(grid[r0 + 1, c0 + 2])), legend_mask


def _extract_fragments(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, GREY):
            continue
        mask = (grid == color) & (~legend_mask)
        if not np.any(mask):
            continue
        r0, c0, r1, c1 = _bbox(mask)
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        fragments.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": local.copy(),
                "area": int(local.sum()),
            }
        )
    return fragments


def _match_pairs(fragments: list[dict[str, object]]) -> list[tuple[int, int]]:
    by_shape: dict[tuple[int, int], list[int]] = {}
    for i, frag in enumerate(fragments):
        by_shape.setdefault(tuple(frag["mask"].shape), []).append(i)

    pairs: list[tuple[int, int]] = []
    for idxs in by_shape.values():
        assert len(idxs) % 2 == 0, "shape group must contain an even number of fragments"
        compat: dict[tuple[int, int], bool] = {}
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a = fragments[idxs[i]]["mask"]
                b = fragments[idxs[j]]["mask"]
                compat[(i, j)] = bool(np.array_equal(a, ~b))

        def backtrack(rem: list[int]) -> list[tuple[int, int]] | None:
            if not rem:
                return []
            first = rem[0]
            for other in rem[1:]:
                key = (min(first, other), max(first, other))
                if not compat.get(key, False):
                    continue
                next_rem = [v for v in rem[1:] if v != other]
                tail = backtrack(next_rem)
                if tail is not None:
                    return [(idxs[first], idxs[other])] + tail
            return None

        matched = backtrack(list(range(len(idxs))))
        assert matched is not None, "complementary pairing failed"
        pairs.extend(matched)

    assert len(pairs) * 2 == len(fragments), "every fragment must be paired"
    return pairs


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    (_, _), (legend_left, legend_right), legend_mask = _find_legend(g)
    fragments = _extract_fragments(g, legend_mask)
    pairs = _match_pairs(fragments)

    out = g.copy()
    for frag in fragments:
        out[g == int(frag["color"])] = BG

    for ia, ib in pairs:
        fa = fragments[ia]
        fb = fragments[ib]
        ba = fa["bbox"]
        bb = fb["bbox"]
        if (ba[0], ba[1]) <= (bb[0], bb[1]):
            anchor, moved = fa, fb
        else:
            anchor, moved = fb, fa

        ar0, ac0, ar1, ac1 = anchor["bbox"]
        anchor_mask = anchor["mask"]
        moved_mask = moved["mask"]
        area_a = int(anchor["area"])
        area_b = int(moved["area"])

        if area_a == area_b:
            color_a = int(anchor["color"])
            color_b = int(moved["color"])
        elif area_a > area_b:
            color_a = legend_left
            color_b = legend_right
        else:
            color_a = legend_right
            color_b = legend_left

        patch = out[ar0 : ar1 + 1, ac0 : ac1 + 1]
        patch[anchor_mask] = color_a
        patch[moved_mask] = color_b

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments into full rectangles and recolor the merged pieces by area order.

    Input structure:
    - The background is black.
    - Several colored fragments are scattered on the grid; each fragment is a single-color piece and different fragments use different colors.
    - Fragments come in complementary pairs: inside equal-sized bounding boxes, one fragment occupies exactly the cells the other fragment leaves empty.
    - A small 1x2 legend sits near the top-right corner inside a grey border; its left color means "larger fragment" and its right color means "smaller fragment".

    Rule:
    - Find fragment pairs by matching equal bounding-box sizes and complementary occupancy masks.
    - For each pair, keep the rectangle at the piece whose bounding-box top row is smaller, and translate the other piece so the two bounding boxes coincide there.
    - Recolor the merged rectangle: the larger-area fragment takes the legend's left color and the smaller-area fragment takes the legend's right color.
    - If the two fragment areas are equal, keep their original colors instead of using the legend colors; erase the moved-away original position to black and keep the legend unchanged.

    Key insight:
    - The hard part is not matching identical shapes but noticing that the two masks are exact complements inside the same window, then using the separate legend to recolor by area ranking rather than by source color.
    """
    return _solve(grid)


def _can_place(rect: tuple[int, int, int, int], others: Iterable[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, h, w = rect
    for or0, oc0, oh, ow in others:
        separated = (
            r0 + h + pad <= or0
            or or0 + oh + pad <= r0
            or c0 + w + pad <= oc0
            or oc0 + ow + pad <= c0
        )
        if not separated:
            return False
    return True


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _random_band_mask(h: int, w: int, equal_area: bool) -> np.ndarray | None:
    if equal_area and (w % 2 == 1):
        return None
    for _ in range(MASK_ATTEMPTS):
        if equal_area:
            width = w // 2
        else:
            widths = [cand for cand in range(2, w - 1) if cand * 2 != w]
            if not widths:
                return None
            width = random.choice(widths)
        max_start = w - width
        starts = [random.randint(0, max_start)]
        for _row in range(1, h):
            prev = starts[-1]
            starts.append(max(0, min(max_start, prev + random.choice((-1, 0, 1)))))
        starts[random.randrange(h)] = 0
        starts[random.randrange(h)] = max_start

        mask = np.zeros((h, w), dtype=bool)
        for r, start in enumerate(starts):
            mask[r, start : start + width] = True

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        if not _touches_all_sides(mask) or not _touches_all_sides(~mask):
            continue
        if not _connected(mask, connectivity=8) or not _connected(~mask, connectivity=8):
            continue
        if equal_area != (int(mask.sum()) * 2 == h * w):
            continue
        return mask
    return None


def _choose_masks() -> list[tuple[np.ndarray, bool]]:
    equal_dims = random.choice(EQUAL_DIMS)
    unequal_choices = [dims for dims in UNEQUAL_DIMS if dims != equal_dims]
    unequal_dims = random.choice(unequal_choices)

    equal_mask = _random_band_mask(equal_dims[0], equal_dims[1], equal_area=True)
    unequal_mask = _random_band_mask(unequal_dims[0], unequal_dims[1], equal_area=False)
    if equal_mask is None or unequal_mask is None:
        raise RuntimeError("failed to construct complementary masks")

    pairs = [(equal_mask, True), (unequal_mask, False)]
    random.shuffle(pairs)
    return pairs


def _place_pairs(shapes: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    heights = [int(mask.shape[0]) for mask in shapes]
    widths = [int(mask.shape[1]) for mask in shapes]
    min_h = max(18, 2 * max(heights) + 5)
    min_w = max(18, sum(widths) + 3 * len(widths) + 4)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        legend_rect = (0, w - 4, 3, 4)
        reserved = [legend_rect]

        order = sorted(range(len(shapes)), key=lambda i: shapes[i].shape[0] * shapes[i].shape[1], reverse=True)
        anchors: list[tuple[int, int] | None] = [None] * len(shapes)
        movers: list[tuple[int, int] | None] = [None] * len(shapes)

        ok = True
        for i in order:
            sh, sw = shapes[i].shape
            row_max = h // 2 - sh
            if row_max < 2:
                ok = False
                break
            candidates = [
                (r0, c0)
                for r0 in range(2, row_max + 1)
                for c0 in range(0, w - sw + 1)
                if _can_place((r0, c0, sh, sw), reserved, pad=1)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            anchors[i] = pos
            reserved.append((pos[0], pos[1], sh, sw))
        if not ok:
            continue

        for i in order:
            sh, sw = shapes[i].shape
            anchor_r, _ = anchors[i]
            candidates = [
                (r0, c0)
                for r0 in range(anchor_r + 2, h - sh + 1)
                for c0 in range(0, w - sw + 1)
                if _can_place((r0, c0, sh, sw), reserved, pad=1)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            movers[i] = pos
            reserved.append((pos[0], pos[1], sh, sw))
        if ok:
            return h, w, [pos for pos in anchors if pos is not None], [pos for pos in movers if pos is not None]
    return None


def _render_grid(
    h: int,
    w: int,
    pair_specs: list[tuple[np.ndarray, bool]],
    anchors: list[tuple[int, int]],
    movers: list[tuple[int, int]],
    legend_colors: tuple[int, int],
    piece_colors: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    legend_left, legend_right = legend_colors
    grid[0:3, w - 4 : w] = GREY
    grid[1, w - 3] = legend_left
    grid[1, w - 2] = legend_right

    for i, ((mask, _equal_area), anchor, mover) in enumerate(zip(pair_specs, anchors, movers)):
        color_a, color_b = piece_colors[i]
        mover_mask = ~mask
        anchor_mask = mask if random.random() < 0.5 else ~mask
        mover_mask = ~anchor_mask
        _draw_mask(grid, anchor[0], anchor[1], anchor_mask, color_a)
        _draw_mask(grid, mover[0], mover[1], mover_mask, color_b)
    return grid


def _gen_attempt() -> np.ndarray | None:
    try:
        pair_specs = _choose_masks()
    except RuntimeError:
        return None
    shapes = [mask for mask, _ in pair_specs]
    layout = _place_pairs(shapes)
    if layout is None:
        return None
    h, w, anchors, movers = layout

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    legend_colors = (colors[0], colors[1])
    piece_colors = [(colors[2], colors[3]), (colors[4], colors[5])]

    grid = _render_grid(h, w, pair_specs, anchors, movers, legend_colors, piece_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
