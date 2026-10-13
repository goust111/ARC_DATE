from __future__ import annotations

from itertools import combinations
import math
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
AZURE = 8
MAROON = 9
FG_POOL = (1, 3, 4, 6, 7)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNER_COMBOS = {
    n: tuple(combinations(((0, 0), (0, -1), (-1, 0), (-1, -1)), n))
    for n in range(1, 5)
}


def _find_divider(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = []
    for c in range(w):
        if np.all(g[:, c] == MAROON) and c + 1 < w and np.all(g[:, c + 1 :] == AZURE):
            candidates.append(c)
    assert len(candidates) == 1, "expected one maroon divider with blank right canvas"
    return int(candidates[0])


def _is_fragment_bbox(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, divider: int) -> bool:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    border = {GREY, RED}
    if c1 >= divider:
        return False

    if not np.isin(g[r0, c0 : c1 + 1], list(border)).all():
        return False
    if not np.isin(g[r1, c0 : c1 + 1], list(border)).all():
        return False
    if not np.isin(g[r0 : r1 + 1, c0], list(border)).all():
        return False
    if not np.isin(g[r0 : r1 + 1, c1], list(border)).all():
        return False

    if c1 - c0 > 1:
        if not np.all(g[r0, c0 + 1 : c1] == GREY):
            return False
        if not np.all(g[r1, c0 + 1 : c1] == GREY):
            return False
    if r1 - r0 > 1:
        if not np.all(g[r0 + 1 : r1, c0] == GREY):
            return False
        if not np.all(g[r0 + 1 : r1, c1] == GREY):
            return False

    if r0 > 0 and not np.all(g[r0 - 1, c0 : c1 + 1] == AZURE):
        return False
    if r1 + 1 < g.shape[0] and not np.all(g[r1 + 1, c0 : c1 + 1] == AZURE):
        return False
    if c0 > 0 and not np.all(g[r0 : r1 + 1, c0 - 1] == AZURE):
        return False
    if c1 + 1 < divider and not np.all(g[r0 : r1 + 1, c1 + 1] == AZURE):
        return False

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.isin(inner, [GREY, RED, AZURE, MAROON]).any():
        return False
    colors = set(int(v) for v in np.unique(inner))
    colors.discard(BLACK)
    return len(colors) == 1


def _find_fragments(g: np.ndarray, divider: int) -> list[dict[str, object]]:
    h, _w = g.shape
    fragments: list[dict[str, object]] = []
    for r0 in range(h):
        for c0 in range(divider):
            if int(g[r0, c0]) not in (GREY, RED):
                continue
            if r0 > 0 and int(g[r0 - 1, c0]) != AZURE:
                continue
            if c0 > 0 and int(g[r0, c0 - 1]) != AZURE:
                continue

            c1 = c0
            while c1 < divider and int(g[r0, c1]) in (GREY, RED):
                c1 += 1
            r1 = r0
            while r1 < h and int(g[r1, c0]) in (GREY, RED):
                r1 += 1

            rr1 = r1 - 1
            cc1 = c1 - 1
            if not _is_fragment_bbox(g, r0, c0, rr1, cc1, divider):
                continue

            corners = (
                int(g[r0, c0]),
                int(g[r0, cc1]),
                int(g[rr1, c0]),
                int(g[rr1, cc1]),
            )
            layer = sum(v == RED for v in corners)
            assert 1 <= layer <= 4, "red corner count must encode the layer"
            fragments.append(
                {
                    "layer": int(layer),
                    "pattern": g[r0 + 1 : rr1, c0 + 1 : cc1].copy(),
                }
            )
    assert fragments, "expected at least one fragment"
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read layer order from red border corners, then stack the fragment interiors on
    the blank right canvas and clear the source side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    divider = _find_divider(g)
    fragments = _find_fragments(g, divider)

    inner_shape = tuple(int(v) for v in np.asarray(fragments[0]["pattern"]).shape)
    assert inner_shape[0] > 0 and inner_shape[1] > 0
    for frag in fragments[1:]:
        assert np.asarray(frag["pattern"]).shape == inner_shape, "all fragments must share one template size"

    composite = np.full(inner_shape, BLACK, dtype=int)
    for frag in sorted(fragments, key=lambda item: int(item["layer"])):
        patch = np.asarray(frag["pattern"], dtype=int)
        mask = patch != BLACK
        composite[mask] = patch[mask]

    right_w = w - divider - 1
    assert right_w >= inner_shape[1], "right canvas must fit the restored template"
    top = (h - inner_shape[0]) // 2
    left = divider + 1 + (right_w - inner_shape[1]) // 2

    out = np.full_like(g, AZURE)
    out[top : top + inner_shape[0], left : left + inner_shape[1]] = composite
    return out


def _grow_blob(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_rect(h: int, w: int) -> np.ndarray:
    r0 = random.randint(0, h - 2)
    r1 = random.randint(r0 + 1, h - 1)
    c0 = random.randint(0, w - 2)
    c1 = random.randint(c0 + 1, w - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _shape_plus(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    cr = random.randrange(h)
    cc = random.randrange(w)
    vr = random.randint(0, cr)
    vb = random.randint(0, h - 1 - cr)
    hl = random.randint(0, cc)
    hr = random.randint(0, w - 1 - cc)
    vth = random.randint(1, min(2, w))
    hth = random.randint(1, min(2, h))
    c_start = max(0, cc - vth // 2)
    c_end = min(w, c_start + vth)
    r_start = max(0, cr - hth // 2)
    r_end = min(h, r_start + hth)
    mask[cr - vr : cr + vb + 1, c_start:c_end] = True
    mask[r_start:r_end, cc - hl : cc + hr + 1] = True
    return mask


def _shape_l(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    corner = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
    vlen = random.randint(2, h)
    hlen = random.randint(2, w)
    thick = random.randint(1, min(2, h, w))
    if corner == (0, 0):
        mask[:vlen, :thick] = True
        mask[:thick, :hlen] = True
    elif corner == (0, 1):
        mask[:vlen, w - thick :] = True
        mask[:thick, w - hlen :] = True
    elif corner == (1, 0):
        mask[h - vlen :, :thick] = True
        mask[h - thick :, :hlen] = True
    else:
        mask[h - vlen :, w - thick :] = True
        mask[h - thick :, w - hlen :] = True
    return mask


def _shape_stair(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    downward = random.choice((True, False))
    start_col = 0 if random.choice((True, False)) else w - 1
    col = start_col
    row = 0 if downward else h - 1
    target_row = h - 1 if downward else 0
    target_col = w - 1 if start_col == 0 else 0
    while True:
        mask[row, col] = True
        if row == target_row and col == target_col:
            break
        move_row = row != target_row and (col == target_col or random.choice((True, False)))
        if move_row:
            row += 1 if downward else -1
        else:
            col += 1 if start_col == 0 else -1
        mask[row, col] = True
    if random.choice((True, False)):
        expanded = mask.copy()
        expanded[1:, :] |= mask[:-1, :]
        mask = expanded
    else:
        expanded = mask.copy()
        expanded[:, 1:] |= mask[:, :-1]
        mask = expanded
    return mask


def _random_mask(h: int, w: int) -> np.ndarray:
    makers = (_shape_rect, _shape_plus, _shape_l, _shape_stair)
    for _ in range(40):
        mode = random.randrange(5)
        if mode == 4:
            area = random.randint(max(4, (h * w) // 4), max(4, (2 * h * w) // 3))
            mask = _grow_blob(h, w, area)
        else:
            mask = makers[mode](h, w)
        if random.choice((True, False)):
            mask = np.fliplr(mask)
        if random.choice((True, False)):
            mask = np.flipud(mask)
        area = int(mask.sum())
        if 3 <= area < h * w:
            return mask
    raise RuntimeError("failed to sample a non-degenerate fragment shape")


def _visible_ok(masks: list[np.ndarray]) -> bool:
    covered = np.zeros_like(masks[0], dtype=bool)
    for mask in reversed(masks):
        if not bool((mask & ~covered).any()):
            return False
        covered |= mask
    return True


def _compose_expected(
    masks: list[np.ndarray],
    colors: list[int],
    h: int,
    left_w: int,
    right_w: int,
) -> np.ndarray:
    inner_h, inner_w = masks[0].shape
    composite = np.full((inner_h, inner_w), BLACK, dtype=int)
    for mask, color in zip(masks, colors):
        composite[mask] = int(color)
    out = np.full((h, left_w + 1 + right_w), AZURE, dtype=int)
    top = (h - inner_h) // 2
    left = left_w + 1 + (right_w - inner_w) // 2
    out[top : top + inner_h, left : left + inner_w] = composite
    return out


def _render_fragment(mask: np.ndarray, color: int, layer: int) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1] = np.where(mask, int(color), BLACK)
    for rr, cc in random.choice(CORNER_COMBOS[int(layer)]):
        patch[rr, cc] = RED
    return patch


def _layout_options(layer_count: int, outer_h: int, outer_w: int, inner_w: int) -> list[tuple[int, int, int, int]]:
    options = []
    for cols in range(1, layer_count + 1):
        rows = math.ceil(layer_count / cols)
        for slot_h in range(outer_h + 1, outer_h + 5):
            for slot_w in range(outer_w + 1, outer_w + 5):
                min_h = rows * slot_h + 2
                min_left = cols * slot_w + 2
                if min_h <= 28 and min_left + 1 + inner_w + 4 <= 34:
                    options.append((rows, cols, slot_h, slot_w))
    return options


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    layer_count = random.randint(1, 4)
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    masks = [_random_mask(inner_h, inner_w) for _ in range(layer_count)]
    if not _visible_ok(masks):
        raise RuntimeError("sampled invisible lower layer")

    colors = random.sample(FG_POOL, layer_count)
    fragments = [
        _render_fragment(mask, color, layer)
        for layer, (mask, color) in enumerate(zip(masks, colors), start=1)
    ]
    outer_h, outer_w = fragments[0].shape

    options = _layout_options(layer_count, outer_h, outer_w, inner_w)
    if not options:
        raise RuntimeError("no valid layout for sampled fragment size")
    rows, cols, slot_h, slot_w = random.choice(options)

    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    side_margin = random.randint(1, 2)
    h = top_margin + rows * slot_h + bottom_margin
    left_w = left_margin + cols * slot_w + side_margin
    right_w = inner_w + random.randint(4, 8)

    grid = np.full((h, left_w + 1 + right_w), AZURE, dtype=int)
    grid[:, left_w] = MAROON

    slot_ids = random.sample(range(rows * cols), layer_count)
    for patch, slot_id in zip(fragments, slot_ids):
        sr = slot_id // cols
        sc = slot_id % cols
        base_r = top_margin + sr * slot_h
        base_c = left_margin + sc * slot_w
        slack_r = slot_h - outer_h
        slack_c = slot_w - outer_w
        r0 = base_r + random.randint(0, slack_r)
        c0 = base_c + random.randint(0, slack_c)
        grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = patch

    expected = _compose_expected(masks, colors, h, left_w, right_w)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
