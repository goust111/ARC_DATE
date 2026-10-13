from __future__ import annotations

import numpy as np


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((int(rr), int(cc)))
            for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(np.array(pts, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[int], np.ndarray]:
    h, w = grid.shape
    best = None
    for r in range(h):
        nz = np.flatnonzero(grid[r] != bg)
        if nz.size < 2:
            continue
        if not bool(np.all(np.diff(nz) == 1)):
            continue
        colors = [int(v) for v in grid[r, nz]]
        if len(colors) != len(set(colors)):
            continue
        score = (-int(r), int(nz.size))
        cand = (score, int(r), int(nz[0]), int(nz[-1]) + 1, colors)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "legend strip not found"
    _, row, c0, c1, colors = best
    mask = np.zeros((h, w), dtype=bool)
    mask[row, c0:c1] = True
    return row, c0, c1, colors, mask


def _is_hollow_frame(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(sub, border))


def _find_canvas_frame(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> tuple[int, int, int, int, int, np.ndarray]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = (grid == color) & (~legend_mask)
        for comp in _components(mask):
            r0, c0, r1, c1 = _bbox(comp)
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[comp[:, 0], comp[:, 1]] = True
            if not _is_hollow_frame(comp_mask):
                continue
            if not bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg)):
                continue
            center_bias = -abs((c0 + c1) - (w - 1))
            score = (int(r1), int(comp.shape[0]), int(center_bias))
            candidates.append((score, color, r0, c0, r1, c1, comp_mask))
    assert candidates, "canvas frame not found"
    _, color, r0, c0, r1, c1, comp_mask = max(candidates, key=lambda item: item[0])
    return int(color), int(r0), int(c0), int(r1), int(c1), comp_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer the scattered same-color objects onto the bottom canvas using the top legend order.

    Input structure:
    - One background color fills most of the grid.
    - A 1-pixel-high horizontal legend strip sits near the top; its distinct colors appear left-to-right without gaps.
    - Away from the legend are several 4-connected monochrome objects, with one or more objects per legend color.
    - Near the bottom is a unique hollow rectangular frame in a special border color; its interior is empty background and defines the output canvas.

    Rule:
    - Read the legend colors from left to right to get the drawing order from bottom layer to top layer.
    - For each legend color, collect every non-legend object pixel of that color and treat the whole set as one layer while preserving all relative positions inside that layer.
    - Compute that color layer's bounding box and translate it so the layer bbox is centered on the framed canvas interior.
    - Paint the centered layers in legend order; later colors overwrite earlier colors on overlaps.
    - Return only the canvas interior, cropped to the frame size without the border.

    Key insight:
    - The top strip is not decorative: it encodes painter-order precedence, while the bottom frame is the target canvas for centered layer compositing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    _legend_row, _c0, _c1, legend_colors, legend_mask = _find_legend(g, bg)
    frame_color, fr0, fc0, fr1, fc1, frame_mask = _find_canvas_frame(g, bg, legend_mask)
    assert frame_color not in legend_colors

    canvas_h = int(fr1 - fr0 - 1)
    canvas_w = int(fc1 - fc0 - 1)
    assert canvas_h >= 1 and canvas_w >= 1

    objects = g.copy()
    objects[legend_mask | frame_mask] = bg

    out = np.full((canvas_h, canvas_w), bg, dtype=int)
    for color in legend_colors:
        coords = np.argwhere(objects == int(color))
        assert coords.size > 0, "every legend color must have object pixels"
        r0, c0, r1, c1 = _bbox(coords)
        layer_h = int(r1 - r0 + 1)
        layer_w = int(c1 - c0 + 1)
        top = (canvas_h - layer_h) // 2
        left = (canvas_w - layer_w) // 2
        rel = coords - np.array([r0, c0], dtype=int)
        rr = top + rel[:, 0]
        cc = left + rel[:, 1]
        assert int(rr.min()) >= 0 and int(rr.max()) < canvas_h
        assert int(cc.min()) >= 0 and int(cc.max()) < canvas_w
        out[rr, cc] = int(color)

    return out


_RNG = np.random.default_rng()
_COLOR_POOL = tuple(range(10))
_GEN_ATTEMPTS = 400
_LAYER_ATTEMPTS = 120
_PLACEMENT_ATTEMPTS = 200


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    pts = [(int(r), int(c))]
    while r != tr or c != tc:
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = _choice(moves)
        pts.append((int(r), int(c)))
    return pts


def _spanning_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    hub = (_randint(0, h - 1), _randint(0, w - 1))
    anchors = {
        (0, _randint(0, w - 1)),
        (h - 1, _randint(0, w - 1)),
        (_randint(0, h - 1), 0),
        (_randint(0, h - 1), w - 1),
    }
    for cell in [hub, *anchors]:
        mask[cell] = True
    for anchor in anchors:
        for cell in _path_cells(hub, anchor):
            mask[cell] = True

    target = _randint(int(mask.sum()), min(h * w, int(mask.sum()) + max(2, (h * w) // 3)))
    frontier = {
        (nr, nc)
        for r, c in np.argwhere(mask)
        for nr, nc in _neighbors4(int(r), int(c), h, w)
        if not mask[nr, nc]
    }
    while int(mask.sum()) < target and frontier:
        nr, nc = _choice(tuple(frontier))
        frontier.remove((int(nr), int(nc)))
        mask[nr, nc] = True
        for rr, cc in _neighbors4(int(nr), int(nc), h, w):
            if not mask[rr, cc]:
                frontier.add((int(rr), int(cc)))

    assert bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())
    return mask


def _mask_has_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _two_component_variant(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    tries = []
    if w >= 3:
        tries.extend([("v", c) for c in range(1, w - 1)])
    if h >= 3:
        tries.extend([("h", r) for r in range(1, h - 1)])
    if not tries:
        return None
    tries = list(_RNG.permutation(tries))
    for axis, idx in tries:
        trial = mask.copy()
        if axis == "v":
            trial[:, int(idx)] = False
        else:
            trial[int(idx), :] = False
        if not _mask_has_full_bbox(trial):
            continue
        comps = _components(trial)
        if len(comps) != 2:
            continue
        if min(int(comp.shape[0]) for comp in comps) < 2:
            continue
        return trial
    return None


def _same_parity_sizes(limit: int) -> list[int]:
    parity = int(limit % 2)
    start = 3 if parity == 1 else 2
    return [v for v in range(start, int(limit) + 1) if v % 2 == parity]


def _make_layer_mask(h: int, w: int, split: bool) -> np.ndarray | None:
    for _ in range(_LAYER_ATTEMPTS):
        base = _spanning_mask(h, w)
        if not split:
            return base
        split_mask = _two_component_variant(base)
        if split_mask is not None:
            return split_mask
    return None


def _paint_expected(
    bg: int, canvas_h: int, canvas_w: int, legend_colors: list[int], layer_masks: list[np.ndarray]
) -> np.ndarray:
    out = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for color, mask in zip(legend_colors, layer_masks, strict=False):
        coords = np.argwhere(mask)
        top = (canvas_h - mask.shape[0]) // 2
        left = (canvas_w - mask.shape[1]) // 2
        out[top + coords[:, 0], left + coords[:, 1]] = int(color)
    return out


def _rects_clear(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for ar0, ac0, ar1, ac1 in rects:
        if not (r1 < ar0 - 1 or r0 > ar1 + 1 or c1 < ac0 - 1 or c0 > ac1 + 1):
            return False
    return True


def _place_layers(
    H: int,
    W: int,
    frame_rect: tuple[int, int, int, int],
    legend_rect: tuple[int, int, int, int],
    layer_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    fr0, _fc0, _fr1, _fc1 = frame_rect
    occupied = [frame_rect, legend_rect]
    placed: list[tuple[int, int]] = []
    order = sorted(range(len(layer_shapes)), key=lambda i: layer_shapes[i][0] * layer_shapes[i][1], reverse=True)
    slots: list[tuple[int, int] | None] = [None] * len(layer_shapes)

    for idx in order:
        lh, lw = layer_shapes[idx]
        candidates = []
        max_row = int(fr0 - lh - 2)
        if max_row < 2:
            return None
        for r in range(2, max_row + 1):
            for c in range(0, W - lw + 1):
                if _rects_clear(occupied, r, c, lh, lw):
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None
        r, c = _choice(candidates)
        occupied.append((int(r), int(c), int(r + lh - 1), int(c + lw - 1)))
        slots[idx] = (int(r), int(c))
        placed.append((int(r), int(c)))

    return [slot for slot in slots if slot is not None]


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    k = _choice([2, 3, 4, 5])
    bg = _choice(_COLOR_POOL)
    others = [c for c in _COLOR_POOL if c != bg]
    legend_colors = _sample(others, k)
    others = [c for c in others if c not in legend_colors]
    frame_color = _choice(others)

    canvas_h = _choice([v for v in range(5, 10)])
    canvas_w = _choice([v for v in range(5, 10)])
    h_opts = _same_parity_sizes(canvas_h)
    w_opts = _same_parity_sizes(canvas_w)

    layer_masks: list[np.ndarray] = []
    for _ in range(k):
        lh = _choice(h_opts)
        lw = _choice(w_opts)
        split = bool(_randint(0, 1))
        mask = _make_layer_mask(lh, lw, split=split)
        if mask is None:
            return None
        layer_masks.append(mask)

    if sum(int(mask.shape[0] * mask.shape[1]) for mask in layer_masks) > 140:
        return None

    frame_h = int(canvas_h + 2)
    frame_w = int(canvas_w + 2)
    max_layer_h = max(int(mask.shape[0]) for mask in layer_masks)
    max_layer_w = max(int(mask.shape[1]) for mask in layer_masks)
    min_h = max(18, frame_h + max_layer_h + 5)
    min_w = max(18, frame_w + max_layer_w + 4, k + 4)
    if min_h > 30 or min_w > 30:
        return None
    H = _randint(min_h, 30)
    W = _randint(min_w, 30)

    grid = np.full((H, W), int(bg), dtype=int)
    legend_len = int(k)
    legend_c0 = _randint(0, W - legend_len)
    legend_c1 = legend_c0 + legend_len
    grid[0, legend_c0:legend_c1] = np.array(legend_colors, dtype=int)
    legend_rect = (0, int(legend_c0), 0, int(legend_c1 - 1))

    bottom_margin = _randint(0, 1)
    fr0 = int(H - frame_h - bottom_margin)
    fc0 = int((W - frame_w) // 2)
    fr1 = int(fr0 + frame_h - 1)
    fc1 = int(fc0 + frame_w - 1)
    grid[fr0, fc0 : fc1 + 1] = int(frame_color)
    grid[fr1, fc0 : fc1 + 1] = int(frame_color)
    grid[fr0 : fr1 + 1, fc0] = int(frame_color)
    grid[fr0 : fr1 + 1, fc1] = int(frame_color)
    frame_rect = (fr0, fc0, fr1, fc1)

    layer_shapes = [(int(mask.shape[0]), int(mask.shape[1])) for mask in layer_masks]
    placements = _place_layers(H, W, frame_rect, legend_rect, layer_shapes)
    if placements is None:
        return None

    for color, mask, (r0, c0) in zip(legend_colors, layer_masks, placements, strict=False):
        coords = np.argwhere(mask)
        grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)

    expected = _paint_expected(bg, canvas_h, canvas_w, legend_colors, layer_masks)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC instance")
