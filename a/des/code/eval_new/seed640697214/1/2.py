from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
ARC_MAX = 30
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(np.asarray(coords, dtype=int))
    return parts


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _extract_monochrome_sprite(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components(grid == color)
    assert len(comps) == 1, f"expected exactly one component for color {color}"
    coords = comps[0]
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return patch == color


def _pack_sprites(shape_mask: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = shape_mask.shape
    out = np.full((h, w * len(colors)), BG, dtype=int)
    for i, color in enumerate(colors):
        block = out[:, i * w : (i + 1) * w]
        block[shape_mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder same-shaped colored sprites by reading the bottom-edge color bar.

    Input structure:
    - bg = black (0).
    - Above the last row, several non-overlapping monochrome sprites are scattered on the canvas.
    - Every sprite has the same binary shape, but each sprite uses a different color.
    - The bottom row contains a left-to-right color bar made of one pixel per sprite color.

    Rule:
    - Read the non-background pixels in the bottom row from left to right to get the target color order.
    - For each listed color, extract the matching sprite from the rows above the bar.
    - Verify that all extracted sprites share the same shape mask, then place them tightly side by side in the bar order.
    - Keep each sprite's original color; the output height is one sprite's height and the output width is the sum of sprite widths.

    Key insight:
    - The bottom-row dots are not extra objects to transform; they are an ordering instruction for otherwise identical colored sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= ARC_MAX

    order = [int(g[-1, c]) for c in range(g.shape[1]) if int(g[-1, c]) != BG]
    assert len(order) >= 2, "bottom row must contain an ordering bar"
    assert len(order) == len(set(order)), "bar colors must be unique"

    upper = g[:-1]
    upper_colors = {int(v) for v in np.unique(upper) if int(v) != BG}
    assert upper_colors == set(order), "sprite colors must match the bar colors exactly"

    reference_mask = None
    for color in order:
        mask = _extract_monochrome_sprite(upper, color)
        if reference_mask is None:
            reference_mask = mask
        else:
            assert mask.shape == reference_mask.shape
            assert np.array_equal(mask, reference_mask), "all sprites must share the same shape"

    assert reference_mask is not None
    return _pack_sprites(reference_mask, order)


def _grow_connected_mask(side_limit: int) -> np.ndarray:
    for _ in range(100):
        h0 = int(RNG.integers(2, side_limit + 1))
        w0 = int(RNG.integers(2, side_limit + 1))
        area_lo = max(3, (h0 * w0 + 1) // 2)
        area_hi = h0 * w0
        target = int(RNG.integers(area_lo, area_hi + 1))

        mask = np.zeros((h0, w0), dtype=bool)
        start = (int(RNG.integers(h0)), int(RNG.integers(w0)))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            present = np.argwhere(mask)
            for r, c in present:
                for dr, dc in NEIGHBORS4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h0 and 0 <= nc < w0 and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(RNG.integers(len(frontier)))]
            mask[nr, nc] = True
            if RNG.random() < 0.25:
                extras = [
                    (nr + dr, nc + dc)
                    for dr, dc in NEIGHBORS4
                    if 0 <= nr + dr < h0 and 0 <= nc + dc < w0 and not mask[nr + dr, nc + dc]
                ]
                if extras and int(mask.sum()) < target:
                    er, ec = extras[int(RNG.integers(len(extras)))]
                    mask[er, ec] = True

        if not mask.any():
            continue
        mask = _crop_mask(mask)
        if RNG.random() < 0.5:
            mask = np.rot90(mask, int(RNG.integers(4)))
        if RNG.random() < 0.5:
            mask = np.fliplr(mask)
        if RNG.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)

        if min(mask.shape) < 2 or int(mask.sum()) < 3:
            continue
        if mask.shape[0] > side_limit or mask.shape[1] > side_limit:
            continue
        if mask.all() and RNG.random() < 0.7:
            continue
        return mask
    raise RuntimeError("failed to sample a connected sprite mask")


def _sample_bar_positions(width: int, count: int) -> np.ndarray | None:
    if width < 2 * count - 1:
        return None
    bases = np.sort(RNG.choice(width - count + 1, size=count, replace=False))
    return bases + np.arange(count)


def _place_bboxes(grid_h: int, grid_w: int, box_h: int, box_w: int, count: int) -> list[tuple[int, int]] | None:
    max_top = grid_h - box_h - 2
    if max_top < 0 or grid_w < box_w:
        return None

    blocked = np.zeros((grid_h - 1, grid_w), dtype=bool)
    positions: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            top = int(RNG.integers(0, max_top + 1))
            left = int(RNG.integers(0, grid_w - box_w + 1))
            r0 = max(0, top - 1)
            r1 = min(grid_h - 1, top + box_h + 1)
            c0 = max(0, left - 1)
            c1 = min(grid_w, left + box_w + 1)
            if blocked[r0:r1, c0:c1].any():
                continue
            blocked[r0:r1, c0:c1] = True
            positions.append((top, left))
            placed = True
            break
        if not placed:
            return None
    return positions


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _expected_output(mask: np.ndarray, order: list[int]) -> np.ndarray:
    return _pack_sprites(mask, order)


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(3, 7))
    side_limit = min(5, 9 - count)
    shape = _grow_connected_mask(side_limit)
    sh, sw = shape.shape

    output_w = sw * count
    if output_w > ARC_MAX:
        return None

    min_h = max(sh + 4, 9)
    min_w = max(sw + 4, 2 * count + 3, 12)
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None

    grid_h = int(RNG.integers(min_h, ARC_MAX + 1))
    grid_w = int(RNG.integers(min_w, ARC_MAX + 1))

    usable_area = (grid_h - 1) * grid_w
    padded_area = count * (sh + 2) * (sw + 2)
    if usable_area < padded_area:
        return None

    positions = _place_bboxes(grid_h, grid_w, sh, sw, count)
    if positions is None:
        return None

    bar_cols = _sample_bar_positions(grid_w, count)
    if bar_cols is None:
        return None

    colors = [int(v) for v in RNG.choice(np.arange(1, 10), size=count, replace=False)]
    order = [int(v) for v in RNG.permutation(colors)]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    pos_by_color = dict(zip(colors, positions))
    for color in colors:
        top, left = pos_by_color[color]
        _paint_mask(grid, shape, top, left, color)

    for col, color in zip(bar_cols, order):
        grid[-1, int(col)] = int(color)

    expected = _expected_output(shape, order)
    if max(expected.shape) > ARC_MAX:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
