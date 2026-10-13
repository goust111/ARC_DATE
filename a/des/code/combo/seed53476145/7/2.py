from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
SEP = 5

INPUT_COLORS = (3, 6, 7, 8, 9)
PANEL_H_RANGE = (8, 14)
PANEL_W_RANGE = (8, 14)
SHAPE_MARGIN = 1
CORE_MIN_AREA = 10
REGION_SIZE_RANGE = (1, 3)
GENERATION_RETRIES = 400
SHAPE_RETRIES = 120
REGION_NAMES = ("ab", "ac", "bc", "a", "b", "c")


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _full_height_separator_cols(g: np.ndarray) -> list[int]:
    h, w = g.shape
    cols = []
    for c in range(w):
        vals = np.unique(g[:, c])
        if vals.size == 1 and int(vals[0]) != BG:
            cols.append(int(c))
    assert len(cols) == 2, f"expected exactly two full-height separator columns, got {cols}"
    assert int(g[0, cols[0]]) == int(g[0, cols[1]]), "separator columns must share one color"
    return cols


def _panel_slices(g: np.ndarray) -> list[tuple[slice, slice]]:
    h, w = g.shape
    sep0, sep1 = _full_height_separator_cols(g)
    widths = (sep0, sep1 - sep0 - 1, w - sep1 - 1)
    assert widths[0] > 0 and widths[0] == widths[1] == widths[2], f"panels must be equal width, got {widths}"
    panel_w = widths[0]
    return [
        (slice(0, h), slice(0, panel_w)),
        (slice(0, h), slice(sep0 + 1, sep1)),
        (slice(0, h), slice(sep1 + 1, w)),
    ]


def _panel_masks(g: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for rs, cs in _panel_slices(g):
        panel = g[rs, cs]
        colors = sorted(int(x) for x in np.unique(panel) if int(x) != BG)
        assert len(colors) == 1, f"each panel must contain exactly one foreground color, got {colors}"
        masks.append(panel != BG)
    return masks


def _coverage_output(masks: list[np.ndarray]) -> np.ndarray:
    assert len(masks) == 3, "expected exactly three panels"
    counts = sum(mask.astype(int) for mask in masks)
    h, w = masks[0].shape
    out = np.full((h, w), BG, dtype=int)
    out[counts == 1] = BLUE
    out[counts == 2] = YELLOW
    out[counts == 3] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the three panel-local foreground masks and color each cell by its coverage count.

    Count mapping:
    - present in all three panels -> red
    - present in exactly two panels -> yellow
    - present in exactly one panel -> blue
    - absent from all panels -> black
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    masks = _panel_masks(g)
    return _coverage_output(masks)


def _allowed_cells(h: int, w: int, margin: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(margin, h - margin)
        for c in range(margin, w - margin)
    }


def _adjacent_empty(
    anchor: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    margin: int,
) -> list[tuple[int, int]]:
    allowed = _allowed_cells(h, w, margin)
    seeds = set()
    for r, c in anchor:
        for rr, cc in _neighbors4(r, c, h, w):
            cell = (rr, cc)
            if cell in allowed and cell not in occupied:
                seeds.add(cell)
    return list(seeds)


def _grow_connected(
    h: int,
    w: int,
    size: int,
    *,
    occupied: set[tuple[int, int]] | None = None,
    seed_candidates: list[tuple[int, int]] | None = None,
    margin: int = SHAPE_MARGIN,
) -> set[tuple[int, int]] | None:
    occupied = set() if occupied is None else set(occupied)
    allowed = _allowed_cells(h, w, margin) - occupied
    if len(allowed) < size:
        return None

    if seed_candidates is None:
        seeds = list(allowed)
    else:
        seeds = [cell for cell in seed_candidates if cell in allowed]
    if not seeds:
        return None

    seed = random.choice(seeds)
    shape = {seed}
    frontier = {cell for cell in _neighbors4(seed[0], seed[1], h, w) if cell in allowed}

    while len(shape) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        shape.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], h, w):
            if nxt in allowed and nxt not in shape:
                frontier.add(nxt)
    return shape


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _color_triplet() -> tuple[int, int, int]:
    branch = random.randint(1, 3)
    if branch == 1:
        color = random.choice(INPUT_COLORS)
        return color, color, color
    if branch == 2:
        a, b = random.sample(INPUT_COLORS, 2)
        return random.choice(
            (
                (a, a, b),
                (a, b, a),
                (b, a, a),
                (a, b, b),
                (b, a, b),
                (b, b, a),
            )
        )
    return tuple(random.sample(INPUT_COLORS, 3))


def _build_related_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    inner_area = (h - 2 * SHAPE_MARGIN) * (w - 2 * SHAPE_MARGIN)
    for _ in range(SHAPE_RETRIES):
        sizes = {name: random.randint(*REGION_SIZE_RANGE) for name in REGION_NAMES}
        total_extra = sum(sizes.values())
        max_extra = max(
            sizes["ab"] + sizes["ac"] + sizes["a"],
            sizes["ab"] + sizes["bc"] + sizes["b"],
            sizes["ac"] + sizes["bc"] + sizes["c"],
        )
        core_min = max(CORE_MIN_AREA, max_extra + 4)
        core_max = min(inner_area - total_extra - 4, inner_area // 2)
        if core_min > core_max:
            continue

        core_area = random.randint(core_min, core_max)
        core = _grow_connected(h, w, core_area)
        if core is None:
            continue

        if len(_adjacent_empty(core, core, h, w, SHAPE_MARGIN)) < len(REGION_NAMES):
            continue

        occupied = set(core)
        regions: dict[str, set[tuple[int, int]]] = {}
        order = list(REGION_NAMES)
        random.shuffle(order)
        ok = True
        for name in order:
            seeds = _adjacent_empty(core, occupied, h, w, SHAPE_MARGIN)
            region = _grow_connected(
                h,
                w,
                sizes[name],
                occupied=occupied,
                seed_candidates=seeds,
            )
            if region is None:
                ok = False
                break
            regions[name] = region
            occupied |= region
        if not ok:
            continue

        a_cells = core | regions["ab"] | regions["ac"] | regions["a"]
        b_cells = core | regions["ab"] | regions["bc"] | regions["b"]
        c_cells = core | regions["ac"] | regions["bc"] | regions["c"]

        if a_cells == b_cells or a_cells == c_cells or b_cells == c_cells:
            continue

        masks = tuple(_mask_from_cells(cells, h, w) for cells in (a_cells, b_cells, c_cells))
        out = _coverage_output(list(masks))
        if not (np.any(out == BLUE) and np.any(out == YELLOW) and np.any(out == RED)):
            continue

        unions = [
            np.count_nonzero(masks[0] | masks[1]),
            np.count_nonzero(masks[0] | masks[2]),
            np.count_nonzero(masks[1] | masks[2]),
        ]
        overlaps = [
            np.count_nonzero(masks[0] & masks[1]),
            np.count_nonzero(masks[0] & masks[2]),
            np.count_nonzero(masks[1] & masks[2]),
        ]
        if any(2 * ov < un for ov, un in zip(overlaps, unions, strict=False)):
            continue

        return masks
    return None


def _render_input(masks: tuple[np.ndarray, np.ndarray, np.ndarray], colors: tuple[int, int, int]) -> np.ndarray:
    h, w = masks[0].shape
    grid = np.full((h, 3 * w + 2), BG, dtype=int)
    grid[:, w] = SEP
    grid[:, 2 * w + 1] = SEP
    starts = (0, w + 1, 2 * w + 2)
    for mask, color, c0 in zip(masks, colors, starts, strict=False):
        grid[:, c0 : c0 + w][mask] = int(color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        h = random.randint(*PANEL_H_RANGE)
        w = random.randint(*PANEL_W_RANGE)
        masks = _build_related_masks(h, w)
        if masks is None:
            continue
        colors = _color_triplet()
        grid = _render_input(masks, colors)

        parsed_masks = _panel_masks(grid)
        assert all(np.array_equal(a, b) for a, b in zip(masks, parsed_masks, strict=False))
        out = main(grid)
        if out.shape != (h, w):
            continue
        if not (np.any(out == BLUE) and np.any(out == YELLOW) and np.any(out == RED)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC instance")
