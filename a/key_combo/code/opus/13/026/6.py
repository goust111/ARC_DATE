import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4)
GEN_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _split_bands(grid: np.ndarray) -> list[np.ndarray]:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(divider_rows) == 3, "expected exactly three full grey divider rows"
    cuts = [-1] + divider_rows + [grid.shape[0]]
    bands = [grid[cuts[i] + 1 : cuts[i + 1]] for i in range(4)]
    band_heights = {band.shape[0] for band in bands}
    assert len(band_heights) == 1, "expected four equal-height bands"
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four horizontal mask bands into one overlay using hole-count priority.

    Input structure:
    - The grid has black background and three full grey horizontal divider rows.
    - The dividers split the grid into four equal-height bands.
    - Each band contains exactly one non-black, non-grey foreground color forming one 4-connected shape.

    Rule:
    - Remove the grey divider rows and treat the four bands as transparent layers in one shared local coordinate system.
    - Count each band's enclosed black holes using 4-connectivity; background regions touching the band border are not holes.
    - Overlay the bands so the shape with fewer holes wins every conflict pixel; ties fall back to band order.

    Key insight:
    - The layer priority is not fixed by position or color; it must be inferred from each band's topology before compositing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bands = _split_bands(g)
    band_info: list[tuple[int, int, np.ndarray]] = []
    for band_index, band in enumerate(bands):
        colors = sorted(set(map(int, np.unique(band))) - {BG, GREY})
        assert len(colors) == 1, "expected one foreground color per band"
        color = colors[0]
        mask = band == color
        assert len(_components(mask)) == 1, "expected one connected foreground shape per band"
        holes = _hole_count(mask)
        band_info.append((holes, band_index, band))

    out = np.zeros_like(bands[0])
    for _holes, _band_index, band in sorted(band_info, key=lambda item: (item[0], item[1]), reverse=True):
        mask = band != BG
        out[mask] = band[mask]
    return out


def _core_left(width: int, core_w: int, anchor_center: int) -> int:
    stem_l = anchor_center - 1
    stem_r = anchor_center + 1
    lo = max(1, stem_l - core_w + 1)
    hi = min(width - core_w - 1, stem_r)
    if lo > hi:
        lo = max(0, stem_l - core_w + 1)
        hi = min(width - core_w, stem_r)
    return random.randint(lo, hi)


def _base_mask(hole_count: int, band_h: int, width: int, anchor_center: int) -> np.ndarray:
    mask = np.zeros((band_h, width), dtype=bool)
    stem_cols = range(anchor_center - 1, anchor_center + 2)
    mask[band_h - 1, list(stem_cols)] = True

    if hole_count == 0:
        core_h, core_w = 3, 5
        top = band_h - core_h
        left = _core_left(width, core_w, anchor_center)
        mask[top : top + core_h, left : left + core_w] = True
        return mask

    if hole_count == 1:
        core_h, core_w = 4, 4
        top = band_h - core_h - 1
        left = _core_left(width, core_w, anchor_center)
        mask[top, left : left + core_w] = True
        mask[top + core_h - 1, left : left + core_w] = True
        mask[top : top + core_h, left] = True
        mask[top : top + core_h, left + core_w - 1] = True
        return mask

    core_w = 2 * hole_count + 1
    top = band_h - 4
    left = _core_left(width, core_w, anchor_center)
    mask[top, left : left + core_w] = True
    mask[top + 2, left : left + core_w] = True
    mask[top + 1, left : left + core_w : 2] = True
    return mask


def _safe_growth(mask: np.ndarray, hole_count: int, extra_steps: int) -> np.ndarray:
    grown = mask.copy()
    h, w = grown.shape
    for _ in range(extra_steps):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(grown):
            for dr, dc in NEIGHBORS4:
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not grown[nr, nc]:
                    frontier.append((nr, nc))
        safe: list[tuple[int, int]] = []
        for nr, nc in sorted(set(frontier)):
            trial = grown.copy()
            trial[nr, nc] = True
            if _hole_count(trial) == hole_count:
                safe.append((nr, nc))
        if not safe:
            break
        rr, cc = random.choice(safe)
        grown[rr, cc] = True
    return grown


def _make_shape(hole_count: int, band_h: int, width: int, anchor_center: int) -> np.ndarray:
    for _ in range(200):
        mask = _base_mask(hole_count, band_h, width, anchor_center)
        extra = random.randint(0, 7)
        mask = _safe_growth(mask, hole_count, extra)
        area = int(mask.sum())
        if not (15 <= area <= 30):
            continue
        if len(_components(mask)) != 1:
            continue
        if _hole_count(mask) != hole_count:
            continue
        return mask
    raise RuntimeError(f"failed to construct shape with {hole_count} holes")


def _render_input(band_h: int, width: int, colors: list[int], masks: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros((band_h * 4 + 3, width), dtype=int)
    row = 0
    for band_index, (color, mask) in enumerate(zip(colors, masks)):
        band = np.zeros((band_h, width), dtype=int)
        band[mask] = color
        grid[row : row + band_h] = band
        row += band_h
        if band_index < 3:
            grid[row] = GREY
            row += 1
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    bands = _split_bands(grid)
    masks = []
    hole_counts = []
    for band in bands:
        colors = sorted(set(map(int, np.unique(band))) - {BG, GREY})
        if len(colors) != 1:
            return False
        mask = band == colors[0]
        if len(_components(mask)) != 1:
            return False
        area = int(mask.sum())
        if not (15 <= area <= 30):
            return False
        masks.append(mask)
        hole_counts.append(_hole_count(mask))
    if sorted(hole_counts) != [0, 1, 2, 3]:
        return False
    overlap = np.sum(np.stack(masks, axis=0), axis=0)
    if int(np.count_nonzero(overlap >= 2)) < 3:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == bands[0].shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        band_h = random.choice((5, 6))
        width = random.randint(11, 13)
        anchor_center = random.randint(3, width - 4)
        colors = random.sample(list(PALETTE), 4)
        hole_order = list(range(4))
        random.shuffle(hole_order)
        masks = [_make_shape(hole_count, band_h, width, anchor_center) for hole_count in hole_order]
        grid = _render_input(band_h, width, colors, masks)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
