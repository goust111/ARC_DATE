from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
TEMPLATE_FG = 5


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _minimal_periods(pattern: np.ndarray) -> tuple[int, int]:
    h, w = pattern.shape
    best: tuple[int, int] | None = None
    best_area: int | None = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(pattern[r, c]) != int(pattern[r % pr, c % pc]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                area = pr * pc
                if best is None or area < best_area or (area == best_area and (pr, pc) < best):
                    best = (pr, pc)
                    best_area = area
    assert best is not None
    return best


def _find_marker_block(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    mask = np.isin(grid, MARKER_COLORS)
    coords = np.argwhere(mask)
    assert coords.shape[0] == 4, "expected exactly four marker-colored cells"
    assert set(map(int, grid[mask])) == set(MARKER_COLORS), "marker colors must be 1,2,3,4"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 == r0 + 1 and c1 == c0 + 1, "marker colors must form one 2x2 block"
    block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(map(int, block.ravel())) == set(MARKER_COLORS), "2x2 block must contain all marker colors"
    return r0, c0, block


def _find_template_patch(grid: np.ndarray) -> np.ndarray:
    grey_coords = np.argwhere(grid == TEMPLATE_FG)
    assert grey_coords.size > 0, "expected a grey template"
    r0, c0, r1, c1 = _bbox(grey_coords)
    h, w = grid.shape

    top = r0 == 0
    bottom = r1 == h - 1
    left = c0 == 0
    right = c1 == w - 1
    assert (top ^ bottom) and (left ^ right), "template must occupy exactly one corner"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(map(int, np.unique(patch))) <= {BG, TEMPLATE_FG}, "template patch must be binary"
    assert np.all(np.any(patch == TEMPLATE_FG, axis=1)), "template must reveal every patch row"
    assert np.all(np.any(patch == TEMPLATE_FG, axis=0)), "template must reveal every patch col"
    return patch


def _recolored_tiling(height: int, width: int, tile: np.ndarray, color: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((height, width), dtype=int)
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    tiled = tile[rows, cols]
    return np.where(tiled == TEMPLATE_FG, int(color), BG).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one corner's binary periodic template into four diagonal quadrants defined by a 2x2 marker block.

    Input structure:
    - The grid has black background, exactly one interior 2x2 marker block, and one binary grey/black template patch in a corner.
    - The marker block uses the four distinct colors 1, 2, 3, and 4 exactly once.
    - The corner patch is a small rectangle whose visible grey cells reveal a row/column periodic texture.

    Rule:
    - Keep the 2x2 marker block unchanged.
    - Interpret the block as a diagonal legend: TL marker colors the lower-right quadrant, TR colors the lower-left, BL colors the upper-right, and BR colors the upper-left.
    - Infer the template's smallest row and column periods, tile that binary pattern independently from each quadrant's own top-left corner, and replace grey template pixels with the quadrant's target color while leaving black template pixels black.

    Key insight:
    - The 2x2 block provides both the spatial partition and the per-quadrant recolor map, while the corner patch provides only the reusable geometry of the periodic texture.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_r0, marker_c0, block = _find_marker_block(g)
    template_patch = _find_template_patch(g)
    pr, pc = _minimal_periods(template_patch)
    tile = template_patch[:pr, :pc].copy()

    h, w = g.shape
    assert 0 < marker_r0 < h - 2 and 0 < marker_c0 < w - 2, "marker block must leave four quadrants"

    out = np.zeros_like(g)
    quadrants = [
        ((0, marker_r0), (0, marker_c0), int(block[1, 1])),
        ((0, marker_r0), (marker_c0 + 2, w), int(block[1, 0])),
        ((marker_r0 + 2, h), (0, marker_c0), int(block[0, 1])),
        ((marker_r0 + 2, h), (marker_c0 + 2, w), int(block[0, 0])),
    ]
    for (r_start, r_stop), (c_start, c_stop), color in quadrants:
        out[r_start:r_stop, c_start:c_stop] = _recolored_tiling(
            r_stop - r_start, c_stop - c_start, tile, color
        )

    out[marker_r0 : marker_r0 + 2, marker_c0 : marker_c0 + 2] = block
    return out


GRID_ROWS = (11, 18)
GRID_COLS = (11, 18)
PATCH_SIDE_CHOICES = (2, 3)
MIN_MARKER_MARGIN = 3
MAX_ATTEMPTS = 400
CORNERS = ("tl", "tr", "bl", "br")
PERIOD_CHOICES = [(pr, pc) for pr in (1, 2, 3) for pc in (1, 2, 3) if (pr, pc) != (1, 1)]
_TEMPLATE_CANDIDATE_CACHE: dict[tuple[int, int, int, int], list[np.ndarray]] = {}


def _repeat_patch(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].astype(int)


def _template_candidates(pr: int, pc: int, patch_h: int, patch_w: int) -> list[np.ndarray]:
    key = (pr, pc, patch_h, patch_w)
    if key in _TEMPLATE_CANDIDATE_CACHE:
        return _TEMPLATE_CANDIDATE_CACHE[key]

    candidates: list[np.ndarray] = []
    for mask_bits in range(1, 1 << (pr * pc)):
        tile = np.zeros((pr, pc), dtype=int)
        for idx in range(pr * pc):
            if (mask_bits >> idx) & 1:
                tile[idx // pc, idx % pc] = TEMPLATE_FG
        if TEMPLATE_FG not in tile or BG not in tile:
            continue
        patch = _repeat_patch(tile, patch_h, patch_w)
        if np.any(np.all(patch == BG, axis=1)) or np.any(np.all(patch == BG, axis=0)):
            continue
        grey_count = int(np.count_nonzero(patch == TEMPLATE_FG))
        black_count = int(np.count_nonzero(patch == BG))
        if grey_count < 2 or black_count < 2:
            continue
        if _minimal_periods(patch) != (pr, pc):
            continue
        candidates.append(patch)

    _TEMPLATE_CANDIDATE_CACHE[key] = candidates
    return candidates


def _random_template_patch() -> np.ndarray:
    for _ in range(100):
        pr, pc = random.choice(PERIOD_CHOICES)
        patch_h = random.randint(max(2, pr), 3)
        patch_w = random.randint(max(2, pc), 3)
        candidates = _template_candidates(pr, pc, patch_h, patch_w)
        if candidates:
            return random.choice(candidates).copy()
    raise RuntimeError("failed to sample a non-degenerate template patch")


def _place_corner_patch(grid: np.ndarray, patch: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    elif corner == "br":
        r0, c0 = h - ph, w - pw
    else:
        raise ValueError(f"unknown corner {corner!r}")
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch


def _quadrant_has_color(out: np.ndarray, row_slice: slice, col_slice: slice, color: int) -> bool:
    sub = out[row_slice, col_slice]
    return bool(np.any(sub == int(color)))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    grid = np.zeros((h, w), dtype=int)

    marker_r0 = random.randint(MIN_MARKER_MARGIN, h - MIN_MARKER_MARGIN - 2)
    marker_c0 = random.randint(MIN_MARKER_MARGIN, w - MIN_MARKER_MARGIN - 2)
    block = np.array(random.sample(MARKER_COLORS, 4), dtype=int).reshape(2, 2)
    grid[marker_r0 : marker_r0 + 2, marker_c0 : marker_c0 + 2] = block

    patch = _random_template_patch()
    corner = random.choice(CORNERS)
    _place_corner_patch(grid, patch, corner)

    grey_coords = np.argwhere(grid == TEMPLATE_FG)
    if grey_coords.size == 0:
        return None
    template_r0, template_c0, template_r1, template_c1 = _bbox(grey_coords)
    if not (
        template_r1 < marker_r0
        or template_r0 > marker_r0 + 1
        or template_c1 < marker_c0
        or template_c0 > marker_c0 + 1
    ):
        return None

    out = main(grid)
    quadrants = [
        (slice(0, marker_r0), slice(0, marker_c0), int(block[1, 1])),
        (slice(0, marker_r0), slice(marker_c0 + 2, w), int(block[1, 0])),
        (slice(marker_r0 + 2, h), slice(0, marker_c0), int(block[0, 1])),
        (slice(marker_r0 + 2, h), slice(marker_c0 + 2, w), int(block[0, 0])),
    ]
    if not all(_quadrant_has_color(out, rs, cs, color) for rs, cs, color in quadrants):
        return None
    if set(map(int, np.unique(grid))) - {BG, TEMPLATE_FG, *MARKER_COLORS}:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
