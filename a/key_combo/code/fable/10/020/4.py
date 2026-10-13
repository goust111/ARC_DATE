from __future__ import annotations

from collections import Counter
import random

import numpy as np


FRAME_COLOR = 2
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[int, int, int, int]] = []

    for seed_h in range(1, H - 1):
        for seed_w in range(1, W - 1):
            secondary = int(g[seed_h, seed_w])
            if secondary == bg:
                continue
            if not np.all(g[seed_h, : seed_w + 1] == secondary):
                continue
            if not np.all(g[: seed_h + 1, seed_w] == secondary):
                continue

            seed = g[:seed_h, :seed_w]
            if seed.size == 0:
                continue
            if np.any(seed == secondary):
                continue

            non_bg = sorted(set(map(int, np.unique(seed[seed != bg]))))
            if len(non_bg) != 1:
                continue
            primary = int(non_bg[0])
            if not np.any(seed == bg):
                continue
            if not np.all((seed == bg) | (seed == primary)):
                continue

            candidates.append((seed_h, seed_w, primary, secondary))

    assert len(candidates) == 1, "expected one top-left L-border legend"
    seed_h, seed_w, primary, _secondary = candidates[0]
    seed_mask = g[:seed_h, :seed_w] == primary
    return seed_mask, primary, candidates[0][3]


def _find_viewport_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[int, int, int, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        mask = g == color
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        expected = np.zeros((H, W), dtype=bool)
        expected[r0, c0 : c1 + 1] = True
        expected[r1, c0 : c1 + 1] = True
        expected[r0 : r1 + 1, c0] = True
        expected[r0 : r1 + 1, c1] = True
        if not np.array_equal(mask, expected):
            continue
        if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue
        candidates.append((color, r0, c0, r1, c1))

    assert len(candidates) == 1, "expected one hollow viewport frame"
    return candidates[0]


def _virtual_crop(
    seed_mask: np.ndarray,
    row0: int,
    col0: int,
    height: int,
    width: int,
    bg: int,
    primary: int,
    secondary: int,
) -> np.ndarray:
    seed_h, seed_w = seed_mask.shape
    rows = (row0 + np.arange(height, dtype=int))[:, None]
    cols = (col0 + np.arange(width, dtype=int))[None, :]

    inner = seed_mask[rows % seed_h, cols % seed_w]
    meta = seed_mask[(rows // seed_h) % seed_h, (cols // seed_w) % seed_w]

    out = np.full((height, width), bg, dtype=int)
    out[inner & meta] = primary
    out[inner & ~meta] = secondary
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the red viewport from the virtual plane obtained by recursively expanding the top-left seed.

    Input structure:
    - A top-left legend rectangle contains a seed mask plus an L-shaped solid border: the seed uses background and one
      primary color, while the legend's last row and last column are a uniform secondary color.
    - Elsewhere on the canvas there is one hollow rectangular frame whose interior is empty background; the frame color
      does not belong to the seed legend.

    Rule:
    - Interpret the seed as a binary mask. Replace each primary seed cell with a primary-colored copy of the seed mask,
      and each background seed cell with a secondary-colored copy of the same seed mask, producing one enlarged tile.
    - Repeat that enlarged tile periodically across the whole coordinate plane, aligned to the input grid's top-left
      corner as the global origin.
    - Return only the hollow frame's interior, filled with the colors seen at those absolute coordinates in the virtual
      periodic pattern.

    Key insight:
    - The large pattern is never drawn in the input: the L-border is just a color legend, and the red rectangle is a
      viewing window whose position selects which patch of the invisible tiled expansion must be output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    seed_mask, primary, secondary = _find_legend(g, bg)
    _frame_color, r0, c0, r1, c1 = _find_viewport_frame(g, bg)

    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1
    assert out_h >= 1 and out_w >= 1, "frame interior must be non-empty"
    return _virtual_crop(seed_mask, r0 + 1, c0 + 1, out_h, out_w, bg, primary, secondary)


def _sample_seed_mask(seed_h: int, seed_w: int) -> np.ndarray | None:
    area = seed_h * seed_w
    lo = max(2, (area + 3) // 4)
    hi = min(area - 2, (3 * area) // 4)
    if lo > hi:
        return None

    for _ in range(80):
        count = random.randint(lo, hi)
        mask = np.zeros((seed_h, seed_w), dtype=bool)
        picks = random.sample(range(area), count)
        mask.reshape(-1)[picks] = True

        if len({tuple(map(int, row)) for row in mask.astype(int)}) < 2:
            continue
        if len({tuple(map(int, col)) for col in mask.astype(int).T}) < 2:
            continue
        return mask
    return None


def _render_input(
    shape: tuple[int, int],
    seed_mask: np.ndarray,
    bg: int,
    primary: int,
    secondary: int,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    H, W = shape
    seed_h, seed_w = seed_mask.shape
    grid = np.full((H, W), bg, dtype=int)
    grid[:seed_h, :seed_w][seed_mask] = primary
    grid[seed_h, : seed_w + 1] = secondary
    grid[:seed_h, seed_w] = secondary

    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    return grid


def _simple_mask_crop(
    seed_mask: np.ndarray,
    row0: int,
    col0: int,
    height: int,
    width: int,
    bg: int,
    primary: int,
) -> np.ndarray:
    seed_h, seed_w = seed_mask.shape
    rows = (row0 + np.arange(height, dtype=int))[:, None]
    cols = (col0 + np.arange(width, dtype=int))[None, :]
    out = np.full((height, width), bg, dtype=int)
    out[seed_mask[rows % seed_h, cols % seed_w]] = primary
    return out


def _valid_view(
    seed_mask: np.ndarray,
    row0: int,
    col0: int,
    height: int,
    width: int,
    bg: int,
    primary: int,
    secondary: int,
) -> bool:
    crop = _virtual_crop(seed_mask, row0, col0, height, width, bg, primary, secondary)
    colors = set(map(int, np.unique(crop)))
    if colors != {bg, primary, secondary}:
        return False

    if np.array_equal(crop, _simple_mask_crop(seed_mask, row0, col0, height, width, bg, primary)):
        return False

    if np.array_equal(crop, _virtual_crop(seed_mask, 0, 0, height, width, bg, primary, secondary)):
        return False

    if height * width < 6:
        return False
    return True


def _sample_frame(
    shape: tuple[int, int],
    seed_mask: np.ndarray,
    bg: int,
    primary: int,
    secondary: int,
) -> tuple[int, int, int, int] | None:
    H, W = shape
    seed_h, seed_w = seed_mask.shape
    forbidden = (0, 0, seed_h + 1, seed_w + 1)

    max_inner_h = min(9, H - 4)
    max_inner_w = min(9, W - 4)
    if max_inner_h < 2 or max_inner_w < 2:
        return None

    for _ in range(240):
        inner_h = random.randint(2, max_inner_h)
        inner_w = random.randint(2, max_inner_w)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if H - outer_h - 1 < 1 or W - outer_w - 1 < 1:
            continue

        r0 = random.randint(1, H - outer_h - 1)
        c0 = random.randint(1, W - outer_w - 1)
        frame_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if _rectangles_intersect(frame_bbox, forbidden):
            continue
        if not _valid_view(seed_mask, r0 + 1, c0 + 1, inner_h, inner_w, bg, primary, secondary):
            continue
        return frame_bbox
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 3, 4, 5, 6, 7, 8, 9])
    primary, secondary = random.sample([c for c in range(10) if c not in {bg, FRAME_COLOR}], 2)

    seed_h = random.randint(2, 4)
    seed_w = random.randint(2, 4)
    seed_mask = _sample_seed_mask(seed_h, seed_w)
    if seed_mask is None:
        return None

    min_h = max(12, seed_h + 8)
    min_w = max(12, seed_w + 8)
    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)

    frame_bbox = _sample_frame((H, W), seed_mask, bg, primary, secondary)
    if frame_bbox is None:
        return None

    grid = _render_input((H, W), seed_mask, bg, primary, secondary, frame_bbox)

    expected = _virtual_crop(
        seed_mask,
        frame_bbox[0] + 1,
        frame_bbox[1] + 1,
        frame_bbox[2] - frame_bbox[0] - 1,
        frame_bbox[3] - frame_bbox[1] - 1,
        bg,
        primary,
        secondary,
    )

    if _get_bg(grid) != bg:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
