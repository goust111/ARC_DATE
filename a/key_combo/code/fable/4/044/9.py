from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
NON_BLACK_COLORS = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 300
PATTERN_MODES = ("noise", "row_bands", "col_bands", "checker")
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quartet_positions(height: int, width: int, r: int, c: int) -> list[tuple[int, int]]:
    positions = {
        (int(r), int(c)),
        (int(height - 1 - r), int(c)),
        (int(r), int(width - 1 - c)),
        (int(height - 1 - r), int(width - 1 - c)),
    }
    return sorted(positions)


def _majority_color(values: list[int]) -> int:
    counts = Counter(values).most_common()
    assert counts, "quartet must contain at least one value"
    if len(counts) > 1:
        assert counts[0][1] > counts[1][1], "quartet must have a unique majority"
    return int(counts[0][0])


def _expected_mirror_tiling(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    expected = np.empty_like(g)

    for r in range((height + 1) // 2):
        for c in range((width + 1) // 2):
            positions = _quartet_positions(height, width, r, c)
            color = _majority_color([int(g[rr, cc]) for rr, cc in positions])
            for rr, cc in positions:
                expected[rr, cc] = color

    return expected


def _crop_residue(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    residue = np.asarray(observed, dtype=int) != np.asarray(expected, dtype=int)
    box = _bbox(residue)
    assert box is not None, "expected at least one residue pixel"

    r0, c0, r1, c1 = box
    cropped_mask = residue[r0 : r1 + 1, c0 : c1 + 1]
    cropped = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    cropped[cropped_mask] = observed[r0 : r1 + 1, c0 : c1 + 1][cropped_mask]
    return cropped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the mirrored background by fourfold voting and crop the intruder residue.

    Input structure:
    - The grid is a 2x2 tiling of one seed tile mirrored across the vertical and horizontal axes.
    - All visible colors are non-black; black is reserved for the output mask.
    - One opaque intruder sprite lies entirely inside a single quadrant and overwrites that quadrant's mirrored tile.

    Rule:
    - For each cell, gather its four mirror-related partners under left-right and top-bottom reflection.
    - The majority color inside that quartet is the expected background tile color.
    - Mark cells whose observed color disagrees with that majority, then crop their minimal bounding box.
    - Output the cropped disagreement pixels in their original colors and paint all matching cells inside that box black.

    Key insight:
    - Color alone is unreliable because the intruder shares the tile palette; the only invariant witness is that a valid mirror quartet has three agreeing cells and at most one dissenting cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    expected = _expected_mirror_tiling(g)
    return _crop_residue(g, expected)


def _build_mirror_tiling(seed: np.ndarray) -> np.ndarray:
    top = np.concatenate([seed, np.fliplr(seed)], axis=1)
    bottom = np.concatenate([np.flipud(seed), np.flipud(np.fliplr(seed))], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _sample_seed(rng: np.random.Generator, qh: int, qw: int, palette: np.ndarray) -> np.ndarray:
    for _ in range(100):
        seed = rng.choice(palette, size=(qh, qw), replace=True).astype(int)
        if np.unique(seed).size < 2:
            continue
        if np.array_equal(seed, np.fliplr(seed)) and np.array_equal(seed, np.flipud(seed)):
            continue
        return seed
    raise RuntimeError("failed to sample a non-degenerate seed tile")


def _sample_sprite_pattern(
    rng: np.random.Generator, height: int, width: int, colors: np.ndarray
) -> np.ndarray:
    mode = PATTERN_MODES[int(rng.integers(0, len(PATTERN_MODES)))]
    if mode == "noise":
        sprite = rng.choice(colors, size=(height, width), replace=True)
    elif mode == "row_bands":
        row_colors = rng.choice(colors, size=height, replace=True)
        sprite = np.repeat(row_colors[:, None], width, axis=1)
    elif mode == "col_bands":
        col_colors = rng.choice(colors, size=width, replace=True)
        sprite = np.repeat(col_colors[None, :], height, axis=0)
    else:
        offset = int(rng.integers(0, len(colors)))
        stripe_r = int(rng.integers(1, 3))
        stripe_c = int(rng.integers(1, 3))
        sprite = np.empty((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                idx = (r // stripe_r + c // stripe_c + offset) % len(colors)
                sprite[r, c] = int(colors[idx])

    sprite = np.asarray(sprite, dtype=int)
    if sprite.size > 1 and np.unique(sprite).size == 1:
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width))
        alternatives = [int(color) for color in colors if int(color) != int(sprite[r, c])]
        sprite[r, c] = alternatives[int(rng.integers(0, len(alternatives)))]
    return sprite


def _force_full_bbox_residue(
    rng: np.random.Generator, sprite: np.ndarray, underlying: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    adjusted = np.asarray(sprite, dtype=int).copy()
    sh, sw = adjusted.shape
    corners = [(0, 0), (0, sw - 1), (sh - 1, 0), (sh - 1, sw - 1)]
    for r, c in corners:
        if int(adjusted[r, c]) == int(underlying[r, c]):
            choices = [int(color) for color in palette if int(color) != int(underlying[r, c])]
            adjusted[r, c] = choices[int(rng.integers(0, len(choices)))]

    if not np.any(adjusted == underlying):
        candidates = [(r, c) for r in range(sh) for c in range(sw) if (r, c) not in set(corners)]
        if candidates:
            r, c = candidates[int(rng.integers(0, len(candidates)))]
            adjusted[r, c] = int(underlying[r, c])

    return adjusted


def _oracle_output(inp: np.ndarray, pristine: np.ndarray) -> np.ndarray:
    return _crop_residue(inp, pristine)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    qh = int(rng.integers(4, 16))
    qw = int(rng.integers(4, 16))
    palette_size = int(rng.integers(2, 6))
    palette = rng.choice(NON_BLACK_COLORS, size=palette_size, replace=False).astype(int)

    seed = _sample_seed(rng, qh, qw, palette)
    pristine = _build_mirror_tiling(seed)

    min_h = 2 if qh <= 4 else 3
    min_w = 2 if qw <= 4 else 3
    max_h = min(qh, 8)
    max_w = min(qw, 8)
    sh = int(rng.integers(min_h, max_h + 1))
    sw = int(rng.integers(min_w, max_w + 1))

    quadrant = QUADRANTS[int(rng.integers(0, len(QUADRANTS)))]
    base_r = quadrant[0] * qh
    base_c = quadrant[1] * qw
    top = base_r + int(rng.integers(0, qh - sh + 1))
    left = base_c + int(rng.integers(0, qw - sw + 1))

    sprite_colors = rng.choice(palette, size=min(len(palette), int(rng.integers(2, min(4, len(palette)) + 1))), replace=False)
    sprite = _sample_sprite_pattern(rng, sh, sw, sprite_colors)
    underlying = pristine[top : top + sh, left : left + sw]
    sprite = _force_full_bbox_residue(rng, sprite, underlying, palette)

    inp = pristine.copy()
    inp[top : top + sh, left : left + sw] = sprite

    expected = _oracle_output(inp, pristine)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-intruder input")
