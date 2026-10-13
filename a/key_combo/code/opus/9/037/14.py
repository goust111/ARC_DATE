from __future__ import annotations

import random

import numpy as np


OCCLUDER = 6
BG = 0
SCENE_COLORS = (1, 2, 4, 5, 7, 8, 9)
QUADRANT_NAMES = ("TL", "TR", "BL", "BR")
MAX_GENERATION_ATTEMPTS = 400

_FOREGROUND_BLOCKS = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 1)),
)


def _quadrant_slices(size: int) -> tuple[tuple[slice, slice], ...]:
    half = size // 2
    return (
        (slice(0, half), slice(0, half)),
        (slice(0, half), slice(half, size)),
        (slice(half, size), slice(0, half)),
        (slice(half, size), slice(half, size)),
    )


def _extract_occluder_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == OCCLUDER)
    assert coords.size > 0, "expected a magenta occluder"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h in (3, 4), "occluder must be a 3x3 or 4x4 square"
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == OCCLUDER), "occluder must be a solid square"
    assert int(coords.shape[0]) == h * w, "occluder color must occur only inside the square"
    return int(r0), int(c0), int(r1), int(c1)


def _damaged_quadrant_index(size: int, bbox: tuple[int, int, int, int]) -> int:
    half = size // 2
    r0, c0, _r1, _c1 = bbox
    row_band = 0 if r0 < half else 1
    col_band = 0 if c0 < half else 1
    return row_band * 2 + col_band


def _restore_scene(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even-sized square grid"

    bbox = _extract_occluder_bbox(g)
    damaged = _damaged_quadrant_index(h, bbox)
    slices = _quadrant_slices(h)
    damaged_quad = g[slices[damaged]].copy()
    occluder_mask = damaged_quad == OCCLUDER
    assert np.any(occluder_mask), "occluder must lie within a single quadrant"

    row_band, col_band = divmod(damaged, 2)
    vertical_source = row_band * 2 + (1 - col_band)
    horizontal_source = (1 - row_band) * 2 + col_band
    diagonal_source = (1 - row_band) * 2 + (1 - col_band)

    source_order = (
        (vertical_source, np.fliplr),
        (horizontal_source, np.flipud),
        (diagonal_source, lambda x: np.rot90(x, 2)),
    )

    repaired_quad = None
    for source_idx, transform in source_order:
        source_quad = g[slices[source_idx]].copy()
        if np.any(source_quad == OCCLUDER):
            continue
        repaired_quad = transform(source_quad)
        break
    assert repaired_quad is not None, "expected at least one undamaged symmetric source quadrant"

    restored = g.copy()
    target = restored[slices[damaged]]
    target[occluder_mask] = repaired_quad[occluder_mask]
    return restored


def _downsample_top_left(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int)[::2, ::2].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the occluded quadrant by mirror symmetry, then output the whole scene shrunk by 2 using 2x2 top-left sampling.

    Input structure:
    - The input is an even-sized square scene split into four equal quadrants by the central horizontal and vertical axes.
    - The undamaged scene is symmetric across both center axes, but one quadrant contains a solid magenta 3x3 or 4x4 occluder wholly inside that quadrant.
    - Normal scene colors are a small foreground palette on black background, and magenta appears only in the occluder.

    Rule:
    - Locate the magenta square and use it to determine the damaged quadrant.
    - Restore the hidden cells from the left-right mirror quadrant when available; otherwise use the top-bottom mirror quadrant, applying the matching flip before copying the occluded pixels back.
    - After the full scene is repaired, shrink it by a factor of 2 by taking the top-left color from every 2x2 block, which preserves the four quadrant positions in the final smaller board.

    Key insight:
    - The occlusion repair is only an intermediate step: the answer is the repaired whole scene after a structured downsampling, so both symmetry recovery and the final 2x2-to-1 reduction are necessary.
    """
    restored = _restore_scene(grid)
    return _downsample_top_left(restored)


def _paint_shape(seed: np.ndarray, color: int) -> None:
    s = seed.shape[0]
    kind = random.choice(("rect", "hline", "vline", "l", "point"))
    if kind == "rect":
        h = random.randint(1, min(3, s))
        w = random.randint(1, min(3, s))
        r0 = random.randint(0, s - h)
        c0 = random.randint(0, s - w)
        seed[r0 : r0 + h, c0 : c0 + w] = color
    elif kind == "hline":
        w = random.randint(2, s)
        r = random.randint(0, s - 1)
        c0 = random.randint(0, s - w)
        seed[r, c0 : c0 + w] = color
    elif kind == "vline":
        h = random.randint(2, s)
        r0 = random.randint(0, s - h)
        c = random.randint(0, s - 1)
        seed[r0 : r0 + h, c] = color
    elif kind == "l":
        h = random.randint(2, min(4, s))
        w = random.randint(2, min(4, s))
        r0 = random.randint(0, s - h)
        c0 = random.randint(0, s - w)
        orientation = random.choice(("tl", "tr", "bl", "br"))
        if orientation == "tl":
            seed[r0, c0 : c0 + w] = color
            seed[r0 : r0 + h, c0] = color
        elif orientation == "tr":
            seed[r0, c0 : c0 + w] = color
            seed[r0 : r0 + h, c0 + w - 1] = color
        elif orientation == "bl":
            seed[r0 + h - 1, c0 : c0 + w] = color
            seed[r0 : r0 + h, c0] = color
        else:
            seed[r0 + h - 1, c0 : c0 + w] = color
            seed[r0 : r0 + h, c0 + w - 1] = color
    else:
        seed[random.randint(0, s - 1), random.randint(0, s - 1)] = color


def _make_seed(size: int, colors: list[int]) -> np.ndarray:
    for _ in range(120):
        seed = np.full((size, size), BG, dtype=int)
        for _ in range(random.randint(size + 1, size + 4)):
            _paint_shape(seed, random.choice(colors))
        for color in colors:
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)
            seed[r, c] = color

        if not np.any(seed == BG):
            seed[random.randint(0, size - 1), random.randint(0, size - 1)] = BG
        if np.count_nonzero(seed != BG) < size + 1:
            continue
        if np.array_equal(seed, np.fliplr(seed)) or np.array_equal(seed, np.flipud(seed)):
            continue
        if np.array_equal(seed, np.rot90(seed, 2)):
            continue
        if any(np.count_nonzero(seed == color) < 2 for color in colors):
            continue
        if len({tuple(row.tolist()) for row in seed}) < 2:
            continue
        return seed
    raise RuntimeError("failed to build a non-degenerate coarse seed")


def _expand_seed(seed: np.ndarray) -> np.ndarray:
    size = seed.shape[0]
    quad = np.full((size * 2, size * 2), BG, dtype=int)
    for r in range(size):
        for c in range(size):
            value = int(seed[r, c])
            block = np.full((2, 2), BG, dtype=int)
            if value != BG:
                for dr, dc in random.choice(_FOREGROUND_BLOCKS):
                    block[dr, dc] = value
            quad[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = block
    return quad


def _build_full_scene(quad: np.ndarray) -> np.ndarray:
    top = np.concatenate((quad, np.fliplr(quad)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _place_occluder(
    full_scene: np.ndarray,
    damaged: int,
    quad_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    candidate_sizes = [3, 4]
    random.shuffle(candidate_sizes)
    quadrant = full_scene[_quadrant_slices(full_scene.shape[0])[damaged]]

    for occ_size in candidate_sizes:
        starts = []
        for r0 in range(1, quad_size - occ_size):
            for c0 in range(1, quad_size - occ_size):
                rows = [r for r in range(r0, r0 + occ_size) if r % 2 == 0]
                cols = [c for c in range(c0, c0 + occ_size) if c % 2 == 0]
                if len(rows) * len(cols) < 2:
                    continue
                sampled_values = quadrant[np.ix_(rows, cols)]
                if not np.any(sampled_values != BG):
                    continue
                starts.append((r0, c0))
        if not starts:
            continue
        local_r0, local_c0 = random.choice(starts)
        row_band, col_band = divmod(damaged, 2)
        global_r0 = row_band * quad_size + local_r0
        global_c0 = col_band * quad_size + local_c0
        grid = full_scene.copy()
        grid[global_r0 : global_r0 + occ_size, global_c0 : global_c0 + occ_size] = OCCLUDER
        return grid, (global_r0, global_c0, global_r0 + occ_size - 1, global_c0 + occ_size - 1)
    raise RuntimeError("failed to place a useful occluder")


def _semantic_checks(
    inp: np.ndarray,
    clean_scene: np.ndarray,
    damaged: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    n = inp.shape[0]
    quad_size = n // 2
    out = main(inp)
    expected = _downsample_top_left(clean_scene)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(_downsample_top_left(inp), expected):
        return False
    if not np.array_equal(clean_scene, np.fliplr(clean_scene)):
        return False
    if not np.array_equal(clean_scene, np.flipud(clean_scene)):
        return False

    slices = _quadrant_slices(n)
    bbox2 = _extract_occluder_bbox(inp)
    if bbox2 != bbox:
        return False
    if _damaged_quadrant_index(n, bbox) != damaged:
        return False

    clean_quads = [clean_scene[sl].copy() for sl in slices]
    coarse_expected = expected[: quad_size // 2, : quad_size // 2]
    if np.array_equal(coarse_expected, np.fliplr(coarse_expected)):
        return False
    if np.array_equal(coarse_expected, np.flipud(coarse_expected)):
        return False

    row_band, col_band = divmod(damaged, 2)
    source_idx = row_band * 2 + (1 - col_band)
    damaged_quad = inp[slices[damaged]].copy()
    source_quad = inp[slices[source_idx]].copy()
    local_mask = damaged_quad == OCCLUDER
    wrong_restored = inp.copy()
    wrong_quad = wrong_restored[slices[damaged]]
    wrong_quad[local_mask] = source_quad[local_mask]
    if np.array_equal(_downsample_top_left(wrong_restored), expected):
        return False

    transformed = np.fliplr(source_quad)
    if not np.array_equal(clean_quads[damaged], transformed):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    coarse_size = random.randint(3, 6)
    color_count = random.randint(2, 3)
    colors = random.sample(SCENE_COLORS, color_count)
    seed = _make_seed(coarse_size, colors)
    quad = _expand_seed(seed)
    if np.array_equal(quad, np.fliplr(quad)) or np.array_equal(quad, np.flipud(quad)):
        return None

    clean_scene = _build_full_scene(quad)
    damaged = random.randint(0, 3)
    try:
        inp, bbox = _place_occluder(clean_scene, damaged, quad.shape[0])
    except RuntimeError:
        return None

    if not _semantic_checks(inp, clean_scene, damaged, bbox):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
