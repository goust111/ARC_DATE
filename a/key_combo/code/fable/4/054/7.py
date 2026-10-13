from __future__ import annotations

import numpy as np


OCCLUDER = 8
SOURCE_COLORS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _mirror_tile(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _upscale2(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _collapse_2x2_blocks(grid: np.ndarray) -> np.ndarray:
    height, width = map(int, grid.shape)
    assert height % 2 == 0 and width % 2 == 0

    out = np.full((height // 2, width // 2), OCCLUDER, dtype=int)
    for r in range(0, height, 2):
        for c in range(0, width, 2):
            block = grid[r : r + 2, c : c + 2]
            visible = np.unique(block[block != OCCLUDER])
            assert len(visible) <= 1, "non-occluded cells inside a 2x2 block must agree"
            if len(visible) == 1:
                out[r // 2, c // 2] = int(visible[0])
    return out


def _quadrant_positions(height: int, width: int, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r, c),
        (r, 2 * width - 1 - c),
        (2 * height - 1 - r, c),
        (2 * height - 1 - r, 2 * width - 1 - c),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the original small picture from a mirrored, 2x-upscaled, partially occluded redundancy stack.

    Input structure:
    - The grid height and width are multiples of four.
    - A hidden source image was mirrored rightward, then mirrored downward to make four linked quadrants.
    - Every tile pixel was expanded into a 2x2 constant-color block, and some cells were overwritten by gray 8.

    Rule:
    - Collapse each 2x2 block back to one pixel by taking its unique visible non-gray color, or gray if the whole block is occluded.
    - For every position in the upper-left quarter, gather its four mirror-linked copies across the coarse 2x2 quadrant layout.
    - The visible copies must agree; output that recovered color in the original upper-left source size.

    Key insight:
    - The source never appears directly: first undo the local 2x enlargement, then merge the four mirrored coarse copies so gray damage in one quadrant is repaired by another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 4 == 0 and width % 4 == 0, "grid dimensions must be multiples of 4"

    coarse = _collapse_2x2_blocks(g)
    coarse_h, coarse_w = map(int, coarse.shape)
    src_h, src_w = coarse_h // 2, coarse_w // 2
    out = np.empty((src_h, src_w), dtype=int)

    for r in range(src_h):
        for c in range(src_w):
            positions = _quadrant_positions(src_h, src_w, r, c)
            visible = {int(coarse[rr, cc]) for rr, cc in positions if int(coarse[rr, cc]) != OCCLUDER}
            assert visible, "every source pixel must survive in at least one mirrored copy"
            assert len(visible) == 1, "mirrored copies must agree on the visible color"
            out[r, c] = visible.pop()

    return out


def _random_nonempty_mask() -> np.ndarray:
    visible_count = int(_RNG.choice(np.array([1, 2, 3, 4], dtype=int), p=[0.22, 0.30, 0.28, 0.20]))
    cells = _RNG.choice(4, size=visible_count, replace=False)
    mask = np.zeros((2, 2), dtype=bool)
    for cell in cells:
        mask[cell // 2, cell % 2] = True
    return mask


def _sample_block_mask() -> np.ndarray:
    visible_count = int(_RNG.choice(np.array([0, 1, 2, 3, 4], dtype=int), p=[0.18, 0.18, 0.24, 0.22, 0.18]))
    if visible_count == 0:
        return np.zeros((2, 2), dtype=bool)
    cells = _RNG.choice(4, size=visible_count, replace=False)
    mask = np.zeros((2, 2), dtype=bool)
    for cell in cells:
        mask[cell // 2, cell % 2] = True
    return mask


def _apply_occlusion(scaled: np.ndarray, src_h: int, src_w: int) -> np.ndarray:
    out = scaled.copy()
    masks = np.zeros((src_h, src_w, 4, 2, 2), dtype=bool)

    for r in range(src_h):
        for c in range(src_w):
            for q in range(4):
                masks[r, c, q] = _sample_block_mask()
            if not masks[r, c].any():
                masks[r, c, int(_RNG.integers(4))] = _random_nonempty_mask()

    groups = [(r, c) for r in range(src_h) for c in range(src_w)]
    forced_groups = _RNG.choice(len(groups), size=4, replace=False)
    for q, group_idx in enumerate(forced_groups):
        r, c = groups[int(group_idx)]
        masks[r, c, q] = False
        if not masks[r, c].any():
            repair_q = (q + 1 + int(_RNG.integers(3))) % 4
            masks[r, c, repair_q] = _random_nonempty_mask()

    fusion_needed = False
    for r in range(src_h):
        for c in range(src_w):
            visible_blocks = [bool(masks[r, c, q].any()) for q in range(4)]
            if 0 < sum(visible_blocks) < 4:
                fusion_needed = True
                break
        if fusion_needed:
            break
    if not fusion_needed:
        r, c = groups[int(_RNG.integers(len(groups)))]
        masks[r, c, int(_RNG.integers(4))] = False
        if not masks[r, c].any():
            masks[r, c, int((_RNG.integers(3) + 1) % 4)] = _random_nonempty_mask()

    for r in range(src_h):
        for c in range(src_w):
            positions = _quadrant_positions(src_h, src_w, r, c)
            for q, (rr, cc) in enumerate(positions):
                block = out[2 * rr : 2 * rr + 2, 2 * cc : 2 * cc + 2]
                block[~masks[r, c, q]] = OCCLUDER

    return out


def _place_required_colors(grid: np.ndarray, palette: np.ndarray) -> None:
    flat = grid.reshape(-1)
    cells = _RNG.choice(flat.size, size=len(palette), replace=False)
    for idx, color in zip(cells, palette):
        flat[int(idx)] = int(color)


def _inject_asymmetry(grid: np.ndarray, palette: np.ndarray) -> None:
    height, width = map(int, grid.shape)
    a, b = map(int, _RNG.choice(palette, size=2, replace=False))

    lr_pairs = [((r, c), (r, width - 1 - c)) for r in range(height) for c in range(width // 2)]
    ud_pairs = [((r, c), (height - 1 - r, c)) for r in range(height // 2) for c in range(width)]
    disjoint = [
        (lr_pair, ud_pair)
        for lr_pair in lr_pairs
        for ud_pair in ud_pairs
        if {lr_pair[0], lr_pair[1]}.isdisjoint({ud_pair[0], ud_pair[1]})
    ]

    if disjoint:
        lr_pair, ud_pair = disjoint[int(_RNG.integers(len(disjoint)))]
    else:
        lr_pair = lr_pairs[int(_RNG.integers(len(lr_pairs)))]
        ud_pair = ud_pairs[int(_RNG.integers(len(ud_pairs)))]

    (r0, c0), (r1, c1) = lr_pair
    grid[r0, c0] = a
    grid[r1, c1] = b

    (r2, c2), (r3, c3) = ud_pair
    grid[r2, c2] = b
    grid[r3, c3] = a


def _make_mosaic(height: int, width: int, palette: np.ndarray) -> np.ndarray:
    grid = _RNG.choice(palette, size=(height, width), replace=True).astype(int)
    _place_required_colors(grid, palette)
    _inject_asymmetry(grid, palette)
    return grid


def _make_rectangles(height: int, width: int, palette: np.ndarray) -> np.ndarray:
    grid = np.full((height, width), int(_RNG.choice(palette)), dtype=int)
    rect_count = int(_RNG.integers(2, min(6, height * width + 1)))
    for _ in range(rect_count):
        r0 = int(_RNG.integers(height))
        r1 = int(_RNG.integers(r0, height))
        c0 = int(_RNG.integers(width))
        c1 = int(_RNG.integers(c0, width))
        color = int(_RNG.choice(palette))
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    accent_count = int(_RNG.integers(1, 4))
    for _ in range(accent_count):
        r = int(_RNG.integers(height))
        c = int(_RNG.integers(width))
        grid[r, c] = int(_RNG.choice(palette))

    _place_required_colors(grid, palette)
    _inject_asymmetry(grid, palette)
    return grid


def _make_bands(height: int, width: int, palette: np.ndarray) -> np.ndarray:
    grid = np.empty((height, width), dtype=int)
    if bool(_RNG.integers(2)):
        for r in range(height):
            grid[r, :] = int(_RNG.choice(palette))
    else:
        for c in range(width):
            grid[:, c] = int(_RNG.choice(palette))

    patch_count = int(_RNG.integers(1, 4))
    for _ in range(patch_count):
        r0 = int(_RNG.integers(height))
        r1 = int(_RNG.integers(r0, height))
        c0 = int(_RNG.integers(width))
        c1 = int(_RNG.integers(c0, width))
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(_RNG.choice(palette))

    _place_required_colors(grid, palette)
    _inject_asymmetry(grid, palette)
    return grid


def _original_is_good(grid: np.ndarray) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 2:
        return False
    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return False
    if int(counts.max()) > int(np.ceil(grid.size * 0.85)):
        return False
    return True


def _sample_original() -> np.ndarray:
    makers = (_make_mosaic, _make_rectangles, _make_bands)
    for _ in range(200):
        height = int(_RNG.integers(2, 8))
        width = int(_RNG.integers(2, 8))
        palette_size = int(_RNG.integers(2, min(6, height * width, len(SOURCE_COLORS)) + 1))
        palette = _RNG.choice(SOURCE_COLORS, size=palette_size, replace=False).astype(int)
        maker = makers[int(_RNG.integers(len(makers)))]
        grid = maker(height, width, palette)
        if _original_is_good(grid):
            return grid
    raise RuntimeError("failed to sample a valid source pattern")


def _gen_attempt() -> np.ndarray | None:
    original = _sample_original()
    mirrored = _mirror_tile(original)
    scaled = _upscale2(mirrored)
    occluded = _apply_occlusion(scaled, *original.shape)

    if not (occluded.shape[0] <= 30 and occluded.shape[1] <= 30):
        return None
    if OCCLUDER not in set(map(int, np.unique(occluded))):
        return None
    if not np.array_equal(main(occluded), original):
        return None
    return occluded


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
