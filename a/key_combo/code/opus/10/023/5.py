from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
TILE_COLORS = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _extract_instance(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("expected a non-empty 2D grid")

    blue_mask = g == BLUE
    tile_mask = (g != BLACK) & (g != BLUE)
    if not blue_mask.any() or not tile_mask.any():
        raise ValueError("expected one blue scaled pattern and one non-blue tile")

    tile = _crop(g, _bbox_from_mask(tile_mask))
    kh, kw = map(int, tile.shape)
    if kh != kw or kh < 2:
        raise ValueError("expected a square tile of side >= 2")
    k = kh

    pattern = _crop(g, _bbox_from_mask(blue_mask))
    ph, pw = map(int, pattern.shape)
    if ph % k != 0 or pw % k != 0:
        raise ValueError("blue pattern bbox must be divisible by the tile size")

    control = np.zeros((ph // k, pw // k), dtype=int)
    for r in range(control.shape[0]):
        for c in range(control.shape[1]):
            block = pattern[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if np.all(block == BLUE):
                control[r, c] = 1
            elif np.all(block == BLACK):
                control[r, c] = 0
            else:
                raise ValueError("each macro-block in the blue pattern must be uniform")
    return tile, control, k


def _render_output(tile: np.ndarray, control: np.ndarray) -> np.ndarray:
    k = int(tile.shape[0])
    ch, cw = map(int, control.shape)
    out = np.zeros((ch * k, cw * k), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(control[r, c]) != 0:
                out[r * k : (r + 1) * k, c * k : (c + 1) * k] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden control sprite by downsampling a large blue pattern, then stamp a multicolor tile into the selected blocks.

    Input structure:
    - The background is black.
    - One object is a large blue pattern whose bbox is partitioned into uniform k x k macro-blocks.
    - Another object is a small multicolor square tile of size k x k using non-blue foreground colors.

    Rule:
    - Crop the blue pattern and the multicolor square tile.
    - Let k be the tile side length, and downsample the blue pattern bbox by reading each k x k macro-block: all-blue becomes a filled control pixel, all-black becomes an empty control pixel.
    - Create an output grid with the same size as the blue pattern bbox, and for every filled control pixel stamp one full copy of the tile into the corresponding k x k output block; empty control pixels leave black blocks.

    Key insight:
    - The large blue object is not part of the answer; it is a scaled-up encoding of a smaller placement mask, and the tile size is both the downsampling factor and the stamp block size.
    """
    tile, control, _k = _extract_instance(grid)
    return _render_output(tile, control)


def _make_tile(rng: np.random.Generator, k: int) -> np.ndarray:
    color_count = int(rng.integers(3, min(5, len(TILE_COLORS)) + 1))
    palette = rng.choice(np.array(TILE_COLORS, dtype=int), size=color_count, replace=False)
    tile = rng.choice(palette, size=(k, k), replace=True).astype(int)

    for color in palette:
        if not np.any(tile == int(color)):
            rr = int(rng.integers(0, k))
            cc = int(rng.integers(0, k))
            tile[rr, cc] = int(color)

    if len(np.unique(tile)) < 3:
        flat = tile.reshape(-1)
        flat[:3] = palette[:3]
        rng.shuffle(flat)
        tile = flat.reshape(k, k)

    if np.all(tile == tile[0, 0]):
        tile[0, 0] = int(palette[0])
        tile[-1, -1] = int(palette[-1])

    row_variety = len({tuple(map(int, row)) for row in tile}) >= 2
    col_variety = len({tuple(map(int, col)) for col in tile.T}) >= 2
    if not row_variety:
        tile[-1] = np.roll(tile[-1], 1)
    if not col_variety:
        tile[:, -1] = np.roll(tile[:, -1], 1)
    return tile


def _make_control(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    area = h * w
    for _ in range(200):
        mask = rng.random((h, w)) < rng.uniform(0.38, 0.62)
        if int(mask.sum()) < 3 or int((~mask).sum()) < 3:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not np.any(mask[:, 1:] != mask[:, :-1]):
            continue
        if not np.any(mask[1:, :] != mask[:-1, :]):
            continue
        if len({tuple(map(int, row)) for row in mask.astype(int)}) < 2:
            continue
        if len({tuple(map(int, col)) for col in mask.astype(int).T}) < 2:
            continue
        return mask.astype(int)
    raise RuntimeError("failed to generate a non-degenerate control mask")


def _place_objects(
    rng: np.random.Generator, pattern: np.ndarray, tile: np.ndarray
) -> np.ndarray:
    ph, pw = map(int, pattern.shape)
    k = int(tile.shape[0])
    gap = int(rng.integers(1, 3))
    border = int(rng.integers(0, 2))
    horizontal = bool(rng.integers(0, 2))
    pattern_first = bool(rng.integers(0, 2))

    if horizontal:
        base_h = max(ph, k) + 2 * border
        base_w = pw + gap + k + 2 * border
        if base_h > 30 or base_w > 30:
            raise ValueError("horizontal layout exceeds ARC size limit")
        extra_h = int(rng.integers(0, min(5, 30 - base_h) + 1))
        extra_w = int(rng.integers(0, min(5, 30 - base_w) + 1))
        H = base_h + extra_h
        W = base_w + extra_w
        grid = np.zeros((H, W), dtype=int)

        left_margin = border + int(rng.integers(0, extra_w + 1))
        if pattern_first:
            p_c = left_margin
            t_c = p_c + pw + gap
        else:
            t_c = left_margin
            p_c = t_c + k + gap
        p_r = border + int(rng.integers(0, H - ph - 2 * border + 1))
        t_r = border + int(rng.integers(0, H - k - 2 * border + 1))
    else:
        base_h = ph + gap + k + 2 * border
        base_w = max(pw, k) + 2 * border
        if base_h > 30 or base_w > 30:
            raise ValueError("vertical layout exceeds ARC size limit")
        extra_h = int(rng.integers(0, min(5, 30 - base_h) + 1))
        extra_w = int(rng.integers(0, min(5, 30 - base_w) + 1))
        H = base_h + extra_h
        W = base_w + extra_w
        grid = np.zeros((H, W), dtype=int)

        top_margin = border + int(rng.integers(0, extra_h + 1))
        if pattern_first:
            p_r = top_margin
            t_r = p_r + ph + gap
        else:
            t_r = top_margin
            p_r = t_r + k + gap
        p_c = border + int(rng.integers(0, W - pw - 2 * border + 1))
        t_c = border + int(rng.integers(0, W - k - 2 * border + 1))

    grid[p_r : p_r + ph, p_c : p_c + pw] = pattern
    grid[t_r : t_r + k, t_c : t_c + k] = tile
    return grid


def _validate_generated(grid: np.ndarray, control: np.ndarray, tile: np.ndarray) -> None:
    parsed_tile, parsed_control, k = _extract_instance(grid)
    if not np.array_equal(parsed_tile, tile):
        raise AssertionError("tile parsing did not recover the generated tile")
    if not np.array_equal(parsed_control, control):
        raise AssertionError("downsampling did not recover the generated control")

    out = main(grid)
    expected = _render_output(tile, control)
    if not np.array_equal(out, expected):
        raise AssertionError("main() disagrees with the construction oracle")

    if out.shape != (control.shape[0] * k, control.shape[1] * k):
        raise AssertionError("output size must match the blue pattern bbox")

    for r in range(control.shape[0]):
        for c in range(control.shape[1]):
            block = out[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if int(control[r, c]) == 1:
                if not np.array_equal(block, tile):
                    raise AssertionError("filled control pixels must stamp the full tile")
            else:
                if np.any(block != BLACK):
                    raise AssertionError("empty control pixels must stay black")

    solid_color_shortcut = np.kron(control, np.full((k, k), int(tile[0, 0]), dtype=int))
    if np.array_equal(out, solid_color_shortcut):
        raise AssertionError("generator collapsed the stamp step into a solid-color shortcut")

    blue_mask_shortcut = np.kron(control, np.full((k, k), BLUE, dtype=int))
    if np.array_equal(out, blue_mask_shortcut):
        raise AssertionError("generator collapsed the result into the original blue pattern")

    if out.shape == tile.shape:
        raise AssertionError("generator collapsed the task into a single-tile masking puzzle")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        k = int(rng.integers(2, 6))
        max_dim = min(5, 20 // k)
        if max_dim < 2:
            continue

        h = int(rng.integers(2, max_dim + 1))
        w = int(rng.integers(2, max_dim + 1))
        if h * w < 6:
            continue
        if max(h, w) < 3:
            continue

        tile = _make_tile(rng, k)
        control = _make_control(rng, h, w)
        pattern = np.kron(control, np.full((k, k), BLUE, dtype=int))
        try:
            grid = _place_objects(rng, pattern, tile)
        except ValueError:
            continue
        _validate_generated(grid, control, tile)
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
