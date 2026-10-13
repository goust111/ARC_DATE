from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEED = 5
SEPARATOR = 9

HORIZONTAL_ONLY = "horizontal"
VERTICAL_ONLY = "vertical"
BOTH_AXES = "both"
NO_AXIS = "none"

COLOR_BY_KIND = {
    HORIZONTAL_ONLY: BLUE,
    VERTICAL_ONLY: RED,
    BOTH_AXES: GREEN,
    NO_AXIS: YELLOW,
}

LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
TILE_SIZE_CHOICES = (3, 4)
REPEAT_CHOICES = (2, 3)
SYMMETRY_KINDS = (HORIZONTAL_ONLY, VERTICAL_ONLY, BOTH_AXES, NO_AXIS)
CORNERS = ((False, False), (False, True), (True, False), (True, True))

MASK_ATTEMPTS = 200
BAND_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in map(int, np.unique(grid)):
        if color == BACKGROUND:
            continue
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color spanning full rows and columns"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _corner_from_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[bool, bool]:
    r0, c0, r1, c1 = bbox
    height, width = map(int, shape)
    if r0 == 0:
        flip_ud = False
    else:
        assert r1 == height - 1, "seed bbox must touch the top or bottom edge of its region"
        flip_ud = True

    if c0 == 0:
        flip_lr = False
    else:
        assert c1 == width - 1, "seed bbox must touch the left or right edge of its region"
        flip_lr = True
    return flip_ud, flip_lr


def _orient_to_top_left(arr: np.ndarray, corner: tuple[bool, bool]) -> np.ndarray:
    out = arr
    if corner[0]:
        out = np.flipud(out)
    if corner[1]:
        out = np.fliplr(out)
    return out.copy()


def _classify_mask(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    if horizontal and vertical:
        return BOTH_AXES
    if horizontal:
        return HORIZONTAL_ONLY
    if vertical:
        return VERTICAL_ONLY
    return NO_AXIS


def _fill_region_from_seed(
    seed_tile: np.ndarray, fill_color: int, reps_r: int, reps_c: int, seed_value: int
) -> np.ndarray:
    tile_h, tile_w = map(int, seed_tile.shape)
    filled = np.full((tile_h * reps_r, tile_w * reps_c), BACKGROUND, dtype=int)
    seed_mask = seed_tile == int(seed_value)

    for rr in range(reps_r):
        for cc in range(reps_c):
            tile_mask = seed_mask
            if rr % 2 == 1:
                tile_mask = np.flipud(tile_mask)
            if cc % 2 == 1:
                tile_mask = np.fliplr(tile_mask)
            patch = filled[rr * tile_h : (rr + 1) * tile_h, cc * tile_w : (cc + 1) * tile_w]
            patch[tile_mask] = int(fill_color)
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-tile each corner seed across separator-defined subregions and recolor it by symmetry class.

    Input structure:
    - Black background pixels are split by one-cell-thick full maroon rows and columns into rectangular subregions.
    - Every subregion contains exactly one grey seed pattern, and the seed's tight bounding box is anchored in one corner of that subregion.
    - The seed bounding-box height and width each divide the enclosing subregion size exactly.

    Rule:
    - For each subregion, crop the grey seed bbox and classify it as horizontal-only symmetric, vertical-only symmetric, symmetric on both axes, or non-symmetric.
    - Map those four classes to Blue, Red, Green, and Yellow respectively.
    - Starting from the corner seed tile, fill the whole subregion with alternating left-right and up-down mirror copies of that tile; recolor former grey cells to the class color and keep tile background black.
    - Preserve the maroon separator lines and the overall grid size.

    Key insight:
    - The seed is both a symmetry label and the primitive tile of a reflection tessellation: once the corner bbox is found, tile-row and tile-column parity determine which flips reproduce the full region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _find_separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected multiple subregions in both directions"

    payload_colors = set(map(int, np.unique(g))) - {BACKGROUND, int(separator)}
    assert len(payload_colors) == 1, "expected exactly one seed color besides background and separators"
    seed_color = int(next(iter(payload_colors)))

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            assert set(map(int, np.unique(region))).issubset({BACKGROUND, seed_color}), "region contains unexpected colors"

            seed_coords = np.argwhere(region == seed_color)
            assert len(seed_coords) > 0, "each region must contain one seed pattern"
            bbox = _bbox(seed_coords)
            corner = _corner_from_bbox(bbox, region.shape)
            norm_region = _orient_to_top_left(region, corner)

            tile_h = bbox[2] - bbox[0] + 1
            tile_w = bbox[3] - bbox[1] + 1
            region_h, region_w = map(int, region.shape)
            assert tile_h < region_h and tile_w < region_w, "seed must be strictly smaller than its region"
            assert region_h % tile_h == 0 and region_w % tile_w == 0, "seed tile must divide region size"

            seed_tile = norm_region[:tile_h, :tile_w].copy()
            assert np.any(seed_tile == seed_color), "seed tile cannot be empty"
            assert not np.any(norm_region[tile_h:, :] == seed_color), "seed cannot extend below its inferred tile"
            assert not np.any(norm_region[:, tile_w:] == seed_color), "seed cannot extend right of its inferred tile"

            kind = _classify_mask(seed_tile == seed_color)
            fill_color = COLOR_BY_KIND[kind]
            filled_norm = _fill_region_from_seed(
                seed_tile, fill_color, region_h // tile_h, region_w // tile_w, seed_color
            )
            out[r0 : r1 + 1, c0 : c1 + 1] = _orient_to_top_left(filled_norm, corner)

    return out


def _mirror_left_half(left: np.ndarray, full_width: int) -> np.ndarray:
    if full_width % 2 == 0:
        return np.hstack([left, np.fliplr(left)])
    return np.hstack([left, np.fliplr(left[:, :-1])])


def _mirror_top_half(top: np.ndarray, full_height: int) -> np.ndarray:
    if full_height % 2 == 0:
        return np.vstack([top, np.flipud(top)])
    return np.vstack([top, np.flipud(top[:-1, :])])


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_mask(height: int, width: int, kind: str) -> np.ndarray:
    min_area = max(3, (height * width) // 3)
    for _ in range(MASK_ATTEMPTS):
        fill_p = random.uniform(0.35, 0.7)
        if kind == BOTH_AXES:
            quarter = np.random.random(((height + 1) // 2, (width + 1) // 2)) < fill_p
            mask = _mirror_top_half(_mirror_left_half(quarter, width), height)
        elif kind == HORIZONTAL_ONLY:
            top = np.random.random(((height + 1) // 2, width)) < fill_p
            mask = _mirror_top_half(top, height)
        elif kind == VERTICAL_ONLY:
            left = np.random.random((height, (width + 1) // 2)) < fill_p
            mask = _mirror_left_half(left, width)
        else:
            mask = np.random.random((height, width)) < fill_p

        area = int(np.count_nonzero(mask))
        if area < min_area or area >= height * width:
            continue
        if not _touches_all_sides(mask):
            continue
        if _classify_mask(mask) != kind:
            continue
        return mask
    raise RuntimeError(f"failed to create a {kind} seed mask for {height}x{width}")


def _sample_band_spec(count: int) -> tuple[list[int], list[int], list[int]] | None:
    for _ in range(BAND_ATTEMPTS):
        tile_sizes = [random.choice(TILE_SIZE_CHOICES) for _ in range(count)]
        repeats = [random.choice(REPEAT_CHOICES) for _ in range(count)]
        sizes = [tile * repeat for tile, repeat in zip(tile_sizes, repeats)]
        if sum(sizes) + count - 1 <= 30:
            return tile_sizes, repeats, sizes
    return None


def _band_starts(sizes: list[int]) -> list[int]:
    starts = []
    pos = 0
    for size in sizes:
        starts.append(pos)
        pos += int(size) + 1
    return starts


def _render_input_region(region_h: int, region_w: int, mask: np.ndarray, corner: tuple[bool, bool]) -> np.ndarray:
    tile_h, tile_w = map(int, mask.shape)
    region = np.full((region_h, region_w), BACKGROUND, dtype=int)
    tile = region[:tile_h, :tile_w]
    tile[mask] = SEED
    return _orient_to_top_left(region, corner)


def _render_output_region(region_h: int, region_w: int, mask: np.ndarray, corner: tuple[bool, bool], kind: str) -> np.ndarray:
    seed_tile = np.full(mask.shape, BACKGROUND, dtype=int)
    seed_tile[mask] = SEED
    filled = _fill_region_from_seed(
        seed_tile, COLOR_BY_KIND[kind], region_h // mask.shape[0], region_w // mask.shape[1], SEED
    )
    return _orient_to_top_left(filled, corner)


def _render_pair(
    row_tiles: list[int],
    row_sizes: list[int],
    col_tiles: list[int],
    col_sizes: list[int],
    kinds: dict[tuple[int, int], str],
    corners: dict[tuple[int, int], tuple[bool, bool]],
) -> tuple[np.ndarray, np.ndarray]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)

    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)
    for idx, start in enumerate(row_starts[:-1]):
        inp[start + row_sizes[idx], :] = SEPARATOR
        out[start + row_sizes[idx], :] = SEPARATOR
    for idx, start in enumerate(col_starts[:-1]):
        inp[:, start + col_sizes[idx]] = SEPARATOR
        out[:, start + col_sizes[idx]] = SEPARATOR

    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            region_h = row_sizes[rr]
            region_w = col_sizes[cc]
            mask = _make_mask(row_tiles[rr], col_tiles[cc], kinds[(rr, cc)])
            corner = corners[(rr, cc)]
            inp[r0 : r0 + region_h, c0 : c0 + region_w] = _render_input_region(region_h, region_w, mask, corner)
            out[r0 : r0 + region_h, c0 : c0 + region_w] = _render_output_region(
                region_h, region_w, mask, corner, kinds[(rr, cc)]
            )
    return inp, out


def _assign_kinds(n_rows: int, n_cols: int) -> dict[tuple[int, int], str]:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    random.shuffle(cells)
    kinds = list(SYMMETRY_KINDS)
    random.shuffle(kinds)
    assigned: dict[tuple[int, int], str] = {}
    for cell, kind in zip(cells, kinds):
        assigned[cell] = kind
    for cell in cells[len(kinds) :]:
        assigned[cell] = random.choice(SYMMETRY_KINDS)
    return assigned


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    row_spec = _sample_band_spec(n_rows)
    col_spec = _sample_band_spec(n_cols)
    if row_spec is None or col_spec is None:
        return None

    row_tiles, _row_repeats, row_sizes = row_spec
    col_tiles, _col_repeats, col_sizes = col_spec
    kinds = _assign_kinds(n_rows, n_cols)
    corners = {(r, c): random.choice(CORNERS) for r in range(n_rows) for c in range(n_cols)}

    inp, expected = _render_pair(row_tiles, row_sizes, col_tiles, col_sizes, kinds, corners)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed168776016/2/4 input")
