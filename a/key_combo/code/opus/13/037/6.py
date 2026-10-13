from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, label_mask, paste, random_free_location


BG = 0
H_BAR_COLOR = 2
V_BAR_COLOR = 3
SPRITE_COLOR = 8

SPRITE_SHAPES = ((3, 3), (3, 4))
COUNT_VALUES = (2, 3, 4)
GRID_MIN = 18
GRID_MAX = 28
RETRY_LIMIT = 400


def _line_length(component: dict) -> int:
    r0, c0, r1, c1 = component["bbox"]
    return max(r1 - r0 + 1, c1 - c0 + 1)


def _is_horizontal_bar(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return (
        int(component["color"]) == H_BAR_COLOR
        and r0 == r1
        and int(component["area"]) == c1 - c0 + 1
    )


def _is_vertical_bar(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return (
        int(component["color"]) == V_BAR_COLOR
        and c0 == c1
        and int(component["area"]) == r1 - r0 + 1
    )


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _all_orientations_distinct(mask: np.ndarray) -> bool:
    variants = [
        mask,
        np.fliplr(mask),
        np.flipud(mask),
        np.flipud(np.fliplr(mask)),
    ]
    return len({tuple(v.astype(int).ravel()) for v in variants}) == 4


def _extract_objects(grid: np.ndarray) -> tuple[int, dict, dict, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) == 3

    horizontal = [component for component in components if _is_horizontal_bar(component)]
    vertical = [component for component in components if _is_vertical_bar(component)]
    sprites = [component for component in components if int(component["color"]) == SPRITE_COLOR]

    assert len(horizontal) == 1
    assert len(vertical) == 1
    assert len(sprites) == 1

    sprite = sprites[0]
    sprite_patch = crop_bbox(g, sprite["bbox"])
    assert sprite_patch.shape in SPRITE_SHAPES
    sprite_mask = sprite_patch == SPRITE_COLOR
    _, n_labels = label_mask(sprite_mask, connectivity=4)
    assert n_labels == 1
    assert _touches_all_bbox_sides(sprite_mask)
    assert _all_orientations_distinct(sprite_mask)

    h_len = _line_length(horizontal[0])
    v_len = _line_length(vertical[0])
    assert h_len in COUNT_VALUES
    assert v_len in COUNT_VALUES
    return bg, sprite, horizontal[0], vertical[0]


def _render_tile_array(
    sprite_patch: np.ndarray,
    rows: int,
    cols: int,
    *,
    apply_row_flips: bool,
    apply_col_flips: bool,
    row_sep_color: int,
    col_sep_color: int,
    vertical_priority: bool,
    bg: int,
) -> np.ndarray:
    tile_h, tile_w = sprite_patch.shape
    out_h = rows * tile_h + (rows - 1)
    out_w = cols * tile_w + (cols - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for rr in range(rows):
        for cc in range(cols):
            tile = sprite_patch.copy()
            if apply_col_flips and (cc % 2 == 1):
                tile = np.fliplr(tile)
            if apply_row_flips and (rr % 2 == 1):
                tile = np.flipud(tile)
            paste(out, tile, rr * (tile_h + 1), cc * (tile_w + 1), transparent=bg)

    for cc in range(1, cols):
        sep_c = cc * (tile_w + 1) - 1
        out[:, sep_c] = col_sep_color

    if vertical_priority:
        for rr in range(1, rows):
            sep_r = rr * (tile_h + 1) - 1
            out[sep_r, :] = row_sep_color
    else:
        row_indices = [rr * (tile_h + 1) - 1 for rr in range(1, rows)]
        col_indices = [cc * (tile_w + 1) - 1 for cc in range(1, cols)]
        for rr in row_indices:
            out[rr, :] = row_sep_color
        for cc in col_indices:
            out[:, cc] = col_sep_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the control-bar lengths and build a biaxially mirrored sprite tiling with colored separators.

    Input structure
    The grid has a black background and exactly three non-overlapping 4-connected
    foreground objects: one cyan sprite with bbox 3x3 or 3x4, one red horizontal
    one-cell-thick control bar, and one green vertical one-cell-thick control bar.

    Rule
    Read the red bar length as the tile-column count C and the green bar length as
    the tile-row count R. Starting at the sprite bbox's top-left corner, place an
    R-by-C array of sprite copies, flipping odd columns left-right and odd rows
    up-down. Insert red separator columns between tile columns and green separator
    rows between tile rows, with green taking priority at separator crossings, and
    erase both control bars.

    Key insight
    Each control bar encodes two visible parameters at once: its geometric length
    gives a repetition count, while its color gives the separator color used along
    that axis.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprite, horizontal, vertical = _extract_objects(g)

    sprite_patch = crop_bbox(g, sprite["bbox"])
    rows = _line_length(vertical)
    cols = _line_length(horizontal)
    tiled = _render_tile_array(
        sprite_patch,
        rows,
        cols,
        apply_row_flips=True,
        apply_col_flips=True,
        row_sep_color=V_BAR_COLOR,
        col_sep_color=H_BAR_COLOR,
        vertical_priority=True,
        bg=bg,
    )

    out = np.full_like(g, bg)
    r0, c0, _r1, _c1 = sprite["bbox"]
    assert r0 + tiled.shape[0] <= g.shape[0]
    assert c0 + tiled.shape[1] <= g.shape[1]
    paste(out, tiled, r0, c0)
    return out


def _build_sprite_mask_pool() -> dict[tuple[int, int], list[np.ndarray]]:
    pools: dict[tuple[int, int], list[np.ndarray]] = {}
    for shape in SPRITE_SHAPES:
        h, w = shape
        valid: list[np.ndarray] = []
        for bits in range(1, 1 << (h * w)):
            mask = np.zeros((h, w), dtype=bool)
            for idx in range(h * w):
                if (bits >> idx) & 1:
                    mask[idx // w, idx % w] = True
            cells = int(mask.sum())
            if cells < 4 or cells >= h * w:
                continue
            if not _touches_all_bbox_sides(mask):
                continue
            _, n_labels = label_mask(mask, connectivity=4)
            if n_labels != 1:
                continue
            if not _all_orientations_distinct(mask):
                continue
            valid.append(mask)
        pools[shape] = valid
    return pools


_SPRITE_MASK_POOL = _build_sprite_mask_pool()


def _sprite_patch_from_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = SPRITE_COLOR
    return patch


def _stamp_forbidden(canvas: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    canvas[r0 : r0 + h, c0 : c0 + w] = 1


def _render_full_output(
    grid_shape: tuple[int, int],
    anchor: tuple[int, int],
    sprite_patch: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    tiled = _render_tile_array(
        sprite_patch,
        rows,
        cols,
        apply_row_flips=True,
        apply_col_flips=True,
        row_sep_color=V_BAR_COLOR,
        col_sep_color=H_BAR_COLOR,
        vertical_priority=True,
        bg=BG,
    )
    paste(out, tiled, anchor[0], anchor[1])
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, sprite, horizontal, vertical = _extract_objects(grid)
    except Exception:
        return False

    sprite_patch = crop_bbox(grid, sprite["bbox"])
    rows = _line_length(vertical)
    cols = _line_length(horizontal)
    r0, c0, _r1, _c1 = sprite["bbox"]
    expected = main(grid)

    if set(np.unique(expected)) - {bg, H_BAR_COLOR, V_BAR_COLOR, SPRITE_COLOR}:
        return False
    if H_BAR_COLOR not in expected or V_BAR_COLOR not in expected:
        return False

    variants = [
        _render_tile_array(
            sprite_patch,
            rows,
            cols,
            apply_row_flips=False,
            apply_col_flips=False,
            row_sep_color=V_BAR_COLOR,
            col_sep_color=H_BAR_COLOR,
            vertical_priority=True,
            bg=bg,
        ),
        _render_tile_array(
            sprite_patch,
            rows,
            cols,
            apply_row_flips=False,
            apply_col_flips=True,
            row_sep_color=V_BAR_COLOR,
            col_sep_color=H_BAR_COLOR,
            vertical_priority=True,
            bg=bg,
        ),
        _render_tile_array(
            sprite_patch,
            rows,
            cols,
            apply_row_flips=True,
            apply_col_flips=False,
            row_sep_color=V_BAR_COLOR,
            col_sep_color=H_BAR_COLOR,
            vertical_priority=True,
            bg=bg,
        ),
        _render_tile_array(
            sprite_patch,
            rows,
            cols,
            apply_row_flips=True,
            apply_col_flips=True,
            row_sep_color=V_BAR_COLOR,
            col_sep_color=H_BAR_COLOR,
            vertical_priority=False,
            bg=bg,
        ),
    ]

    for variant in variants:
        alt = np.full_like(grid, bg)
        paste(alt, variant, r0, c0)
        if np.array_equal(alt, expected):
            return False

    if cols < max(COUNT_VALUES):
        alt_cols = cols + 1
        alt_w = alt_cols * sprite_patch.shape[1] + (alt_cols - 1)
        if c0 + alt_w <= grid.shape[1]:
            alt = _render_full_output(grid.shape, (r0, c0), sprite_patch, rows, alt_cols)
            if np.array_equal(alt, expected):
                return False
    if rows < max(COUNT_VALUES):
        alt_rows = rows + 1
        alt_h = alt_rows * sprite_patch.shape[0] + (alt_rows - 1)
        if r0 + alt_h <= grid.shape[0]:
            alt = _render_full_output(grid.shape, (r0, c0), sprite_patch, alt_rows, cols)
            if np.array_equal(alt, expected):
                return False

    h_r0, h_c0, h_r1, h_c1 = horizontal["bbox"]
    v_r0, v_c0, v_r1, v_c1 = vertical["bbox"]
    tile_h = rows * sprite_patch.shape[0] + (rows - 1)
    tile_w = cols * sprite_patch.shape[1] + (cols - 1)
    if not (h_r1 < r0 - 1 or h_r0 > r0 + tile_h):
        return False
    if not (v_c1 < c0 - 1 or v_c0 > c0 + tile_w):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    sprite_shape = random.choice(SPRITE_SHAPES)
    sprite_mask = random.choice(_SPRITE_MASK_POOL[sprite_shape])
    sprite_patch = _sprite_patch_from_mask(sprite_mask)

    rows = random.choice(COUNT_VALUES)
    cols = random.choice(COUNT_VALUES)
    tile_h = rows * sprite_patch.shape[0] + (rows - 1)
    tile_w = cols * sprite_patch.shape[1] + (cols - 1)

    min_h = max(GRID_MIN, tile_h + 5)
    min_w = max(GRID_MIN, tile_w + 5)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)

    h_bar = np.full((1, cols), H_BAR_COLOR, dtype=int)
    v_bar = np.full((rows, 1), V_BAR_COLOR, dtype=int)

    for _ in range(60):
        sprite_r = random.randint(0, grid_h - tile_h)
        sprite_c = random.randint(0, grid_w - tile_w)

        occupancy = np.zeros((grid_h, grid_w), dtype=int)
        _stamp_forbidden(occupancy, sprite_r, sprite_c, tile_h, tile_w)

        try:
            h_r, h_c = random_free_location(
                occupancy,
                h_bar,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
            paste(occupancy, np.ones_like(h_bar), h_r, h_c)
            v_r, v_c = random_free_location(
                occupancy,
                v_bar,
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        paste(grid, sprite_patch, sprite_r, sprite_c, transparent=BG)
        paste(grid, h_bar, h_r, h_c)
        paste(grid, v_bar, v_r, v_c)

        if np.array_equal(main(grid), _render_full_output(grid.shape, (sprite_r, sprite_c), sprite_patch, rows, cols)) and _semantic_checks(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/037/6")
