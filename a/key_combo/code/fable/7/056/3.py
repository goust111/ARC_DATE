from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_for_fable_7_056_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste


BG = 0
GRAY = 5
LEGEND_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GEN_ATTEMPTS = 400

SHAPE_LIBRARY = {
    2: [
        np.array([[1, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    ],
    7: [
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    ],
    8: [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool),
    ],
    9: [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ],
}


def _random_orient(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    if int(rng.integers(2)):
        out = np.flipud(out)
    if int(rng.integers(2)):
        out = np.fliplr(out)
    return np.rot90(out, int(rng.integers(4))).copy()


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    colored = np.argwhere((grid != bg) & (grid != GRAY))
    assert colored.size > 0, "expected a colored legend row"
    legend_rows = np.unique(colored[:, 0])
    assert legend_rows.size == 1, "legend must occupy exactly one row"
    row = int(legend_rows[0])
    cols = np.sort(colored[:, 1])
    assert np.all(np.diff(cols) == 1), "legend pixels must be contiguous"
    colors = [int(grid[row, int(c)]) for c in cols]
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    gray_pixels = np.argwhere(grid == GRAY)
    assert gray_pixels.size > 0, "expected at least one gray sprite"
    assert row < int(gray_pixels[:, 0].min()), "legend must be above all sprites"
    return row, int(cols[0]), colors


def _gray_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites = find_components(
        grid,
        bg=bg,
        colors=[GRAY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert sprites, "expected gray sprites"
    areas = [int(sprite["area"]) for sprite in sprites]
    assert len(areas) == len(set(areas)), "sprite areas must be unique"
    return sprites


def _sprite_patch(grid: np.ndarray, sprite: dict, color: int) -> np.ndarray:
    patch = crop_bbox(grid, sprite["bbox"])
    return np.where(patch == GRAY, int(color), BG)


def _pack_output(
    grid: np.ndarray,
    legend_row: int,
    legend_c0: int,
    legend: list[int],
    sprites: list[dict],
    order_key,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)
    ordered = sorted(sprites, key=order_key)
    kept = ordered[: len(legend)]
    x = 0
    for color, sprite in zip(legend, kept):
        patch = _sprite_patch(g, sprite, color)
        ph, pw = patch.shape
        assert x + pw <= w, "packed sprites must fit horizontally"
        paste(out, patch, h - ph, x, transparent=BG)
        x += pw
    out[legend_row, legend_c0 : legend_c0 + len(legend)] = np.array(legend, dtype=int)
    return out


def _x_key(sprite: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = sprite["bbox"]
    return int(c0), int(r0)


def _bbox_area_key(sprite: dict) -> tuple[int, int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    return int((r1 - r0 + 1) * (c1 - c0 + 1)), int(c0), int(r0)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the smallest gray sprites by legend rank, keep only the covered ranks, and bottom-pack them.

    Input structure:
    - Black background with several separated gray 4-connected sprites of distinct pixel areas.
    - Near the top sits one compact horizontal legend row made of unique colored single pixels.
    - Legend pixels are the only non-black, non-gray cells.

    Rule:
    - Read the legend colors from left to right.
    - Rank the gray sprites by pixel area from smallest to largest.
    - Recolor the k-th smallest sprite with the k-th legend color, discard every sprite whose rank exceeds the legend length, and erase all original sprite positions.
    - Pack the retained recolored sprites from left to right along the bottom edge using their cropped bbox sprites, while keeping the legend at its original position.

    Key insight:
    - The same area ranking controls three things at once: which sprites survive, which color each survivor gets, and the left-to-right packing order, so position-based shortcuts fail.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend_row, legend_c0, legend = _parse_legend(g, bg)
    sprites = _gray_sprites(g, bg)
    return _pack_output(
        g,
        legend_row,
        legend_c0,
        legend,
        sprites,
        order_key=lambda sprite: (int(sprite["area"]), int(sprite["bbox"][1]), int(sprite["bbox"][0])),
    )


def _build_sprite(area: int, rng: np.random.Generator) -> np.ndarray:
    template = SHAPE_LIBRARY[area][int(rng.integers(len(SHAPE_LIBRARY[area])))]
    sprite = _random_orient(template, rng)
    return np.where(sprite, GRAY, BG).astype(int)


def _area_rank(items: list[dict]) -> list[int]:
    return [int(item["area"]) for item in sorted(items, key=lambda item: int(item["area"]))]


def _bbox_area_rank(items: list[dict]) -> list[int]:
    ranked = sorted(
        items,
        key=lambda item: (
            int((item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1)),
            int(item["bbox"][1]),
            int(item["bbox"][0]),
        ),
    )
    return [int(item["area"]) for item in ranked]


def _place_sprites(
    grid: np.ndarray,
    sprites: list[np.ndarray],
    x_order: list[int],
    left_margin: int,
    gaps: list[int],
    rng: np.random.Generator,
) -> None:
    widths = [sprite.shape[1] for sprite in sprites]
    span = left_margin + sum(widths[idx] for idx in x_order) + sum(gaps)
    x0 = int(rng.integers(0, grid.shape[1] - span + 1))
    row_options = []
    for idx in x_order:
        sh, _sw = sprites[idx].shape
        min_r = 3
        max_r = grid.shape[0] - sh - 2
        assert min_r <= max_r
        row_options.append((min_r, max_r))
    rows = None
    for _ in range(20):
        trial = [int(rng.integers(lo, hi + 1)) for lo, hi in row_options]
        bottoms = [trial[pos] + sprites[idx].shape[0] for pos, idx in enumerate(x_order)]
        if len(set(bottoms)) > 1:
            rows = trial
            break
    assert rows is not None, "input sprites should not already share one baseline"
    x = x0 + left_margin
    for pos, idx in enumerate(x_order):
        sprite = sprites[idx]
        r = rows[pos]
        paste(grid, sprite, r, x, transparent=BG)
        x += widths[idx]
        if pos < len(gaps):
            x += gaps[pos]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg = int(get_bg(grid))
        legend_row, legend_c0, legend = _parse_legend(grid, bg)
        sprites = _gray_sprites(grid, bg)
        if len(legend) < 2 or len(sprites) < 3:
            return False
        x_rank = [int(item["area"]) for item in sorted(sprites, key=_x_key)]
        area_rank = _area_rank(sprites)
        bbox_rank = _bbox_area_rank(sprites)
        if x_rank == area_rank:
            return False
        if bbox_rank == area_rank:
            return False
        correct = main(grid)
        wrong_x = _pack_output(grid, legend_row, legend_c0, legend, sprites, order_key=_x_key)
        wrong_bbox = _pack_output(grid, legend_row, legend_c0, legend, sprites, order_key=_bbox_area_key)
        if np.array_equal(correct, wrong_x):
            return False
        if np.array_equal(correct, wrong_bbox):
            return False
        if correct.shape != grid.shape or max(correct.shape) > 30:
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_sprites = int(rng.integers(3, 6))
    extra_areas = rng.choice(np.array([2, 3, 4, 7, 8, 9], dtype=int), size=num_sprites - 2, replace=False)
    areas = [5, 6, *map(int, extra_areas)]
    rng.shuffle(areas)
    sprites = [_build_sprite(area, rng) for area in areas]

    area_sorted = sorted(range(num_sprites), key=lambda idx: areas[idx])
    x_order = list(rng.permutation(num_sprites))
    if x_order == area_sorted:
        x_order = x_order[1:] + x_order[:1]

    if rng.random() < 0.6:
        legend_len = int(rng.integers(2, num_sprites))
    else:
        legend_len = num_sprites
    legend = list(map(int, rng.choice(LEGEND_COLORS, size=legend_len, replace=False)))

    kept_width = sum(sprites[idx].shape[1] for idx in area_sorted[:legend_len])
    left_margin = int(rng.integers(1, 4))
    gaps = [int(rng.integers(2, 5)) for _ in range(num_sprites - 1)]
    scattered_width = left_margin + sum(sprites[idx].shape[1] for idx in x_order) + sum(gaps)
    width_min = max(kept_width + 2, scattered_width, legend_len + 3)
    height_min = max(max(sprite.shape[0] for sprite in sprites) + 6, 12)
    if width_min > 30 or height_min > 30:
        return None

    height = int(rng.integers(height_min, min(30, height_min + 6) + 1))
    width = int(rng.integers(width_min, min(30, width_min + 6) + 1))
    grid = np.full((height, width), BG, dtype=int)

    legend_c0 = int(rng.integers(1, width - legend_len))
    grid[0, legend_c0 : legend_c0 + legend_len] = np.array(legend, dtype=int)
    _place_sprites(grid, sprites, x_order, left_margin, gaps, rng)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/7/056/3")
