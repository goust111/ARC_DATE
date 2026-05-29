# Variant of b0039139
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 三步串行：(1)裁剪模板到bbox，(2)用色板重着色，(3)按连通分量数平铺并加分隔带... → 反向流程：从输出推断输入

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, random_sprite


# Design analysis for the D11 inversion variant
# =============================================
# a) Concrete behavior trace
#    - Old task input had four sections; old output was a strip made from N copies of one recolored crop,
#      with 1-cell background separators between copies.
#    - New task uses that strip itself as the input. The output is the recovered original template section:
#      the repeated primitive tile is extracted, its foreground is mapped back to 4, its background to 0,
#      then a 1-cell black border is added around the recovered crop.
#    - Example: if the input is [tile][bg-col][tile][bg-col][tile], the old logic would have *produced*
#      this strip. The new logic instead detects the repeated tile, drops the separators, restores colors,
#      and emits the bordered template. The difference is visible because the task direction is reversed:
#      a long repeated strip shrinks back to one canonical template.
# b) Complexity calibration
#    - Pure "find any repeated block" would be too weak because periodic tiles could admit multiple cuts.
#    - To keep the inversion legible but unambiguous, the generator enforces a unique decomposition into
#      identical blocks plus single full separator rows/cols, with repeat count still limited to 2-6.
# c) Implicit assumptions review
#    - The old `_split_sections` assumption is invalid: the input no longer contains blue separators or
#      swatches/stencils, so the new solver must inspect the whole grid directly.
#    - The old `_solid_color` assumption is invalid: foreground/background are no longer provided
#      explicitly, so the new solver must infer the separator/background color from a valid tiling.
#    - The old crop step operated on the template section. In the inverse task, crop restoration happens
#      after extracting one tile from the repeated strip.
# d) Placement legality
#    - The generator still uses the original crop size, density, color, and repeat-count ranges.
#    - Each sample is built by tiling one crop-sized tile with 1-cell separators, so no overlap or
#      out-of-bounds placement can occur.
#    - An oracle check `main(input) == expected_output` is kept, and ambiguous decompositions are rejected.
# e) Visual observability
#    - Humans can see that the input is made of repeated identical blocks, and that the output keeps only
#      one block, recolored back to yellow-on-black with a black frame.
#    - The repeat axis varies across samples, so the inversion is not tied to one orientation.
# f) Visual cleanliness
#    - The input contains only the repeated strip; there are no layered annotations competing for space.
#      The output is a single compact template section, so the transformation is visually crisp.
# g) Color inference
#    - No hidden ordering by color id is used. The recovered background is whichever color forms the full
#      separator row/col in the valid decomposition, and every non-background tile pixel becomes 4.
# 9) New assumptions introduced by `main()`
#    - The input must admit exactly one valid decomposition into identical tiles separated by single full
#      rows/cols of a background color.
#    - The primitive tile must contain at least one non-background pixel.
#    - `generate_input()` guarantees these assumptions by filtering to unique decompositions only.


# --- CFG: mostly inherited from the seed task ---
REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
SEPARATOR_WIDTH = 1
GEN_RETRIES = 400


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along one axis with a 1-cell bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], SEPARATOR_WIDTH), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((SEPARATOR_WIDTH, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _try_decompose(grid: np.ndarray, bg_color: int, n_repeats: int, horizontal: bool) -> np.ndarray | None:
    """Return the repeated primitive tile if the grid matches this decomposition, else None."""
    h, w = grid.shape
    if horizontal:
        remaining = w - (n_repeats - 1) * SEPARATOR_WIDTH
        if remaining <= 0 or remaining % n_repeats != 0:
            return None
        tile_w = remaining // n_repeats
        if not (CROP_SIZE_RANGE[0] <= tile_w <= CROP_SIZE_RANGE[1]):
            return None
        tile = grid[:, :tile_w]
        pos = tile_w
        for _ in range(1, n_repeats):
            if pos + SEPARATOR_WIDTH > w or not np.all(grid[:, pos : pos + SEPARATOR_WIDTH] == bg_color):
                return None
            block = grid[:, pos + SEPARATOR_WIDTH : pos + SEPARATOR_WIDTH + tile_w]
            if block.shape[1] != tile_w or not np.array_equal(block, tile):
                return None
            pos += tile_w + SEPARATOR_WIDTH
        return tile if pos == w and np.any(tile != bg_color) else None

    remaining = h - (n_repeats - 1) * SEPARATOR_WIDTH
    if remaining <= 0 or remaining % n_repeats != 0:
        return None
    tile_h = remaining // n_repeats
    if not (CROP_SIZE_RANGE[0] <= tile_h <= CROP_SIZE_RANGE[1]):
        return None
    tile = grid[:tile_h, :]
    pos = tile_h
    for _ in range(1, n_repeats):
        if pos + SEPARATOR_WIDTH > h or not np.all(grid[pos : pos + SEPARATOR_WIDTH, :] == bg_color):
            return None
        block = grid[pos + SEPARATOR_WIDTH : pos + SEPARATOR_WIDTH + tile_h, :]
        if block.shape[0] != tile_h or not np.array_equal(block, tile):
            return None
        pos += tile_h + SEPARATOR_WIDTH
    return tile if pos == h and np.any(tile != bg_color) else None


def _enumerate_decompositions(grid: np.ndarray) -> list[dict[str, object]]:
    """Enumerate all valid strip decompositions; the generator keeps only unique cases."""
    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for bg_color in np.unique(grid):
        for n_repeats in range(REPEAT_COUNT_RANGE[0], REPEAT_COUNT_RANGE[1] + 1):
            for horizontal in (True, False):
                tile = _try_decompose(grid, int(bg_color), n_repeats, horizontal)
                if tile is None:
                    continue
                key = (
                    horizontal,
                    int(bg_color),
                    int(n_repeats),
                    int(tile.shape[0]),
                    int(tile.shape[1]),
                    tuple(int(x) for x in tile.flat),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "horizontal": horizontal,
                        "background": int(bg_color),
                        "n_repeats": int(n_repeats),
                        "tile": tile.copy(),
                    }
                )
    return candidates


def _restore_template(tile: np.ndarray, background: int) -> np.ndarray:
    """Map the extracted tile back to a yellow-on-black template with a 1-cell black border."""
    shape_mask = tile != background
    shape_bbox = bbox(shape_mask)
    if shape_bbox is None:
        return np.empty((0, 0), dtype=int)
    cropped_mask = crop_bbox(shape_mask, shape_bbox)
    template = np.zeros((cropped_mask.shape[0] + 2, cropped_mask.shape[1] + 2), dtype=int)
    template[1:-1, 1:-1] = np.where(cropped_mask, TEMPLATE_COLOR, 0)
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the canonical template by undoing a repeated recolored strip.

    Input structure:
    - The whole input grid is a former output strip: one tile repeated 2-6 times.
    - Repetitions run either left-to-right or top-to-bottom.
    - Adjacent copies are separated by a single full row or column in the tile's background color.

    Rule:
    - Find the unique decomposition of the grid into identical tiles plus 1-cell separators.
    - Use the separator color as the tile background and convert every other tile pixel back to color 4.
    - Crop that recovered mask to its bbox and wrap it in a 1-cell black border to rebuild the template.

    Key insight:
    - The inverse task is solved by detecting periodic structure first; once one primitive tile is isolated,
      the recoloring and border restoration become deterministic.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _enumerate_decompositions(g)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one valid decomposition, found {len(candidates)}")
    chosen = candidates[0]
    return _restore_template(chosen["tile"], int(chosen["background"]))


# --------------- generator (c) ---------------

def _make_template(shape: np.ndarray) -> np.ndarray:
    template = np.zeros((shape.shape[0] + 2, shape.shape[1] + 2), dtype=int)
    template[1:-1, 1:-1] = shape
    return template


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )
    if not np.any(shape == 0):
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    inp = _tile_repeat(tile, n_repeats, bg_color, horizontal)
    expected = _make_template(shape)

    candidates = _enumerate_decompositions(inp)
    if len(candidates) != 1:
        return None
    chosen = candidates[0]
    if bool(chosen["horizontal"]) != horizontal:
        return None
    if int(chosen["background"]) != bg_color:
        return None
    if int(chosen["n_repeats"]) != n_repeats:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
