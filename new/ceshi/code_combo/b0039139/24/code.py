# Combo variant of b0039139 (index 24)
# combo_spec: D2/1+D10/2
# Name: 不裁剪模板配紧凑裁剪输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the fused variant:
# - Concrete behavior trace:
#   Original task crops the template down to the yellow shape bbox before recoloring,
#   so a 4x5 template holding a 2x3 shape becomes a 2x3 tile immediately. Here we do
#   the opposite: recolor the full 4x5 template, keep its black frame as output-bg,
#   tile those larger copies, and only then trim the outer all-bg frame. With N=3 and
#   horizontal layout, the top/bottom border rows disappear in the final crop, but the
#   internal border columns survive between copies, so the output is visibly wider than
#   the original task on the same input.
# - Complexity calibration:
#   The only new freedom is whether the initial template margin survives internally after
#   global trimming. Keeping the template border fixed at 1 cell and N in [2, 6] makes
#   the visual effect stable: every sample shows repeated thick bg gutters without adding
#   ambiguous extra object types or hidden bookkeeping.
# - Hidden assumptions audit:
#   The original solver assumed `bbox(template != 0)` was the working tile. That premise
#   is intentionally removed. The new solver instead assumes only that the template has at
#   least one nonzero cell and that fg != bg, so the final tight crop still has foreground
#   to anchor on. The generator guarantees both.
# - Placement legality:
#   The generator still builds four disjoint sections separated by full blue rows/cols,
#   keeps the template wholly inside section 1 with a 1-cell black frame, and places the
#   stencil blobs with a black moat so the component count is unambiguous.
# - Visual observability and cleanliness:
#   Because N >= 2, skipping the initial crop cannot collapse back to the base behavior:
#   after final trimming, every internal join retains border+separator+border, producing
#   3-cell bg gutters (or 3 bg rows in vertical layouts). A human can infer that only the
#   exterior margin was removed, since the repeated internal padding remains obvious across
#   samples and depends on geometry rather than color ids.


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg separators."""
    if n <= 1:
        return tile.copy()
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _tight_crop_background(grid: np.ndarray, bg_color: int) -> np.ndarray:
    """Remove outer rows/cols that are entirely `bg_color`."""
    fg_bbox = bbox(grid != bg_color)
    if fg_bbox is None:
        return np.empty((0, 0), dtype=int)
    return crop_bbox(grid, fg_bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the full template section, repeat it by the stencil count, then trim only the outside background frame.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into four sections.
    - Section 1 is a black-background template carrying a yellow shape plus its surrounding black margin.
    - Section 2 is a black stencil whose 4-connected non-black components encode the repeat count.
    - Sections 3 and 4 are solid-color swatches for the output foreground and output background.

    Rule:
    - Do not crop section 1 to the shape bbox; use the entire section as the template footprint.
    - Recolor every nonzero template cell to the foreground swatch and every zero template cell to the background swatch.
    - Count the stencil components, repeat that full-size recolored template along the separator orientation, and place one background-colored separator row or column between copies.
    - After the repetition is built, tight-crop the whole output once by removing any outer rows or columns that are entirely background-colored.

    Key insight: The template margin is preserved during tiling, so the end crop removes only the global outer frame; the internal template padding survives and becomes the widened gutters between repeated copies.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    if not bool(np.any(template_mask)):
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile = np.where(template_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    tiled = _tile_repeat(tile, n_repeats, background, orientation == "horizontal")
    return _tight_crop_background(tiled, background)


# --------------- generator (combo variant) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # inner shape bbox height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bbox
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
GEN_RETRIES = 300                    # outer generate_input retry limit


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    """Create a solid color swatch. shared_dim is the fixed axis; varying axis is random."""
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for s in sections[1:]:
            parts.extend([sep, s])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for s in sections[1:]:
        parts.extend([sep, s])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs of STENCIL_COLOR on a black grid with a black moat between them."""
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= n_blobs:
            break
        br, bc = (1, 2) if random.choice([True, False]) else (2, 1)

        if h - 1 - br < 1 or w - 1 - bc < 1:
            continue
        r = random.randint(1, h - 1 - br)
        c = random.randint(1, w - 1 - bc)

        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r : r + br, c : c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


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

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    # Section 1 keeps the full template frame: the 1-cell black border is intentional input signal.
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    tile = np.where(template != 0, fg_color, bg_color).astype(int)
    output = _tight_crop_background(_tile_repeat(tile, n_repeats, bg_color, horizontal), bg_color)

    shared_dim = template.shape[0] if horizontal else template.shape[1]
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    # Guard against accidental degeneration back to the base task:
    # for N >= 2 the combo output must be larger on the repeat axis than the cropped-template output.
    base_tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    base_output = _tile_repeat(base_tile, n_repeats, bg_color, horizontal)
    repeat_axis = 1 if horizontal else 0
    if output.shape[repeat_axis] <= base_output.shape[repeat_axis]:
        return None

    try:
        if np.array_equal(main(inp), output):
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
