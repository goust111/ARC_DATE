# Combo variant of b0039139 (index 14)
# combo_spec: D11/0+D12/0
# Name: 先平铺后统一着色配透明标记

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import bbox, components_mask, crop_bbox, random_sprite  # noqa: E402


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The four-section layout is preserved: template, stencil, foreground swatch,
#      background swatch, separated by solid color-1 rows or columns.
#    - The template is now three-state inside its black border:
#        * 4 = foreground-bearing template mass.
#        * 1 = transparent marker that survives cropping and replication.
#        * 0 = ordinary template background.
#    - The combined rule changes the processing order:
#        1. Crop the template to the bbox of all marked cells (4 and 1).
#        2. Repeat that raw cropped template using raw 0 separators.
#        3. Recolor the full repeated mosaic at once with 4->fg, 0->bg, 1->0.
#    - Small trace with horizontal repetition, fg=8, bg=6, repeats=2:
#        cropped raw template = [[4, 1, 0],
#                                [0, 4, 4]]
#        raw tiled mosaic    = [[4, 1, 0, 0, 4, 1, 0],
#                                [0, 4, 4, 0, 0, 4, 4]]
#        final output        = [[8, 0, 6, 6, 8, 0, 6],
#                                [6, 8, 8, 6, 6, 8, 8]]
#      The original binary-template solver would instead paint both 1 cells as 8,
#      so transparency remains visually observable in the final output.
# b) Complexity calibration
#    - To keep the fusion visible, each generated cropped template must contain all
#      of {4, 1, 0}. This yields three distinct semantics in one compact pattern:
#      painted mass, transparent holes, and recolorable background.
#    - Repeat count stays in the original 2-6 range so the stencil still offers a
#      simple single-purpose cue rather than introducing a second latent variable.
# c) Hidden assumptions in the original solver
#    - `template != 0` can no longer mean "foreground pixel"; it now means only
#      "participates in the crop". Recoloring must inspect raw values afterward.
#    - Tiling after recoloring is no longer the intended execution order. The combo
#      requires repeating the raw template first so transparent markers are copied as
#      markers, not as already-rendered output cells.
#    - Separator strips between repeated copies are raw 0 before recoloring; they are
#      allowed to become the background swatch color during the final paint pass.
# d) Placement legality
#    - The template remains embedded in a larger black section with a guaranteed
#      1-cell outer border, so cropping is unambiguous and never touches separators.
#    - The stencil still uses isolated 2-cell 4-connected blobs with a 1-cell black
#      moat, preserving stable component counting and preventing accidental merges.
# e) Visual observability
#    - Transparent markers become repeated black holes in the output, while ordinary
#      0 cells inside the crop become the chosen background color. That contrast is
#      visible across multiple copies and makes the three-state template inferable.
# f) Visual cleanliness
#    - No new legends or overlay objects are introduced. The only new signal is the
#      extra color-1 pixels inside the template section, which remain spatially local.
# g) Color-change inferability
#    - The color story stays small and structured: stencil uses 3, template uses
#      0/1/4, and output colors come from exactly two swatches plus fixed black for
#      transparency. Human solvers can match these roles from repeated evidence.
# h) Quantitative anti-degeneration target
#    - Generator validation rejects any sample whose fused output matches the original
#      binary-template output. Because every kept sample contains at least one cropped
#      transparent marker, the fused rule changes visible pixels on every example.
# i) New solver assumptions to audit
#    - The solver assumes only {0,1,4} occur in the cropped template; the generator
#      enforces that by constructing the template directly from those values.
#    - The solver assumes at least one marked template pixel exists; generation starts
#      from a nonempty connected support sprite and preserves at least one 4 cell.
#    - The solver assumes transparency should remain 0 after the final paint pass;
#      that is fixed by construction and reinforced by validation checks.

TEMPLATE_COLOR = 4
TRANSPARENT_MARKER = 1
OUTPUT_DEFAULT_COLOR = 0
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == SEPARATOR_COLOR))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEPARATOR_COLOR))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _crop_marked_template(template: np.ndarray) -> np.ndarray:
    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)
    return crop_bbox(template, template_bbox).astype(int)


def _render_raw_mosaic(raw_mosaic: np.ndarray, foreground: int, background: int) -> np.ndarray:
    painted = np.full(raw_mosaic.shape, background, dtype=int)
    painted[raw_mosaic == TEMPLATE_COLOR] = foreground
    painted[raw_mosaic == TRANSPARENT_MARKER] = OUTPUT_DEFAULT_COLOR
    return painted


def _tile_repeat_raw(tile: np.ndarray, n: int, horizontal: bool) -> np.ndarray:
    """Repeat an uncolored template using raw-0 separators before recoloring."""
    if n <= 1:
        return tile.copy()

    if horizontal:
        sep = np.zeros((tile.shape[0], 1), dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.zeros((1, tile.shape[1]), dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _tile_repeat_painted(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Legacy helper used only for anti-degeneration checks in the generator."""
    if n <= 1:
        return tile.copy()

    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _legacy_binary_output(raw_tile: np.ndarray, repeats: int, foreground: int, background: int, horizontal: bool) -> np.ndarray:
    legacy_tile = np.where(raw_tile != 0, foreground, background).astype(int)
    return _tile_repeat_painted(legacy_tile, repeats, background, horizontal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the full raw three-state template section first, then recolor the full mosaic.

    Input structure:
    - Solid blue (1) separator rows or columns divide the input into four sections.
    - Section 1 is a black-background template containing yellow paint cells (4),
      blue transparent-marker cells (1), and black ordinary background cells (0).
    - Section 2 is the repeat-count stencil; sections 3 and 4 are solid swatches for
      the output foreground and output background colors.

    Rule:
    - Keep the entire template section, including its black frame and any transparent markers.
    - Count 4-connected non-black components in section 2, then repeat that raw
      template section that many times with raw black separator strips between copies.
    - After tiling, recolor the entire repeated mosaic in one pass: 4 -> foreground
      swatch, 0 -> background swatch, and 1 stays at the default output color 0.

    Key insight: The stencil still contributes only the repetition count, but the
    template must be interpreted as a three-state raw section whose transparent markers
    and black frame survive replication before the final recoloring step.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    raw_template = template.astype(int)
    if not np.any(raw_template != 0):
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    raw_mosaic = _tile_repeat_raw(raw_template, n_repeats, horizontal=(orientation == "horizontal"))
    return _render_raw_mosaic(raw_mosaic, foreground, background)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
TEMPLATE_ATTEMPTS = 160
GEN_RETRIES = 300
TRANSPARENT_RATIO_RANGE = (0.2, 0.45)


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)

    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
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


def _make_template_core(crop_h: int, crop_w: int) -> np.ndarray | None:
    for _ in range(TEMPLATE_ATTEMPTS):
        density = random.uniform(*DENSITY_RANGE)
        support = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        support_mask = support == TEMPLATE_COLOR
        support_cells = int(support_mask.sum())
        if support_cells < 2:
            continue

        coords = [tuple(coord) for coord in np.argwhere(support_mask)]
        random.shuffle(coords)

        max_transparent = min(
            support_cells - 1,
            max(1, int(round(support_cells * TRANSPARENT_RATIO_RANGE[1]))),
        )
        min_transparent = max(1, int(np.ceil(support_cells * TRANSPARENT_RATIO_RANGE[0])))
        min_transparent = min(min_transparent, max_transparent)
        n_transparent = random.randint(min_transparent, max_transparent)

        template_core = np.zeros((crop_h, crop_w), dtype=int)
        template_core[support_mask] = TEMPLATE_COLOR
        for r, c in coords[:n_transparent]:
            template_core[r, c] = TRANSPARENT_MARKER

        raw_tile = _crop_marked_template(template_core)
        if raw_tile.size == 0:
            continue
        if not np.any(raw_tile == TEMPLATE_COLOR):
            continue
        if not np.any(raw_tile == TRANSPARENT_MARKER):
            continue
        if not np.any(raw_tile == 0):
            continue
        return template_core

    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    template_core = _make_template_core(crop_h, crop_w)
    if template_core is None:
        return None

    cropped_tile = _crop_marked_template(template_core)
    if cropped_tile.size == 0:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = template_core

    raw_mosaic = _tile_repeat_raw(template, n_repeats, horizontal)
    output = _render_raw_mosaic(raw_mosaic, fg_color, bg_color)
    if not np.any(output == OUTPUT_DEFAULT_COLOR):
        return None
    if not np.any(output == bg_color):
        return None

    legacy_output = _legacy_binary_output(cropped_tile, n_repeats, fg_color, bg_color, horizontal)
    if np.array_equal(output, legacy_output):
        return None

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        solved = main(inp)
    except Exception:
        return None

    return inp if np.array_equal(solved, output) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
