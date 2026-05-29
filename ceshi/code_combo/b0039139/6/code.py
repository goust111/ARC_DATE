# Combo variant of b0039139 (index 6)
# combo_spec: D4/1+D7/0
# Name: 二维网格中分量位置决定副本布局
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original task: crop the template, recolor it, count stencil components, then lay the
#    same tile along one axis only. For example, 4 components with column separators would
#    become 4 tiles in one row.
#    Fused task: the stencil is no longer "count only". Its components occupy a visible
#    2D arrangement of row-bands and column-bands. After cropping/recoloring the template,
#    the solver recovers that 2D component layout and places one tile in each matching
#    output cell. Example: if the stencil has components in a 2x3 band layout, the output
#    is a 2x3 tile grid with one background row/column between neighboring tiles.
#    Difference: the same count can now yield different outputs (notably 2x3 vs 3x2),
#    because component positions decide the row/column factorization.
# b) Complexity calibration:
#    The layout choices are restricted to a small set of dense 2D grids: 2x2, 2x3, 3x2.
#    This is enough to make the fusion visible while keeping the hypothesis space small.
#    Prime counts and 1xN degeneracies are excluded so the new rule cannot collapse back
#    to the original single-axis repetition behavior.
# c) Hidden assumptions broken from the original solver:
#    - "Component count alone is sufficient" is no longer true.
#    - "Output orientation follows separator direction" is no longer true.
#    - "Stencil only needs to preserve connected-component count" is no longer true.
#    The new solver therefore ignores section orientation after splitting and instead
#    infers output rows/cols from occupied row/column bands in the stencil.
# d) Placement legality:
#    The generator uses one 2-cell component per stencil cell, with fixed 1-cell black
#    borders and 1-cell black gaps between neighboring row/column bands. This guarantees
#    no overlap, no out-of-bounds placements, and exactly one component per output slot.
# e) Visual observability:
#    Humans can see the 2D band layout directly in the stencil. A 2x3 stencil and a 3x2
#    stencil have the same count but different visible geometry, and the output mirrors
#    that geometry. This makes the fusion inferable from a few examples.
# f) Visual cleanliness:
#    The four input sections remain cleanly separated by blue lines. The stencil uses only
#    black background plus green 2-cell blobs; the template uses yellow on black; swatches
#    use colors outside {1,3,4}. No region mixes multiple encodings.
# g) New solver assumptions and how the generator satisfies them:
#    - Occupied stencil rows form distinct consecutive row-bands; generator enforces this
#      with 1-cell blank gaps between lattice rows.
#    - Occupied stencil cols form distinct consecutive col-bands; generator does the same.
#    - Each band intersection contains exactly one component; generator places one blob in
#      every lattice cell.
#    - The shared dimension required by section joining may exceed the template's minimal
#      size; generator pads the template with extra black margin on the shared axis.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


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


def _consecutive_groups(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in map(int, indices[1:]):
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _stencil_bands(stencil_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_groups = _consecutive_groups(np.where(np.any(stencil_mask, axis=1))[0])
    col_groups = _consecutive_groups(np.where(np.any(stencil_mask, axis=0))[0])
    return row_groups, col_groups


def _band_index(bands: list[tuple[int, int]], lo: int, hi: int) -> int:
    for idx, (start, end) in enumerate(bands):
        if not (hi < start or lo > end):
            return idx
    center = (int(lo) + int(hi)) / 2.0
    distances = [abs(center - (start + end) / 2.0) for start, end in bands]
    return int(np.argmin(distances))


def _factor_rows_cols(n: int) -> tuple[int, int]:
    best = (1, n)
    best_gap = abs(n - 1)
    for rows in range(1, n + 1):
        if n % rows != 0:
            continue
        cols = n // rows
        gap = abs(cols - rows)
        if gap < best_gap:
            best = (rows, cols)
            best_gap = gap
    return best


def _place_tiles(tile: np.ndarray, rows: int, cols: int, background: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.full((rows * th + (rows - 1), cols * tw + (cols - 1)), background, dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (th + 1)
            c0 = cc * (tw + 1)
            out[r0 : r0 + th, c0 : c0 + tw] = tile
    return out


def _layout_from_stencil(stencil_mask: np.ndarray) -> tuple[int, int, list[tuple[int, int, np.ndarray]]]:
    components = components_mask(stencil_mask, connectivity=4)
    if not components:
        return 1, 1, [(0, 0, np.empty((0, 2), dtype=int))]

    row_bands, col_bands = _stencil_bands(stencil_mask)
    rows = len(row_bands)
    cols = len(col_bands)

    placements: dict[tuple[int, int], np.ndarray] = {}
    for comp in components:
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        rr = _band_index(row_bands, int(r0), int(r1))
        cc = _band_index(col_bands, int(c0), int(c1))
        placements[(rr, cc)] = comp

    if rows * cols != len(components) or len(placements) != len(components):
        rows, cols = _factor_rows_cols(len(components))
        ordered = sorted(
            components,
            key=lambda comp: (int(comp[:, 0].min()), int(comp[:, 1].min())),
        )
        mapped = []
        for idx, comp in enumerate(ordered):
            mapped.append((idx // cols, idx % cols, comp))
        return rows, cols, mapped

    mapped = [(rr, cc, comp) for (rr, cc), comp in sorted(placements.items())]
    return rows, cols, mapped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and copy it into a 2D grid recovered from the stencil.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into four sections.
    - Section 1 is a black-background template; section 2 is a green component stencil.
    - Sections 3 and 4 are solid swatches giving the foreground and background colors.

    Rule:
    - Crop the template to the bbox of its non-black pixels and recolor that crop using
      the foreground/background swatches.
    - In the stencil, treat each 4-connected non-black component as one copy request.
    - Recover the stencil's occupied row-bands and column-bands, then place one tile at
      every corresponding output grid cell, separating adjacent rows/cols by one
      background-colored line.

    Key insight:
    - The stencil is no longer a pure counter. Its 2D spatial arrangement determines the
      output's row/column layout, so identical component counts can yield different grids.
    """
    g = np.asarray(grid, dtype=int)
    _orientation, sections = _split_sections(g)
    template, stencil, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    stencil_mask = stencil != 0
    rows, cols, placements = _layout_from_stencil(stencil_mask)
    if not placements:
        return tile

    out = np.full(
        (rows * tile.shape[0] + (rows - 1), cols * tile.shape[1] + (cols - 1)),
        background,
        dtype=int,
    )
    for rr, cc, _comp in placements:
        r0 = rr * (tile.shape[0] + 1)
        c0 = cc * (tile.shape[1] + 1)
        out[r0 : r0 + tile.shape[0], c0 : c0 + tile.shape[1]] = tile
    return out


# --------------- generator (fused combo) ---------------

GRID_SHAPES = [(2, 2), (2, 3), (3, 2)]
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = [2, 5, 6, 7, 8, 9]
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 7)
GEN_RETRIES = 300


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for sec in sections[1:]:
            parts.extend([sep, sec])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for sec in sections[1:]:
        parts.extend([sep, sec])
    return np.vstack(parts)


def _embed_template(shape: np.ndarray, shared_dim: int, horizontal: bool) -> np.ndarray:
    crop_h, crop_w = shape.shape
    base = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    base[1 : crop_h + 1, 1 : crop_w + 1] = shape

    if horizontal:
        if shared_dim < base.shape[0]:
            raise ValueError("shared_dim too small for template height")
        pad_top = (shared_dim - base.shape[0]) // 2
        pad_bottom = shared_dim - base.shape[0] - pad_top
        return np.pad(base, ((pad_top, pad_bottom), (0, 0)), constant_values=0)

    if shared_dim < base.shape[1]:
        raise ValueError("shared_dim too small for template width")
    pad_left = (shared_dim - base.shape[1]) // 2
    pad_right = shared_dim - base.shape[1] - pad_left
    return np.pad(base, ((0, 0), (pad_left, pad_right)), constant_values=0)


def _build_stencil(rows: int, cols: int, shared_dim: int, horizontal: bool) -> np.ndarray:
    stencil_h = 3 * rows + 1
    stencil_w = 3 * cols + 1

    if horizontal:
        grid = np.zeros((shared_dim, stencil_w), dtype=int)
        row_offset = (shared_dim - stencil_h) // 2
        col_offset = 0
    else:
        grid = np.zeros((stencil_h, shared_dim), dtype=int)
        row_offset = 0
        col_offset = (shared_dim - stencil_w) // 2

    for rr in range(rows):
        for cc in range(cols):
            r0 = row_offset + 1 + rr * 3
            c0 = col_offset + 1 + cc * 3
            if random.choice([True, False]):
                grid[r0, c0 : c0 + 2] = STENCIL_COLOR
            else:
                grid[r0 : r0 + 2, c0] = STENCIL_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    rows, cols = random.choice(GRID_SHAPES)

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
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _place_tiles(tile, rows, cols, bg_color)

    min_shared = max((crop_h + 2) if horizontal else (crop_w + 2), 3 * (rows if horizontal else cols) + 1)
    shared_dim = min_shared + random.randint(0, 3)

    template = _embed_template(shape, shared_dim, horizontal)
    stencil = _build_stencil(rows, cols, shared_dim, horizontal)
    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

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
