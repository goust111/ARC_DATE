# Combo variant of b0039139 (index 12)
# combo_spec: D7/1+D11/3
# Name: 分量形状决定变换加交织着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for combo D7/1 + D11/3
# ---------------------------------------
# a) Concrete behavior and trace
#    - The input still has four blue-separated sections: template, stencil, foreground swatch,
#      background swatch.
#    - Old task:
#      1. crop the yellow template to its non-black bbox,
#      2. recolor it once with the two swatches,
#      3. count green stencil components,
#      4. repeat identical copies with one background-colored gap.
#    - Fused task:
#      1. crop the template and pad it to a square canvas,
#      2. sort stencil components along the output axis,
#      3. for each component, use its shape to choose a transform:
#         * 1x3 horizontal bar -> unchanged
#         * 3x1 vertical bar   -> rotate 90 degrees clockwise
#         * 2x2 square block   -> mirror left-right
#      4. while emitting copies, alternate the palette:
#         * copy 1,3,5,... uses fg=section3 and bg=section4
#         * copy 2,4,6,... uses fg=section4 and bg=section3
#      5. the separator after each copy uses that copy's local background color.
#    - Small trace:
#      if the sorted stencil kinds are [horizontal, square, vertical, horizontal], then the
#      old task outputs [A, A, A, A] with one fixed palette, while the fused task outputs
#      [A-id in palette A, A-mirror in palette B, A-rot90 in palette A, A-id in palette B].
#      Both geometry and color now vary per copy.
# b) Complexity calibration
#    - Geometry is limited to three visibly distinct shape classes, so the transform lookup
#      stays learnable.
#    - Palette alternation is binary and tied to copy parity, not to an arbitrary index list.
#    - The generator uses 4-6 copies, guarantees all three geometry classes appear, and makes
#      one class repeat across odd/even positions. This gives repeated evidence for both the
#      transform rule and the odd/even recolor rule in a single sample.
# c) Hidden-assumption audit of main()
#    - `_split_sections` still relies on full-width/full-height blue separators; unchanged.
#    - The original solver only needed the number of stencil components. That assumption is
#      broken here because every component now contributes its own transform and its position
#      along the axis determines both order and parity.
#    - The original solver recolored the template once. That no longer holds because color
#      assignment depends on copy index, so recoloring must happen inside the repetition loop.
#    - Rotation can change tile dimensions for non-square crops; therefore the solver pads the
#      cropped mask to a square before any transform. The generator also samples square crops.
#    - Component classification by bbox aspect ratio assumes stencil objects are exactly the
#      three allowed solids above; the generator enforces that.
# d) Placement legality
#    - Stencil components are laid out sequentially along the repeat axis with a black border
#      and gaps of at least two cells, so no two components can touch or merge.
#    - The orthogonal offset is randomized only within safe in-bounds ranges.
#    - Template, stencil, and swatches share the required common axis and are joined only by
#      one-cell blue separators, matching the source layout.
# e) Visual observability
#    - The template crop is forced to occupy its full square bbox and to differ under identity,
#      90-degree rotation, and mirror. Without this, the D7 branch could become invisible.
#    - A repeated shape class is intentionally placed on both odd and even positions, so a
#      human can see that parity changes colors independently of geometry.
#    - Ordering is recovered from visible spatial position only; no hidden color-id ordering or
#      numeric metadata is used.
# f) Visual cleanliness
#    - The stencil is sparse green-on-black with one component per local region.
#    - The output remains a single straight strip with one-cell gaps, so copies are easy to
#      segment despite alternating backgrounds.
#    - Only two palette colors are used in the output, preventing clutter.
# g) Color-change inferability
#    - The odd/even palette flip is the only color rule: there are exactly two swatch colors
#      and every output copy uses one of the two possible assignments.
#    - Because the generator uses at least four copies, both palette phases recur multiple
#      times, making the alternation visible from a small number of examples.
# h) Degeneracy checks
#    - `generate_input()` rejects any sample where the fused output matches the original rule,
#      the D7-only rule, or the D11-only rule on the same input. This directly guards against
#      "passes main() but does not really express the fusion" failures.
# i) New assumptions introduced by the fused solver
#    - Every stencil component must be classifiable as horizontal / vertical / square.
#    - All transformed copies must share one canvas size so they can be concatenated with
#      one-cell separators; the square-padding step and square generator crop guarantee this.
#    - The parity rule assumes a stable copy order, so the generator keeps components separated
#      and monotone along the output axis, and the solver sorts by visible position.


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1

REPEAT_COUNT_RANGE = (4, 6)
CROP_SIZE_RANGE = (3, 5)
DENSITY_RANGE = (0.50, 0.80)
COLOR_POOL = list(range(2, 10))
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
GEN_RETRIES = 500
SHAPE_RETRIES = 200

COMPONENT_SHAPES = {
    "horizontal": np.full((1, 3), STENCIL_COLOR, dtype=int),
    "vertical": np.full((3, 1), STENCIL_COLOR, dtype=int),
    "square": np.full((2, 2), STENCIL_COLOR, dtype=int),
}


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = SEPARATOR_COLOR

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


def _square_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    if h == w:
        return mask.astype(bool)
    side = max(h, w)
    out = np.zeros((side, side), dtype=bool)
    r0 = (side - h) // 2
    c0 = (side - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _component_kind(coords: np.ndarray) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "square"


def _sorted_components(pattern: np.ndarray, orientation: str) -> list[np.ndarray]:
    comps = components_mask(pattern != 0, connectivity=4)
    if orientation == "horizontal":
        return sorted(comps, key=lambda comp: (int(comp[:, 1].min()), int(comp[:, 0].min())))
    return sorted(comps, key=lambda comp: (int(comp[:, 0].min()), int(comp[:, 1].min())))


def _transform_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vertical":
        return np.rot90(mask, k=3)
    if kind == "square":
        return np.fliplr(mask)
    return mask


def _palette_for_index(index: int, fg_color: int, bg_color: int) -> tuple[int, int]:
    if index % 2 == 0:
        return int(fg_color), int(bg_color)
    return int(bg_color), int(fg_color)


def _render_copy(mask: np.ndarray, kind: str, fg_color: int, bg_color: int) -> np.ndarray:
    transformed = _transform_mask(mask, kind)
    return np.where(transformed, int(fg_color), int(bg_color)).astype(int)


def _join_tiles(tiles: list[np.ndarray], gap_colors: list[int], horizontal: bool) -> np.ndarray:
    if len(tiles) == 1:
        return tiles[0]

    pieces: list[np.ndarray] = [tiles[0]]
    for idx, tile in enumerate(tiles[1:]):
        gap_color = int(gap_colors[idx])
        if horizontal:
            sep = np.full((tiles[0].shape[0], 1), gap_color, dtype=int)
        else:
            sep = np.full((1, tiles[0].shape[1]), gap_color, dtype=int)
        pieces.extend([sep, tile])

    return np.hstack(pieces) if horizontal else np.vstack(pieces)


def _decode_problem(
    grid: np.ndarray,
) -> tuple[str, np.ndarray | None, list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return orientation, None, [], _solid_color(foreground_block), _solid_color(background_block)

    base_mask = crop_bbox(template != 0, template_bbox)
    square_mask = _square_mask(base_mask)
    components = _sorted_components(repetition_pattern, orientation)
    return (
        orientation,
        square_mask,
        components,
        _solid_color(foreground_block),
        _solid_color(background_block),
    )


def _render_sequence(
    base_mask: np.ndarray,
    components: list[np.ndarray],
    fg_color: int,
    bg_color: int,
    horizontal: bool,
    *,
    use_transform: bool,
    use_alternating_palette: bool,
) -> np.ndarray:
    if not components:
        use_kind = "horizontal"
        local_fg, local_bg = _palette_for_index(0, fg_color, bg_color) if use_alternating_palette else (fg_color, bg_color)
        return _render_copy(base_mask, use_kind, local_fg, local_bg)

    tiles: list[np.ndarray] = []
    gap_colors: list[int] = []
    for idx, comp in enumerate(components):
        kind = _component_kind(comp) if use_transform else "horizontal"
        local_fg, local_bg = (
            _palette_for_index(idx, fg_color, bg_color)
            if use_alternating_palette
            else (int(fg_color), int(bg_color))
        )
        tiles.append(_render_copy(base_mask, kind, local_fg, local_bg))
        if idx < len(components) - 1:
            gap_colors.append(int(local_bg))
    return _join_tiles(tiles, gap_colors, horizontal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a strip of transformed template copies whose colors alternate by copy parity.

    Input structure:
    - Blue (1) full rows or columns split the input into four sections.
    - Section 1 is a yellow-on-black template, section 2 is a green stencil, and sections 3
      and 4 are solid swatches that provide the two output colors.

    Rule:
    - Crop section 1 to the bbox of its non-black cells, pad that crop to a square canvas,
      and sort the 4-connected stencil components in section 2 along the output axis.
    - For each component, use its shape to choose the copy transform: horizontal bar keeps
      the tile, vertical bar rotates it 90 degrees clockwise, and square block mirrors it.
    - Emit one copy per component. Odd-position copies use section3 as foreground and section4
      as background; even-position copies swap those roles. Insert a one-cell separator after
      each copy using that copy's background color.

    Key insight: The stencil now controls two independent per-copy properties at once:
    visible component geometry selects the transform, while visible copy order selects the
    alternating color mapping.
    """
    orientation, base_mask, components, fg_color, bg_color = _decode_problem(grid)
    if base_mask is None:
        return np.empty((0, 0), dtype=int)
    return _render_sequence(
        base_mask,
        components,
        fg_color,
        bg_color,
        orientation == "horizontal",
        use_transform=True,
        use_alternating_palette=True,
    )


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


def _sample_shape(size: int) -> np.ndarray | None:
    target_bbox = (0, 0, size - 1, size - 1)
    for _ in range(SHAPE_RETRIES):
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            size,
            size,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        mask = shape != 0
        if bbox(mask) != target_bbox:
            continue
        if int(mask.sum()) <= 1 or int(mask.sum()) >= size * size - 1:
            continue

        variants = {
            mask.tobytes(),
            np.rot90(mask, k=3).tobytes(),
            np.fliplr(mask).tobytes(),
        }
        if len(variants) < 3:
            continue
        return shape
    return None


def _sample_component_kinds(n_blobs: int) -> list[str]:
    base = ["horizontal", "vertical", "square"]
    while True:
        kinds = base.copy()
        while len(kinds) < n_blobs:
            kinds.append(random.choice(base))
        random.shuffle(kinds)

        parity_hits = {kind: set() for kind in base}
        for idx, kind in enumerate(kinds):
            parity_hits[kind].add(idx % 2)

        if all(parity_hits[kind] for kind in base) and any(len(parities) == 2 for parities in parity_hits.values()):
            return kinds


def _build_stencil(kinds: list[str], shared_dim: int, horizontal: bool) -> np.ndarray:
    components = [COMPONENT_SHAPES[kind] for kind in kinds]
    axis_sizes = [comp.shape[1] if horizontal else comp.shape[0] for comp in components]
    leading_gap = random.randint(1, 2)
    trailing_gap = random.randint(1, 2) + random.randint(0, STENCIL_VARYING_EXTRA)
    inner_gaps = [random.randint(2, 3) for _ in range(len(components) - 1)]
    varying_dim = leading_gap + sum(axis_sizes) + sum(inner_gaps) + trailing_gap

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    pos = leading_gap
    for idx, comp in enumerate(components):
        ch, cw = comp.shape
        if horizontal:
            row = random.randint(1, h - ch - 1)
            col = pos
            grid[row : row + ch, col : col + cw] = comp
            pos += cw
        else:
            row = pos
            col = random.randint(1, w - cw - 1)
            grid[row : row + ch, col : col + cw] = comp
            pos += ch
        if idx < len(inner_gaps):
            pos += inner_gaps[idx]

    return grid


def _variant_signatures(grid: np.ndarray) -> dict[str, np.ndarray]:
    orientation, base_mask, components, fg_color, bg_color = _decode_problem(grid)
    assert base_mask is not None
    horizontal = orientation == "horizontal"
    return {
        "combo": _render_sequence(
            base_mask,
            components,
            fg_color,
            bg_color,
            horizontal,
            use_transform=True,
            use_alternating_palette=True,
        ),
        "d7_only": _render_sequence(
            base_mask,
            components,
            fg_color,
            bg_color,
            horizontal,
            use_transform=True,
            use_alternating_palette=False,
        ),
        "d11_only": _render_sequence(
            base_mask,
            components,
            fg_color,
            bg_color,
            horizontal,
            use_transform=False,
            use_alternating_palette=True,
        ),
        "original": _render_sequence(
            base_mask,
            components,
            fg_color,
            bg_color,
            horizontal,
            use_transform=False,
            use_alternating_palette=False,
        ),
    }


def _variant_is_visible(grid: np.ndarray) -> bool:
    orientation, _base_mask, components, fg_color, bg_color = _decode_problem(grid)
    del orientation, fg_color, bg_color
    if len(components) < REPEAT_COUNT_RANGE[0]:
        return False

    kinds = [_component_kind(comp) for comp in components]
    if len(set(kinds)) < 3:
        return False

    parity_hits = {kind: set() for kind in ("horizontal", "vertical", "square")}
    for idx, kind in enumerate(kinds):
        parity_hits[kind].add(idx % 2)
    if not any(len(parities) == 2 for parities in parity_hits.values()):
        return False

    variants = _variant_signatures(grid)
    combo = variants["combo"]
    if np.array_equal(combo, variants["d7_only"]):
        return False
    if np.array_equal(combo, variants["d11_only"]):
        return False
    if np.array_equal(combo, variants["original"]):
        return False

    colors = set(map(int, combo.flatten()))
    return len(colors) == 2


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)
    kinds = _sample_component_kinds(n_repeats)

    crop_size = random.randint(*CROP_SIZE_RANGE)
    shape = _sample_shape(crop_size)
    if shape is None:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    base_mask = shape != 0
    output = _render_sequence(
        base_mask,
        [np.argwhere(COMPONENT_SHAPES[kind] != 0) for kind in kinds],
        fg_color,
        bg_color,
        horizontal,
        use_transform=True,
        use_alternating_palette=True,
    )

    template = np.zeros((crop_size + 2, crop_size + 2), dtype=int)
    template[1 : crop_size + 1, 1 : crop_size + 1] = shape

    shared_dim = crop_size + 2
    stencil = _build_stencil(kinds, shared_dim, horizontal)
    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output) and _variant_is_visible(inp):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid combo input after {GEN_RETRIES} attempts")
