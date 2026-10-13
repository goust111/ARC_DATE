from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
ALL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _coords_to_tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in np.asarray(mask, dtype=bool))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[str, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        for cand in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(cand)
            key = (tuple(map(int, tight.shape)), _mask_key(tight))
            if key in seen:
                continue
            seen.add(key)
            variants.append(tight)
    return variants


def _canonical_shape(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(variant) for variant in _dihedral_variants(mask))


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _row_groups(active_rows: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx, active in enumerate(np.asarray(active_rows, dtype=bool)):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(active_rows) - 1))
    return groups


def _enclosed_component(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    enclosed: list[np.ndarray] = []
    for comp in _components(mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 1] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == w - 1):
            continue
        enclosed.append(comp)
    if not enclosed:
        return None
    chosen = max(enclosed, key=lambda comp: int(comp.shape[0]))
    out = np.zeros_like(mask, dtype=bool)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
]
SHAPE_SIGNATURES = [_canonical_shape(shape) for shape in SHAPE_LIBRARY]
assert len(set(SHAPE_SIGNATURES)) == len(SHAPE_LIBRARY)
SHAPE_VARIANTS = [_dihedral_variants(shape) for shape in SHAPE_LIBRARY]


def _parse_palette(grid: np.ndarray, bg: int) -> tuple[dict[tuple[str, ...], int], np.ndarray]:
    grey_rows = np.any(grid == GREY, axis=1)
    groups = _row_groups(grey_rows)
    mapping: dict[tuple[str, ...], int] = {}
    palette_mask = np.zeros_like(grid, dtype=bool)

    for r0, r1 in groups:
        band = grid[r0 : r1 + 1, :]
        comps = _components(band != bg)
        if not comps:
            continue

        infos = []
        for comp in comps:
            abs_comp = comp.copy()
            abs_comp[:, 0] += r0
            min_col = int(abs_comp[:, 1].min())
            max_col = int(abs_comp[:, 1].max())
            has_grey = bool(np.any(grid[abs_comp[:, 0], abs_comp[:, 1]] == GREY))
            infos.append((min_col, max_col, has_grey, abs_comp))

        grey_infos = [info for info in infos if info[2]]
        if not grey_infos:
            continue
        shape_info = min(grey_infos, key=lambda info: info[0])
        block_candidates = [info for info in infos if not info[2] and info[0] > shape_info[1]]
        if not block_candidates:
            continue
        block_info = min(block_candidates, key=lambda info: info[0])

        shape_coords = shape_info[3]
        block_coords = block_info[3]
        shape_mask = _coords_to_tight_mask(shape_coords)
        block_colors = np.unique(grid[block_coords[:, 0], block_coords[:, 1]])
        block_colors = block_colors[block_colors != bg]
        if block_colors.size != 1:
            continue

        signature = _canonical_shape(shape_mask)
        mapping[signature] = int(block_colors[0])
        palette_mask[shape_coords[:, 0], shape_coords[:, 1]] = True
        palette_mask[block_coords[:, 0], block_coords[:, 1]] = True

    if not mapping:
        raise AssertionError("palette legend not found")
    return mapping, palette_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region using the palette color keyed by its interior hole shape.

    Input structure:
    - The background is the dominant color. The left legend area contains one grey feature shape and one colored key pixel per row.
    - The right canvas contains several separated non-background components, each formed by a colored border enclosing a filled interior with one background-colored hole.
    - Legend shapes are small connected masks within 3x3, while hole masks may appear rotated or mirrored inside larger framed regions.

    Rule:
    - Read each legend row as a shape-to-color mapping by extracting the grey shape mask and the colored key next to it.
    - For every closed region on the canvas, recover the enclosed interior, isolate its background hole mask, and normalize that hole up to rotation and reflection.
    - Recolor the entire enclosed interior, including the hole cells, with the unique palette color whose legend shape matches the normalized hole mask. Keep borders and the legend unchanged.

    Key insight:
    - The decisive object is the hole's negative-space shape, not the outer frame; matching is done after canonicalizing shapes over the full dihedral symmetry group.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, palette_mask = _parse_palette(g, bg)

    out = g.copy()
    canvas_mask = (g != bg) & (~palette_mask)
    for comp in _components(canvas_mask):
        r0, c0, r1, c1 = _bbox_coords(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        boundary_vals = np.concatenate(
            [
                sub[0, :],
                sub[-1, :],
                sub[:, 0],
                sub[:, -1],
            ]
        )
        boundary_vals = boundary_vals[boundary_vals != bg]
        if boundary_vals.size == 0:
            continue
        border_color = int(np.bincount(boundary_vals, minlength=10).argmax())

        interior_mask = _enclosed_component(sub != border_color)
        if interior_mask is None:
            continue

        hole_mask = interior_mask & (sub == bg)
        hole_components = _components(hole_mask)
        if not hole_components:
            continue
        hole_tight = _coords_to_tight_mask(max(hole_components, key=lambda coords: int(coords.shape[0])))
        fill_color = mapping[_canonical_shape(hole_tight)]

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[interior_mask] = int(fill_color)

    return out


def _pick_variant(shape_idx: int) -> np.ndarray:
    variants = SHAPE_VARIANTS[shape_idx]
    return variants[int(_RNG.integers(len(variants)))]


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paint_hole_frame(
    shape_mask: np.ndarray,
    border_color: int,
    fill_color: int,
    hole_top_pad: int,
    hole_left_pad: int,
    hole_bottom_pad: int,
    hole_right_pad: int,
) -> np.ndarray:
    hole_h, hole_w = map(int, shape_mask.shape)
    inner_h = hole_top_pad + hole_h + hole_bottom_pad
    inner_w = hole_left_pad + hole_w + hole_right_pad
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:-1, 1:-1] = fill_color
    hole_r0 = 1 + hole_top_pad
    hole_c0 = 1 + hole_left_pad
    patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w][shape_mask] = BG
    return patch


def _build_palette_rows(
    grid: np.ndarray,
    shape_indices: list[int],
    palette_colors: list[int],
    top_margin: int,
    left_margin: int,
    color_col: int,
) -> None:
    for row_idx, (shape_idx, color) in enumerate(zip(shape_indices, palette_colors)):
        band_top = top_margin + 4 * row_idx
        shape_mask = _pick_variant(shape_idx)
        sh, sw = map(int, shape_mask.shape)
        off_r = int(_RNG.integers(0, 4 - sh))
        off_c = int(_RNG.integers(0, 4 - sw))
        sr0 = band_top + off_r
        sc0 = left_margin + off_c
        grid[sr0 : sr0 + sh, sc0 : sc0 + sw][shape_mask] = GREY

        occupied_rows = np.where(shape_mask.any(axis=1))[0]
        block_row = sr0 + int(_RNG.choice(occupied_rows))
        grid[block_row, color_col] = int(color)


def _assigned_shape_indices(num_shapes: int, region_count: int) -> list[int]:
    assignments = list(range(num_shapes))
    while len(assignments) < region_count:
        assignments.append(int(_RNG.integers(num_shapes)))
    _RNG.shuffle(assignments)
    return assignments


def _gen_attempt() -> np.ndarray | None:
    num_shapes = int(_RNG.integers(3, 5))
    shape_indices = list(map(int, _RNG.choice(len(SHAPE_LIBRARY), size=num_shapes, replace=False)))

    palette_candidates = [color for color in ALL_COLORS]
    _RNG.shuffle(palette_candidates)
    palette_colors = palette_candidates[:num_shapes]
    remaining = [color for color in palette_candidates[num_shapes:] if color != GREY]
    if len(remaining) < 2:
        return None
    border_color = int(remaining[0])
    filler_pool = [int(color) for color in remaining[1:]]
    if not filler_pool:
        return None

    top_margin = int(_RNG.integers(1, 4))
    left_margin = int(_RNG.integers(1, 3))
    legend_gap = int(_RNG.integers(1, 3))
    color_col = left_margin + 3 + legend_gap
    palette_bottom = top_margin + 4 * (num_shapes - 1) + 2
    sep_gap = int(_RNG.integers(3, 5))
    canvas_left = color_col + 1 + sep_gap

    min_height = max(16, palette_bottom + 3)
    min_width = max(20, canvas_left + 10)
    if min_height > 30 or min_width > 30:
        return None

    height = int(_RNG.integers(min_height, 31))
    width = int(_RNG.integers(min_width, 31))
    grid = np.full((height, width), BG, dtype=int)

    _build_palette_rows(grid, shape_indices, palette_colors, top_margin, left_margin, color_col)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[:, : canvas_left - 1] = True

    region_count = num_shapes
    assignments = _assigned_shape_indices(num_shapes, region_count)
    region_specs = []

    for palette_slot in assignments:
        shape_idx = shape_indices[palette_slot]
        hole_mask = _pick_variant(shape_idx)
        hole_top_pad = int(_RNG.integers(1, 3))
        hole_left_pad = int(_RNG.integers(1, 3))
        hole_bottom_pad = int(_RNG.integers(1, 3))
        hole_right_pad = int(_RNG.integers(1, 3))
        filler_color = int(_RNG.choice(filler_pool))

        patch = _paint_hole_frame(
            hole_mask,
            border_color=border_color,
            fill_color=filler_color,
            hole_top_pad=hole_top_pad,
            hole_left_pad=hole_left_pad,
            hole_bottom_pad=hole_bottom_pad,
            hole_right_pad=hole_right_pad,
        )
        region_specs.append((int(patch.shape[0] * patch.shape[1]), palette_slot, patch))

    region_specs.sort(key=lambda item: -item[0])
    placements: list[tuple[int, int, int, int, int]] = []

    for _area, palette_slot, patch in region_specs:
        ph, pw = map(int, patch.shape)

        placed = False
        for _ in range(120):
            top = int(_RNG.integers(1, height - ph))
            left = int(_RNG.integers(canvas_left, width - pw))
            if not _can_place(occupied, top, left, ph, pw, pad=1):
                continue
            grid[top : top + ph, left : left + pw] = patch
            occupied[top : top + ph, left : left + pw] = True
            placements.append((top, left, ph, pw, palette_slot))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for top, left, ph, pw, palette_slot in placements:
        expected[top + 1 : top + ph - 1, left + 1 : left + pw - 1] = palette_colors[palette_slot]

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
