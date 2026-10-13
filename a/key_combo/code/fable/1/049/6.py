from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Box = tuple[int, int, int, int]

NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 500


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one cell"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty tiles"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final tile band"
    spans.append((start, length - 1))
    return spans


def _single_coord(tile: np.ndarray, color: int) -> Coord:
    coords = np.argwhere(tile == int(color))
    assert len(coords) == 1, "expected exactly one anchor pixel"
    return tuple(map(int, coords[0]))


def _array_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in np.asarray(arr).tolist())


def _dihedral_variants(tile: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(tile, k)
        variants.append(rotated.copy())
        variants.append(np.fliplr(rotated).copy())
    return variants


def _try_parse_scene(grid: np.ndarray, separator: int) -> dict | None:
    height, width = map(int, grid.shape)
    sep_rows = [r for r in range(height) if np.all(grid[r, :] == int(separator))]
    sep_cols = [c for c in range(width) if np.all(grid[:, c] == int(separator))]
    if not sep_rows or not sep_cols:
        return None

    try:
        row_spans = _spans(height, sep_rows)
        col_spans = _spans(width, sep_cols)
    except AssertionError:
        return None

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    if len(tile_heights) != 1 or len(tile_widths) != 1:
        return None

    tile_h = next(iter(tile_heights))
    tile_w = next(iter(tile_widths))
    if tile_h != tile_w or tile_h < 2:
        return None

    try:
        bg = _dominant_color(grid[grid != int(separator)])
    except AssertionError:
        return None

    tiles: list[tuple[Box, np.ndarray]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tiles.append(((r0, c0, r1, c1), grid[r0 : r1 + 1, c0 : c1 + 1]))

    template_indices = [i for i, (_box, tile) in enumerate(tiles) if int(np.sum(tile != bg)) > 2]
    if len(template_indices) != 1:
        return None
    template_index = template_indices[0]
    template_tile = tiles[template_index][1]

    template_colors = Counter(map(int, template_tile[template_tile != bg]))
    outside_colors = Counter()
    for i, (_box, tile) in enumerate(tiles):
        if i == template_index:
            continue
        outside_colors.update(map(int, tile[tile != bg]))

    anchors = sorted(
        color
        for color, count in template_colors.items()
        if count == 1 and outside_colors[color] > 0
    )
    if len(anchors) != 2:
        return None
    anchor_a, anchor_b = anchors

    variants = _dihedral_variants(template_tile)
    anchor_states = [(_single_coord(variant, anchor_a), _single_coord(variant, anchor_b)) for variant in variants]
    if len(set(anchor_states)) != 8:
        return None

    placements: list[tuple[Box, np.ndarray]] = []
    for i, (box, tile) in enumerate(tiles):
        if i == template_index:
            continue

        values = tile[tile != bg]
        if values.size == 0:
            continue
        if Counter(map(int, values)) != Counter({anchor_a: 1, anchor_b: 1}):
            return None

        pair = (_single_coord(tile, anchor_a), _single_coord(tile, anchor_b))
        matches = [variant for variant, state in zip(variants, anchor_states) if state == pair]
        if len(matches) != 1:
            return None
        placements.append((box, matches[0]))

    if not placements:
        return None

    return {
        "separator": int(separator),
        "bg": int(bg),
        "tile_size": int(tile_h),
        "template_index": int(template_index),
        "template_tile": template_tile.copy(),
        "placements": placements,
    }


def _find_scene(grid: np.ndarray) -> dict:
    matches = []
    for separator in map(int, np.unique(grid)):
        scene = _try_parse_scene(np.asarray(grid, dtype=int), separator)
        if scene is not None:
            matches.append(scene)
    assert len(matches) == 1, "expected exactly one valid separator tiling interpretation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct every residue-only tile by stamping the template at the unique pose fixed by its two anchors.

    Input structure:
    - Full separator rows and columns partition the grid into equal square tiles.
    - Exactly one tile contains the complete multicolor template, including one unique A-colored pixel and one unique B-colored pixel.
    - Other nonempty tiles contain only one A pixel and one B pixel on background; empty tiles contain only background.

    Rule:
    - Infer the separator color and tile geometry from the full-row and full-column lines.
    - Use the unique full template tile to enumerate its 8 dihedral transforms, and for each residue tile find the only transform whose A/B anchor coordinates match that tile's two pixels.
    - Replace each residue tile with the full transformed template while leaving the template tile, blank tiles, and separator lines unchanged.

    Key insight:
    - The two colored residue pixels are not decorations: they are transformed anchor points from the missing template, and their ordered positions uniquely determine the needed rotation or reflection.
    """
    g = np.asarray(grid, dtype=int)
    scene = _find_scene(g)

    out = g.copy()
    for (r0, c0, r1, c1), variant in scene["placements"]:
        out[r0 : r1 + 1, c0 : c1 + 1] = variant
    return out


def _feasible_layouts() -> list[tuple[int, int, list[int]]]:
    layouts: list[tuple[int, int, list[int]]] = []
    for tile_rows in range(2, 5):
        for tile_cols in range(2, 5):
            if tile_rows * tile_cols < 6:
                continue
            max_size = min(
                (30 - (tile_rows - 1)) // tile_rows,
                (30 - (tile_cols - 1)) // tile_cols,
            )
            sizes = [size for size in range(5, 8) if size <= max_size]
            if sizes:
                layouts.append((tile_rows, tile_cols, sizes))
    return layouts


GEN_LAYOUTS = _feasible_layouts()


def _build_base_grid(tile_rows: int, tile_cols: int, tile_size: int, bg: int, separator: int) -> np.ndarray:
    height = tile_rows * tile_size + (tile_rows - 1)
    width = tile_cols * tile_size + (tile_cols - 1)
    grid = np.full((height, width), int(bg), dtype=int)
    for r in range(1, tile_rows):
        grid[r * tile_size + (r - 1), :] = int(separator)
    for c in range(1, tile_cols):
        grid[:, c * tile_size + (c - 1)] = int(separator)
    return grid


def _tile_origin(tile_r: int, tile_c: int, tile_size: int) -> Coord:
    step = tile_size + 1
    return int(tile_r * step), int(tile_c * step)


def _random_connected_mask(size: int) -> np.ndarray:
    target_area = random.randint(size + 2, min(size * size - 2, size * size // 2 + size))
    mask = np.zeros((size, size), dtype=bool)
    mask[random.randrange(size), random.randrange(size)] = True

    while int(mask.sum()) < target_area:
        frontier: set[Coord] = set()
        for r, c in np.argwhere(mask):
            rr, cc = int(r), int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True

    return mask


def _random_template_tile(tile_size: int, bg: int, body: int, accent: int, anchor_a: int, anchor_b: int) -> np.ndarray | None:
    for _ in range(300):
        mask = _random_connected_mask(tile_size)
        coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
        if len(coords) < 7:
            continue
        if len({r for r, _c in coords}) < 3 or len({c for _r, c in coords}) < 3:
            continue

        anchors = random.sample(coords, 2)
        remaining = [coord for coord in coords if coord not in anchors]
        accent_count = random.choice((2, 3))
        if len(remaining) < accent_count + 2:
            continue
        accents = random.sample(remaining, accent_count)

        tile = np.full((tile_size, tile_size), int(bg), dtype=int)
        tile[mask] = int(body)
        for r, c in accents:
            tile[r, c] = int(accent)
        tile[anchors[0]] = int(anchor_a)
        tile[anchors[1]] = int(anchor_b)

        if int(np.count_nonzero(tile == int(body))) < 2 or int(np.count_nonzero(tile == int(accent))) < 2:
            continue

        variants = _dihedral_variants(tile)
        if len({_array_key(variant) for variant in variants}) != 8:
            continue
        anchor_states = {
            (_single_coord(variant, anchor_a), _single_coord(variant, anchor_b))
            for variant in variants
        }
        if len(anchor_states) != 8:
            continue
        return tile

    return None


def _sample_transform_ids(count: int) -> list[int]:
    chosen = {random.choice((0, 2, 4, 6)), random.choice((1, 3, 5, 7))}
    while len(chosen) < count:
        chosen.add(random.randrange(8))
    return random.sample(list(chosen), count)


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows, tile_cols, size_choices = random.choice(GEN_LAYOUTS)
    tile_size = random.choice(size_choices)
    bg, separator, body, accent, anchor_a, anchor_b = random.sample(range(10), 6)

    template = _random_template_tile(tile_size, bg, body, accent, anchor_a, anchor_b)
    if template is None:
        return None

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    template_pos = random.choice(positions)
    other_positions = [pos for pos in positions if pos != template_pos]

    max_targets = min(len(other_positions) - 1, 6)
    if max_targets < 3:
        return None
    min_targets = 4 if max_targets >= 4 else 3
    target_count = random.randint(min_targets, max_targets)
    target_positions = random.sample(other_positions, target_count)
    transform_ids = _sample_transform_ids(target_count)

    expected = _build_base_grid(tile_rows, tile_cols, tile_size, bg, separator)
    inp = expected.copy()

    template_r0, template_c0 = _tile_origin(template_pos[0], template_pos[1], tile_size)
    expected[template_r0 : template_r0 + tile_size, template_c0 : template_c0 + tile_size] = template
    inp[template_r0 : template_r0 + tile_size, template_c0 : template_c0 + tile_size] = template

    variants = _dihedral_variants(template)
    for (tile_r, tile_c), transform_id in zip(target_positions, transform_ids):
        variant = variants[transform_id]
        r0, c0 = _tile_origin(tile_r, tile_c, tile_size)
        expected[r0 : r0 + tile_size, c0 : c0 + tile_size] = variant

        residue = np.full((tile_size, tile_size), int(bg), dtype=int)
        residue[_single_coord(variant, anchor_a)] = int(anchor_a)
        residue[_single_coord(variant, anchor_b)] = int(anchor_b)
        inp[r0 : r0 + tile_size, c0 : c0 + tile_size] = residue

    try:
        reconstructed = main(inp)
    except Exception:
        return None
    if not np.array_equal(reconstructed, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        generated = _generate_attempt()
        if generated is not None:
            inp, _expected = generated
            return inp
    raise RuntimeError("failed to generate a valid dual-anchor stamp puzzle")
