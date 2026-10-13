import random
from collections import Counter, deque

import numpy as np


BG = 0
DIVIDER = 5
OUTPUT_COLOR_BY_KIND = {"H": 1, "V": 3, "HV": 4, "D": 6}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INPUT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask):
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                continue
            cell = (nr, nc)
            if cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen) == int(mask.sum())


def _mode(values):
    counts = Counter(int(v) for v in values)
    top = counts.most_common(2)
    assert top
    if len(top) > 1:
        assert top[0][1] > top[1][1], "majority vote must be unique"
    return int(top[0][0])


def _spans_from_cuts(length, cuts):
    cuts = sorted(int(c) for c in cuts)
    spans = []
    prev = -1
    for cut in cuts + [length]:
        if cut - prev <= 1:
            return None
        spans.append((prev + 1, cut))
        prev = cut
    return spans


def _find_layout(grid):
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        row_spans = _spans_from_cuts(h, rows)
        col_spans = _spans_from_cuts(w, cols)
        if row_spans is None or col_spans is None:
            continue
        if len(row_spans) not in (3, 4) or len(col_spans) not in (3, 4):
            continue
        if len({r1 - r0 for r0, r1 in row_spans}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_spans}) != 1:
            continue
        candidates.append((int(color), row_spans, col_spans))
    assert len(candidates) == 1, "expected one separator-grid interpretation"
    return candidates[0]


def _erase_isolated_pixels(tile, bg):
    out = tile.copy()
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            color = int(tile[r, c])
            if color == bg:
                continue
            same_neighbor = False
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(tile.shape, nr, nc) and int(tile[nr, nc]) == color:
                    same_neighbor = True
                    break
            if not same_neighbor:
                out[r, c] = bg
    return out


def _consensus_tile(tiles):
    tile_h, tile_w = tiles[0].shape
    consensus = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            consensus[r, c] = _mode(tile[r, c] for tile in tiles)
    return consensus


def _symmetry_kind(mask):
    cropped = _crop_to_bbox(mask)
    horizontal = np.array_equal(cropped, np.flipud(cropped))
    vertical = np.array_equal(cropped, np.fliplr(cropped))
    diagonal = False
    if cropped.shape[0] == cropped.shape[1]:
        reflected_anti = np.fliplr(np.flipud(cropped)).T
        diagonal = (
            np.array_equal(cropped, cropped.T)
            or np.array_equal(cropped, reflected_anti)
            or np.array_equal(cropped, np.rot90(cropped))
        )
    if horizontal and vertical:
        return "HV"
    if diagonal:
        return "D"
    if horizontal:
        return "H"
    if vertical:
        return "V"
    raise AssertionError("consensus shape must match a supported symmetry class")


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the repeated tile consensus after removing singleton noise, then recolor it by symmetry and mark the damaged cells.

    Input structure:
    - A single separator color forms full-width and full-height 1-cell divider lines that partition the grid into an equal tile array.
    - Each tile contains the same monochrome foreground shape on a black background, plus optional isolated noise pixels of other colors.
    - Exactly one tile is damaged: some foreground cells from the shared shape are erased to background.

    Rule:
    - Split the grid into tiles using the separator lines, and within each tile erase every non-background pixel that has no 4-neighbor of the same color.
    - Perform per-coordinate majority voting across the cleaned tiles to reconstruct the consensus tile.
    - Find the unique cleaned tile that still differs from that consensus; its missing foreground coordinates are the damaged cells.
    - Recolor the consensus shape by its symmetry class: horizontal only -> blue, vertical only -> green, diagonal/quarter-turn family -> magenta, both horizontal and vertical -> yellow.
    - Output one tile-sized grid containing that recolored consensus shape, except that the damaged coordinates are shown in grey.

    Key insight:
    - The singleton noise must be erased before consensus voting, otherwise repeated noise can fake extra consensus pixels and change the symmetry-based recoloring branch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _separator, row_spans, col_spans = _find_layout(g)
    bg = BG
    tiles = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tiles.append(g[r0:r1, c0:c1].copy())

    cleaned_tiles = [_erase_isolated_pixels(tile, bg) for tile in tiles]
    consensus = _consensus_tile(cleaned_tiles)

    damaged = []
    for index, tile in enumerate(cleaned_tiles):
        diff = tile != consensus
        if not np.any(diff):
            continue
        missing = (consensus != bg) & (tile == bg)
        assert np.array_equal(diff, missing), "the damaged tile should only lose consensus pixels"
        damaged.append((index, missing))
    assert len(damaged) == 1, "expected exactly one damaged tile"
    missing_mask = damaged[0][1]
    assert int(missing_mask.sum()) >= 1

    consensus_mask = consensus != bg
    kind = _symmetry_kind(consensus_mask)
    out = np.full(consensus.shape, bg, dtype=int)
    out[consensus_mask] = OUTPUT_COLOR_BY_KIND[kind]
    out[missing_mask] = DIVIDER
    return out


def _random_connected_cells(height, width, target, start):
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))
    while frontier and len(cells) < target:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _cells_to_mask(shape, cells):
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mirror_horizontal(cells, height):
    mirrored = set()
    for r, c in cells:
        mirrored.add((r, c))
        mirrored.add((height - 1 - r, c))
    return mirrored


def _mirror_vertical(cells, width):
    mirrored = set()
    for r, c in cells:
        mirrored.add((r, c))
        mirrored.add((r, width - 1 - c))
    return mirrored


def _generate_h_or_v_patch(height, width, kind):
    if kind == "H":
        seed_h, seed_w = (height + 1) // 2, width
        start = (seed_h - 1, random.randrange(seed_w))
        limit = max(3, min(seed_h * seed_w, seed_h + seed_w + 1))
        target = random.randint(max(2, seed_w // 2), limit)
        seed = _random_connected_cells(seed_h, seed_w, target, start)
        mask = _cells_to_mask((height, width), _mirror_horizontal(seed, height))
    else:
        seed_h, seed_w = height, (width + 1) // 2
        start = (random.randrange(seed_h), seed_w - 1)
        limit = max(3, min(seed_h * seed_w, seed_h + seed_w + 1))
        target = random.randint(max(2, seed_h // 2), limit)
        seed = _random_connected_cells(seed_h, seed_w, target, start)
        mask = _cells_to_mask((height, width), _mirror_vertical(seed, width))
    return _crop_to_bbox(mask)


def _generate_hv_patch(height, width):
    seed_h, seed_w = (height + 1) // 2, (width + 1) // 2
    start = (seed_h - 1, seed_w - 1)
    limit = max(3, min(seed_h * seed_w, seed_h + seed_w + 1))
    target = random.randint(2, limit)
    seed = _random_connected_cells(seed_h, seed_w, target, start)
    cells = _mirror_horizontal(seed, height)
    cells = _mirror_vertical(cells, width)
    return _crop_to_bbox(_cells_to_mask((height, width), cells))


def _diag_orbits(size):
    orbits = []
    for r in range(size):
        for c in range(r, size):
            if r == c:
                orbit = ((r, c),)
            else:
                orbit = ((r, c), (c, r))
            orbits.append(orbit)
    return orbits


def _orbit_touches(cells, orbit):
    for r, c in orbit:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in cells:
                return True
    return False


def _generate_diag_patch(size):
    orbits = _diag_orbits(size)
    diagonal_orbits = [orbit for orbit in orbits if len(orbit) == 1]
    start = random.choice(diagonal_orbits)
    used = {start[0]}
    cells = set(start)
    max_area = min(size * size - 1, size * 2 + 5)
    target = random.randint(size + 1, max(size + 1, max_area))
    while len(cells) < target:
        choices = [orbit for orbit in orbits if orbit[0] not in used and _orbit_touches(cells, orbit)]
        if not choices:
            break
        orbit = random.choice(choices)
        used.add(orbit[0])
        cells.update(orbit)
    mask = _cells_to_mask((size, size), cells)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _crop_to_bbox(mask)


def _place_patch(tile_h, tile_w, patch):
    patch_h, patch_w = patch.shape
    off_r = random.randint(0, tile_h - patch_h)
    off_c = random.randint(0, tile_w - patch_w)
    mask = np.zeros((tile_h, tile_w), dtype=bool)
    mask[off_r : off_r + patch_h, off_c : off_c + patch_w] = patch
    return mask


def _generate_base_mask(tile_h, tile_w, kind):
    for _ in range(200):
        if kind in ("H", "V"):
            patch_h = random.randint(3, tile_h)
            patch_w = random.randint(3, tile_w)
            patch = _generate_h_or_v_patch(patch_h, patch_w, kind)
        elif kind == "HV":
            patch_h = random.randint(3, tile_h)
            patch_w = random.randint(3, tile_w)
            if patch_h == patch_w:
                continue
            patch = _generate_hv_patch(patch_h, patch_w)
        else:
            side = random.randint(4, min(tile_h, tile_w))
            patch = _generate_diag_patch(side)

        mask = _place_patch(tile_h, tile_w, patch)
        area = int(mask.sum())
        if area < 5 or area > tile_h * tile_w - 3:
            continue
        if not _is_connected(mask):
            continue
        if _symmetry_kind(mask) != kind:
            continue
        return mask
    return None


def _noise_candidates(tile, bg):
    shape_mask = tile != bg
    candidates = []
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            if int(tile[r, c]) != bg:
                continue
            if any(
                _in_bounds(tile.shape, r + dr, c + dc) and shape_mask[r + dr, c + dc]
                for dr, dc in NEIGHBORS4
            ):
                continue
            candidates.append((r, c))
    return candidates


def _sample_connected_subset(mask, size):
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    coord_set = set(coords)
    start = random.choice(coords)
    subset = {start}
    frontier = {cell for cell in coord_set if cell != start and any((cell[0] + dr, cell[1] + dc) == start for dr, dc in NEIGHBORS4)}
    while frontier and len(subset) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        subset.add(cell)
        for dr, dc in NEIGHBORS4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in coord_set and nbr not in subset:
                frontier.add(nbr)
    if len(subset) == size:
        return subset
    return None


def _apply_damage(tile, bg):
    shape_mask = tile != bg
    area = int(shape_mask.sum())
    for _ in range(200):
        remove_size = random.randint(2, min(4, area - 2))
        removed = _sample_connected_subset(shape_mask, remove_size)
        if removed is None:
            continue
        damaged = tile.copy()
        for r, c in removed:
            damaged[r, c] = bg
        cleaned = _erase_isolated_pixels(damaged, bg)
        missing_mask = (tile != bg) & (cleaned == bg)
        if int(missing_mask.sum()) != remove_size:
            continue
        remaining = cleaned != bg
        if int(remaining.sum()) < 2 or not _is_connected(remaining):
            continue
        return damaged, missing_mask
    return None, None


def _add_noise(tile, candidate_positions, color_choices, trap_pos, trap_color, include_trap):
    out = tile.copy()
    used_positions = set()
    used_colors = set()
    if include_trap:
        out[trap_pos] = trap_color
        used_positions.add(trap_pos)
        used_colors.add(trap_color)

    if random.random() < 0.75:
        choices = []
        for pos in candidate_positions:
            if pos in used_positions:
                continue
            if any(abs(pos[0] - other[0]) + abs(pos[1] - other[1]) == 1 for other in used_positions):
                continue
            choices.append(pos)
        if choices:
            pos = random.choice(choices)
            colors = [color for color in color_choices if color not in used_colors]
            if colors:
                out[pos] = random.choice(colors)
    return out


def _render_grid(tile_rows, tile_cols, tile_h, tile_w, tiles):
    grid_h = tile_rows * tile_h + (tile_rows - 1)
    grid_w = tile_cols * tile_w + (tile_cols - 1)
    grid = np.full((grid_h, grid_w), DIVIDER, dtype=int)
    index = 0
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            r0 = tr * (tile_h + 1)
            c0 = tc * (tile_w + 1)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tiles[index]
            index += 1
    return grid


def _expected_output(base_mask, missing_mask, kind):
    out = np.full(base_mask.shape, BG, dtype=int)
    out[base_mask] = OUTPUT_COLOR_BY_KIND[kind]
    out[missing_mask] = DIVIDER
    return out


def _generate_case():
    tile_rows = random.choice([3, 4])
    tile_cols = 3
    tile_h = random.randint(4, 6)
    tile_w = random.randint(4, 6)
    kind = random.choice(["H", "V", "D", "HV"])
    shape_color = random.choice(INPUT_COLORS)

    base_mask = _generate_base_mask(tile_h, tile_w, kind)
    if base_mask is None:
        return None

    mother = np.full((tile_h, tile_w), BG, dtype=int)
    mother[base_mask] = shape_color

    noise_spots = _noise_candidates(mother, BG)
    if len(noise_spots) < 3:
        return None

    damaged_tile, missing_mask = _apply_damage(mother, BG)
    if damaged_tile is None:
        return None

    n_tiles = tile_rows * tile_cols
    damaged_index = random.randrange(n_tiles)
    trap_pos = random.choice(noise_spots)
    noise_palette = [color for color in INPUT_COLORS if color != shape_color]
    trap_color = random.choice(noise_palette)
    trap_count = random.randint(n_tiles // 2 + 1, n_tiles - 1)
    trap_tiles = set(random.sample(range(n_tiles), trap_count))

    tiles = []
    for index in range(n_tiles):
        tile = damaged_tile.copy() if index == damaged_index else mother.copy()
        tile = _add_noise(
            tile,
            noise_spots,
            noise_palette,
            trap_pos,
            trap_color,
            index in trap_tiles,
        )
        tiles.append(tile)

    grid = _render_grid(tile_rows, tile_cols, tile_h, tile_w, tiles)
    expected = _expected_output(base_mask, missing_mask, kind)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid consensus-repair puzzle")
