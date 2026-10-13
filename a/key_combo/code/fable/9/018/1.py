from __future__ import annotations

from collections import Counter, deque
from itertools import combinations

import numpy as np


PALETTE = tuple(range(10))
GEN_RETRIES = 400
GAP_ROWS = 1
MIN_CANDIDATE_HEIGHT = 7
MAX_SHAPE_SPAN = 4
MIN_SHAPE_CELLS = 3
MAX_SHAPE_CELLS = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected nonempty values for majority vote"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), -1, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    queue = deque([cells[0]])
    seen = {cells[0]}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _enumerate_shapes() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(MAX_SHAPE_SPAN) for c in range(MAX_SHAPE_SPAN)]
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for n_cells in range(MIN_SHAPE_CELLS, MAX_SHAPE_CELLS + 1):
        for combo in combinations(cells, n_cells):
            if not _is_connected(combo):
                continue
            normalized = _normalize_coords(np.array(combo, dtype=int))
            if normalized not in seen:
                seen.add(normalized)
                shapes.append(normalized)
    return shapes


ALL_SHAPES = _enumerate_shapes()
SHAPES_BY_AREA: dict[int, list[tuple[tuple[int, int], ...]]] = {}
for _shape in ALL_SHAPES:
    SHAPES_BY_AREA.setdefault(len(_shape), []).append(_shape)


TARGET_SHAPES = [
    shape
    for shape in ALL_SHAPES
    if any(_shape_dims(other) == _shape_dims(shape) for other in SHAPES_BY_AREA[len(shape)] if other != shape)
    and any(_shape_dims(other) != _shape_dims(shape) for other in SHAPES_BY_AREA[len(shape)] if other != shape)
]


def _connected_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _parse_lattice(grid: np.ndarray) -> tuple[int, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    for sep in sorted(set(map(int, g.reshape(-1)))):
        full_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
        if len(full_rows) < 2:
            continue
        tile_h = int(full_rows[0])
        if tile_h <= 0:
            continue
        step_h = tile_h + 1
        row_seps = []
        expected = tile_h
        for r in full_rows:
            if r == expected:
                row_seps.append(int(r))
                expected += step_h
            elif row_seps:
                break
        if len(row_seps) < 2:
            continue
        lattice_h = row_seps[-1] + tile_h + 1
        if lattice_h >= h:
            continue

        top = g[:lattice_h, :]
        full_cols = [c for c in range(w) if np.all(top[:, c] == sep)]
        if len(full_cols) < 2:
            continue
        tile_w = int(full_cols[0])
        if tile_w <= 0:
            continue
        step_w = tile_w + 1
        col_seps = []
        expected = tile_w
        for c in full_cols:
            if c == expected:
                col_seps.append(int(c))
                expected += step_w
            elif col_seps:
                break
        if len(col_seps) < 2:
            continue

        lattice_w = col_seps[-1] + tile_w + 1
        if lattice_w != w:
            continue
        if tile_h != tile_w:
            continue

        block_rows = len(row_seps) + 1
        block_cols = len(col_seps) + 1
        if block_rows < 3 or block_cols < 3:
            continue

        blocks = np.empty((block_rows, block_cols, tile_h, tile_w), dtype=int)
        ok = True
        for br in range(block_rows):
            r0 = br * step_h
            for bc in range(block_cols):
                c0 = bc * step_w
                patch = top[r0 : r0 + tile_h, c0 : c0 + tile_w]
                if patch.shape != (tile_h, tile_w) or np.any(patch == sep):
                    ok = False
                    break
                blocks[br, bc] = patch
            if not ok:
                break
        if ok:
            return int(sep), blocks, int(tile_h), int(block_rows), int(lattice_h)

    raise AssertionError("failed to locate a valid repeated-tile lattice")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the repeated motif, use the damage mask as a shape key, and recolor the repaired cells with the matching candidate color.

    Input structure:
    - The upper part of the grid is a lattice of equal square blocks separated
      by one-cell solid rows and columns of one separator color.
    - All blocks are copies of one motif except exactly one damaged block,
      where some motif foreground cells were overwritten by the motif background.
    - Below the lattice are several scattered monochromatic candidate shapes
      with distinct colors that do not appear in the motif.

    Rule:
    - Recover the motif by taking the unique majority color at each local block
      coordinate across all copies, then find the single block that differs.
    - Normalize the differing local coordinates into a shape mask and find the
      unique lower candidate with the same normalized shape.
    - Output one motif-sized grid: draw the recovered motif, then recolor the
      damaged local coordinates with the matched candidate's color.

    Key insight:
    - The missing-cell pattern is not noise: its normalized shape is the key
      that selects which candidate color should mark the repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, blocks, tile_size, _block_rows, lattice_h = _parse_lattice(g)
    motif = np.empty((tile_size, tile_size), dtype=int)
    for r in range(tile_size):
        for c in range(tile_size):
            motif[r, c] = _unique_mode(blocks[:, :, r, c])

    diffs = blocks != motif
    diff_counts = diffs.reshape(blocks.shape[0], blocks.shape[1], -1).sum(axis=2)
    damaged_blocks = np.argwhere(diff_counts > 0)
    assert damaged_blocks.shape[0] == 1, "expected exactly one damaged block"
    damaged_br, damaged_bc = map(int, damaged_blocks[0])
    damage_mask = diffs[damaged_br, damaged_bc]
    assert np.any(damage_mask), "damage mask must be nonempty"

    bg = _dominant_color(motif)
    damaged_block = blocks[damaged_br, damaged_bc]
    assert np.all(damaged_block[damage_mask] == bg), "damage must overwrite motif cells with the motif background"
    assert np.all(motif[damage_mask] != bg), "only motif foreground cells may be damaged"

    candidate_region = g[lattice_h:, :]
    motif_colors = set(map(int, np.unique(motif)))
    candidate_objects = []
    for color in sorted(set(map(int, candidate_region.reshape(-1)))):
        if color in motif_colors or color == separator:
            continue
        comps = _connected_components_of_color(candidate_region, color)
        if not comps:
            continue
        assert len(comps) == 1, "each candidate color must form one object"
        candidate_objects.append({"color": int(color), "coords": comps[0]})

    assert len(candidate_objects) >= 2, "expected multiple candidate shapes"
    target_shape = _normalize_coords(np.argwhere(damage_mask))
    matches = [obj for obj in candidate_objects if _normalize_coords(obj["coords"]) == target_shape]
    assert len(matches) == 1, "expected a unique candidate matching the damage mask"

    out = motif.copy()
    out[damage_mask] = int(matches[0]["color"])
    return out


def _choose_geometry(rng: np.random.Generator) -> tuple[int, int, int]:
    choices = []
    for tile_size in range(4, 7):
        for block_rows in range(3, 5):
            lattice_h = block_rows * tile_size + (block_rows - 1)
            if lattice_h + GAP_ROWS + MIN_CANDIDATE_HEIGHT <= 30:
                for block_cols in range(3, 5):
                    lattice_w = block_cols * tile_size + (block_cols - 1)
                    if lattice_w <= 30:
                        choices.append((tile_size, block_rows, block_cols))
    idx = int(rng.integers(len(choices)))
    return choices[idx]


def _choose_colors(rng: np.random.Generator) -> tuple[int, int, list[int], list[int]]:
    perm = rng.permutation(PALETTE).tolist()
    bg = int(perm[0])
    separator = int(perm[1])
    motif_count = int(rng.integers(2, 4))
    motif_colors = [int(c) for c in perm[2 : 2 + motif_count]]
    candidate_colors = [int(c) for c in perm[2 + motif_count : 6 + motif_count]]
    return bg, separator, motif_colors, candidate_colors


def _choose_target_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    idx = int(rng.integers(len(TARGET_SHAPES)))
    return TARGET_SHAPES[idx]


def _build_candidate_shapes(
    rng: np.random.Generator, target_shape: tuple[tuple[int, int], ...]
) -> list[tuple[tuple[int, int], ...]]:
    same_area = [shape for shape in SHAPES_BY_AREA[len(target_shape)] if shape != target_shape]
    same_bbox = [shape for shape in same_area if _shape_dims(shape) == _shape_dims(target_shape)]
    same_area_diff_bbox = [shape for shape in same_area if _shape_dims(shape) != _shape_dims(target_shape)]
    assert same_bbox and same_area_diff_bbox, "target shape must admit same-area and same-bbox distractors"

    chosen = [target_shape]
    chosen.append(same_area_diff_bbox[int(rng.integers(len(same_area_diff_bbox)))])
    bbox_choices = [shape for shape in same_bbox if shape not in chosen]
    if not bbox_choices:
        bbox_choices = same_bbox
    chosen.append(bbox_choices[int(rng.integers(len(bbox_choices)))])

    other_pool = [shape for shape in ALL_SHAPES if shape not in chosen]
    chosen.append(other_pool[int(rng.integers(len(other_pool)))])
    rng.shuffle(chosen)
    return chosen


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], values: list[int]) -> None:
    for (r, c), color in zip(coords, values):
        grid[r, c] = int(color)


def _build_motif(
    rng: np.random.Generator,
    tile_size: int,
    bg: int,
    motif_colors: list[int],
    target_shape: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    motif = np.full((tile_size, tile_size), int(bg), dtype=int)
    shape_h, shape_w = _shape_dims(target_shape)
    top = int(rng.integers(0, tile_size - shape_h + 1))
    left = int(rng.integers(0, tile_size - shape_w + 1))
    damage_coords = np.array([(top + r, left + c) for r, c in target_shape], dtype=int)

    area = tile_size * tile_size
    max_fg = max(len(target_shape) + 3, area // 2)
    max_fg = min(max_fg, area - 2)
    min_fg = len(target_shape) + 3
    fg_total = int(rng.integers(min_fg, max_fg + 1))
    extra_needed = fg_total - len(target_shape)

    occupied = {tuple(map(int, rc)) for rc in damage_coords}
    remaining = [(r, c) for r in range(tile_size) for c in range(tile_size) if (r, c) not in occupied]
    rng.shuffle(remaining)
    extra_coords = remaining[:extra_needed]
    all_coords = [tuple(map(int, rc)) for rc in damage_coords] + extra_coords

    values = motif_colors.copy()
    while len(values) < len(all_coords):
        values.append(int(motif_colors[int(rng.integers(len(motif_colors)))]))
    rng.shuffle(values)
    _paint_cells(motif, all_coords, values[: len(all_coords)])
    assert np.any(motif != bg) and np.any(motif == bg)
    return motif, damage_coords


def _render_lattice(motif: np.ndarray, separator: int, block_rows: int, block_cols: int) -> np.ndarray:
    tile_size = motif.shape[0]
    step = tile_size + 1
    lattice_h = block_rows * tile_size + (block_rows - 1)
    lattice_w = block_cols * tile_size + (block_cols - 1)
    grid = np.full((lattice_h, lattice_w), int(separator), dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * step
            c0 = bc * step
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = motif
    return grid


def _place_patches(
    rng: np.random.Generator, region_h: int, region_w: int, patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((region_h, region_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(patches)), key=lambda i: int(np.count_nonzero(patches[i])), reverse=True)
    placed = [None] * len(patches)

    for idx in order:
        patch = patches[idx]
        mask = patch != -1
        ph, pw = patch.shape
        options = []
        coords = np.argwhere(mask)
        for r0 in range(region_h - ph + 1):
            for c0 in range(region_w - pw + 1):
                fits = True
                for rr, cc in coords:
                    r = r0 + int(rr)
                    c = c0 + int(cc)
                    r1 = max(0, r - 1)
                    r2 = min(region_h, r + 2)
                    c1 = max(0, c - 1)
                    c2 = min(region_w, c + 2)
                    if occupied[r1:r2, c1:c2].any():
                        fits = False
                        break
                if fits:
                    options.append((r0, c0))
        if not options:
            return None
        choice = options[int(rng.integers(len(options)))]
        placed[idx] = choice
        placements.append(choice)
        for rr, cc in np.argwhere(mask):
            occupied[choice[0] + int(rr), choice[1] + int(cc)] = True

    return [placed[i] for i in range(len(patches))]


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    motif: np.ndarray,
    candidate_shapes: list[tuple[tuple[int, int], ...]],
    target_shape: tuple[tuple[int, int], ...],
    target_color: int,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    target_area = len(target_shape)
    target_dims = _shape_dims(target_shape)
    if sum(len(shape) == target_area for shape in candidate_shapes) < 2:
        return False
    if sum(_shape_dims(shape) == target_dims for shape in candidate_shapes) < 2:
        return False
    if not np.any(expected == target_color):
        return False
    if np.all(expected == target_color):
        return False
    if len(set(map(int, np.unique(motif)))) < 2:
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    tile_size, block_rows, block_cols = _choose_geometry(rng)
    bg, separator, motif_colors, candidate_colors = _choose_colors(rng)
    target_shape = _choose_target_shape(rng)
    candidate_shapes = _build_candidate_shapes(rng, target_shape)

    motif, damage_coords = _build_motif(rng, tile_size, bg, motif_colors, target_shape)
    damage_mask = np.zeros_like(motif, dtype=bool)
    damage_mask[damage_coords[:, 0], damage_coords[:, 1]] = True

    lattice = _render_lattice(motif, separator, block_rows, block_cols)
    step = tile_size + 1
    damaged_br = int(rng.integers(block_rows))
    damaged_bc = int(rng.integers(block_cols))
    r0 = damaged_br * step
    c0 = damaged_bc * step
    lattice[r0 : r0 + tile_size, c0 : c0 + tile_size][damage_mask] = bg

    region_h_max = 30 - lattice.shape[0] - GAP_ROWS
    if region_h_max < MIN_CANDIDATE_HEIGHT:
        return None
    candidate_h = int(rng.integers(MIN_CANDIDATE_HEIGHT, region_h_max + 1))
    candidate_w = lattice.shape[1]

    shuffled_colors = candidate_colors.copy()
    rng.shuffle(shuffled_colors)
    patches = [_shape_patch(shape, color) for shape, color in zip(candidate_shapes, shuffled_colors)]
    placements = _place_patches(rng, candidate_h, candidate_w, patches)
    if placements is None:
        return None

    candidate_region = np.full((candidate_h, candidate_w), int(bg), dtype=int)
    for patch, (pr, pc) in zip(patches, placements):
        ph, pw = patch.shape
        mask = patch != -1
        candidate_region[pr : pr + ph, pc : pc + pw][mask] = patch[mask]

    grid = np.full((lattice.shape[0] + GAP_ROWS + candidate_h, lattice.shape[1]), int(bg), dtype=int)
    grid[: lattice.shape[0], :] = lattice
    grid[lattice.shape[0] + GAP_ROWS :, :] = candidate_region

    target_index = candidate_shapes.index(target_shape)
    target_color = int(shuffled_colors[target_index])
    expected = motif.copy()
    expected[damage_mask] = target_color

    return grid if _validate_semantics(grid, expected, motif, candidate_shapes, target_shape, target_color) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
