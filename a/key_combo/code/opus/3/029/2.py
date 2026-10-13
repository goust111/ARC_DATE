from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


BG = 5
BLUE = 1
WALL = 2
CONNECTOR_COLORS = (3, 4, 7)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_DIR = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}
TREE_SHAPES = {
    3: [
        [(0, 1), (1, 2)],
        [(0, 1), (0, 2)],
    ],
    4: [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (0, 2), (2, 3)],
        [(0, 1), (1, 2), (0, 3)],
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _connector_dir(sprite: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    assert len(dirs) == 1
    return dirs[0]


def _interior_regions(grid: np.ndarray, blockers: tuple[int, ...]) -> list[np.ndarray]:
    mask = ~np.isin(grid, blockers)
    return [coords for coords in _components(mask) if not _touches_edge(coords, grid.shape)]


def _region_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _extract_seed_tile(grid: np.ndarray, region_coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    region = _region_mask(grid.shape, region_coords)
    blue_mask = region & (grid == BLUE)
    blue_components = _components(blue_mask)
    assert blue_components

    bboxes = [_bbox_coords(coords) for coords in blue_components]
    sr0, sc0, sr1, sc1 = min(bboxes, key=lambda b: (b[0], b[1], b[2] - b[0], b[3] - b[1]))
    tile = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert set(map(int, np.unique(tile))) <= {BG, BLUE}

    tile_h, tile_w = tile.shape
    for rr, cc in np.argwhere(blue_mask):
        assert tile[(int(rr) - sr0) % tile_h, (int(cc) - sc0) % tile_w] == BLUE
    return tile, (sr0, sc0)


def _tile_region_in_place(grid: np.ndarray, region_coords: np.ndarray, tile: np.ndarray, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    tile_h, tile_w = tile.shape
    for rr, cc in region_coords:
        r = int(rr)
        c = int(cc)
        if grid[r, c] != BG:
            continue
        if tile[(r - ar) % tile_h, (c - ac) % tile_w] == BLUE:
            grid[r, c] = BLUE


def _finish_assembled(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[np.isin(out, CONNECTOR_COLORS)] = BG
    for region_coords in _interior_regions(out, (WALL,)):
        tile, anchor = _extract_seed_tile(out, region_coords)
        _tile_region_in_place(out, region_coords, tile, anchor)
    return out


def _assemble_scattered(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wallish = np.isin(g, (WALL, *CONNECTOR_COLORS))
    components = _components(wallish)
    assert len(components) >= 2

    sprites: list[np.ndarray] = []
    occurrences: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for idx, coords in enumerate(components):
        sprite = _crop_bbox(g, _bbox_coords(coords))
        sprites.append(sprite)
        for rr, cc in np.argwhere(np.isin(sprite, CONNECTOR_COLORS)):
            r = int(rr)
            c = int(cc)
            direction = _connector_dir(sprite, r, c)
            occurrences[int(sprite[r, c])].append((idx, (r, c), direction))

    first_occurrence: dict[int, tuple[int, int]] = {}
    for rr, cc in np.argwhere(np.isin(g, CONNECTOR_COLORS)):
        color = int(g[int(rr), int(cc)])
        first_occurrence.setdefault(color, (int(rr), int(cc)))

    adjacency: list[list[tuple[int, tuple[int, int], int]]] = [[] for _ in sprites]
    ordered_colors = sorted(occurrences, key=lambda color: first_occurrence[color])
    for order_idx, color in enumerate(ordered_colors):
        occs = occurrences[color]
        assert len(occs) == 2
        (a_idx, a_pos, a_dir), (b_idx, b_pos, b_dir) = occs
        assert a_idx != b_idx
        assert b_dir == (-a_dir[0], -a_dir[1])
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])
        adjacency[a_idx].append((b_idx, delta, order_idx))
        adjacency[b_idx].append((a_idx, (-delta[0], -delta[1]), order_idx))

    placements: dict[int, tuple[int, int]] = {0: (0, 0)}
    queue = deque([0])
    while queue:
        idx = queue.popleft()
        base_r, base_c = placements[idx]
        for nbr, (dr, dc), _order_idx in sorted(adjacency[idx], key=lambda item: item[2]):
            pos = (base_r + dr, base_c + dc)
            if nbr in placements:
                assert placements[nbr] == pos
            else:
                placements[nbr] = pos
                queue.append(nbr)
    assert len(placements) == len(sprites)

    placed = []
    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        for rr, cc in np.argwhere(sprite != bg):
            placed.append((base_r + int(rr), base_c + int(cc)))

    r0 = min(r for r, _ in placed)
    c0 = min(c for _, c in placed)
    r1 = max(r for r, _ in placed)
    c1 = max(c for _, c in placed)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)

    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        rr = base_r - r0
        cc = base_c - c0
        view = out[rr : rr + sprite.shape[0], cc : cc + sprite.shape[1]]
        mask = sprite != bg
        assert view.shape == sprite.shape
        assert not np.any((view != bg) & mask)
        view[mask] = sprite[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Dock equal-colored wall connectors, erase them to merge the cavities, then phase-tile the merged cavity from the blue seed.

    Input structure:
    - The background is grey, and the foreground contains 3-4 separate red wall objects.
    - Each object is a hollow frame with a rectangular cavity, blue seed pixels inside, and 1-2 colored connector pixels on non-corner outer wall cells.
    - Every connector color appears exactly twice on different objects, and matched connectors lie on opposite exposed sides.

    Rule:
    - Crop each wall object to its bbox and translate the objects so every equal-colored connector pair becomes side-adjacent in the outward direction indicated by the connector's wall side.
    - After docking, erase all connector pixels to grey; this opens one-cell passages that merge previously separate cavities.
    - For each enclosed non-wall cavity region in the assembled shape, take one visible blue seed copy, use its bbox as the periodic tile, and repeat that tile across the whole merged cavity while preserving the seed copy's absolute phase.

    Key insight:
    - The connectors do two jobs at once: they determine the assembly translation and, once erased, they change cavity topology, so the final tiling region only exists after docking and connector removal.
    """
    assembled = _assemble_scattered(np.asarray(grid, dtype=int))
    return _finish_assembled(assembled)


def _rect(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    h, w = shape
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _coord_from_side(h: int, w: int, side: str, slot: int) -> tuple[int, int]:
    if side == "T":
        return 0, slot
    if side == "B":
        return h - 1, slot
    if side == "L":
        return slot, 0
    return slot, w - 1


def _side_slots(h: int, w: int, side: str) -> list[int]:
    if side in ("T", "B"):
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _connected_mask(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_tile(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        tile_h, tile_w = ((2, 2), (2, 3), (3, 2))[int(rng.integers(0, 3))]
        area = tile_h * tile_w
        blue_count = int(rng.integers(max(2, area // 2), area))
        coords = [(r, c) for r in range(tile_h) for c in range(tile_w)]
        choice = rng.choice(len(coords), size=blue_count, replace=False)
        mask = np.zeros((tile_h, tile_w), dtype=bool)
        for idx in np.atleast_1d(choice):
            r, c = coords[int(idx)]
            mask[r, c] = True
        if mask.all():
            continue
        if not _connected_mask(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        tile = np.full((tile_h, tile_w), BG, dtype=int)
        tile[mask] = BLUE
        return tile
    raise AssertionError("failed to generate a connected seed tile")


def _random_tree_edges(rng: np.random.Generator, n_objects: int) -> list[tuple[int, int, int]]:
    shape = TREE_SHAPES[n_objects][int(rng.integers(0, len(TREE_SHAPES[n_objects])))]
    colors = list(CONNECTOR_COLORS)
    rng.shuffle(colors)
    return [(parent, child, colors[idx]) for idx, (parent, child) in enumerate(shape)]


def _sample_dims(rng: np.random.Generator, n_objects: int, tile: np.ndarray) -> list[tuple[int, int]]:
    tile_h, tile_w = tile.shape
    dims: list[tuple[int, int]] = []
    for _ in range(n_objects):
        cavity_h = int(rng.integers(max(4, tile_h + 1), 8))
        cavity_w = int(rng.integers(max(4, tile_w + 1), 8))
        dims.append((cavity_h + 2, cavity_w + 2))
    return dims


def _place_objects(
    rng: np.random.Generator,
    dims: list[tuple[int, int]],
    edges: list[tuple[int, int, int]],
    max_span: int = 22,
) -> tuple[dict[int, tuple[int, int]], list[list[tuple[int, str, int]]]] | None:
    n_objects = len(dims)
    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    connectors: list[list[tuple[int, str, int]]] = [[] for _ in range(n_objects)]
    used_sides = [set() for _ in range(n_objects)]

    def placement_ok(child: int, pos: tuple[int, int], parent: int) -> bool:
        child_rect = _rect(pos, dims[child])
        rects = [child_rect]
        for idx, base in positions.items():
            other = _rect(base, dims[idx])
            if idx == parent:
                if _rects_too_close(child_rect, other, 0):
                    return False
            else:
                if _rects_too_close(child_rect, other, 1):
                    return False
            rects.append(other)
        r0 = min(rect[0] for rect in rects)
        c0 = min(rect[1] for rect in rects)
        r1 = max(rect[2] for rect in rects)
        c1 = max(rect[3] for rect in rects)
        return (r1 - r0 + 1) <= max_span and (c1 - c0 + 1) <= max_span

    def rec(edge_idx: int) -> bool:
        if edge_idx == len(edges):
            return True
        parent, child, color = edges[edge_idx]
        assert parent in positions and child not in positions

        parent_h, parent_w = dims[parent]
        child_h, child_w = dims[child]
        side_choices = list(SIDE_TO_DIR)
        rng.shuffle(side_choices)
        for side in side_choices:
            if side in used_sides[parent]:
                continue
            child_side = OPPOSITE[side]
            if child_side in used_sides[child]:
                continue

            parent_slots = _side_slots(parent_h, parent_w, side)
            child_slots = _side_slots(child_h, child_w, child_side)
            rng.shuffle(parent_slots)
            rng.shuffle(child_slots)
            for parent_slot in parent_slots:
                pr, pc = _coord_from_side(parent_h, parent_w, side, parent_slot)
                for child_slot in child_slots:
                    cr, cc = _coord_from_side(child_h, child_w, child_side, child_slot)
                    dr, dc = SIDE_TO_DIR[side]
                    base_r, base_c = positions[parent]
                    new_pos = (base_r + pr + dr - cr, base_c + pc + dc - cc)
                    if not placement_ok(child, new_pos, parent):
                        continue

                    positions[child] = new_pos
                    connectors[parent].append((color, side, parent_slot))
                    connectors[child].append((color, child_side, child_slot))
                    used_sides[parent].add(side)
                    used_sides[child].add(child_side)
                    if rec(edge_idx + 1):
                        return True
                    used_sides[parent].remove(side)
                    used_sides[child].remove(child_side)
                    connectors[parent].pop()
                    connectors[child].pop()
                    del positions[child]
        return False

    if rec(0):
        return positions, connectors
    return None


def _aligned_starts(start: int, span: int, tile_len: int, phase: int) -> list[int]:
    out = []
    end = start + span - tile_len
    for pos in range(start, end + 1):
        if (pos - phase) % tile_len == 0:
            out.append(pos)
    return out


def _choose_seed_positions(
    rng: np.random.Generator,
    dims: list[tuple[int, int]],
    positions: dict[int, tuple[int, int]],
    tile: np.ndarray,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    tile_h, tile_w = tile.shape
    row_phases = []
    col_phases = []
    for idx, (h, w) in enumerate(dims):
        base_r, base_c = positions[idx]
        cavity_r = base_r + 1
        cavity_c = base_c + 1
        row_phases.append(
            {
                residue
                for residue in range(tile_h)
                if _aligned_starts(cavity_r, h - 2, tile_h, residue)
            }
        )
        col_phases.append(
            {
                residue
                for residue in range(tile_w)
                if _aligned_starts(cavity_c, w - 2, tile_w, residue)
            }
        )

    common_rows = set.intersection(*row_phases)
    common_cols = set.intersection(*col_phases)
    if not common_rows or not common_cols:
        return None

    phase = (
        int(sorted(common_rows)[int(rng.integers(0, len(common_rows)))]),
        int(sorted(common_cols)[int(rng.integers(0, len(common_cols)))]),
    )
    seeds: list[tuple[int, int]] = []
    for idx, (h, w) in enumerate(dims):
        base_r, base_c = positions[idx]
        cavity_r = base_r + 1
        cavity_c = base_c + 1
        rows = _aligned_starts(cavity_r, h - 2, tile_h, phase[0])
        cols = _aligned_starts(cavity_c, w - 2, tile_w, phase[1])
        assert rows and cols
        seed_r = int(rows[int(rng.integers(0, len(rows)))])
        seed_c = int(cols[int(rng.integers(0, len(cols)))])
        seeds.append((seed_r - base_r, seed_c - base_c))
    return phase, seeds


def _make_sprite(
    shape: tuple[int, int],
    connector_specs: list[tuple[int, str, int]],
    tile: np.ndarray,
    seed_top_left: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    sprite = np.full((h, w), BG, dtype=int)
    sprite[0, :] = WALL
    sprite[-1, :] = WALL
    sprite[:, 0] = WALL
    sprite[:, -1] = WALL
    for color, side, slot in connector_specs:
        rr, cc = _coord_from_side(h, w, side, slot)
        sprite[rr, cc] = int(color)

    sr, sc = seed_top_left
    tile_h, tile_w = tile.shape
    view = sprite[sr : sr + tile_h, sc : sc + tile_w]
    view[tile == BLUE] = BLUE
    return sprite


def _render_layout(
    dims: list[tuple[int, int]],
    positions: dict[int, tuple[int, int]],
    connectors: list[list[tuple[int, str, int]]],
    tile: np.ndarray,
    seeds: list[tuple[int, int]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    rects = [_rect(positions[idx], dims[idx]) for idx in range(len(dims))]
    r0 = min(rect[0] for rect in rects)
    c0 = min(rect[1] for rect in rects)
    r1 = max(rect[2] for rect in rects)
    c1 = max(rect[3] for rect in rects)
    assembled = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)

    sprites: list[np.ndarray] = []
    for idx, shape in enumerate(dims):
        sprite = _make_sprite(shape, connectors[idx], tile, seeds[idx])
        sprites.append(sprite)
        base_r, base_c = positions[idx]
        rr = base_r - r0
        cc = base_c - c0
        view = assembled[rr : rr + shape[0], cc : cc + shape[1]]
        mask = sprite != BG
        assert not np.any((view != BG) & mask)
        view[mask] = sprite[mask]
    return assembled, sprites


def _scatter_sprites(rng: np.random.Generator, sprites: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(100):
        height = int(rng.integers(20, 31))
        width = int(rng.integers(20, 31))
        order = sorted(range(len(sprites)), key=lambda idx: sprites[idx].size, reverse=True)
        placements: dict[int, tuple[int, int]] = {}
        rects: list[tuple[int, int, int, int]] = []
        for idx in order:
            h, w = sprites[idx].shape
            choices: list[tuple[int, int]] = []
            for r in range(0, height - h + 1):
                for c in range(0, width - w + 1):
                    rect = (r, c, r + h - 1, c + w - 1)
                    if all(not _rects_too_close(rect, other, 2) for other in rects):
                        choices.append((r, c))
            if not choices:
                break
            placements[idx] = choices[int(rng.integers(0, len(choices)))]
            rects.append(_rect(placements[idx], sprites[idx].shape))
        else:
            grid = np.full((height, width), BG, dtype=int)
            for idx, sprite in enumerate(sprites):
                r, c = placements[idx]
                view = grid[r : r + sprite.shape[0], c : c + sprite.shape[1]]
                mask = sprite != BG
                assert not np.any((view != BG) & mask)
                view[mask] = sprite[mask]
            return grid
    return None


def _tile_premerge_cavities(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    blockers = (WALL, *CONNECTOR_COLORS)
    for region in _interior_regions(out, blockers):
        tile, anchor = _extract_seed_tile(out, region)
        _tile_region_in_place(out, region, tile, anchor)
    return out


def _validate_instance(inp: np.ndarray, assembled: np.ndarray, expected: np.ndarray) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    pre_regions = _interior_regions(assembled, (WALL, *CONNECTOR_COLORS))
    post_regions = _interior_regions(expected, (WALL,))
    if len(pre_regions) < 3 or len(post_regions) != 1:
        return False

    shortcut = _tile_premerge_cavities(assembled)
    shortcut[np.isin(shortcut, CONNECTOR_COLORS)] = BG
    if np.array_equal(shortcut, expected):
        return False

    if np.count_nonzero(expected == BLUE) <= np.count_nonzero(assembled == BLUE) + 2:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile = _random_tile(rng)
    n_objects = int(rng.choice([3, 4]))
    dims = _sample_dims(rng, n_objects, tile)
    edges = _random_tree_edges(rng, n_objects)
    placement = _place_objects(rng, dims, edges)
    if placement is None:
        return None
    positions, connectors = placement

    phase_and_seeds = _choose_seed_positions(rng, dims, positions, tile)
    if phase_and_seeds is None:
        return None
    _phase, seeds = phase_and_seeds
    assembled, sprites = _render_layout(dims, positions, connectors, tile, seeds)
    expected = _finish_assembled(assembled)

    inp = _scatter_sprites(rng, sprites)
    if inp is None:
        return None
    if not _validate_instance(inp, assembled, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid connector-cavity tiling input")
