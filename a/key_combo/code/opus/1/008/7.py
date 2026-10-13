from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
TILE_COLORS = (1, 2, 3, 4)

OUTPUT_DIMS = ((4, 5), (4, 6), (5, 6))
GEN_ATTEMPTS = 500
PARTITION_ATTEMPTS = 120
SCATTER_ATTEMPTS = 200
ENUM_LIMIT = 1024


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(tuple(sorted(coords)))
    return comps


def _bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _w = _normalize(coords)
    return _normalize(tuple((c, h - 1 - r) for r, c in norm))


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _h, w = _normalize(coords)
    return _normalize(tuple((r, w - 1 - c) for r, c in norm))


def _dihedral_orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    base, _h, _w = _normalize(coords)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int]] = []

    cur = base
    for _ in range(4):
        norm, h, w = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w))
        cur = _rot90(norm)[0]

    cur = _flip_h(base)[0]
    for _ in range(4):
        norm, h, w = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w))
        cur = _rot90(norm)[0]

    return out


def _canonical_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(ori for ori, _h, _w in _dihedral_orientations(coords))


def _has_hole(coords: tuple[tuple[int, int], ...]) -> bool:
    norm, h, w = _normalize(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    bg = ~mask
    if not np.any(bg):
        return False

    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if bg[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    stack.append((r, c))
                    while stack:
                        rr, cc = stack.pop()
                        for nr, nc in _neighbors4(rr, cc):
                            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _is_irregular_piece(coords: tuple[tuple[int, int], ...]) -> bool:
    norm, h, w = _normalize(coords)
    area = len(norm)
    if h == 1 or w == 1:
        return False
    if area == h * w:
        return False
    if _has_hole(norm):
        return False
    return True


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort()
    return out


def _placement_options(
    orients: list[tuple[tuple[tuple[int, int], ...], int, int]], H: int, W: int
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    opts: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    seen_masks: set[int] = set()
    for coords, h, w in orients:
        if h > H or w > W:
            continue
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                abs_coords = tuple(sorted((r0 + r, c0 + c) for r, c in coords))
                mask = 0
                for r, c in abs_coords:
                    mask |= 1 << (r * W + c)
                mask = int(mask)
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
                opts.append((mask, abs_coords))
    return opts


def _enumerate_assemblies(
    piece_shapes: list[tuple[tuple[int, int], ...]], limit: int = ENUM_LIMIT
) -> list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]]:
    n = len(piece_shapes)
    total = sum(len(shape) for shape in piece_shapes)
    orient_cache = [_dihedral_orientations(shape) for shape in piece_shapes]
    solutions: list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]] = []

    for H, W in _factor_pairs(total):
        if H > 30 or W > 30:
            continue

        placements_by_piece: list[list[tuple[int, tuple[tuple[int, int], ...]]]] = []
        valid = True
        for orients in orient_cache:
            opts = _placement_options(orients, H, W)
            if not opts:
                valid = False
                break
            placements_by_piece.append(opts)
        if not valid:
            continue

        all_mask = int((1 << int(H * W)) - 1)
        current: list[tuple[tuple[int, int], ...] | None] = [None] * n

        def search(used: int, remaining: tuple[int, ...]) -> None:
            if len(solutions) >= limit:
                return
            if used == all_mask:
                if not remaining:
                    solutions.append((H, W, tuple(current)))  # type: ignore[arg-type]
                return
            if not remaining:
                return

            empty = int((~used) & all_mask)
            cell = (empty & -empty).bit_length() - 1

            candidates: list[tuple[int, int, list[tuple[int, tuple[tuple[int, int], ...]]]]] = []
            for piece_id in remaining:
                opts = []
                for mask, abs_coords in placements_by_piece[piece_id]:
                    if ((mask >> cell) & 1) and not (mask & used):
                        opts.append((mask, abs_coords))
                if opts:
                    candidates.append((len(opts), piece_id, opts))

            if not candidates:
                return

            candidates.sort(key=lambda item: (item[0], item[1]))
            for _count, piece_id, opts in candidates:
                next_remaining = tuple(pid for pid in remaining if pid != piece_id)
                for mask, abs_coords in opts:
                    current[piece_id] = abs_coords
                    search(used | mask, next_remaining)
                    current[piece_id] = None

        search(0, tuple(range(n)))
        if len(solutions) >= limit:
            break

    return solutions


def _piece_color(
    abs_coords: tuple[tuple[int, int], ...], tile: np.ndarray
) -> tuple[int, bool]:
    th, tw = tile.shape
    counts: dict[int, int] = {}
    for r, c in abs_coords:
        color = int(tile[r % th, c % tw])
        counts[color] = counts.get(color, 0) + 1

    best_count = max(counts.values())
    modes = [color for color, count in counts.items() if count == best_count]
    top_r, top_c = min(abs_coords)
    top_color = int(tile[top_r % th, top_c % tw])
    if len(modes) == 1:
        return modes[0], False
    return top_color, True


def _render_from_piece_cells(
    piece_cells: tuple[tuple[tuple[int, int], ...], ...], H: int, W: int, tile: np.ndarray
) -> tuple[np.ndarray, bool]:
    out = np.zeros((H, W), dtype=int)
    tie_any = False
    for abs_coords in piece_cells:
        color, is_tie = _piece_color(abs_coords, tile)
        tie_any |= is_tie
        for r, c in abs_coords:
            out[r, c] = color
    assert np.all(out != 0), "assembled rectangle must be fully covered"
    return out, tie_any


def _render_from_label_grid(label_grid: np.ndarray, tile: np.ndarray) -> tuple[np.ndarray, bool]:
    H, W = label_grid.shape
    piece_cells: list[tuple[tuple[int, int], ...]] = []
    for piece_id in sorted(int(x) for x in np.unique(label_grid)):
        coords = tuple(sorted(map(tuple, np.argwhere(label_grid == piece_id))))
        piece_cells.append(coords)
    return _render_from_piece_cells(tuple(piece_cells), H, W, tile)


def _grid_key(arr: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return arr.shape[0], arr.shape[1], tuple(int(x) for x in arr.ravel())


def _dihedral_arrays(arr: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    return out


def _solution_signature(
    H: int, W: int, piece_cells: tuple[tuple[tuple[int, int], ...], ...]
) -> tuple[int, int, tuple[int, ...]]:
    label_grid = np.full((H, W), -1, dtype=int)
    for piece_id, abs_coords in enumerate(piece_cells):
        for r, c in abs_coords:
            assert label_grid[r, c] == -1
            label_grid[r, c] = piece_id
    assert np.all(label_grid >= 0)
    return min(_grid_key(arr) for arr in _dihedral_arrays(label_grid))


def _parse_input(grid: np.ndarray) -> tuple[list[tuple[tuple[int, int], ...]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"
    assert int(np.min(g)) >= 0

    piece_comps = sorted(_components(g == GRAY), key=lambda comp: _bbox(comp)[:2])
    assert 4 <= len(piece_comps) <= 6, "expected 4-6 gray fragments"

    piece_shapes: list[tuple[tuple[int, int], ...]] = []
    for comp in piece_comps:
        assert 4 <= len(comp) <= 8, "fragment size out of range"
        rel, _h, _w = _normalize(comp)
        assert _is_irregular_piece(rel), "fragments must be irregular non-rectangles"
        piece_shapes.append(rel)

    tile_comps = _components((g != BG) & (g != GRAY))
    assert len(tile_comps) == 1, "expected exactly one colored tile template"
    tile_coords = tile_comps[0]
    r0, c0, r1, c1 = _bbox(tile_coords)
    tile = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert tile.size == len(tile_coords), "tile template must be a filled rectangle"
    assert tile.shape in ((2, 2), (2, 3), (3, 2)), "tile template must be 2x2 or 2x3"
    tile_colors = {int(x) for x in np.unique(tile)}
    assert BG not in tile_colors and GRAY not in tile_colors
    assert 2 <= len(tile_colors) <= 4

    return piece_shapes, tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble gray jigsaw fragments into a rectangle, then recolor each piece from a periodically tiled color map.

    Input structure:
    - The grid has a black background.
    - Four to six disconnected gray (5) 4-connected fragments are scattered across the canvas; each fragment is a small irregular monochrome piece.
    - One separate filled multicolor rectangle of size 2x2 or 2x3 serves as the tile template; it uses non-black, non-gray colors.

    Rule:
    - Identify the gray fragments and the unique colored rectangular tile template.
    - Reassemble the gray fragments into one tight rectangle, allowing rotations and horizontal reflection of each fragment.
    - Repeat the tile template periodically from the assembled rectangle's top-left corner to cover the whole rectangle.
    - For each fragment, read the repeated tile colors on all cells occupied by that fragment in the assembly.
    - Recolor the whole fragment to the most frequent of those tile colors; if there is a tie, use the tile color at that fragment's top-left occupied cell.
    - Output only the assembled recolored rectangle.

    Key insight:
    - The final colors are not encoded inside the fragments: first the jigsaw assembly fixes every fragment's spatial coordinates, then an independent periodic tile assigns colors by position.
    """
    piece_shapes, tile = _parse_input(grid)
    solutions = _enumerate_assemblies(piece_shapes)
    assert solutions, "expected at least one valid rectangular assembly"

    best_out = None
    best_key = None
    for H, W, piece_cells in solutions:
        out, _tie_any = _render_from_piece_cells(piece_cells, H, W, tile)
        key = _grid_key(out)
        if best_key is None or key < best_key:
            best_key = key
            best_out = out

    assert best_out is not None
    return best_out


def _sample_areas(total: int, n_pieces: int) -> list[int]:
    areas: list[int] = []
    remaining = total
    for i in range(n_pieces):
        slots = n_pieces - i - 1
        lo = max(4, remaining - 8 * slots)
        hi = min(8, remaining - 4 * slots)
        areas.append(random.randint(lo, hi))
        remaining -= areas[-1]
    random.shuffle(areas)
    return areas


def _choose_seeds(H: int, W: int, n_pieces: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(H) for c in range(W)]
    seeds = [random.choice(cells)]
    while len(seeds) < n_pieces:
        best = None
        best_score = None
        for cell in cells:
            if cell in seeds:
                continue
            dist = min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in seeds)
            score = dist + random.random() * 0.5
            if best_score is None or score > best_score:
                best = cell
                best_score = score
        assert best is not None
        seeds.append(best)
    return seeds


def _grow_partition(H: int, W: int, areas: list[int]) -> np.ndarray | None:
    n_pieces = len(areas)
    for _ in range(PARTITION_ATTEMPTS):
        labels = np.full((H, W), -1, dtype=int)
        frontiers = [set() for _ in range(n_pieces)]
        sizes = [0] * n_pieces
        seeds = _choose_seeds(H, W, n_pieces)

        for piece_id, (r, c) in enumerate(seeds):
            labels[r, c] = piece_id
            sizes[piece_id] = 1
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W and labels[nr, nc] < 0:
                    frontiers[piece_id].add((nr, nc))

        ok = True
        while any(sizes[i] < areas[i] for i in range(n_pieces)):
            candidate_ids = [i for i in range(n_pieces) if sizes[i] < areas[i]]
            candidate_ids = [i for i in candidate_ids if any(labels[r, c] < 0 for r, c in frontiers[i])]
            if not candidate_ids:
                ok = False
                break

            weights = []
            for piece_id in candidate_ids:
                remaining = areas[piece_id] - sizes[piece_id]
                live_frontier = sum(1 for r, c in frontiers[piece_id] if labels[r, c] < 0)
                weights.append((remaining**1.7) * (live_frontier + 1))
            piece_id = random.choices(candidate_ids, weights=weights, k=1)[0]

            frontier_cells = [(r, c) for r, c in frontiers[piece_id] if labels[r, c] < 0]
            if not frontier_cells:
                ok = False
                break

            scored: list[tuple[float, tuple[int, int]]] = []
            for r, c in frontier_cells:
                free_neighbors = 0
                same_neighbors = 0
                edge_bonus = 0.0
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < H and 0 <= nc < W:
                        if labels[nr, nc] < 0:
                            free_neighbors += 1
                        elif labels[nr, nc] == piece_id:
                            same_neighbors += 1
                if r in (0, H - 1):
                    edge_bonus += 0.25
                if c in (0, W - 1):
                    edge_bonus += 0.25
                score = free_neighbors * 1.8 - same_neighbors * 0.35 + edge_bonus + random.random() * 0.6
                scored.append((score, (r, c)))

            scored.sort(reverse=True)
            top_score = scored[0][0]
            top_cells = [cell for score, cell in scored if score >= top_score - 0.4]
            r, c = random.choice(top_cells)

            labels[r, c] = piece_id
            sizes[piece_id] += 1
            for frontier in frontiers:
                frontier.discard((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W and labels[nr, nc] < 0:
                    frontiers[piece_id].add((nr, nc))

        if ok and np.all(labels >= 0):
            return labels

    return None


def _shapes_from_label_grid(label_grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    shapes: list[tuple[tuple[int, int], ...]] = []
    for piece_id in sorted(int(x) for x in np.unique(label_grid)):
        coords = tuple(sorted(map(tuple, np.argwhere(label_grid == piece_id))))
        rel, _h, _w = _normalize(coords)
        shapes.append(rel)
    return shapes


def _sample_tile() -> np.ndarray:
    h, w = (2, random.choice((2, 3)))
    distinct = random.randint(2, min(4, h * w))
    colors = random.sample(TILE_COLORS, distinct)
    cells = colors + [random.choice(colors) for _ in range(h * w - distinct)]
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(h, w)


def _construction_output(label_grid: np.ndarray, tile: np.ndarray) -> tuple[np.ndarray, bool]:
    best_out = None
    best_key = None
    best_tie = False
    for transformed in _dihedral_arrays(label_grid):
        out, tie_any = _render_from_label_grid(transformed, tile)
        key = _grid_key(out)
        if best_key is None or key < best_key:
            best_key = key
            best_out = out
            best_tie = tie_any
    assert best_out is not None
    return best_out, best_tie


def _enumeration_summary(
    piece_shapes: list[tuple[tuple[int, int], ...]]
) -> tuple[list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]], set[tuple[int, int, tuple[int, ...]]]]:
    solutions = _enumerate_assemblies(piece_shapes)
    assert solutions, "generator constructed an unsatisfiable puzzle"

    signatures: set[tuple[int, int, tuple[int, ...]]] = set()
    for H, W, piece_cells in solutions:
        signatures.add(_solution_signature(H, W, piece_cells))
    return solutions, signatures


def _best_output_from_solutions(
    solutions: list[tuple[int, int, tuple[tuple[tuple[int, int], ...], ...]]], tile: np.ndarray
) -> tuple[np.ndarray, bool]:
    best_out = None
    best_key = None
    best_tie = False
    for H, W, piece_cells in solutions:
        out, tie_any = _render_from_piece_cells(piece_cells, H, W, tile)
        key = _grid_key(out)
        if best_key is None or key < best_key:
            best_key = key
            best_out = out
            best_tie = tie_any
    assert best_out is not None
    return best_out, best_tie


def _boxes_separated(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    total_bbox_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)

    min_side = max(16, max(max_h, max_w) + 4, int(np.ceil(np.sqrt(total_bbox_area * 2.6))))
    max_side = min(26, min_side + 6)

    for _ in range(SCATTER_ATTEMPTS):
        H = random.randint(min_side, max_side)
        W = random.randint(min_side, max_side)
        order = list(range(len(patches)))
        random.shuffle(order)

        positions: dict[int, tuple[int, int]] = {}
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            h, w = patches[idx].shape
            candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            random.shuffle(candidates)
            found = False
            for r0, c0 in candidates:
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_boxes_separated(box, prev, gap=1) for prev in boxes):
                    positions[idx] = (r0, c0)
                    boxes.append(box)
                    found = True
                    break
            if not found:
                ok = False
                break

        if not ok:
            continue

        grid = np.zeros((H, W), dtype=int)
        for idx, patch in enumerate(patches):
            r0, c0 = positions[idx]
            h, w = patch.shape
            view = grid[r0 : r0 + h, c0 : c0 + w]
            mask = patch != 0
            if np.any(view[mask] != 0):
                ok = False
                break
            view[mask] = patch[mask]
        if ok:
            return grid

    return None


def _patch_from_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    oriented, h, w = random.choice(_dihedral_orientations(shape))
    patch = np.zeros((h, w), dtype=int)
    for r, c in oriented:
        patch[r, c] = color
    return patch


def _build_partition_instance() -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]] | None:
    H, W = random.choice(OUTPUT_DIMS)
    total = H * W
    feasible_counts = [n for n in (4, 5, 6) if 4 * n <= total <= 8 * n]
    n_pieces = random.choice(feasible_counts)

    areas = _sample_areas(total, n_pieces)
    if len(set(areas)) < 2:
        return None

    label_grid = _grow_partition(H, W, areas)
    if label_grid is None:
        return None

    shapes = _shapes_from_label_grid(label_grid)
    if any(not _is_irregular_piece(shape) for shape in shapes):
        return None
    if any(_has_hole(shape) for shape in shapes):
        return None

    canonicals = [_canonical_shape(shape) for shape in shapes]
    if len(set(canonicals)) != len(canonicals):
        return None

    return label_grid, shapes


def _gen_attempt() -> np.ndarray | None:
    built = _build_partition_instance()
    if built is None:
        return None
    label_grid, shapes = built

    try:
        solutions, signatures = _enumeration_summary(shapes)
    except AssertionError:
        return None
    if len(signatures) != 1 or len(solutions) < 4:
        return None

    require_tie = random.random() < 0.45
    tile = None
    expected = None
    expected_tie = False
    solver_out = None

    for _ in range(80):
        candidate_tile = _sample_tile()
        candidate_expected, candidate_tie = _construction_output(label_grid, candidate_tile)
        if require_tie and not candidate_tie:
            continue
        if len(set(map(int, candidate_expected.ravel()))) < 2:
            continue

        candidate_solver_out, _solver_tie = _best_output_from_solutions(solutions, candidate_tile)
        if not np.array_equal(candidate_solver_out, candidate_expected):
            continue

        tile = candidate_tile
        expected = candidate_expected
        expected_tie = candidate_tie
        solver_out = candidate_solver_out
        break

    if tile is None or expected is None or solver_out is None:
        return None
    assert np.array_equal(expected, solver_out)
    if require_tie:
        assert expected_tie

    patches = [_patch_from_shape(shape, GRAY) for shape in shapes]
    patches.append(tile.copy())
    grid = _scatter_patches(patches)
    if grid is None:
        return None

    parsed_shapes, parsed_tile = _parse_input(grid)
    if {_canonical_shape(shape) for shape in parsed_shapes} != {_canonical_shape(shape) for shape in shapes}:
        return None
    if not np.array_equal(parsed_tile, tile):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
