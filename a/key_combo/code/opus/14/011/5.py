import random
from collections import defaultdict, deque

import numpy as np


BG = 5
FG = 3
MARKER_COLORS = (1, 2, 4)
DIRS4 = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
PHASE_OFFSETS = (
    {"N": (-2, 0), "E": (0, 1), "S": (1, -1), "W": (-1, -2)},
    {"N": (-2, -1), "E": (-1, 1), "S": (1, 0), "W": (0, -2)},
)
GEN_ATTEMPTS = 600


def _neighbors4(r, c):
    for dr, dc in DIRS4.values():
        yield r + dr, c + dc


def _component_count(mask: np.ndarray) -> int:
    points = [tuple(coord) for coord in np.argwhere(mask)]
    if not points:
        return 0
    unseen = set(points)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
    return count


def _solve_unique_cover(mask: np.ndarray, limit: int = 2):
    fg_cells = [tuple(coord) for coord in np.argwhere(mask)]
    fg_set = set(fg_cells)
    candidates = []
    cell_to_candidates = defaultdict(list)

    h, w = mask.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if not bool(mask[r : r + 2, c : c + 2].all()):
                continue
            cells = ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))
            idx = len(candidates)
            candidates.append(cells)
            for cell in cells:
                cell_to_candidates[cell].append(idx)

    if any(cell not in cell_to_candidates for cell in fg_set):
        return 0, None

    covered = set()
    partial = []
    solution = None
    count = 0

    def dfs():
        nonlocal solution, count
        if count >= limit:
            return
        if len(covered) == len(fg_set):
            count += 1
            if count == 1:
                solution = list(partial)
            return

        best_options = None
        for cell in fg_cells:
            if cell in covered:
                continue
            options = []
            for idx in cell_to_candidates[cell]:
                tile = candidates[idx]
                if any(tile_cell in covered for tile_cell in tile):
                    continue
                options.append(idx)
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        for idx in best_options:
            tile = candidates[idx]
            partial.append(tile)
            covered.update(tile)
            dfs()
            for tile_cell in tile:
                covered.remove(tile_cell)
            partial.pop()

    dfs()
    return count, solution


def _tile_marker_counts(tile, marker_lookup):
    counts = defaultdict(int)
    for r, c in tile:
        for nr, nc in _neighbors4(r, c):
            color = marker_lookup.get((nr, nc))
            if color is not None:
                counts[color] += 1
    return counts


def _tile_winner(tile, marker_lookup, markers_sorted):
    counts = _tile_marker_counts(tile, marker_lookup)
    assert counts, "every tile must touch at least one marker"
    best = max(counts.values())
    tied_colors = {color for color, value in counts.items() if value == best}
    for r, c, color in markers_sorted:
        if color in tied_colors:
            return color
    raise AssertionError("tie-break marker not found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 2x2 tile in the green mask by the marker color with the strongest shared edge contact.

    Input structure:
    - The background is grey, with one 4-connected irregular green foreground mask.
    - The green mask is exactly coverable by 2x2 square tiles in a unique way.
    - Red, blue, and yellow single-pixel markers lie outside the mask on background cells.

    Rule:
    - Split the green mask into its unique exact cover of non-overlapping 2x2 squares.
    - For each tile, count all side-adjacencies from its four cells to the external marker pixels, grouped by marker color.
    - Recolor the whole tile with the color having the largest adjacency count; if colors tie, use the color of the earliest marker in row-major order among the tied colors.
    - Keep background and marker pixels unchanged.

    Key insight:
    - All tiles have the same shape, so the answer comes from recovering the exact tiling first and then using each tile's position-dependent marker contacts to choose its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g == FG
    assert np.any(mask), "expected a green mask"
    assert _component_count(mask) == 1, "green mask must be 4-connected"

    marker_positions = np.argwhere(np.isin(g, MARKER_COLORS))
    assert len(marker_positions) > 0, "expected external marker pixels"
    marker_lookup = {(int(r), int(c)): int(g[r, c]) for r, c in marker_positions}
    markers_sorted = sorted((r, c, color) for (r, c), color in marker_lookup.items())

    cover_count, tiles = _solve_unique_cover(mask, limit=2)
    assert cover_count == 1 and tiles is not None, "expected a unique 2x2 exact cover"

    out = g.copy()
    for tile in tiles:
        color = _tile_winner(tile, marker_lookup, markers_sorted)
        for r, c in tile:
            out[r, c] = color
    return out


def _marker_cell(node):
    r, c = node
    return 2 + 3 * r, 2 + 3 * c


def _tile_anchor(node, direction):
    mr, mc = _marker_cell(node)
    dr, dc = PHASE_OFFSETS[(node[0] + node[1]) % 2][direction]
    return mr + dr, mc + dc


def _tile_cells_from_anchor(anchor):
    r0, c0 = anchor
    return ((r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1))


def _connected_marker_subset(rows, cols, count):
    all_nodes = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = {random.choice(all_nodes)}
    tree_edges = []

    while len(chosen) < count:
        frontier = []
        for node in chosen:
            for direction, (dr, dc) in DIRS4.items():
                nxt = (node[0] + dr, node[1] + dc)
                if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in chosen:
                    frontier.append((node, direction, nxt))
        if not frontier:
            return None
        edge = random.choice(frontier)
        chosen.add(edge[2])
        tree_edges.append(edge)

    return chosen, tree_edges


def _all_internal_edges(nodes):
    node_set = set(nodes)
    edges = []
    for node in node_set:
        for direction, (dr, dc) in DIRS4.items():
            nxt = (node[0] + dr, node[1] + dc)
            if nxt in node_set and node < nxt:
                edges.append((node, direction, nxt))
    return edges


def _render_shape(tiles, markers):
    occupied = set()
    for tile in tiles:
        occupied.update(tile)
    occupied.update(markers)

    r0 = min(r for r, _ in occupied)
    c0 = min(c for _, c in occupied)
    r1 = max(r for r, _ in occupied)
    c1 = max(c for _, c in occupied)

    base_tiles = [
        tuple((r - r0, c - c0) for r, c in tile)
        for tile in tiles
    ]
    base_markers = tuple((r - r0, c - c0) for r, c in markers)
    base_h = r1 - r0 + 1
    base_w = c1 - c0 + 1

    valid_pads = []
    for top in range(3):
        for bottom in range(3):
            for left in range(3):
                for right in range(3):
                    h = base_h + top + bottom
                    w = base_w + left + right
                    area_ratio = 4 * len(base_tiles) / float(h * w)
                    if h > 30 or w > 30:
                        continue
                    if h < 8 or w < 8:
                        continue
                    if 0.38 <= area_ratio <= 0.55:
                        valid_pads.append((top, bottom, left, right))

    if not valid_pads:
        return None

    top, bottom, left, right = random.choice(valid_pads)
    shifted_tiles = [
        tuple((r + top, c + left) for r, c in tile)
        for tile in base_tiles
    ]
    shifted_markers = tuple((r + top, c + left) for r, c in base_markers)
    return shifted_tiles, shifted_markers, base_h + top + bottom, base_w + left + right


def _touches_other_tile(tile_a, tile_b):
    tile_b_set = set(tile_b)
    for r, c in tile_a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in tile_b_set:
                return True
    return False


def _nearest_marker_colors(tiles, markers_sorted):
    colors = []
    for tile in tiles:
        best = None
        best_color = None
        for r, c, color in markers_sorted:
            dist = min(abs(tr - r) + abs(tc - c) for tr, tc in tile)
            candidate = (dist, r, c)
            if best is None or candidate < best:
                best = candidate
                best_color = color
        colors.append(best_color)
    return colors


def _choose_marker_colors(markers, tiles):
    marker_positions = sorted(markers)
    tile_sets = [set(tile) for tile in tiles]

    for _ in range(300):
        marker_lookup = {pos: random.choice(MARKER_COLORS) for pos in marker_positions}
        if len(set(marker_lookup.values())) < 2:
            continue

        markers_sorted = [(r, c, marker_lookup[(r, c)]) for r, c in marker_positions]
        outputs = []
        conflict_tiles = 0
        simple_tiles = 0

        for tile in tiles:
            counts = _tile_marker_counts(tile, marker_lookup)
            if not counts:
                break
            if len(counts) >= 2:
                conflict_tiles += 1
            else:
                simple_tiles += 1
            outputs.append(_tile_winner(tile, marker_lookup, markers_sorted))
        else:
            if conflict_tiles < 2 or simple_tiles < 1:
                continue
            if len(set(outputs)) < 2:
                continue

            adjacent_diff = False
            for i, tile_a in enumerate(tiles):
                for j in range(i + 1, len(tiles)):
                    if outputs[i] == outputs[j]:
                        continue
                    if _touches_other_tile(tile_a, tiles[j]):
                        adjacent_diff = True
                        break
                if adjacent_diff:
                    break
            if not adjacent_diff:
                continue

            shortcut = _nearest_marker_colors(tiles, markers_sorted)
            if shortcut == outputs:
                continue

            return marker_lookup

    return None


def _build_attempt():
    lattice_rows, lattice_cols = random.choice(((3, 3), (3, 4), (4, 3)))
    marker_count = random.randint(4, min(8, lattice_rows * lattice_cols))
    subset = _connected_marker_subset(lattice_rows, lattice_cols, marker_count)
    if subset is None:
        return None
    nodes, tree_edges = subset
    nodes = set(nodes)

    edge_pairs = {frozenset((u, v)) for u, _, v in tree_edges}
    edges = list(tree_edges)
    extra_edges = [
        edge
        for edge in _all_internal_edges(nodes)
        if frozenset((edge[0], edge[2])) not in edge_pairs
    ]
    random.shuffle(extra_edges)
    extra_count = random.randint(0, len(extra_edges))
    edges.extend(extra_edges[:extra_count])

    used_cells = set()
    tiles = []
    for node, direction, _ in edges:
        tile = _tile_cells_from_anchor(_tile_anchor(node, direction))
        tile_set = set(tile)
        if used_cells & tile_set:
            return None
        used_cells.update(tile_set)
        tiles.append(tile)

    target_tile_count = random.randint(max(10, len(tiles)), min(18, len(tiles) + 8))
    leaf_options = [
        (node, direction)
        for node in nodes
        for direction, (dr, dc) in DIRS4.items()
        if (node[0] + dr, node[1] + dc) not in nodes
    ]
    random.shuffle(leaf_options)
    for node, direction in leaf_options:
        if len(tiles) >= target_tile_count:
            break
        tile = _tile_cells_from_anchor(_tile_anchor(node, direction))
        tile_set = set(tile)
        if used_cells & tile_set:
            continue
        used_cells.update(tile_set)
        tiles.append(tile)

    if len(tiles) < 10:
        return None

    markers = tuple(_marker_cell(node) for node in nodes)
    rendered = _render_shape(tiles, markers)
    if rendered is None:
        return None
    shifted_tiles, shifted_markers, height, width = rendered

    mask = np.zeros((height, width), dtype=bool)
    for tile in shifted_tiles:
        for r, c in tile:
            mask[r, c] = True

    if _component_count(mask) != 1:
        return None

    cover_count, solved_tiles = _solve_unique_cover(mask, limit=2)
    if cover_count != 1 or solved_tiles is None:
        return None

    marker_lookup = _choose_marker_colors(shifted_markers, solved_tiles)
    if marker_lookup is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[mask] = FG
    for (r, c), color in marker_lookup.items():
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    output = main(grid)
    tile_colors = []
    markers_sorted = sorted((r, c, marker_lookup[(r, c)]) for r, c in shifted_markers)
    for tile in solved_tiles:
        expected = _tile_winner(tile, marker_lookup, markers_sorted)
        tile_colors.append(expected)
        if any(output[r, c] != expected for r, c in tile):
            return None

    if len(set(tile_colors)) < 2:
        return None
    if np.array_equal(grid, output):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid exact-tiling adjacency-coloring input")
