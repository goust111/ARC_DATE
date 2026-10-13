import random
from collections import Counter, deque

import numpy as np


GREY = 5
MAROON = 9
GREEN = 3

HORIZONTAL = 0
VERTICAL = 1
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _find_regions(wall_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = wall_mask.shape
    region_of = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if wall_mask[r, c] or region_of[r, c] != -1:
                continue
            rid = len(regions)
            q = deque([(r, c)])
            region_of[r, c] = rid
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if wall_mask[nr, nc] or region_of[nr, nc] != -1:
                        continue
                    region_of[nr, nc] = rid
                    q.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert coords_arr.shape[0] == area, "each pane should be a solid rectangle"
            regions.append({"coords": coords_arr, "bbox": (r0, c0, r1, c1)})

    return regions, region_of


def _extract_channels(grid: np.ndarray, region_of: np.ndarray, num_regions: int) -> dict[int, list[dict]]:
    h, w = grid.shape
    channels = {rid: [] for rid in range(num_regions)}

    for r, c in np.argwhere(grid == MAROON):
        left = region_of[r, c - 1] if c - 1 >= 0 else -1
        right = region_of[r, c + 1] if c + 1 < w else -1
        up = region_of[r - 1, c] if r - 1 >= 0 else -1
        down = region_of[r + 1, c] if r + 1 < h else -1

        if left != -1 and right != -1 and up == -1 and down == -1:
            assert left != right
            channels[left].append({"dir": (0, 1), "neighbor": right, "gate": (int(r), int(c))})
            channels[right].append({"dir": (0, -1), "neighbor": left, "gate": (int(r), int(c))})
            continue
        if up != -1 and down != -1 and left == -1 and right == -1:
            assert up != down
            channels[up].append({"dir": (1, 0), "neighbor": down, "gate": (int(r), int(c))})
            channels[down].append({"dir": (-1, 0), "neighbor": up, "gate": (int(r), int(c))})
            continue
        raise AssertionError("each maroon pixel must be a single-channel opening")

    return channels


def _arrow_direction(green_coords: np.ndarray) -> tuple[int, int]:
    pts = [tuple(map(int, pt)) for pt in green_coords]
    center = None
    for r, c in pts:
        touches = sum(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in pts if (rr, cc) != (r, c))
        if touches == 2:
            center = (r, c)
            break
    assert center is not None, "arrow center not found"

    others = [(r, c) for r, c in pts if (r, c) != center]
    avg_r = (others[0][0] + others[1][0]) / 2.0
    avg_c = (others[0][1] + others[1][1]) / 2.0
    dr = _sign(center[0] - avg_r)
    dc = _sign(center[1] - avg_c)
    assert (dr, dc) in {(-1, 0), (1, 0), (0, -1), (0, 1)}
    return dr, dc


def _line_color(values: np.ndarray) -> int:
    kept = [int(v) for v in values if int(v) != GREEN]
    assert kept
    colors = set(kept)
    assert len(colors) == 1
    return kept[0]


def _infer_source_pattern(grid: np.ndarray, region: dict) -> tuple[int, int, int]:
    r0, c0, r1, c1 = region["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = sorted({int(v) for v in patch.ravel()} - {GREEN})
    assert len(colors) == 2, "source pane must contain exactly two stripe colors besides the arrow"

    row_score = sum(len({int(v) for v in row if int(v) != GREEN}) > 1 for row in patch)
    col_score = sum(len({int(v) for v in col if int(v) != GREEN}) > 1 for col in patch.T)
    orientation = HORIZONTAL if row_score < col_score else VERTICAL

    if orientation == HORIZONTAL:
        first = _line_color(patch[0, :])
        second = next((int(_line_color(row)) for row in patch[1:, :] if _line_color(row) != first), None)
    else:
        first = _line_color(patch[:, 0])
        second = next((int(_line_color(col)) for col in patch[:, 1:].T if _line_color(col) != first), None)
    if second is None:
        second = colors[0] if colors[1] == first else colors[1]
    assert first != second
    return orientation, first, second


def _stripe_patch(height: int, width: int, orientation: int, color0: int, color1: int) -> np.ndarray:
    patch = np.empty((height, width), dtype=int)
    if orientation == HORIZONTAL:
        for r in range(height):
            patch[r, :] = color0 if r % 2 == 0 else color1
    else:
        for c in range(width):
            patch[:, c] = color0 if c % 2 == 0 else color1
    return patch


def _paint_region(out: np.ndarray, region: dict, orientation: int, color0: int, color1: int) -> None:
    r0, c0, r1, c1 = region["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = _stripe_patch(r1 - r0 + 1, c1 - c0 + 1, orientation, color0, color1)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall_mask = (g == GREY) | (g == MAROON)

    regions, region_of = _find_regions(wall_mask)
    channels = _extract_channels(g, region_of, len(regions))

    green_coords = np.argwhere(g == GREEN)
    assert green_coords.shape == (3, 2), "there must be exactly one 3-pixel green arrow"
    source_id = int(region_of[tuple(green_coords[0])])
    assert source_id >= 0
    assert all(int(region_of[tuple(pt)]) == source_id for pt in green_coords)

    arrow_dir = _arrow_direction(green_coords)
    source_edges = [edge for edge in channels[source_id] if edge["dir"] == arrow_dir]
    assert len(source_edges) == 1, "arrow must point to exactly one channel"

    source_orientation, color0, color1 = _infer_source_pattern(g, regions[source_id])
    blank_region = {
        rid: np.all(g[region["coords"][:, 0], region["coords"][:, 1]] == bg)
        for rid, region in enumerate(regions)
    }
    blank_region[source_id] = False

    out = g.copy()
    _paint_region(out, regions[source_id], source_orientation, color0, color1)

    filled_orientation = {source_id: source_orientation}
    queue = deque([(source_id, source_edges[0]["neighbor"])])

    while queue:
        parent, rid = queue.popleft()
        if rid in filled_orientation or not blank_region[rid]:
            continue

        orientation = VERTICAL if filled_orientation[parent] == HORIZONTAL else HORIZONTAL
        _paint_region(out, regions[rid], orientation, color0, color1)
        filled_orientation[rid] = orientation

        for edge in channels[rid]:
            if edge["neighbor"] not in filled_orientation:
                queue.append((rid, edge["neighbor"]))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source pane's two-color stripes through maroon wall openings.

    The grey and maroon wall pixels stay fixed. The unique green arrow marks the
    source pane and indicates the first channel. Each channel crossing rotates
    the stripe orientation by 90 degrees, and the cascade continues through
    later channels into still-empty panes. The green arrow is overwritten by
    the restored stripe pattern in the output.
    """
    return _solve(grid)


GEN_RETRIES = 400
GRID_BANDS_MIN = 2
GRID_BANDS_MAX = 4
PANE_SIZE_MIN = 4
PANE_SIZE_MAX = 7
GRID_SIZE_MAX = 30

BG_CHOICES = [0, 1, 2, 4, 6, 7, 8]


def _region_bbox(row_lines: list[int], col_lines: list[int], cell: tuple[int, int]) -> tuple[int, int, int, int]:
    rr, cc = cell
    return row_lines[rr] + 1, col_lines[cc] + 1, row_lines[rr + 1] - 1, col_lines[cc + 1] - 1


def _cell_neighbors(cell: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    rr, cc = cell
    nbrs = []
    if rr > 0:
        nbrs.append((rr - 1, cc))
    if rr + 1 < n_rows:
        nbrs.append((rr + 1, cc))
    if cc > 0:
        nbrs.append((rr, cc - 1))
    if cc + 1 < n_cols:
        nbrs.append((rr, cc + 1))
    return nbrs


def _grow_tree(rng: random.Random, n_rows: int, n_cols: int, target_size: int) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    start = (rng.randrange(n_rows), rng.randrange(n_cols))
    selected = {start}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(selected) < target_size:
        frontier = []
        for cell in selected:
            for nbr in _cell_neighbors(cell, n_rows, n_cols):
                if nbr not in selected:
                    frontier.append((cell, nbr))
        assert frontier
        edge = rng.choice(frontier)
        selected.add(edge[1])
        edges.append(edge)

    return selected, edges


def _tree_adjacency(selected: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj = {cell: [] for cell in selected}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    return _sign(dst[0] - src[0]), _sign(dst[1] - src[1])


def _gate_candidates(bbox: tuple[int, int, int, int], axis: int) -> list[int]:
    if axis == 0:
        lo, hi = bbox[0], bbox[2]
    else:
        lo, hi = bbox[1], bbox[3]
    if hi - lo >= 2:
        return list(range(lo + 1, hi))
    return list(range(lo, hi + 1))


def _arrow_cells(bbox: tuple[int, int, int, int], direction: tuple[int, int], gate_pos: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gr, gc = gate_pos
    if direction == (0, 1):
        cells = [(gr, c1), (gr - 1, c1 - 1), (gr + 1, c1 - 1)]
    elif direction == (0, -1):
        cells = [(gr, c0), (gr - 1, c0 + 1), (gr + 1, c0 + 1)]
    elif direction == (1, 0):
        cells = [(r1, gc), (r1 - 1, gc - 1), (r1 - 1, gc + 1)]
    elif direction == (-1, 0):
        cells = [(r0, gc), (r0 + 1, gc - 1), (r0 + 1, gc + 1)]
    else:
        raise AssertionError("invalid arrow direction")

    for r, c in cells:
        assert r0 <= r <= r1 and c0 <= c <= c1
    return cells


def _paint_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], orientation: int, color0: int, color1: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = _stripe_patch(r1 - r0 + 1, c1 - c0 + 1, orientation, color0, color1)


def _tree_distances(source: tuple[int, int], adj: dict[tuple[int, int], list[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    dist = {source: 0}
    q = deque([source])
    while q:
        cell = q.popleft()
        for nbr in adj[cell]:
            if nbr in dist:
                continue
            dist[nbr] = dist[cell] + 1
            q.append(nbr)
    return dist


def _validate_sample(
    grid: np.ndarray,
    bg: int,
    source_orientation: int,
    color0: int,
    color1: int,
    row_lines: list[int],
    col_lines: list[int],
    selected: set[tuple[int, int]],
    distances: dict[tuple[int, int], int],
) -> bool:
    try:
        out = _solve(grid)
    except AssertionError:
        return False

    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            bbox = _region_bbox(row_lines, col_lines, (rr, cc))
            r0, c0, r1, c1 = bbox
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            if (rr, cc) in selected:
                orientation = source_orientation if distances[(rr, cc)] % 2 == 0 else 1 - source_orientation
                expect = _stripe_patch(r1 - r0 + 1, c1 - c0 + 1, orientation, color0, color1)
                if not np.array_equal(patch, expect):
                    return False
            else:
                if not np.all(patch == bg):
                    return False

    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(n) for v, n in zip(values, counts)}
    if count_map[bg] != max(count_map.values()):
        return False
    if count_map[bg] == max(v for k, v in count_map.items() if k != bg):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(GRID_BANDS_MIN, GRID_BANDS_MAX)
    n_cols = rng.randint(GRID_BANDS_MIN, GRID_BANDS_MAX)
    pane_heights = [rng.randint(PANE_SIZE_MIN, PANE_SIZE_MAX) for _ in range(n_rows)]
    pane_widths = [rng.randint(PANE_SIZE_MIN, PANE_SIZE_MAX) for _ in range(n_cols)]

    h = sum(pane_heights) + n_rows + 1
    w = sum(pane_widths) + n_cols + 1
    if h > GRID_SIZE_MAX or w > GRID_SIZE_MAX:
        return None

    total_cells = n_rows * n_cols
    target_size = rng.randint(2, total_cells)
    selected, edges = _grow_tree(rng, n_rows, n_cols, target_size)
    adj = _tree_adjacency(selected, edges)
    leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1]
    if not leaves:
        return None
    source = rng.choice(leaves)
    source_neighbor = adj[source][0]
    source_dir = _direction_between(source, source_neighbor)

    source_orientation = rng.choice([HORIZONTAL, VERTICAL])
    bg = rng.choice(BG_CHOICES)
    palette = [color for color in range(10) if color not in {bg, GREY, MAROON, GREEN}]
    color0, color1 = rng.sample(palette, 2)

    row_lines = [0]
    for size in pane_heights:
        row_lines.append(row_lines[-1] + size + 1)
    col_lines = [0]
    for size in pane_widths:
        col_lines.append(col_lines[-1] + size + 1)

    grid = np.full((h, w), bg, dtype=int)
    for r in row_lines:
        grid[r, :] = GREY
    for c in col_lines:
        grid[:, c] = GREY

    gate_positions: dict[frozenset, tuple[int, int]] = {}
    for a, b in edges:
        key = frozenset((a, b))
        if a[0] == b[0]:
            row_band = a[0]
            left = a if a[1] < b[1] else b
            bbox = _region_bbox(row_lines, col_lines, left)
            gate_row = rng.choice(_gate_candidates(bbox, axis=0))
            gate_col = col_lines[left[1] + 1]
            gate_positions[key] = (gate_row, gate_col)
        else:
            col_band = a[1]
            top = a if a[0] < b[0] else b
            bbox = _region_bbox(row_lines, col_lines, top)
            gate_col = rng.choice(_gate_candidates(bbox, axis=1))
            gate_row = row_lines[top[0] + 1]
            gate_positions[key] = (gate_row, gate_col)

    source_key = frozenset((source, source_neighbor))
    source_bbox = _region_bbox(row_lines, col_lines, source)
    if source_dir in {(0, 1), (0, -1)}:
        candidates = _gate_candidates(source_bbox, axis=0)
        gate_pos = (rng.choice(candidates), gate_positions[source_key][1])
    else:
        candidates = _gate_candidates(source_bbox, axis=1)
        gate_pos = (gate_positions[source_key][0], rng.choice(candidates))
    gate_positions[source_key] = gate_pos

    for pos in gate_positions.values():
        grid[pos] = MAROON

    _paint_bbox(grid, source_bbox, source_orientation, color0, color1)
    for r, c in _arrow_cells(source_bbox, source_dir, gate_positions[source_key]):
        grid[r, c] = GREEN

    distances = _tree_distances(source, adj)
    if not _validate_sample(grid, bg, source_orientation, color0, color1, row_lines, col_lines, selected, distances):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
