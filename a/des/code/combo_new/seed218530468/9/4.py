import itertools
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SOURCE_POOL = (1, 2, 3, 6, 7)  # blue, red, green, magenta, orange
MIX_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _corner_score(r0: int, r1: int, c0: int, c1: int, h: int, w: int) -> int:
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def _find_legend(grid: np.ndarray) -> tuple[dict[frozenset[int], int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != BLACK):
        r0, r1, c0, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww != 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BLACK) or np.any(patch == GREY):
            continue
        if patch.size != len(coords):
            continue
        score = _corner_score(r0, r1, c0, c1, h, w)
        candidates.append((score, hh, r0, c0, patch.copy()))
    assert candidates, "legend not found"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, hh, r0, c0, patch = candidates[0]

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + hh, c0 : c0 + 3] = True

    mapping: dict[frozenset[int], int] = {}
    for row in patch:
        left, mix, right = map(int, row)
        assert left not in (BLACK, GREY)
        assert right not in (BLACK, GREY)
        assert mix not in (BLACK, GREY)
        assert left != right
        key = frozenset((left, right))
        if key in mapping:
            assert mapping[key] == mix
        mapping[key] = mix
    assert mapping, "legend mapping must be non-empty"
    return mapping, legend_mask


def _network_sources(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[tuple[int, int], int]]]:
    network_mask = (grid != BLACK) & (~legend_mask)
    comps = _components(network_mask)
    assert len(comps) == 1, "pipe network must form one component"

    source_cells = []
    for r, c in np.argwhere(network_mask & (grid != GREY)):
        pos = (int(r), int(c))
        source_cells.append((pos, int(grid[pos])))
    assert 2 <= len(source_cells) <= 4, "expected 2 to 4 source cells"
    colors = [color for _, color in source_cells]
    assert len(set(colors)) == len(colors), "source colors must be unique"
    return network_mask, source_cells


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        base = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = base
                q.append((nr, nc))
    return dist


def _resolve_winners(colors: list[int], mix_map: dict[frozenset[int], int]) -> int:
    if len(colors) == 1:
        return int(colors[0])
    if len(colors) == 2:
        key = frozenset(colors)
        assert key in mix_map, "missing legend entry for color pair"
        return int(mix_map[key])

    pair_outputs = []
    for a, b in itertools.combinations(colors, 2):
        key = frozenset((int(a), int(b)))
        assert key in mix_map, "missing legend entry for multi-way tie"
        pair_outputs.append(int(mix_map[key]))
    if len(set(pair_outputs)) == 1:
        return pair_outputs[0]
    raise AssertionError("unsupported tie among more than two source colors")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a grey pipe network by synchronous color diffusion using a corner legend for two-color collisions.

    Input structure:
    - The background is black.
    - One connected pipe network is drawn with 1-pixel-wide grey horizontal and vertical segments.
    - Some dead-end pipe endpoints are replaced by distinct source colors from a small fixed palette.
    - A separate legend near one corner contains stacked 1x3 horizontal bars: the left and right cells are source colors, and the middle cell is their collision color.

    Rule:
    - Keep the legend unchanged and treat the remaining non-black component as the pipe network.
    - Start simultaneous unit-speed diffusion from every colored source through the network's grey cells.
    - For each grey pipe cell, compare its shortest path distance to every source through the network.
    - If exactly one source is nearest, color that cell with the source color; if two sources tie for nearest arrival, color that cell with the legend's mixed color for that pair.
    - Source cells keep their original colors.

    Key insight:
    The visual diffusion is equivalent to labeling each pipe cell by the set of nearest sources in the network graph, with the corner legend decoding every two-source tie.
    """
    g = np.asarray(grid, dtype=int)
    mix_map, legend_mask = _find_legend(g)
    network_mask, sources = _network_sources(g, legend_mask)

    dist_maps = [(color, _bfs_dist(network_mask, pos)) for pos, color in sources]

    out = g.copy()
    for r, c in np.argwhere(network_mask & (g == GREY)):
        rr = int(r)
        cc = int(c)
        best = None
        winners: list[int] = []
        for color, dist in dist_maps:
            d = int(dist[rr, cc])
            assert d >= 0, "all pipe cells must be reachable from a source"
            if best is None or d < best:
                best = d
                winners = [int(color)]
            elif d == best:
                winners.append(int(color))
        out[rr, cc] = _resolve_winners(winners, mix_map)
    return out


def _shuffle_in_place(items: list):
    order = _RNG.permutation(len(items))
    shuffled = [items[int(i)] for i in order]
    items[:] = shuffled


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                out[nr, nc] = True
    return out


def _legend_origin(h: int, w: int, rows: int, corner: str) -> tuple[int, int]:
    max_off_r = 1 if h - rows >= 2 else 0
    max_off_c = 1 if w - 3 >= 2 else 0
    off_r = int(_RNG.integers(0, max_off_r + 1))
    off_c = int(_RNG.integers(0, max_off_c + 1))
    if corner == "tl":
        return off_r, off_c
    if corner == "tr":
        return off_r, w - 3 - off_c
    if corner == "bl":
        return h - rows - off_r, off_c
    return h - rows - off_r, w - 3 - off_c


def _render_legend(
    grid: np.ndarray,
    pair_order: list[tuple[int, int]],
    mix_map: dict[frozenset[int], int],
) -> np.ndarray:
    h, w = grid.shape
    corner = ("tl", "tr", "bl", "br")[int(_RNG.integers(0, 4))]
    r0, c0 = _legend_origin(h, w, len(pair_order), corner)
    mask = np.zeros_like(grid, dtype=bool)
    for idx, pair in enumerate(pair_order):
        left, right = pair
        if float(_RNG.random()) < 0.5:
            left, right = right, left
        grid[r0 + idx, c0 : c0 + 3] = [left, mix_map[frozenset(pair)], right]
        mask[r0 + idx, c0 : c0 + 3] = True
    return mask


def _cell_degree(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in occupied for dr, dc in DIR4)


def _path_if_clear(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    forbidden: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = forbidden.shape
    dr, dc = direction
    r, c = start
    prev = start
    added: list[tuple[int, int]] = []
    for _ in range(length):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if forbidden[r, c] or (r, c) in occupied:
            return None
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in occupied and (nr, nc) != prev:
                return None
        added.append((r, c))
        prev = (r, c)
    return added


def _cells_to_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_tree_mask(h: int, w: int, forbidden: np.ndarray, source_count: int) -> np.ndarray | None:
    allowed = np.argwhere(~forbidden)
    if len(allowed) == 0:
        return None

    min_cells = 10 + 4 * source_count
    max_cells = min(75, max(min_cells, (h * w - int(forbidden.sum())) // 4))
    if max_cells < min_cells:
        return None

    for _ in range(80):
        root_r, root_c = allowed[int(_RNG.integers(0, len(allowed)))]
        occupied: set[tuple[int, int]] = {(int(root_r), int(root_c))}
        target = int(_RNG.integers(min_cells, max_cells + 1))
        stall = 0
        while len(occupied) < target and stall < 180:
            candidates = [cell for cell in occupied if _cell_degree(cell, occupied) < 4]
            if not candidates:
                break
            start = candidates[int(_RNG.integers(0, len(candidates)))]
            dirs = list(DIR4)
            _shuffle_in_place(dirs)
            progressed = False
            for direction in dirs:
                next_r = start[0] + direction[0]
                next_c = start[1] + direction[1]
                if not (0 <= next_r < h and 0 <= next_c < w):
                    continue
                if forbidden[next_r, next_c] or (next_r, next_c) in occupied:
                    continue
                max_len_here = min(6, target - len(occupied))
                if max_len_here <= 0:
                    break
                lengths = list(range(1, max_len_here + 1))
                _shuffle_in_place(lengths)
                for length in lengths:
                    path = _path_if_clear(start, direction, int(length), occupied, forbidden)
                    if path:
                        occupied.update(path)
                        progressed = True
                        break
                if progressed:
                    break
            stall = 0 if progressed else stall + 1

        if len(occupied) < min_cells:
            continue

        degs = {_cell: _cell_degree(_cell, occupied) for _cell in occupied}
        leaves = [cell for cell, degree in degs.items() if degree == 1]
        if len(leaves) < source_count:
            continue
        if source_count >= 3 and max(degs.values()) < 3:
            continue
        return _cells_to_mask(occupied, h, w)
    return None


def _seed_combo_stats(mask: np.ndarray, seeds: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    dist_maps = [_bfs_dist(mask, seed) for seed in seeds]
    seed_set = set(seeds)
    mixed = 0
    unique = 0
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seed_set:
            continue
        dists = [int(dist[cell]) for dist in dist_maps]
        best = min(dists)
        winners = sum(dist == best for dist in dists)
        if winners == 1:
            unique += 1
        elif winners == 2:
            mixed += 1
        else:
            return None
    if mixed == 0 or unique == 0:
        return None
    return mixed, unique


def _choose_seeds(mask: np.ndarray, source_count: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    degs = {cell: _cell_degree(cell, cells) for cell in cells}
    leaves = [cell for cell, degree in degs.items() if degree == 1]
    if len(leaves) < source_count:
        return None

    valid: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
    for combo in itertools.combinations(leaves, source_count):
        stats = _seed_combo_stats(mask, combo)
        if stats is not None:
            valid.append((combo, stats))
    if not valid:
        return None

    weights = np.array([1 + mixed for _, (mixed, _unique) in valid], dtype=float)
    weights /= weights.sum()
    idx = int(_RNG.choice(len(valid), p=weights))
    return valid[idx]


def _mix_mapping(source_colors: list[int]) -> tuple[list[tuple[int, int]], dict[frozenset[int], int]]:
    pairs = list(itertools.combinations(source_colors, 2))
    _shuffle_in_place(pairs)

    available = [color for color in MIX_POOL if color not in source_colors and color not in (BLACK, GREY)]
    assert available, "need at least one mix color"
    sequence: list[int] = []
    while len(sequence) < len(pairs):
        block = available[:]
        _shuffle_in_place(block)
        sequence.extend(block)

    mapping: dict[frozenset[int], int] = {}
    for idx, pair in enumerate(pairs):
        mapping[frozenset(pair)] = int(sequence[idx])
    return pairs, mapping


def _gen_attempt() -> np.ndarray | None:
    source_count = int(_RNG.integers(2, 5))
    pair_count = source_count * (source_count - 1) // 2

    h_min = max(12, pair_count + 7)
    w_min = 12
    h = int(_RNG.integers(h_min, 26))
    w = int(_RNG.integers(w_min, 26))

    source_colors = [int(v) for v in _RNG.choice(np.array(SOURCE_POOL), size=source_count, replace=False)]
    pair_order, mix_map = _mix_mapping(source_colors)

    grid = np.full((h, w), BLACK, dtype=int)
    legend_mask = _render_legend(grid, pair_order, mix_map)
    forbidden = _expand_mask(legend_mask)

    tree_mask = _generate_tree_mask(h, w, forbidden, source_count)
    if tree_mask is None:
        return None

    choice = _choose_seeds(tree_mask, source_count)
    if choice is None:
        return None
    seeds, _stats = choice

    color_order = source_colors[:]
    _shuffle_in_place(color_order)
    grid[tree_mask] = GREY
    for pos, color in zip(seeds, color_order):
        grid[pos] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out[tree_mask] == GREY):
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[legend_mask], grid[legend_mask]):
        return None
    if not np.any((out != grid) & tree_mask):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-diffusion input")
