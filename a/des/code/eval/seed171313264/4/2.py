import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
TRACK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

COUNT_OPTIONS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
MIN_TRACK_SEPARATION = 3
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(grid.shape, cr, cc):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _neighbor_positions(shape: tuple[int, int], coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    coord_set = set(coords)
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in _neighbors4(shape, r, c):
            if (nr, nc) not in coord_set:
                out.add((nr, nc))
    return out


def _find_bulbs(grid: np.ndarray) -> list[dict]:
    bulbs: list[dict] = []
    for coords in _color_components(grid, GREY):
        if len(coords) != 4:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        bulbs.append(
            {
                "coords": sorted(coords),
                "bbox": (r0, c0, r1, c1),
                "neighbor_positions": _neighbor_positions(grid.shape, coords),
            }
        )
    assert bulbs, "expected at least one 2x2 grey bulb"
    return bulbs


def _track_colors_from_bulbs(grid: np.ndarray, bulbs: list[dict]) -> dict[int, int]:
    track_to_bulb: dict[int, int] = {}
    for bulb_id, bulb in enumerate(bulbs):
        colors = {
            int(grid[r, c])
            for r, c in bulb["neighbor_positions"]
            if int(grid[r, c]) not in (BG, GREY)
        }
        assert len(colors) == 1, "each bulb should touch exactly one track color"
        track_color = next(iter(colors))
        assert track_color not in track_to_bulb, "each track should end at a unique bulb"
        track_to_bulb[track_color] = bulb_id
    return track_to_bulb


def _find_sources(grid: np.ndarray, track_colors: set[int]) -> dict[int, dict]:
    sources: dict[int, dict] = {}
    for color in sorted(set(map(int, grid.flatten())) - {BG, GREY} - track_colors):
        for coords in _color_components(grid, color):
            if len(coords) != 2:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1) * (c1 - c0 + 1) != 2:
                continue
            neighbors = _neighbor_positions(grid.shape, coords)
            adjacent_tracks = {
                int(grid[r, c])
                for r, c in neighbors
                if int(grid[r, c]) in track_colors
            }
            if len(adjacent_tracks) != 1:
                continue
            track_color = next(iter(adjacent_tracks))
            assert track_color not in sources, "each track should have exactly one source"
            sources[track_color] = {
                "signal_color": int(color),
                "coords": sorted(coords),
                "bbox": (r0, c0, r1, c1),
            }
    assert sources and len(sources) == len(track_colors), "failed to find all sources"
    return sources


def _track_nodes(grid: np.ndarray, track_color: int, track_colors: set[int]) -> set[tuple[int, int]]:
    nodes = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(track_color))}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            cell = int(grid[r, c])
            if cell not in track_colors or cell == track_color:
                continue
            if 0 < c < w - 1 and int(grid[r, c - 1]) == track_color and int(grid[r, c + 1]) == track_color:
                nodes.add((r, c))
            if 0 < r < h - 1 and int(grid[r - 1, c]) == track_color and int(grid[r + 1, c]) == track_color:
                nodes.add((r, c))
    return nodes


def _ordered_track(
    grid: np.ndarray,
    track_color: int,
    node_set: set[tuple[int, int]],
    source_coords: list[tuple[int, int]],
    bulb_coords: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    source_neighbors = _neighbor_positions(grid.shape, source_coords)
    bulb_neighbors = _neighbor_positions(grid.shape, bulb_coords)
    starts = sorted(node_set & source_neighbors)
    ends = sorted(node_set & bulb_neighbors)
    assert len(starts) == 1, f"track {track_color} should touch its source at one endpoint"
    assert len(ends) == 1, f"track {track_color} should touch its bulb at one endpoint"
    start, end = starts[0], ends[0]

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in node_set:
        nbrs = []
        for nr, nc in _neighbors4(grid.shape, r, c):
            if (nr, nc) in node_set:
                nbrs.append((nr, nc))
        adjacency[(r, c)] = nbrs

    assert len(adjacency[start]) == 1 and len(adjacency[end]) == 1
    for node, nbrs in adjacency.items():
        if node not in (start, end):
            assert len(nbrs) == 2, f"track {track_color} should be a single non-branching path"

    path = [start]
    prev = None
    cur = start
    seen = {start}
    while cur != end:
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(nxts) == 1, f"track {track_color} path should advance uniquely"
        nxt = nxts[0]
        assert nxt not in seen or nxt == end, f"track {track_color} should not loop"
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert seen == node_set, f"track {track_color} path should cover every node"
    return path, {pos: idx for idx, pos in enumerate(path)}


def _extract_tracks(grid: np.ndarray):
    bulbs = _find_bulbs(grid)
    track_to_bulb = _track_colors_from_bulbs(grid, bulbs)
    track_colors = set(track_to_bulb)
    sources = _find_sources(grid, track_colors)

    tracks: dict[int, dict] = {}
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for track_color in sorted(track_colors):
        nodes = _track_nodes(grid, track_color, track_colors)
        path, index = _ordered_track(
            grid,
            track_color,
            nodes,
            sources[track_color]["coords"],
            bulbs[track_to_bulb[track_color]]["coords"],
        )
        tracks[track_color] = {
            "path": path,
            "index": index,
            "signal_color": sources[track_color]["signal_color"],
            "source_bbox": sources[track_color]["bbox"],
            "bulb_id": track_to_bulb[track_color],
        }
        for pos in path:
            owners[pos].append(track_color)

    crossings = {pos: tuple(colors) for pos, colors in owners.items() if len(colors) == 2}
    assert all(len(colors) <= 2 for colors in owners.values()), "only pairwise crossings are supported"
    return bulbs, tracks, crossings


def _route_signal(grid: np.ndarray, tracks: dict[int, dict], crossings: dict[tuple[int, int], tuple[int, int]], track_color: int):
    cur_track = track_color
    idx = 0
    straight_count = 0
    switch_count = 0
    seen: set[tuple[int, int]] = set()

    while True:
        path = tracks[cur_track]["path"]
        if idx >= len(path) - 1:
            return tracks[cur_track]["bulb_id"], straight_count, switch_count

        state = (cur_track, idx)
        assert state not in seen, "signal path should not cycle"
        seen.add(state)

        pos = path[idx]
        if pos in crossings and int(grid[pos]) != cur_track:
            a, b = crossings[pos]
            other = b if a == cur_track else a
            other_idx = tracks[other]["index"][pos]
            switch_count += 1
            cur_track = other
            idx = other_idx + 1
            continue

        if pos in crossings:
            straight_count += 1
        idx += 1


def _solve(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and np.all((0 <= g) & (g <= 9))
    bulbs, tracks, crossings = _extract_tracks(g)

    out = g.copy()
    fills: dict[int, int] = {}
    final_bulbs: list[int] = []
    total_straights = 0
    total_switches = 0

    source_order = sorted(tracks.items(), key=lambda item: item[1]["source_bbox"])
    for track_color, info in source_order:
        bulb_id, straight_count, switch_count = _route_signal(g, tracks, crossings, track_color)
        fills[bulb_id] = int(info["signal_color"])
        final_bulbs.append(int(bulb_id))
        total_straights += int(straight_count)
        total_switches += int(switch_count)

    for bulb_id, color in fills.items():
        for r, c in bulbs[bulb_id]["coords"]:
            out[r, c] = int(color)

    if collect_stats:
        return out, {
            "n_tracks": len(tracks),
            "n_crossings": len(crossings),
            "straight_count": total_straights,
            "switch_count": total_switches,
            "final_bulbs": final_bulbs,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source's signal along its rail.

    The solver reconstructs every rail from its source endpoint to its bulb
    endpoint. At a crossing, the visible pixel color identifies the upper rail:
    the upper rail continues straight; the lower rail transfers onto the other
    rail and then keeps moving along that rail's source-to-bulb direction.
    """
    return _solve(grid, collect_stats=False)


def _place_positions(n: int, lo: int, hi: int, min_sep: int) -> list[int] | None:
    span = hi - lo + 1
    needed = n + (n - 1) * (min_sep - 1)
    if span < needed:
        return None
    compressed = span - (n - 1) * (min_sep - 1)
    picks = sorted(random.sample(range(compressed), n))
    return [lo + picks[i] + i * (min_sep - 1) for i in range(n)]


def _apply_transform(grid: np.ndarray, code: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if code & 1:
        out = np.flipud(out)
    if code & 2:
        out = np.fliplr(out)
    if code & 4:
        out = out.T
    return np.array(out, copy=True)


def _render_candidate(h_rows: list[int], v_cols: list[int], h_colors: list[int], v_colors: list[int], signal_color: int, height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    for row in h_rows:
        grid[row, 0:2] = signal_color
        grid[row : row + 2, width - 2 : width] = GREY
    for col in v_cols:
        grid[0:2, col] = signal_color
        grid[height - 2 : height, col : col + 2] = GREY

    for row, color in zip(h_rows, h_colors):
        grid[row, 2 : width - 2] = int(color)
    for col, color in zip(v_cols, v_colors):
        grid[2 : height - 2, col] = int(color)

    for row, h_color in zip(h_rows, h_colors):
        for col, v_color in zip(v_cols, v_colors):
            grid[row, col] = int(random.choice((h_color, v_color)))

    return grid


def _candidate_sizes(n_h: int, n_v: int) -> tuple[int, int]:
    min_h = max(10, n_h + (n_h - 1) * (MIN_TRACK_SEPARATION - 1) + 6)
    min_w = max(10, n_v + (n_v - 1) * (MIN_TRACK_SEPARATION - 1) + 6)
    height = random.randint(min_h, min_h + 7)
    width = random.randint(min_w, min_w + 7)
    return height, width


def _gen_attempt() -> np.ndarray | None:
    n_h, n_v = random.choice(COUNT_OPTIONS)
    height, width = _candidate_sizes(n_h, n_v)

    h_rows = _place_positions(n_h, 2, height - 3, MIN_TRACK_SEPARATION)
    v_cols = _place_positions(n_v, 2, width - 3, MIN_TRACK_SEPARATION)
    if h_rows is None or v_cols is None:
        return None

    color_pool = TRACK_COLORS.copy()
    random.shuffle(color_pool)
    total_tracks = n_h + n_v
    if total_tracks + 1 > len(color_pool):
        return None
    signal_color = int(color_pool.pop())
    track_colors = color_pool[:total_tracks]
    h_colors = [int(c) for c in track_colors[:n_h]]
    v_colors = [int(c) for c in track_colors[n_h:]]

    grid = _render_candidate(h_rows, v_cols, h_colors, v_colors, signal_color, height, width)
    grid = _apply_transform(grid, random.randrange(8))

    try:
        out, stats = _solve(grid, collect_stats=True)
    except AssertionError:
        return None

    lit_bulbs = len(set(stats["final_bulbs"]))
    if stats["n_crossings"] < 1:
        return None
    if stats["straight_count"] < 1 or stats["switch_count"] < 1:
        return None
    if lit_bulbs < 2:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-signal puzzle")
