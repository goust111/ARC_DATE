import numpy as np
from collections import Counter


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _fg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(grid[nr, nc]) == bg or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            comps.append(comp)
    return comps


def _build_adj(coords: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pts = set(coords)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {pt: [] for pt in coords}
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in pts:
                adj[(r, c)].append((nr, nc))
    return adj


def _trace_path(
    start: tuple[int, int], adj: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [pt for pt in adj[cur] if pt != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return []
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _path_start(
    grid: np.ndarray,
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    color_counts: Counter,
) -> tuple[int, int] | None:
    endpoints = [pt for pt, nbrs in adj.items() if len(nbrs) == 1]
    if len(endpoints) != 2:
        return None

    scored: list[tuple[int, tuple[int, int]]] = []
    for endpoint in endpoints:
        mate = adj[endpoint][0]
        score = int(color_counts[int(grid[endpoint])] == 1) + int(
            color_counts[int(grid[mate])] == 1
        )
        scored.append((score, endpoint))

    best = [endpoint for score, endpoint in scored if score > 0]
    if len(best) != 1:
        return None
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify domino chains and swap each domino's colors when the previous domino is horizontal.

    Input structure
    - bg is the most frequent color.
    - Foreground cells form 2-cell dominoes. A chain is a linear sequence of dominoes connected by
      orthogonal adjacency between consecutive dominoes.
    - Each chain's start domino is the unique end whose first domino contains a globally unique color.

    Rule
    - Isolated dominoes stay unchanged.
    - Traverse every non-isolated chain from its marked start.
    - The start domino stays unchanged.
    - For each later domino: if the previous domino is horizontal, swap the two colors of the current
      domino; if the previous domino is vertical, keep the current domino unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    out = g.copy()
    color_counts = Counter(map(int, g[g != bg].ravel()))

    for comp in _fg_components(g, bg):
        if len(comp) <= 2:
            continue

        adj = _build_adj(comp)
        degrees = sorted(len(nbrs) for nbrs in adj.values())
        if len(comp) % 2 != 0 or degrees[:2] != [1, 1] or any(deg != 2 for deg in degrees[2:]):
            continue

        start = _path_start(g, adj, color_counts)
        if start is None:
            continue

        path = _trace_path(start, adj)
        if len(path) != len(comp):
            continue

        domino_count = len(path) // 2
        for domino_idx in range(1, domino_count):
            prev_a = path[2 * (domino_idx - 1)]
            prev_b = path[2 * (domino_idx - 1) + 1]
            cur_a = path[2 * domino_idx]
            cur_b = path[2 * domino_idx + 1]
            prev_horizontal = prev_a[0] == prev_b[0]
            if prev_horizontal:
                out[cur_a] = int(g[cur_b])
                out[cur_b] = int(g[cur_a])

    return out


_RNG = np.random.default_rng()

_GEN_RETRIES = 300
_GRID_H_RANGE = (11, 18)
_GRID_W_RANGE = (11, 18)
_CHAIN_COUNT_RANGE = (1, 3)
_CHAIN_LEN_RANGE = (3, 6)
_ISOLATED_COUNT_RANGE = (0, 2)
_REPEAT_COLOR_COUNT = 3
_INTERIOR_MARGIN = 1
_PATH_BUILD_ATTEMPTS = 48


def _global_forbidden(occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    forbidden = set(occupied)
    for r, c in occupied:
        forbidden.update(neighbors4(r, c))
    return forbidden


def _cell_is_valid(
    cell: tuple[int, int],
    h: int,
    w: int,
    path_set: set[tuple[int, int]],
    prev: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (_INTERIOR_MARGIN <= r < h - _INTERIOR_MARGIN and _INTERIOR_MARGIN <= c < w - _INTERIOR_MARGIN):
        return False
    if cell in path_set or cell in occupied:
        return False
    for nbr in neighbors4(r, c):
        if nbr in occupied:
            return False
        if nbr in path_set and nbr != prev:
            return False
    return True


def _build_induced_path(
    h: int,
    w: int,
    length: int,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    forbidden = _global_forbidden(occupied)
    starts = [
        (r, c)
        for r in range(_INTERIOR_MARGIN, h - _INTERIOR_MARGIN)
        for c in range(_INTERIOR_MARGIN, w - _INTERIOR_MARGIN)
        if (r, c) not in forbidden
    ]

    if not starts:
        return None

    for _ in range(_PATH_BUILD_ATTEMPTS):
        start = starts[int(rng.integers(len(starts)))]
        path = [start]
        path_set = {start}

        while len(path) < length:
            r, c = path[-1]
            candidates: list[tuple[int, float, tuple[int, int]]] = []
            for nxt in neighbors4(r, c):
                if not _cell_is_valid(nxt, h, w, path_set, path[-1], occupied):
                    continue
                trial_set = set(path_set)
                trial_set.add(nxt)
                onward = 0
                for after in neighbors4(*nxt):
                    if _cell_is_valid(after, h, w, trial_set, nxt, occupied):
                        onward += 1
                candidates.append((onward, float(rng.random()), nxt))

            if not candidates:
                break

            candidates.sort(reverse=True)
            best_onward = candidates[0][0]
            shortlist = [nxt for onward, _noise, nxt in candidates if onward == best_onward]
            nxt = shortlist[int(rng.integers(len(shortlist)))]
            path.append(nxt)
            path_set.add(nxt)

        if len(path) == length:
            return path
    return None


def _make_repeat_counts(
    total: int, palette: list[int], rng: np.random.Generator
) -> dict[int, int] | None:
    if total < 2:
        return None
    counts = {color: 0 for color in palette}
    remaining = total
    while remaining > 0:
        if remaining in (2, 3):
            chunk = remaining
        else:
            chunk = int(rng.choice([2, 3]))
            if remaining - chunk == 1:
                chunk = 5 - chunk
        counts[int(rng.choice(palette))] += chunk
        remaining -= chunk
    return counts


def _assign_regular_groups(
    group_sizes: list[int],
    palette: list[int],
    counts: dict[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, ...]] | None:
    assignments: list[tuple[int, ...]] = [tuple()] * len(group_sizes)

    def rec(idx: int) -> bool:
        if idx == len(group_sizes):
            return True

        options: list[tuple[int, ...]] = []
        if group_sizes[idx] == 1:
            options = [(color,) for color in palette if counts[color] > 0]
        else:
            for a in palette:
                if counts[a] == 0:
                    continue
                for b in palette:
                    if a == b or counts[b] == 0:
                        continue
                    options.append((a, b))
        rng.shuffle(options)

        for option in options:
            for color in option:
                counts[color] -= 1
            if min(counts.values()) >= 0 and rec(idx + 1):
                assignments[idx] = option
                return True
            for color in option:
                counts[color] += 1
        return False

    if rec(0):
        return assignments
    return None


def _paint_structure(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: list[int],
) -> None:
    for (r, c), color in zip(path, colors):
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    h = int(rng.integers(*_GRID_H_RANGE))
    w = int(rng.integers(*_GRID_W_RANGE))
    bg = int(rng.integers(10))

    color_pool = [color for color in range(10) if color != bg]
    rng.shuffle(color_pool)

    chain_count = int(rng.integers(*_CHAIN_COUNT_RANGE))
    isolated_count = int(rng.integers(*_ISOLATED_COUNT_RANGE))
    if chain_count + _REPEAT_COLOR_COUNT > len(color_pool):
        return None

    unique_colors = color_pool[:chain_count]
    repeat_palette = color_pool[chain_count : chain_count + _REPEAT_COLOR_COUNT]
    if len(repeat_palette) < _REPEAT_COLOR_COUNT:
        return None

    occupied: set[tuple[int, int]] = set()
    chain_paths: list[list[tuple[int, int]]] = []
    for _ in range(chain_count):
        length = 2 * int(rng.integers(*_CHAIN_LEN_RANGE))
        path = _build_induced_path(h, w, length, occupied, rng)
        if path is None:
            return None
        chain_paths.append(path)
        occupied.update(path)

    isolated_paths: list[list[tuple[int, int]]] = []
    for _ in range(isolated_count):
        path = _build_induced_path(h, w, 2, occupied, rng)
        if path is None:
            return None
        isolated_paths.append(path)
        occupied.update(path)

    regular_groups: list[int] = []
    chain_unique_indices: list[int] = []
    for path in chain_paths:
        unique_idx = int(rng.integers(0, 2))
        chain_unique_indices.append(unique_idx)
        regular_groups.append(1)
        regular_groups.extend([2] * (len(path) // 2 - 1))
    regular_groups.extend([2] * len(isolated_paths))

    total_regular = sum(regular_groups)
    repeat_counts = _make_repeat_counts(total_regular, repeat_palette, rng)
    if repeat_counts is None:
        return None
    regular_assignments = _assign_regular_groups(regular_groups, repeat_palette, repeat_counts, rng)
    if regular_assignments is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    group_idx = 0

    for chain_idx, path in enumerate(chain_paths):
        colors = [-1] * len(path)
        unique_idx = chain_unique_indices[chain_idx]
        colors[unique_idx] = unique_colors[chain_idx]
        start_group = regular_assignments[group_idx]
        group_idx += 1
        colors[1 - unique_idx] = start_group[0]
        for domino_idx in range(1, len(path) // 2):
            group = regular_assignments[group_idx]
            group_idx += 1
            colors[2 * domino_idx] = group[0]
            colors[2 * domino_idx + 1] = group[1]
        if any(colors[i] == colors[i + 1] for i in range(0, len(colors), 2)):
            return None
        _paint_structure(grid, path, colors)

    for path in isolated_paths:
        group = regular_assignments[group_idx]
        group_idx += 1
        if group[0] == group[1]:
            return None
        _paint_structure(grid, path, [group[0], group[1]])

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None

    fg_counts = Counter(map(int, grid[grid != bg].ravel()))
    comps = _fg_components(grid, bg)
    long_chain_count = 0
    for comp in comps:
        if len(comp) == 2:
            continue
        adj = _build_adj(comp)
        start = _path_start(grid, adj, fg_counts)
        if start is None:
            return None
        path = _trace_path(start, adj)
        if len(path) != len(comp) or len(path) % 2 != 0:
            return None
        long_chain_count += 1

    if long_chain_count != chain_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid domino-chain input")
