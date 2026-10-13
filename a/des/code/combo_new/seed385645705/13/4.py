from __future__ import annotations

import random

import numpy as np


GREY = 5
GEN_RETRIES = 400
PATH_RETRIES = 120

LEGEND_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    3: (
        ((0, 0), (0, 1), (1, 0)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ),
    5: (
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in neighbor_fn(r, c):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _normalize_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    if random.choice((False, True)):
        pts = [(r, -c) for r, c in pts]
    for _ in range(random.randrange(4)):
        pts = [(c, -r) for r, c in pts]
    return _normalize_cells(tuple(pts))


def _legend_entries_from_component(g: np.ndarray, pts: list[tuple[int, int]]) -> list[tuple[tuple[int, int], int]]:
    return [((int(r), int(c)), int(g[r, c])) for r, c in pts]


def _legend_lookup(entries: list[tuple[tuple[int, int], int]]) -> dict[frozenset[int], int]:
    colors = [color for _pos, color in entries]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"

    adj: dict[int, set[int]] = {color: set() for color in colors}
    for i, ((r1, c1), color1) in enumerate(entries):
        for j in range(i + 1, len(entries)):
            (r2, c2), color2 = entries[j]
            if max(abs(r1 - r2), abs(c1 - c2)) == 1:
                adj[color1].add(color2)
                adj[color2].add(color1)

    lookup: dict[frozenset[int], int] = {}
    for i, color1 in enumerate(colors):
        for j in range(i + 1, len(colors)):
            color2 = colors[j]
            if color2 not in adj[color1]:
                continue
            common = (adj[color1] & adj[color2]) - {color1, color2}
            if len(common) == 1:
                lookup[frozenset((color1, color2))] = next(iter(common))

    assert lookup, "legend must define at least one mixable pair"
    return lookup


def _find_legend_component(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    mask = (g != bg) & (g != GREY)
    candidates = []
    for pts in _components(mask, connectivity=8):
        colors = [int(g[r, c]) for r, c in pts]
        if 3 <= len(pts) <= 5 and len(set(colors)) == len(pts):
            r0 = min(r for r, _ in pts)
            c0 = min(c for _, c in pts)
            candidates.append((r0 + c0, r0, c0, -len(pts), pts))

    assert candidates, "expected a distinct multi-pixel legend in the top-left area"
    candidates.sort(key=lambda item: item[:4])
    return candidates[0][4]


def _path_component(g: np.ndarray, bg: int, legend_pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    legend_mask = np.zeros(g.shape, dtype=bool)
    for r, c in legend_pts:
        legend_mask[r, c] = True

    mask = (g != bg) & (~legend_mask)
    candidates = [pts for pts in _components(mask, connectivity=4) if any(int(g[r, c]) == GREY for r, c in pts)]
    assert len(candidates) == 1, "expected exactly one grey path component"
    return candidates[0]


def _ordered_path(path_pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(path_pts)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        nbs = [pos for pos in _neighbors4(r, c) if pos in cells]
        assert 1 <= len(nbs) <= 2, "path must be one-pixel wide and branchless"
        adj[(r, c)] = nbs

    endpoints = sorted(pos for pos, nbs in adj.items() if len(nbs) == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    ordered = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while True:
        nxts = [pos for pos in adj[cur] if pos != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "branchless path should have a unique forward step"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(cells), "path ordering must visit every path cell exactly once"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey valve-to-valve path segment using the color-mixing legend in the top-left corner.

    Input structure:
    - The background is the dominant color.
    - A disconnected top-left legend consists of 3 to 5 distinct non-grey colored pixels; legend adjacency is read in the 8-neighborhood.
    - The main object is a one-pixel-wide 4-connected path whose ordinary cells are grey and whose endpoints and internal checkpoints are colored valve pixels taken from the legend palette.

    Rule:
    - Extract the legend component and, for every adjacent legend color pair that has exactly one third legend color adjacent to both, record that unordered pair's mix result.
    - Remove the legend, linearize the remaining branchless path from one endpoint valve to the other, and split the ordered path at every valve pixel.
    - Repaint each grey run between consecutive valves with the lookup color for that valve pair, while keeping the valve pixels themselves unchanged.

    Key insight:
    - The puzzle is two-stage: first infer a visible pair-to-color lookup from the isolated legend, then exploit the fact that the maze trail is really a single ordered path whose valve positions define independent recoloring segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)

    legend_pts = _find_legend_component(g, bg)
    legend_entries = _legend_entries_from_component(g, legend_pts)
    legend_lookup = _legend_lookup(legend_entries)
    legend_colors = {color for _pos, color in legend_entries}

    path_pts = _path_component(g, bg, legend_pts)
    ordered = _ordered_path(path_pts)

    valve_indices = [idx for idx, (r, c) in enumerate(ordered) if int(g[r, c]) != GREY]
    assert valve_indices, "path must contain valve pixels"
    assert valve_indices[0] == 0 and valve_indices[-1] == len(ordered) - 1, "path endpoints must be valves"

    out = g.copy()
    for left_idx, right_idx in zip(valve_indices, valve_indices[1:], strict=False):
        assert right_idx - left_idx >= 2, "consecutive valves must enclose at least one grey path cell"
        left_color = int(g[ordered[left_idx]])
        right_color = int(g[ordered[right_idx]])
        assert left_color in legend_colors and right_color in legend_colors, "valves must use legend colors"
        assert left_color != right_color, "consecutive valves must use a mixable color pair"

        fill_color = legend_lookup[frozenset((left_color, right_color))]
        for idx in range(left_idx + 1, right_idx):
            r, c = ordered[idx]
            assert int(g[r, c]) == GREY, "only grey cells may appear between valves"
            out[r, c] = fill_color

    return out


def _path_rect_candidates(path_len: int) -> list[tuple[int, int]]:
    candidates = []
    for height in range(4, 16):
        for width in range(4, 16):
            area = height * width
            if area >= path_len + 6 and area <= max(42, 3 * path_len):
                candidates.append((height, width))
    return candidates


def _choose_legend_spec() -> tuple[list[tuple[int, int]], list[int]]:
    n_colors = random.choice((3, 4, 5))
    base = random.choice(LEGEND_SHAPES[n_colors])
    cells = list(_transform_cells(base))

    palette = [color for color in range(10) if color != GREY]
    colors = random.sample(palette, n_colors + 1)
    bg = colors[0]
    legend_colors = colors[1:]
    random.shuffle(legend_colors)
    return cells, [bg, *legend_colors]


def _legend_graph_neighbors(entries: list[tuple[tuple[int, int], int]]) -> dict[int, list[int]]:
    lookup = _legend_lookup(entries)
    colors = {color for _pos, color in entries}
    neighbors = {color: set() for color in colors}
    for pair in lookup:
        a, b = tuple(pair)
        neighbors[a].add(b)
        neighbors[b].add(a)
    return {color: sorted(vals) for color, vals in neighbors.items() if vals}


def _choose_valve_sequence(entries: list[tuple[tuple[int, int], int]], valve_count: int) -> list[int]:
    neighbors = _legend_graph_neighbors(entries)
    start = random.choice(sorted(neighbors))
    seq = [start]
    for _ in range(valve_count - 1):
        options = neighbors[seq[-1]]
        if len(seq) >= 2:
            back = seq[-2]
            non_back = [color for color in options if color != back]
            if non_back and random.random() < 0.65:
                options = non_back
        seq.append(random.choice(options))
    return seq


def _candidate_steps(
    tail: tuple[int, int], allowed: np.ndarray, used: set[tuple[int, int]]
) -> list[tuple[int, float, tuple[int, int]]]:
    candidates = []
    h, w = allowed.shape
    for nr, nc in _neighbors4(*tail):
        if not (0 <= nr < h and 0 <= nc < w) or not allowed[nr, nc] or (nr, nc) in used:
            continue

        touching = [pos for pos in _neighbors4(nr, nc) if pos in used]
        if touching != [tail] and sorted(touching) != [tail]:
            continue

        onward = 0
        for rr, cc in _neighbors4(nr, nc):
            if not (0 <= rr < h and 0 <= cc < w) or not allowed[rr, cc] or (rr, cc) in used:
                continue
            extra_touches = [pos for pos in _neighbors4(rr, cc) if pos in used and pos != tail]
            if extra_touches:
                continue
            onward += 1
        candidates.append((onward, random.random(), (nr, nc)))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates


def _sample_path(allowed: np.ndarray, path_len: int) -> list[tuple[int, int]] | None:
    free_cells = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    if len(free_cells) < path_len:
        return None

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == path_len:
            return True
        for _score, _noise, step in _candidate_steps(path[-1], allowed, used):
            used.add(step)
            path.append(step)
            if dfs(path, used):
                return True
            path.pop()
            used.remove(step)
        return False

    for _ in range(PATH_RETRIES):
        start = random.choice(free_cells)
        path = [start]
        used = {start}
        if dfs(path, used):
            return path
    return None


def _valve_positions_ok(path: list[tuple[int, int]], valve_indices: list[int]) -> bool:
    valve_cells = [path[idx] for idx in valve_indices]
    for i, (r1, c1) in enumerate(valve_cells):
        for r2, c2 in valve_cells[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _build_attempt() -> np.ndarray | None:
    legend_cells, palette = _choose_legend_spec()
    bg = palette[0]
    legend_colors = palette[1:]

    legend_h = 1 + max(r for r, _ in legend_cells)
    legend_w = 1 + max(c for _, c in legend_cells)
    legend_r0 = random.randint(0, 1)
    legend_c0 = random.randint(0, 1)

    valve_count = random.randint(3, 6)
    segment_lengths = [random.randint(1, 4) for _ in range(valve_count - 1)]
    path_len = valve_count + sum(segment_lengths)

    path_rect_h, path_rect_w = random.choice(_path_rect_candidates(path_len))
    path_top = legend_r0 + legend_h + random.randint(2, 4)
    path_left = legend_c0 + legend_w + random.randint(2, 4)
    height = path_top + path_rect_h + random.randint(1, 3)
    width = path_left + path_rect_w + random.randint(1, 3)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), bg, dtype=int)
    legend_entries: list[tuple[tuple[int, int], int]] = []
    for (dr, dc), color in zip(legend_cells, legend_colors, strict=False):
        rr = legend_r0 + dr
        cc = legend_c0 + dc
        grid[rr, cc] = color
        legend_entries.append(((rr, cc), color))

    valve_sequence = _choose_valve_sequence(legend_entries, valve_count)
    valve_indices = [0]
    cursor = 0
    for seg_len in segment_lengths:
        cursor += seg_len + 1
        valve_indices.append(cursor)

    allowed = np.zeros_like(grid, dtype=bool)
    allowed[path_top : path_top + path_rect_h, path_left : path_left + path_rect_w] = True
    path = _sample_path(allowed, path_len)
    if path is None or not _valve_positions_ok(path, valve_indices):
        return None

    for idx, (r, c) in enumerate(path):
        color = GREY
        if idx in valve_indices:
            color = valve_sequence[valve_indices.index(idx)]
        grid[r, c] = color

    non_grey_non_bg = (grid != bg) & (grid != GREY)
    comps8 = _components(non_grey_non_bg, connectivity=8)
    legend_like = [pts for pts in comps8 if len(pts) >= 3]
    if len(legend_like) != 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
