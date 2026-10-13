import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_core_and_path(grid: np.ndarray):
    core_comps = _components(grid == AZURE)
    assert len(core_comps) == 1, "expected exactly one azure core object"
    core = set(core_comps[0])

    path_mask = (grid != BLACK) & (grid != AZURE)
    path_comps = _components(path_mask)
    assert len(path_comps) == 1, "expected exactly one attached path"
    path = set(path_comps[0])
    assert path, "path must be non-empty"
    return core, path


def _ordered_path_and_colors(grid: np.ndarray, core: set[tuple[int, int]], path: set[tuple[int, int]]):
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    start_candidates: list[tuple[int, int]] = []
    for cell in path:
        r, c = cell
        nbrs = [nb for nb in _neighbors4(r, c) if nb in path]
        adjacency[cell] = nbrs
        if any(nb in core for nb in _neighbors4(r, c)):
            start_candidates.append(cell)

    assert len(start_candidates) == 1, "path must touch the core at exactly one cell"
    start = start_candidates[0]

    if len(path) == 1:
        ordered = [start]
    else:
        endpoints = [cell for cell, nbrs in adjacency.items() if len(nbrs) == 1]
        assert len(endpoints) == 2 and start in endpoints, "path must be a simple open chain"
        ordered: list[tuple[int, int]] = []
        prev = None
        cur = start
        while True:
            ordered.append(cur)
            next_cells = [nb for nb in adjacency[cur] if nb != prev]
            if not next_cells:
                break
            assert len(next_cells) == 1, "path branches are not allowed"
            prev, cur = cur, next_cells[0]
        assert len(ordered) == len(path), "path traversal must cover every path cell"

    colors = [int(grid[r, c]) for r, c in ordered if int(grid[r, c]) != GREY]
    assert all(color not in (BLACK, GREY, AZURE) for color in colors)
    return ordered, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colored markers along the attached grey path, erase the path,
    then grow successive 4-neighborhood contour rings around the azure core.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    core, path = _extract_core_and_path(g)
    ordered_path, colors = _ordered_path_and_colors(g, core, path)

    out = g.copy()
    for r, c in ordered_path:
        out[r, c] = BLACK

    filled = set(core)
    for color in colors:
        layer: set[tuple[int, int]] = set()
        for r, c in filled:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled and out[nr, nc] == BLACK:
                    layer.add((nr, nc))
        if not layer:
            break
        for r, c in layer:
            out[r, c] = color
        filled |= layer

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_point(point: tuple[int, int], rot: int, flip: bool) -> tuple[int, int]:
    r, c = point
    if flip:
        c = -c
    for _ in range(rot % 4):
        r, c = c, -r
    return r, c


def _transform_cells(cells: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    return {_transform_point(cell, rot, flip) for cell in cells}


def _random_polyomino(rng: random.Random, size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(rng.choice(frontier))

    rot = rng.randrange(4)
    flip = rng.choice((False, True))
    return _normalize(_transform_cells(cells, rot, flip))


def _anchor_candidates(core: set[tuple[int, int]]):
    candidates = []
    for r, c in core:
        for dr, dc in DIR4:
            outer = (r + dr, c + dc)
            if outer in core:
                continue
            core_touch = sum((outer[0] + rr, outer[1] + cc) in core for rr, cc in DIR4)
            if core_touch == 1:
                candidates.append((outer, (dr, dc)))
    return candidates


def _turn(direction: tuple[int, int], side: int) -> tuple[int, int]:
    dr, dc = direction
    if side < 0:
        return -dc, dr
    return dc, -dr


def _composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _valid_next(
    nxt: tuple[int, int],
    prev: tuple[int, int],
    path_set: set[tuple[int, int]],
    core: set[tuple[int, int]],
) -> bool:
    if nxt in core or nxt in path_set:
        return False
    if any(nb in core for nb in _neighbors4(*nxt)):
        return False
    neighbor_paths = [nb for nb in _neighbors4(*nxt) if nb in path_set]
    return neighbor_paths == [prev] or neighbor_paths == []


def _build_path(
    core: set[tuple[int, int]],
    start: tuple[int, int],
    initial_dir: tuple[int, int],
    length: int,
    segment_count: int,
    rng: random.Random,
):
    assert length >= 1
    path = [start]
    if length == 1:
        return path

    path_set = {start}
    current = start
    direction = initial_dir
    for seg_len in _composition(length - 1, segment_count, rng):
        for _ in range(seg_len):
            nxt = (current[0] + direction[0], current[1] + direction[1])
            if not _valid_next(nxt, current, path_set, core):
                return None
            path.append(nxt)
            path_set.add(nxt)
            current = nxt
        if len(path) < length:
            direction = _turn(direction, rng.choice((-1, 1)))
    return path


def _marker_positions(path_len: int, count: int, rng: random.Random) -> list[int]:
    slots = path_len - 1
    bases = sorted(rng.sample(range(slots - count + 1), count))
    return [base + idx + 1 for idx, base in enumerate(bases)]


def _random_colors(count: int, rng: random.Random) -> list[int]:
    if count <= len(MARKER_PALETTE) and rng.random() < 0.45:
        return rng.sample(list(MARKER_PALETTE), count)
    return [rng.choice(MARKER_PALETTE) for _ in range(count)]


def _place_scene(
    core: set[tuple[int, int]],
    path: list[tuple[int, int]],
    colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    k = len(colors)
    core_rs = [r for r, _ in core]
    core_cs = [c for _, c in core]
    all_cells = set(core) | set(path)
    all_rs = [r for r, _ in all_cells]
    all_cs = [c for _, c in all_cells]

    min_needed_r = min(min(all_rs), min(core_rs) - k)
    max_needed_r = max(max(all_rs), max(core_rs) + k)
    min_needed_c = min(min(all_cs), min(core_cs) - k)
    max_needed_c = max(max(all_cs), max(core_cs) + k)

    pad_top = rng.randint(1, 4)
    pad_bottom = rng.randint(1, 4)
    pad_left = rng.randint(1, 4)
    pad_right = rng.randint(1, 4)

    h = max_needed_r - min_needed_r + 1 + pad_top + pad_bottom
    w = max_needed_c - min_needed_c + 1 + pad_left + pad_right
    off_r = pad_top - min_needed_r
    off_c = pad_left - min_needed_c

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in core:
        grid[r + off_r, c + off_c] = AZURE
    for r, c in path:
        grid[r + off_r, c + off_c] = GREY

    for idx, color in zip(_marker_positions(len(path), len(colors), rng), colors):
        r, c = path[idx]
        grid[r + off_r, c + off_c] = color

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    core_size = rng.randint(3, 6)
    core = _random_polyomino(rng, core_size)
    anchors = _anchor_candidates(core)
    if not anchors:
        return None

    start, initial_dir = rng.choice(anchors)
    marker_count = rng.randint(1, 6)
    path_len = rng.randint(max(4, 2 * marker_count + 1), 3 * marker_count + 8)
    max_segments = min(4, path_len - 1)
    if max_segments < 1:
        return None

    path = None
    for _ in range(120):
        segment_count = rng.randint(1, max_segments)
        built = _build_path(core, start, initial_dir, path_len, segment_count, rng)
        if built is not None:
            path = built
            break
    if path is None:
        return None

    colors = _random_colors(marker_count, rng)
    grid = _place_scene(core, path, colors, rng)

    parsed_core, parsed_path = _extract_core_and_path(grid)
    _, parsed_colors = _ordered_path_and_colors(grid, parsed_core, parsed_path)
    if parsed_colors != colors:
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
