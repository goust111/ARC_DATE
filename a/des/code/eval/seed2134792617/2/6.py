from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WIRE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(
    grid: np.ndarray,
    bg: int,
    wire_color: int,
) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    special_mask = (grid != bg) & (grid != wire_color)
    candidates = []
    for comp in _components(special_mask):
        if len(comp) < 2:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        block = c1 - c0 + 1
        height = r1 - r0 + 1
        if c0 not in (0, w - block):
            continue
        if r0 != 0 and r1 != h - 1:
            continue
        if height % block != 0:
            continue
        if len(comp) != height * block:
            continue

        colors = []
        ok = True
        for i in range(height // block):
            patch = grid[r0 + i * block : r0 + (i + 1) * block, c0 : c1 + 1]
            vals = set(map(int, np.unique(patch)))
            if len(vals) != 1:
                ok = False
                break
            color = next(iter(vals))
            if color in {bg, wire_color}:
                ok = False
                break
            colors.append(color)

        if not ok or len(colors) != len(set(colors)):
            continue

        if any(
            grid[nr, nc] == wire_color
            for r, c in comp
            for nr, nc in _neighbors4(r, c, h, w)
            if (nr, nc) not in comp
        ):
            continue
        candidates.append((colors, comp))

    assert len(candidates) == 1, "expected one corner legend"
    colors, comp = candidates[0]
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r, c in comp:
        legend_mask[r, c] = True
    return colors, legend_mask


def _bfs_dist(network_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = network_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if network_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let colored sources propagate simultaneously through a grey wire network.

    Input structure
    - black background;
    - a one-pixel-wide grey orthogonal wire network;
    - several colored one-pixel sources replacing dead-end wire pixels;
    - a corner legend: a top-to-bottom stack of color blocks encoding priority.

    Rule
    - Each source spreads one step at a time through the wire.
    - At bends it keeps following the wire; at T/cross junctions it branches.
    - If two colors reach the same wire pixel at the same time, the legend order
      breaks the tie: earlier in the legend wins and the lower-priority signal
      stops there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    vals, counts = np.unique(g[g != bg], return_counts=True)
    assert len(vals) >= 3, "expected wire plus legend/source colors"
    wire_color = int(vals[counts.argmax()])
    assert int(counts.max()) > 2 and int(np.sum(counts == counts.max())) == 1

    priority_colors, legend_mask = _parse_legend(g, bg, wire_color)
    priority_rank = {color: idx for idx, color in enumerate(priority_colors)}

    source_cells = []
    for r, c in np.argwhere((g != bg) & (g != wire_color) & (~legend_mask)):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        assert color in priority_rank
        source_cells.append((rr, cc, color))

    assert len(source_cells) >= 2
    assert len(source_cells) == len({color for _r, _c, color in source_cells})
    assert {color for _r, _c, color in source_cells} == set(priority_colors)

    network_mask = g == wire_color
    for r, c, _color in source_cells:
        network_mask[r, c] = True

    for r, c, _color in source_cells:
        deg = sum(network_mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
        assert deg == 1, "sources must sit on dead ends"

    best_dist = np.full((h, w), 10**9, dtype=int)
    best_rank = np.full((h, w), 10**9, dtype=int)
    best_color = np.full((h, w), bg, dtype=int)
    for r, c, color in source_cells:
        dist = _bfs_dist(network_mask, (r, c))
        reachable = dist >= 0
        rank = priority_rank[color]
        better = reachable & (
            (dist < best_dist) | ((dist == best_dist) & (rank < best_rank))
        )
        best_dist[better] = dist[better]
        best_rank[better] = rank
        best_color[better] = color

    assert np.all(best_dist[network_mask] < 10**9), "all wires must be reachable"
    out = g.copy()
    out[network_mask] = best_color[network_mask]
    return out


def _legend_cells(
    h: int,
    w: int,
    corner: str,
    count: int,
    block: int,
) -> list[tuple[int, int]]:
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - block
    elif corner == "bl":
        r0, c0 = h - block * count, 0
    else:
        r0, c0 = h - block * count, w - block
    return [
        (r, c)
        for r in range(r0, r0 + block * count)
        for c in range(c0, c0 + block)
    ]


def _legend_forbidden(
    h: int,
    w: int,
    corner: str,
    count: int,
    block: int,
    pad: int = 2,
) -> np.ndarray:
    cells = _legend_cells(h, w, corner, count, block)
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _occupied_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    return [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc]]


def _degree(mask: np.ndarray, cell: tuple[int, int]) -> int:
    return len(_occupied_neighbors(mask, cell))


def _leaf_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if _degree(mask, (int(r), int(c))) == 1
    ]


def _is_turn(mask: np.ndarray, cell: tuple[int, int]) -> bool:
    neighbors = _occupied_neighbors(mask, cell)
    if len(neighbors) != 2:
        return False
    (r0, c0), (r1, c1) = neighbors
    r, c = cell
    dr0, dc0 = r0 - r, c0 - c
    dr1, dc1 = r1 - r, c1 - c
    return not (dr0 == -dr1 and dc0 == -dc1)


def _max_extension_length(
    mask: np.ndarray,
    forbidden: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    max_degree: int,
) -> int:
    h, w = mask.shape
    if _degree(mask, start) >= max_degree:
        return 0

    dr, dc = direction
    prev = start
    length = 0
    while True:
        nr = start[0] + dr * (length + 1)
        nc = start[1] + dc * (length + 1)
        if not (0 < nr < h - 1 and 0 < nc < w - 1):
            break
        if forbidden[nr, nc] or mask[nr, nc]:
            break
        bad = False
        for ar, ac in _neighbors4(nr, nc, h, w):
            if mask[ar, ac] and (ar, ac) != prev:
                bad = True
                break
        if bad:
            break
        length += 1
        prev = (nr, nc)
    return length


def _grow_segment(
    mask: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> None:
    dr, dc = direction
    for step in range(1, length + 1):
        mask[start[0] + dr * step, start[1] + dc * step] = True


def _pick_root(
    h: int,
    w: int,
    forbidden: np.ndarray,
    lengths: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if forbidden[r, c]:
                continue
            cells = [(r, c)]
            ok = True
            for (dr, dc), length in lengths.items():
                for step in range(1, length + 1):
                    rr = r + dr * step
                    cc = c + dc * step
                    if not (0 < rr < h - 1 and 0 < cc < w - 1):
                        ok = False
                        break
                    if forbidden[rr, cc]:
                        ok = False
                        break
                    cells.append((rr, cc))
                if not ok:
                    break
            if ok and len(cells) == len(set(cells)):
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _trace_distance(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    return _bfs_dist(mask, start)


def _force_turn(
    mask: np.ndarray,
    forbidden: np.ndarray,
    protected: set[tuple[int, int]],
    max_degree: int,
) -> bool:
    leaves = [cell for cell in _leaf_cells(mask) if cell not in protected]
    random.shuffle(leaves)
    for leaf in leaves:
        neighbors = _occupied_neighbors(mask, leaf)
        if len(neighbors) != 1:
            continue
        nr, nc = neighbors[0]
        back = (nr - leaf[0], nc - leaf[1])
        options = [
            direction
            for direction in DIRS4
            if direction != back and direction != (-back[0], -back[1])
        ]
        random.shuffle(options)
        for direction in options:
            max_len = _max_extension_length(mask, forbidden, leaf, direction, max_degree)
            if max_len >= 2:
                _grow_segment(mask, leaf, direction, random.randint(2, max_len))
                return True
    return False


def _branch_from_internal(
    mask: np.ndarray,
    forbidden: np.ndarray,
    protected: set[tuple[int, int]],
    max_degree: int,
) -> bool:
    cells = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in protected and 2 <= _degree(mask, (int(r), int(c))) < max_degree
    ]
    random.shuffle(cells)
    for start in cells:
        options = []
        for direction in DIRS4:
            if (start[0] + direction[0], start[1] + direction[1]) in _occupied_neighbors(mask, start):
                continue
            max_len = _max_extension_length(mask, forbidden, start, direction, max_degree)
            if max_len >= 2:
                options.append((direction, max_len))
        if options:
            direction, max_len = random.choice(options)
            _grow_segment(mask, start, direction, random.randint(2, max_len))
            return True
    return False


def _random_extension(
    mask: np.ndarray,
    forbidden: np.ndarray,
    protected: set[tuple[int, int]],
    max_degree: int,
) -> bool:
    starts = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in protected and _degree(mask, (int(r), int(c))) < max_degree
    ]
    random.shuffle(starts)
    for start in starts:
        options = []
        occupied = set(_occupied_neighbors(mask, start))
        for direction in DIRS4:
            if (start[0] + direction[0], start[1] + direction[1]) in occupied:
                continue
            max_len = _max_extension_length(mask, forbidden, start, direction, max_degree)
            if max_len >= 1:
                options.append((direction, max_len))
        if options:
            direction, max_len = random.choice(options)
            _grow_segment(mask, start, direction, random.randint(1, max_len))
            return True
    return False


def _root_competition_ok(
    mask: np.ndarray,
    root: tuple[int, int],
    protected_sources: list[tuple[int, int]],
    chosen_sources: list[tuple[int, int]],
) -> bool:
    root_dist = _trace_distance(mask, root)
    protected_steps = [int(root_dist[r, c]) for r, c in protected_sources]
    if len(set(protected_steps)) != 1:
        return False
    tie_dist = protected_steps[0]
    if tie_dist <= 0:
        return False
    all_steps = [int(root_dist[r, c]) for r, c in chosen_sources]
    return min(all_steps) == tie_dist and sum(step == tie_dist for step in all_steps) >= 2


def _build_input(
    mask: np.ndarray,
    source_colors: dict[tuple[int, int], int],
    priority: list[int],
    corner: str,
    block: int,
) -> np.ndarray:
    h, w = mask.shape
    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = WIRE
    for (r, c), color in source_colors.items():
        grid[r, c] = color

    cells = _legend_cells(h, w, corner, len(priority), block)
    r0, c0, r1, c1 = _bbox(cells)
    for i, color in enumerate(priority):
        grid[r0 + i * block : r0 + (i + 1) * block, c0 : c1 + 1] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    source_count = random.randint(3, 5)
    legend_block = random.choice((1, 2))
    corner = random.choice(CORNERS)
    need_cross = random.choice((False, True))
    max_degree = 4 if need_cross else 3

    forbidden = _legend_forbidden(h, w, corner, source_count, legend_block, pad=2)

    core_dirs = random.sample(DIRS4, 4 if need_cross else 3)
    tie_len = random.randint(2, 4)
    lengths = {direction: random.randint(2, 5) for direction in core_dirs}
    lengths[core_dirs[0]] = tie_len
    lengths[core_dirs[1]] = tie_len

    root = _pick_root(h, w, forbidden, lengths)
    if root is None:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[root] = True
    core_leaves = []
    for direction in core_dirs:
        _grow_segment(mask, root, direction, lengths[direction])
        dr, dc = direction
        core_leaves.append((root[0] + dr * lengths[direction], root[1] + dc * lengths[direction]))

    protected = {core_leaves[0], core_leaves[1]}
    if not _force_turn(mask, forbidden, protected, max_degree):
        return None

    branch_tries = 0
    while len(_leaf_cells(mask)) < source_count and branch_tries < 12:
        if not _branch_from_internal(mask, forbidden, protected, max_degree):
            break
        branch_tries += 1
    if len(_leaf_cells(mask)) < source_count:
        return None

    for _ in range(random.randint(0, 3)):
        _random_extension(mask, forbidden, protected, max_degree)

    if need_cross:
        if not any(_degree(mask, (int(r), int(c))) == 4 for r, c in np.argwhere(mask)):
            return None
    else:
        if any(_degree(mask, (int(r), int(c))) > 3 for r, c in np.argwhere(mask)):
            return None
        if not any(_degree(mask, (int(r), int(c))) == 3 for r, c in np.argwhere(mask)):
            return None

    if not any(_is_turn(mask, (int(r), int(c))) for r, c in np.argwhere(mask)):
        return None

    leaves = _leaf_cells(mask)
    if not protected.issubset(leaves):
        return None

    root_dist = _trace_distance(mask, root)
    extra_leaf_choices = [
        leaf
        for leaf in leaves
        if leaf not in protected and int(root_dist[leaf]) >= tie_len
    ]
    need_extra = source_count - len(protected)
    if len(extra_leaf_choices) < need_extra:
        return None

    chosen_sources = list(protected) + random.sample(extra_leaf_choices, need_extra)
    if not _root_competition_ok(mask, root, list(protected), chosen_sources):
        return None

    colors = random.sample(PALETTE, source_count)
    random.shuffle(chosen_sources)
    source_colors = {cell: color for cell, color in zip(chosen_sources, colors, strict=False)}
    priority = random.sample(colors, len(colors))

    grid = _build_input(mask, source_colors, priority, corner, legend_block)
    if _most_common_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(grid == WIRE):
        return None
    if np.any(out == WIRE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
