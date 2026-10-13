import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PATH_COLORS = (AZURE, GREEN, YELLOW, MAGENTA)
GATE_COLORS = (GREEN, YELLOW, MAGENTA)
CORNER_TO_OFFSET = {
    "TL": (0, 0),
    "TR": (0, -1),
    "BL": (-1, 0),
    "BR": (-1, -1),
}

GEN_SIZE_CHOICES = (3, 4)
GEN_COARSE_MIN = 6
GEN_COARSE_MAX = 10
GEN_MAX_ATTEMPTS = 500
GEN_PATH_ATTEMPTS = 120
GEN_SHAPE_ATTEMPTS = 120
GEN_GATE_MARGIN = 2


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _adjacent_path_cells(mask: np.ndarray, r0: int, c0: int, path_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = mask.shape
    adj: set[tuple[int, int]] = set()
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            gr = r0 + rr
            gc = c0 + cc
            for dr, dc in DIRS4:
                cell = (gr + dr, gc + dc)
                if cell in path_set:
                    adj.add(cell)
    return adj


def _apply_gates(shape: np.ndarray, gates: list[int]) -> np.ndarray:
    out = shape.copy()
    for color in gates:
        if color == GREEN:
            out = np.rot90(out, -1)
        elif color == YELLOW:
            out = np.fliplr(out)
        elif color == MAGENTA:
            out = np.flipud(out)
        else:
            raise AssertionError(color)
    return out


def _trace_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = [cell for cell in _neighbors(cur[0], cur[1], h, w) if mask[cell] and cell != prev]
        assert len(nxts) == 1, "path must be non-branching"
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _gate_sequence(grid: np.ndarray, path: list[tuple[int, int]]) -> list[int]:
    colors = [int(grid[r, c]) for r, c in path]
    gates: list[int] = []
    i = 0
    while i < len(colors):
        color = colors[i]
        j = i + 1
        while j < len(colors) and colors[j] == color:
            j += 1
        if color != AZURE:
            assert color in GATE_COLORS, color
            assert j - i == 2, "each gate should occupy exactly two path cells"
            gates.append(color)
        i = j
    return gates


def _render_output(grid: np.ndarray, placeholder_bbox: tuple[int, int, int, int], shape: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = placeholder_bbox
    out = np.asarray(grid, dtype=int).copy()
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert shape.shape == (h, w)
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[shape] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the colored path, accumulate gate transforms, and draw the transformed red shape at the grey placeholder."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_comps = _components(g == RED)
    grey_comps = _components(g == GREY)
    assert len(red_comps) == 1, "expected one red source shape"
    assert len(grey_comps) == 1, "expected one grey destination block"

    red_mask = g == RED
    grey_mask = g == GREY
    walk_mask = np.isin(g, PATH_COLORS)
    walk_comps = _components(walk_mask)
    assert len(walk_comps) == 1, "expected one connected path"

    path_coords = walk_comps[0]
    degrees = {
        cell: sum(1 for nbr in _neighbors(cell[0], cell[1], g.shape[0], g.shape[1]) if walk_mask[nbr])
        for cell in path_coords
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path should have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    start_candidates = [
        cell
        for cell in endpoints
        if any(red_mask[nr, nc] for nr, nc in _neighbors(cell[0], cell[1], g.shape[0], g.shape[1]))
    ]
    end_candidates = [
        cell
        for cell in endpoints
        if any(grey_mask[nr, nc] for nr, nc in _neighbors(cell[0], cell[1], g.shape[0], g.shape[1]))
    ]
    assert len(start_candidates) == 1 and len(end_candidates) == 1
    start = start_candidates[0]
    end = end_candidates[0]
    assert start != end

    path = _trace_path(walk_mask, start, end)
    gates = _gate_sequence(g, path)

    red_bbox = _bbox(red_comps[0])
    rr0, rc0, rr1, rc1 = red_bbox
    source_shape = g[rr0 : rr1 + 1, rc0 : rc1 + 1] == RED

    grey_bbox = _bbox(grey_comps[0])
    gr0, gc0, gr1, gc1 = grey_bbox
    placeholder = g[gr0 : gr1 + 1, gc0 : gc1 + 1]
    assert bool(np.all(placeholder == GREY)), "destination placeholder should be a solid grey block"

    final_shape = _apply_gates(source_shape, gates)
    return _render_output(g, grey_bbox, final_shape)


def _inward_neighbor(cell: tuple[int, int], ch: int, cw: int) -> tuple[int, int]:
    r, c = cell
    if r == 0:
        return 1, c
    if r == ch - 1:
        return ch - 2, c
    if c == 0:
        return r, 1
    if c == cw - 1:
        return r, cw - 2
    raise AssertionError(cell)


def _boundary_cells(ch: int, cw: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(2, cw - 2)],
        "bottom": [(ch - 1, c) for c in range(2, cw - 2)],
        "left": [(r, 0) for r in range(2, ch - 2)],
        "right": [(r, cw - 1) for r in range(2, ch - 2)],
    }


def _interior_neighbors(cell: tuple[int, int], ch: int, cw: int) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (nr, nc)
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
        if 1 <= nr < ch - 1 and 1 <= nc < cw - 1
    ]


def _find_coarse_path(
    ch: int,
    cw: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    start_inner = _inward_neighbor(start, ch, cw)
    end_inner = _inward_neighbor(end, ch, cw)
    if start_inner == end_inner:
        return None

    interior_cells = (ch - 2) * (cw - 2)
    target_nodes = random.randint(6, min(12, interior_cells + 2))
    max_nodes = min(interior_cells + 2, target_nodes + random.randint(1, 4))

    path = [start_inner]
    visited = {start_inner}

    def dfs(cur: tuple[int, int]) -> list[tuple[int, int]] | None:
        if cur == end_inner:
            total_nodes = len(path) + 2
            if total_nodes >= target_nodes:
                return [start] + path[:] + [end]
            return None

        nbrs = _interior_neighbors(cur, ch, cw)
        random.shuffle(nbrs)
        nbrs.sort(key=lambda cell: (abs(cell[0] - end_inner[0]) + abs(cell[1] - end_inner[1]), random.random()))
        for nxt in nbrs:
            if nxt in visited:
                continue
            total_nodes = len(path) + 3
            if total_nodes > max_nodes:
                continue
            min_possible = total_nodes + abs(nxt[0] - end_inner[0]) + abs(nxt[1] - end_inner[1]) - 1
            if min_possible > max_nodes:
                continue
            if nxt == end_inner and total_nodes < target_nodes:
                continue
            visited.add(nxt)
            path.append(nxt)
            found = dfs(nxt)
            if found is not None:
                return found
            path.pop()
            visited.remove(nxt)
        return None

    for _ in range(GEN_PATH_ATTEMPTS):
        found = dfs(start_inner)
        if found is not None and _count_turns(found) >= 2:
            return found
    return None


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for (r0, c0), (r1, c1), (r2, c2) in zip(path, path[1:], path[2:], strict=False):
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            turns += 1
    return turns


def _render_fine_path(coarse_path: list[tuple[int, int]], ch: int, cw: int) -> tuple[int, int, list[tuple[int, int]]]:
    h = 2 * ch - 1
    w = 2 * cw - 1
    trace = [(2 * coarse_path[0][0], 2 * coarse_path[0][1])]
    for (r0, c0), (r1, c1) in zip(coarse_path, coarse_path[1:], strict=False):
        fr, fc = 2 * r0, 2 * c0
        tr, tc = 2 * r1, 2 * c1
        trace.append(((fr + tr) // 2, (fc + tc) // 2))
        trace.append((tr, tc))
    return h, w, trace


def _placement_options(side: str, endpoint: tuple[int, int], size: int, h: int, w: int) -> list[tuple[int, int, str]]:
    r, c = endpoint
    options: list[tuple[int, int, str]] = []
    if side == "top":
        options.extend([(0, c - size, "TR"), (0, c + 1, "TL")])
    elif side == "bottom":
        options.extend([(h - size, c - size, "BR"), (h - size, c + 1, "BL")])
    elif side == "left":
        options.extend([(r - size, 0, "BL"), (r + 1, 0, "TL")])
    else:
        options.extend([(r - size, w - size, "BR"), (r + 1, w - size, "TR")])
    out = []
    for r0, c0, corner in options:
        if 0 <= r0 and 0 <= c0 and r0 + size <= h and c0 + size <= w:
            out.append((r0, c0, corner))
    random.shuffle(out)
    return out


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return not (ar + asz <= br or br + bsz <= ar or ac + asz <= bc or bc + bsz <= ac)


def _bbox_hits_path(r0: int, c0: int, size: int, path_set: set[tuple[int, int]]) -> bool:
    for r in range(r0, r0 + size):
        for c in range(c0, c0 + size):
            if (r, c) in path_set:
                return True
    return False


def _sample_gate_colors(count: int) -> list[int]:
    if count == 1:
        return [random.choice(GATE_COLORS)]
    if count == 2 and random.random() < 0.5:
        return random.sample(list(GATE_COLORS), 2)
    if count >= 3 and random.random() < 0.5:
        colors = list(GATE_COLORS)
        while len(colors) < count:
            colors.append(random.choice(GATE_COLORS))
        random.shuffle(colors)
        return colors
    return [random.choice(GATE_COLORS) for _ in range(count)]


def _choose_gate_positions(trace: list[tuple[int, int]]) -> list[int] | None:
    candidates = list(range(GEN_GATE_MARGIN, len(trace) - GEN_GATE_MARGIN - 1))
    if not candidates:
        return None
    max_count = min(5, (len(trace) - 2 * GEN_GATE_MARGIN) // 3)
    if max_count <= 0:
        return None
    count = random.randint(1, max_count)
    random.shuffle(candidates)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) >= 3 for prev in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        return None
    chosen.sort()
    return chosen


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        options = []
        if r != tr:
            options.append((r + (1 if tr > r else -1), c))
        if c != tc:
            options.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _grow_mask(mask: np.ndarray, target: int) -> bool:
    h, w = mask.shape
    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[nr, nc] for nr, nc in _neighbors(r, c, h, w)):
                    frontier.append((r, c))
        if not frontier:
            return False
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return True


def _seed_shape(size: int, corner: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    corner_r = 0 if corner in ("TL", "TR") else size - 1
    corner_c = 0 if corner in ("TL", "BL") else size - 1
    opposite_r = size - 1 - corner_r
    opposite_c = size - 1 - corner_c
    row_anchor = (opposite_r, random.randrange(size))
    col_anchor = (random.randrange(size), opposite_c)
    for cell in _monotone_path((corner_r, corner_c), row_anchor):
        mask[cell] = True
    for cell in _monotone_path((corner_r, corner_c), col_anchor):
        mask[cell] = True
    return mask


def _make_shape(size: int, required_corner: str, gates: list[int]) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        mask = _seed_shape(size, required_corner)
        min_fill = max(size + 1, int(mask.sum()))
        max_fill = size * size - 1
        if min_fill > max_fill:
            continue
        target = random.randint(min_fill, max_fill)
        if not _grow_mask(mask, target):
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        transformed = _apply_gates(mask, gates)
        if np.array_equal(mask, transformed):
            continue
        return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _side_of_endpoint(cell: tuple[int, int], h: int, w: int) -> str:
    r, c = cell
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError(cell)


def _candidate_grid() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.choice(GEN_SIZE_CHOICES)
    ch = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)
    cw = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)
    sides = list(_boundary_cells(ch, cw).keys())
    random.shuffle(sides)
    start_side = sides[0]
    end_side = random.choice([side for side in sides[1:] if side != start_side])

    boundary = _boundary_cells(ch, cw)
    start = random.choice(boundary[start_side])
    end = random.choice(boundary[end_side])
    coarse_path = _find_coarse_path(ch, cw, start, end)
    if coarse_path is None:
        return None

    h, w, trace = _render_fine_path(coarse_path, ch, cw)
    start_fine = trace[0]
    end_fine = trace[-1]
    gate_positions = _choose_gate_positions(trace)
    if gate_positions is None:
        return None
    gate_colors = _sample_gate_colors(len(gate_positions))

    path_set = set(trace)
    start_options = _placement_options(_side_of_endpoint(start_fine, h, w), start_fine, size, h, w)
    end_options = _placement_options(_side_of_endpoint(end_fine, h, w), end_fine, size, h, w)
    if not start_options or not end_options:
        return None

    for sr0, sc0, required_corner in start_options:
        if _bbox_hits_path(sr0, sc0, size, path_set):
            continue
        source_shape = _make_shape(size, required_corner, gate_colors)
        if source_shape is None:
            continue
        start_adj = _adjacent_path_cells(source_shape, sr0, sc0, path_set)
        if start_fine not in start_adj or end_fine in start_adj:
            continue

        for er0, ec0, _ in end_options:
            if _bbox_hits_path(er0, ec0, size, path_set):
                continue
            if _rects_overlap((sr0, sc0, size), (er0, ec0, size)):
                continue
            placeholder_mask = np.ones((size, size), dtype=bool)
            end_adj = _adjacent_path_cells(placeholder_mask, er0, ec0, path_set)
            if end_fine not in end_adj or start_fine in end_adj:
                continue

            grid = np.full((h, w), BG, dtype=int)
            for r, c in trace:
                grid[r, c] = AZURE
            for idx, color in zip(gate_positions, gate_colors, strict=False):
                for r, c in trace[idx : idx + 2]:
                    grid[r, c] = color

            _paint_mask(grid, source_shape, sr0, sc0, RED)
            grid[er0 : er0 + size, ec0 : ec0 + size] = GREY

            expected = _render_output(grid, (er0, ec0, er0 + size - 1, ec0 + size - 1), _apply_gates(source_shape, gate_colors))
            if not np.array_equal(main(grid), expected):
                continue
            return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        generated = _candidate_grid()
        if generated is not None:
            grid, _ = generated
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
