from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SOIL = 3
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 300


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _flood_exterior(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _boundary_edges(
    component_padded: list[tuple[int, int]],
    exterior: np.ndarray,
) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    edges: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for r, c in component_padded:
        candidates = [
            ((r, c), (r, c + 1), (r - 1, c)),
            ((r, c + 1), (r + 1, c + 1), (r, c + 1)),
            ((r + 1, c + 1), (r + 1, c), (r + 1, c)),
            ((r + 1, c), (r, c), (r, c - 1)),
        ]
        for start, end, outside in candidates:
            if not exterior[outside[0], outside[1]]:
                continue
            assert start not in edges, "boundary should have one outgoing edge per vertex"
            edges[start] = (end, outside)

    incoming: dict[tuple[int, int], tuple[int, int]] = {}
    for start, (end, _outside) in edges.items():
        assert end not in incoming, "boundary should have one incoming edge per vertex"
        incoming[end] = start
    assert set(incoming) == set(edges), "boundary vertices must form closed cycle(s)"
    return edges


def _boundary_walk_from_edges(
    edges: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
    seed_padded: tuple[int, int],
) -> list[tuple[int, int]]:
    start_vertices = [start for start, (_end, outside) in edges.items() if outside == seed_padded]
    assert len(start_vertices) == 1, "seed must correspond to exactly one boundary edge"
    start = start_vertices[0]
    out_cells: list[tuple[int, int]] = []
    cur = start
    for _ in range(len(edges)):
        assert cur in edges, "boundary walk must stay on the traced contour"
        nxt, outside = edges[cur]
        out_cells.append(outside)
        cur = nxt
        if cur == start:
            break
    assert cur == start, "boundary walk must close"
    assert len(out_cells) == len(edges), "outer boundary should be a single cycle"

    compressed: list[tuple[int, int]] = []
    for cell in out_cells:
        if not compressed or cell != compressed[-1]:
            compressed.append(cell)
    if len(compressed) > 1 and compressed[-1] == compressed[0]:
        compressed.pop()
    assert compressed and compressed[0] == seed_padded
    return [(int(r - 1), int(c - 1)) for r, c in compressed]


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nbr = (int(r + dr), int(c + dc))
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _has_hole(cells: set[tuple[int, int]]) -> bool:
    cells = _normalize_cells(cells)
    _r0, _c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    padded = np.pad(mask, 1, constant_values=False)
    exterior = _flood_exterior(~padded)
    interior_bg = (~padded) & (~exterior)
    return bool(interior_bg.any())


def _local_geometry(
    cells: set[tuple[int, int]],
) -> tuple[
    set[tuple[int, int]],
    dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
    set[tuple[int, int]],
    list[tuple[int, int]],
]:
    soil = _normalize_cells(cells)
    _r0, _c0, r1, c1 = _bbox(soil)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in soil:
        mask[r, c] = True
    padded = np.pad(mask, 1, constant_values=False)
    exterior = _flood_exterior(~padded)
    component_padded = [(int(r + 1), int(c + 1)) for r, c in sorted(soil)]
    edges = _boundary_edges(component_padded, exterior)
    ring = {(int(outside[0] - 1), int(outside[1] - 1)) for _start, (_end, outside) in edges.items()}
    seed_candidates = [
        cell
        for cell in sorted(ring)
        if sum((int(cell[0] + dr), int(cell[1] + dc)) in soil for dr, dc in DIR4) == 1
    ]
    return soil, edges, ring, seed_candidates


def _component_seed(grid: np.ndarray, comp: set[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in comp:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, SOIL):
                candidates.add((nr, nc))
    assert len(candidates) == 1, "each soil object must have exactly one seed"
    seed = next(iter(candidates))
    assert sum((int(seed[0] + dr), int(seed[1] + dc)) in comp for dr, dc in DIR4) == 1
    return seed, int(grid[seed[0], seed[1]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed clockwise along the outer contour of its neighboring green soil object.

    Input structure:
    - The background is black and the soil color is green.
    - The grid contains several irregular 4-connected green soil objects.
    - Each soil object has exactly one non-black, non-green single-pixel seed on its outer contour, and that seed is 4-adjacent to exactly one soil pixel.

    Rule:
    - For each soil object, count its green pixels to get N.
    - Starting from that object's seed, follow the exterior contour clockwise on the non-soil side and paint the next N contour positions in the seed color, excluding the seed itself.
    - If the drawable contour is shorter than N, continue cycling around the same contour until N positions have been emitted.
    - Keep the green soil pixels unchanged and keep every seed pixel unchanged.

    Key insight:
    - The seed provides both the start point and the paint color, while the soil area controls the vine length and the vine follows the object's outer boundary rather than a straight ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    soil_mask = g == SOIL
    components = _components4(soil_mask)
    assert components, "expected at least one soil object"

    other_markers = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != SOIL))
    }
    padded_soil = np.pad(soil_mask, 1, constant_values=False)
    exterior = _flood_exterior(~padded_soil)

    out = g.copy()
    used_seeds: set[tuple[int, int]] = set()
    for comp in components:
        comp_set = {(int(r), int(c)) for r, c in comp}
        seed, seed_color = _component_seed(g, comp_set)
        used_seeds.add(seed)

        edges = _boundary_edges(
            component_padded=[(int(r + 1), int(c + 1)) for r, c in sorted(comp_set)],
            exterior=exterior,
        )
        path = _boundary_walk_from_edges(edges, (int(seed[0] + 1), int(seed[1] + 1)))
        assert path[0] == seed, "seed must be the first contour position"
        drawable = path[1:]
        assert drawable, "contour must contain at least one drawable position"

        area = len(comp_set)
        for i in range(area):
            r, c = drawable[i % len(drawable)]
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "drawable contour must stay on-grid"
            assert int(g[r, c]) == BG, "vine may only occupy background contour cells"
            out[r, c] = int(seed_color)

    assert used_seeds == other_markers, "all non-soil colored pixels must be seeds"
    return out


def _mutate_loop_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set(cells)
    for _ in range(random.randint(0, 3)):
        if random.random() < 0.5:
            removable = [cell for cell in cells if sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in DIR4) <= 2]
            random.shuffle(removable)
            for cell in removable:
                proposal = cells - {cell}
                if len(proposal) >= 10 and _is_connected(proposal) and not _has_hole(proposal):
                    cells = proposal
                    break
        else:
            frontier = {
                (int(r + dr), int(c + dc))
                for r, c in cells
                for dr, dc in DIR4
                if (int(r + dr), int(c + dc)) not in cells
            }
            frontier = list(frontier)
            random.shuffle(frontier)
            for cell in frontier:
                proposal = cells | {cell}
                proposal = _normalize_cells(proposal)
                _r0, _c0, r1, c1 = _bbox(proposal)
                if max(r1 + 1, c1 + 1) <= 7 and not _has_hole(proposal):
                    cells = proposal
                    break
    return _normalize_cells(cells)


def _sample_loop_shape() -> set[tuple[int, int]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(4, 5)
        w = random.randint(4, 5)
        cells = {(r, c) for r in range(h) for c in range(w)}
        cells = _mutate_loop_shape(cells)
        if not _is_connected(cells) or _has_hole(cells):
            continue
        return cells
    return None


def _sample_simple_shape() -> set[tuple[int, int]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        length = random.randint(4, 10)
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < length:
            endpoints = [path[0], path[-1]]
            choices: list[tuple[int, tuple[int, int]]] = []
            for which, end in enumerate(endpoints):
                for dr, dc in DIR4:
                    cand = (int(end[0] + dr), int(end[1] + dc))
                    if cand in used:
                        continue
                    if sum((int(cand[0] + dr2), int(cand[1] + dc2)) in used for dr2, dc2 in DIR4) != 1:
                        continue
                    choices.append((which, cand))
            if not choices:
                break
            which, cand = random.choice(choices)
            if which == 0:
                path.insert(0, cand)
            else:
                path.append(cand)
            used.add(cand)
        if len(path) != length:
            continue
        cells = _normalize_cells(set(path))
        if _has_hole(cells):
            continue
        return cells
    return None


def _make_object(mode: str, color: int) -> dict[str, object] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = _sample_loop_shape() if mode == "loop" else _sample_simple_shape()
        if cells is None:
            continue
        soil, edges, ring, seed_candidates = _local_geometry(cells)
        if not seed_candidates:
            continue

        temp_seed = random.choice(seed_candidates)
        path = _boundary_walk_from_edges(
            edges,
            (int(temp_seed[0] + 1), int(temp_seed[1] + 1)),
        )
        if len(set(path)) != len(path):
            continue

        area = len(soil)
        drawable_len = len(path) - 1
        if drawable_len < 1:
            continue
        if mode == "loop" and area <= drawable_len:
            continue
        if mode == "simple" and area > drawable_len:
            continue

        seed = random.choice(seed_candidates)
        reserve = set(soil) | set(ring)
        min_r = min(r for r, _c in reserve)
        min_c = min(c for _r, c in reserve)
        shift_r = -min_r
        shift_c = -min_c
        shifted_soil = {(int(r + shift_r), int(c + shift_c)) for r, c in soil}
        shifted_reserve = {(int(r + shift_r), int(c + shift_c)) for r, c in reserve}
        shifted_seed = (int(seed[0] + shift_r), int(seed[1] + shift_c))
        height = max(r for r, _c in shifted_reserve) + 1
        width = max(c for _r, c in shifted_reserve) + 1
        if max(height, width) > 9:
            continue
        return {
            "soil": shifted_soil,
            "reserve": shifted_reserve,
            "seed": shifted_seed,
            "color": int(color),
            "height": int(height),
            "width": int(width),
        }
    return None


def _choose_modes(n_objects: int) -> list[str]:
    if n_objects == 1:
        return [random.choice(["loop", "simple"])]
    n_loop = random.choice([n_objects // 2, (n_objects + 1) // 2])
    modes = ["loop"] * n_loop + ["simple"] * (n_objects - n_loop)
    random.shuffle(modes)
    return modes


def _place_objects(objects: list[dict[str, object]], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied_reserve: set[tuple[int, int]] = set()
    for obj in sorted(objects, key=lambda item: len(item["reserve"]), reverse=True):
        oh = int(obj["height"])
        ow = int(obj["width"])
        positions = [(r, c) for r in range(h - oh + 1) for c in range(w - ow + 1)]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            translated_reserve = {
                (int(top + r), int(left + c))
                for r, c in obj["reserve"]
            }
            if translated_reserve & occupied_reserve:
                continue
            for r, c in obj["soil"]:
                grid[top + r, left + c] = SOIL
            sr, sc = obj["seed"]
            grid[top + sr, left + sc] = int(obj["color"])
            occupied_reserve |= translated_reserve
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    colors = random.sample(SEED_COLORS, n_objects)
    modes = _choose_modes(n_objects)

    objects: list[dict[str, object]] = []
    for mode, color in zip(modes, colors):
        obj = _make_object(mode, color)
        if obj is None:
            return None
        objects.append(obj)

    min_h = max(int(obj["height"]) for obj in objects)
    min_w = max(int(obj["width"]) for obj in objects)
    lower_h = max(12, min_h + 1, 7 + 2 * n_objects)
    lower_w = max(12, min_w + 1, 7 + 3 * n_objects)
    if lower_h > 30 or lower_w > 30:
        return None

    h = random.randint(lower_h, 30)
    w = random.randint(lower_w, 30)
    grid = _place_objects(objects, h, w)
    if grid is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        assert isinstance(out, np.ndarray)
        return grid
    raise RuntimeError("failed to generate a valid input")
