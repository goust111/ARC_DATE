from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
PATH_COLOR = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_DISTANCE = 5
CELL_GAP = 4
MIN_CELL_SIZE = 11
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 200

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_shape(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    return norm


def _rotate90(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, -r) for r, c in coords]


def _mirror(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, -c) for r, c in coords]


def _all_transforms(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    variants: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = list(coords)
    for _ in range(4):
        for raw in (cur, _mirror(cur)):
            norm = _normalize_shape(raw)
            key = tuple(norm)
            if key not in seen:
                seen.add(key)
                variants.append(norm)
        cur = _rotate90(cur)
    return variants


def _shape_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _choose_fitting_transform(
    coords: list[tuple[int, int]], max_h: int, max_w: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    candidates = []
    for variant in _all_transforms(coords):
        h, w = _shape_dims(variant)
        if h <= max_h and w <= max_w:
            candidates.append(variant)
    if not candidates:
        return None
    return list(rng.choice(candidates))


def _random_path_shape(max_h: int, max_w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    shape_types = ["line", "elbow", "zigzag"]
    for _ in range(100):
        kind = rng.choice(shape_types)
        coords: list[tuple[int, int]]
        if kind == "line":
            max_len = min(15, max(max_h, max_w))
            if max_len < 3:
                continue
            length = rng.randint(3, max_len)
            coords = [(0, c) for c in range(length)]
        elif kind == "elbow":
            pairs = [
                (a, b)
                for a in range(2, min(8, max(max_h, max_w)) + 1)
                for b in range(2, min(8, max(max_h, max_w)) + 1)
                if a + b - 1 <= 15
            ]
            if not pairs:
                continue
            a, b = rng.choice(pairs)
            coords = [(0, c) for c in range(a)] + [(r, a - 1) for r in range(1, b)]
        else:
            triples = [
                (a, b, c)
                for a in range(3, min(8, max(max_h, max_w)) + 1)
                for b in range(2, min(6, max(max_h, max_w)) + 1)
                for c in range(2, a + 1)
                if a + b + c - 2 <= 15
            ]
            if not triples:
                continue
            a, b, c = rng.choice(triples)
            coords = [(0, x) for x in range(a)]
            coords += [(y, a - 1) for y in range(1, b)]
            coords += [(b - 1, a - 1 - x) for x in range(1, c)]

        fitted = _choose_fitting_transform(coords, max_h, max_w, rng)
        if fitted is not None:
            return fitted
    return None


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_scene(
    grid: np.ndarray,
) -> tuple[list[set[tuple[int, int]]], dict[int, dict[str, object]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components_of_color(g, PATH_COLOR)
    assert comps, "expected at least one grey path"

    cell_to_comp: dict[tuple[int, int], int] = {}
    for comp_id, comp in enumerate(comps):
        for cell in comp:
            cell_to_comp[cell] = comp_id

    sources = [tuple(map(int, p)) for p in np.argwhere((g != BG_COLOR) & (g != PATH_COLOR))]
    assert len(sources) == len(comps), "expected one source per path"

    associations: dict[int, dict[str, object]] = {}
    for sr, sc in sources:
        color = int(g[sr, sc])
        candidates: list[tuple[int, int, int, int]] = []
        for dr, dc in DIRS.values():
            rr, cc = sr, sc
            for dist in range(1, MAX_DISTANCE + 1):
                rr += dr
                cc += dc
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                value = int(g[rr, cc])
                if value == BG_COLOR:
                    continue
                if value == PATH_COLOR:
                    candidates.append((cell_to_comp[(rr, cc)], dr, dc, dist))
                break
        assert len(candidates) == 1, "source-to-path relation must be unique"
        comp_id, dr, dc, dist = candidates[0]
        assert comp_id not in associations, "each path must have exactly one source"
        associations[comp_id] = {
            "source": (sr, sc),
            "color": color,
            "dir": (dr, dc),
            "dist": dist,
        }

    assert len(associations) == len(comps), "every path must receive a source"
    return comps, associations, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored bands away from each grey path using the opposite side and distance encoded by its source pixel.

    Input structure:
    - The background is black.
    - One or more grey 4-connected path objects appear as 1-cell-wide straight or bent polylines.
    - Each grey path has exactly one colored source pixel on one cardinal side, 1 to 5 cells away through black space.

    Rule:
    - For each colored source, scan orthogonally until the first non-black cell; it must uniquely hit one grey path.
    - The hit distance gives the band length, and that scan direction is the path's outward growth direction because it is opposite the source side.
    - From every pixel of the matched path, paint black cells step by step along that direction for that many cells, stopping before any non-black input pixel.
    - Keep the grey paths unchanged and clear all source pixels to black.

    Key insight:
    A source pixel jointly encodes three visible cues: which path it belongs to, which color to use, and how far the opposite-side band should extend.
    """
    g = np.asarray(grid, dtype=int)
    comps, associations, sources = _extract_scene(g)

    out = g.copy()
    for sr, sc in sources:
        out[sr, sc] = BG_COLOR

    for comp_id, comp in enumerate(comps):
        info = associations[comp_id]
        dr, dc = info["dir"]
        dist = int(info["dist"])
        color = int(info["color"])
        for pr, pc in comp:
            for step in range(1, dist + 1):
                rr = pr + dr * step
                cc = pc + dc * step
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                if int(g[rr, cc]) != BG_COLOR:
                    break
                out[rr, cc] = color

    return out


def _random_split(total: int, parts: int, min_size: int, rng: random.Random) -> list[int]:
    sizes = [min_size] * parts
    remaining = total - parts * min_size
    for _ in range(remaining):
        sizes[rng.randrange(parts)] += 1
    return sizes


def _valid_source_candidates(
    path_cells: set[tuple[int, int]], cell_h: int, cell_w: int, side: str, dist: int
) -> list[tuple[int, int]]:
    side_dr, side_dc = DIRS[side]
    scan_dr, scan_dc = -side_dr, -side_dc
    candidates: set[tuple[int, int]] = set()

    for pr, pc in path_cells:
        sr = pr + side_dr * dist
        sc = pc + side_dc * dist
        if not (0 <= sr < cell_h and 0 <= sc < cell_w):
            continue
        if (sr, sc) in path_cells:
            continue
        clear = True
        for step in range(1, dist):
            mid = (pr + side_dr * step, pc + side_dc * step)
            if mid in path_cells:
                clear = False
                break
        if not clear:
            continue

        test = np.zeros((cell_h, cell_w), dtype=int)
        for rr, cc in path_cells:
            test[rr, cc] = PATH_COLOR
        test[sr, sc] = 1
        try:
            comps, associations, _ = _extract_scene(test)
        except AssertionError:
            continue
        if len(comps) != 1 or 0 not in associations:
            continue
        assoc = associations[0]
        if tuple(assoc["dir"]) != (scan_dr, scan_dc):
            continue
        if int(assoc["dist"]) != dist:
            continue
        candidates.add((sr, sc))

    return sorted(candidates)


def _make_object(
    cell_h: int, cell_w: int, side: str, rng: random.Random
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(OBJECT_ATTEMPTS):
        shape = _random_path_shape(cell_h - 2, cell_w - 2, rng)
        if shape is None:
            continue
        shape_h, shape_w = _shape_dims(shape)
        max_dist = min(MAX_DISTANCE, (cell_h - shape_h) // 2, (cell_w - shape_w) // 2)
        if max_dist < 1:
            continue
        dist = rng.randint(1, max_dist)

        min_r = dist
        max_r = cell_h - shape_h - dist
        min_c = dist
        max_c = cell_w - shape_w - dist
        if min_r > max_r or min_c > max_c:
            continue

        off_r = rng.randint(min_r, max_r)
        off_c = rng.randint(min_c, max_c)
        path_cells = {(off_r + r, off_c + c) for r, c in shape}
        candidates = _valid_source_candidates(path_cells, cell_h, cell_w, side, dist)
        if not candidates:
            continue
        return path_cells, rng.choice(candidates)
    return None


def _build_grid(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(2, 4)
    layouts = [(1, 2), (2, 1), (2, 2)]
    valid_layouts = [layout for layout in layouts if layout[0] * layout[1] >= num_objects]
    rows, cols = rng.choice(valid_layouts)

    min_h = rows * MIN_CELL_SIZE + (rows - 1) * CELL_GAP
    min_w = cols * MIN_CELL_SIZE + (cols - 1) * CELL_GAP
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    row_sizes = _random_split(grid_h - (rows - 1) * CELL_GAP, rows, MIN_CELL_SIZE, rng)
    col_sizes = _random_split(grid_w - (cols - 1) * CELL_GAP, cols, MIN_CELL_SIZE, rng)

    row_starts = []
    cur = 0
    for size in row_sizes:
        row_starts.append(cur)
        cur += size + CELL_GAP

    col_starts = []
    cur = 0
    for size in col_sizes:
        col_starts.append(cur)
        cur += size + CELL_GAP

    cells = [
        (row_starts[r], col_starts[c], row_sizes[r], col_sizes[c])
        for r in range(rows)
        for c in range(cols)
    ]
    chosen_cells = rng.sample(cells, num_objects)
    directions = rng.sample(list(DIRS.keys()), num_objects)
    colors = rng.sample(SOURCE_COLORS, num_objects)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (top, left, cell_h, cell_w) in enumerate(chosen_cells):
        built = _make_object(cell_h, cell_w, directions[idx], rng)
        if built is None:
            return None
        path_cells, source = built
        for rr, cc in path_cells:
            grid[top + rr, left + cc] = PATH_COLOR
        sr, sc = source
        grid[top + sr, left + sc] = colors[idx]

    try:
        comps, associations, sources = _extract_scene(grid)
    except AssertionError:
        return None
    if len(comps) != num_objects or len(associations) != num_objects or len(sources) != num_objects:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
