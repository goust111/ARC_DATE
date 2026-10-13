from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random

import numpy as np


Coord = tuple[int, int]


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _on_border(cell: Coord, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors4(mask: np.ndarray, cell: Coord) -> list[Coord]:
    r, c = cell
    h, w = mask.shape
    out: list[Coord] = []
    if r > 0 and mask[r - 1, c]:
        out.append((r - 1, c))
    if r + 1 < h and mask[r + 1, c]:
        out.append((r + 1, c))
    if c > 0 and mask[r, c - 1]:
        out.append((r, c - 1))
    if c + 1 < w and mask[r, c + 1]:
        out.append((r, c + 1))
    return out


def _choose_next(prev: Coord | None, curr: Coord, nbrs: list[Coord]) -> Coord | None:
    if prev is None:
        assert len(nbrs) == 1, "path start must be a degree-1 endpoint"
        return nbrs[0]

    others = [cell for cell in nbrs if cell != prev]
    if not others:
        return None

    dr = curr[0] - prev[0]
    dc = curr[1] - prev[1]
    forward = (curr[0] + dr, curr[1] + dc)
    if forward in others:
        return forward
    if len(others) == 1:
        return others[0]
    raise AssertionError("ambiguous continuation at non-crossing junction")


def _trace_path(mask: np.ndarray, neighbor_map: dict[Coord, list[Coord]], start: Coord) -> list[Coord]:
    path = [start]
    seen = {start}
    prev: Coord | None = None
    curr = start
    while True:
        nxt = _choose_next(prev, curr, neighbor_map[curr])
        if nxt is None:
            break
        assert nxt not in seen, "paths must be simple chains"
        path.append(nxt)
        seen.add(nxt)
        prev, curr = curr, nxt
    return path


def _extract_paths(grid: np.ndarray, bg: int) -> tuple[list[list[Coord]], list[int]]:
    mask = grid != bg
    if not np.any(mask):
        return [], []

    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    neighbor_map = {cell: _neighbors4(mask, cell) for cell in occupied}

    starts = sorted(
        cell
        for cell in occupied
        if len(neighbor_map[cell]) == 1 and _on_border(cell, grid.shape)
    )
    assert starts, "expected at least one border-starting path"

    paths: list[list[Coord]] = []
    colors: list[int] = []
    for start in starts:
        coords = _trace_path(mask, neighbor_map, start)
        assert len(coords) >= 2, "path must have at least two pixels"
        end = coords[-1]
        assert len(neighbor_map[end]) == 1, "path must terminate at an endpoint"
        assert not _on_border(end, grid.shape), "path end must be interior"
        paths.append(coords)
        colors.append(int(grid[start]))

    coverage: dict[Coord, int] = defaultdict(int)
    for coords in paths:
        for cell in coords:
            coverage[cell] += 1

    for cell in occupied:
        deg = len(neighbor_map[cell])
        expected = 2 if deg == 4 else 1
        assert deg in (1, 2, 4), "only endpoints, turns/straights, and 4-way crossings are allowed"
        assert coverage[cell] == expected, "every occupied cell must belong to the expected number of paths"

    return paths, colors


def _solve_from_paths(
    shape: tuple[int, int],
    paths: list[list[Coord]],
    original_colors: list[int],
    bg: int,
    tie_grid: np.ndarray | None = None,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    if not paths:
        return out

    members: dict[Coord, list[int]] = defaultdict(list)
    for path_id, coords in enumerate(paths):
        for cell in coords:
            members[cell].append(path_id)

    intersections = {
        cell: ids
        for cell, ids in members.items()
        if len(ids) == 2
    }
    assert all(len(ids) <= 2 for ids in members.values()), "crossings may involve at most two paths"

    ordered_intersections: list[list[Coord]] = []
    intersection_order: list[dict[Coord, int]] = []
    coord_index: list[dict[Coord, int]] = []
    for coords in paths:
        coord_index.append({cell: idx for idx, cell in enumerate(coords)})
        ordered = [cell for cell in coords if cell in intersections]
        ordered_intersections.append(ordered)
        intersection_order.append({cell: idx for idx, cell in enumerate(ordered)})

    visiting: set[tuple[int, int]] = set()

    @lru_cache(maxsize=None)
    def outgoing_color(path_id: int, intersection_idx: int) -> int:
        node = (path_id, intersection_idx)
        assert node not in visiting, "crossing order cannot create a color-dependency cycle"
        visiting.add(node)
        cell = ordered_intersections[path_id][intersection_idx]
        a, b = intersections[cell]
        other = b if a == path_id else a
        other_idx = intersection_order[other][cell]
        if other_idx == 0:
            value = int(original_colors[other])
        else:
            value = outgoing_color(other, other_idx - 1)
        visiting.remove(node)
        return value

    for path_id, coords in enumerate(paths):
        if not ordered_intersections[path_id]:
            for r, c in coords:
                out[r, c] = int(original_colors[path_id])
            continue

        cuts = [coord_index[path_id][cell] for cell in ordered_intersections[path_id]]
        for r, c in coords[: cuts[0]]:
            out[r, c] = int(original_colors[path_id])

        for order_idx, cut in enumerate(cuts):
            color = outgoing_color(path_id, order_idx)
            next_cut = cuts[order_idx + 1] if order_idx + 1 < len(cuts) else len(coords)
            for r, c in coords[cut + 1 : next_cut]:
                out[r, c] = int(color)

    for cell, ids in intersections.items():
        a, b = ids
        len_a = len(paths[a])
        len_b = len(paths[b])
        if len_a > len_b:
            color = int(original_colors[a])
        elif len_b > len_a:
            color = int(original_colors[b])
        elif tie_grid is not None:
            color = int(tie_grid[cell])
        else:
            color = int(original_colors[a])
        out[cell] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored paths from their border endpoints and swap downstream colors at each crossing.

    Input structure:
    - Black background.
    - One-pixel-wide orthogonal paths with one border endpoint and one interior endpoint.
    - Crossings are rendered as a shared pixel whose visible color belongs to one of the two paths.

    Rule:
    - Recover each path by following the non-background skeleton from the border endpoint.
    - At each 4-way crossing, the color carried by each path swaps onto the other path's downstream segment.
    - The crossing pixel itself takes the original color of the longer of the two crossing paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    if not np.any(g != bg):
        return g.copy()

    paths, colors = _extract_paths(g, bg)
    return _solve_from_paths(g.shape, paths, colors, bg, tie_grid=g)


# --------------- generator ---------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
MAX_ATTEMPTS = 500


def _partition_total(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _pick_offset(curr: int, lo: int, hi: int) -> int:
    wide = [v for v in range(lo, hi + 1) if v != curr and abs(v - curr) >= 2]
    if wide:
        return random.choice(wide)
    near = [v for v in range(lo, hi + 1) if v != curr]
    assert near, "secondary axis needs at least two valid positions"
    return random.choice(near)


def _sample_vertical_path(h: int, w: int) -> list[Coord] | None:
    turns = random.choices([0, 1, 2, 3], weights=[1, 4, 3, 1])[0]
    segments = turns + 1
    if h - 2 < segments or w < 4:
        return None

    total = random.randint(max(segments + 1, h // 3), h - 2)
    vertical_steps = _partition_total(total, segments)
    start_top = random.choice([True, False])
    sign = 1 if start_top else -1
    r = 0 if start_top else h - 1
    c = random.randint(1, w - 2)
    columns = [c]
    for _ in range(turns):
        columns.append(_pick_offset(columns[-1], 1, w - 2))

    path = [(r, c)]
    for seg_idx, steps in enumerate(vertical_steps):
        for _ in range(steps):
            r += sign
            path.append((r, c))
        if seg_idx + 1 < len(columns):
            target_c = columns[seg_idx + 1]
            step = 1 if target_c > c else -1
            while c != target_c:
                c += step
                path.append((r, c))
    return path


def _sample_horizontal_path(h: int, w: int) -> list[Coord] | None:
    turns = random.choices([0, 1, 2, 3], weights=[1, 4, 3, 1])[0]
    segments = turns + 1
    if w - 2 < segments or h < 4:
        return None

    total = random.randint(max(segments + 1, w // 3), w - 2)
    horizontal_steps = _partition_total(total, segments)
    start_left = random.choice([True, False])
    sign = 1 if start_left else -1
    c = 0 if start_left else w - 1
    r = random.randint(1, h - 2)
    rows = [r]
    for _ in range(turns):
        rows.append(_pick_offset(rows[-1], 1, h - 2))

    path = [(r, c)]
    for seg_idx, steps in enumerate(horizontal_steps):
        for _ in range(steps):
            c += sign
            path.append((r, c))
        if seg_idx + 1 < len(rows):
            target_r = rows[seg_idx + 1]
            step = 1 if target_r > r else -1
            while r != target_r:
                r += step
                path.append((r, c))
    return path


def _orientation_at(path: list[Coord], idx: int) -> str | None:
    if idx == 0 or idx == len(path) - 1:
        return None
    prev = path[idx - 1]
    curr = path[idx]
    nxt = path[idx + 1]
    if prev[0] == curr[0] == nxt[0]:
        return "h"
    if prev[1] == curr[1] == nxt[1]:
        return "v"
    return None


def _validate_paths(paths: list[list[Coord]], shape: tuple[int, int]) -> dict[str, object] | None:
    h, w = shape
    if len(set(map(len, paths))) != len(paths):
        return None

    members: dict[Coord, list[int]] = defaultdict(list)
    path_indices: list[dict[Coord, int]] = []
    for path_id, path in enumerate(paths):
        if len(path) < 4 or len(set(path)) != len(path):
            return None
        if not _on_border(path[0], shape) or _on_border(path[-1], shape):
            return None
        if not all(0 <= r < h and 0 <= c < w for r, c in path):
            return None
        path_indices.append({cell: idx for idx, cell in enumerate(path)})
        for cell in path:
            members[cell].append(path_id)

    if any(len(ids) > 2 for ids in members.values()):
        return None

    intersections = [cell for cell, ids in members.items() if len(ids) == 2]
    for cell in intersections:
        a, b = members[cell]
        ia = path_indices[a][cell]
        ib = path_indices[b][cell]
        oa = _orientation_at(paths[a], ia)
        ob = _orientation_at(paths[b], ib)
        if oa is None or ob is None or oa == ob or _on_border(cell, shape):
            return None

    for i, cell_a in enumerate(intersections):
        for cell_b in intersections[i + 1 :]:
            if max(abs(cell_a[0] - cell_b[0]), abs(cell_a[1] - cell_b[1])) <= 1:
                return None

    mask = np.zeros(shape, dtype=bool)
    for r, c in members:
        mask[r, c] = True

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r + 1 < h and mask[r + 1, c]:
                if not (set(members[(r, c)]) & set(members[(r + 1, c)])):
                    return None
            if c + 1 < w and mask[r, c + 1]:
                if not (set(members[(r, c)]) & set(members[(r, c + 1)])):
                    return None

    degree_map = {cell: len(_neighbors4(mask, cell)) for cell in members}
    if any(deg not in (1, 2, 4) for deg in degree_map.values()):
        return None

    border_endpoints = [cell for cell, deg in degree_map.items() if deg == 1 and _on_border(cell, shape)]
    inner_endpoints = [cell for cell, deg in degree_map.items() if deg == 1 and not _on_border(cell, shape)]
    if len(border_endpoints) != len(paths) or len(inner_endpoints) != len(paths):
        return None

    if np.count_nonzero(mask) > (h * w) // 3:
        return None

    per_path_intersections = [
        sum(1 for cell in path if len(members[cell]) == 2)
        for path in paths
    ]
    return {
        "members": members,
        "intersections": intersections,
        "intersection_count": len(intersections),
        "per_path_intersections": per_path_intersections,
    }


def _path_kinds(count: int) -> list[str]:
    if count == 2:
        kinds = ["vertical", "horizontal"]
    elif count == 3:
        kinds = ["vertical", "horizontal", random.choice(["vertical", "horizontal"])]
    else:
        kinds = ["vertical", "vertical", "horizontal", "horizontal"]
    random.shuffle(kinds)
    return kinds


def _mode_requirements(mode: str, info: dict[str, object]) -> bool:
    total = int(info["intersection_count"])
    per_path = list(info["per_path_intersections"])
    if total == 0:
        return False
    if mode == "single":
        return 1 <= total <= 2
    if mode == "cascade":
        return total >= 2 and max(per_path) >= 2
    return any(v == 0 for v in per_path) and any(v > 0 for v in per_path)


def _render_input(shape: tuple[int, int], paths: list[list[Coord]], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    members: dict[Coord, list[int]] = defaultdict(list)
    for path_id, path in enumerate(paths):
        for cell in path:
            members[cell].append(path_id)

    for path_id, path in enumerate(paths):
        for r, c in path:
            if len(members[(r, c)]) == 1:
                grid[r, c] = int(colors[path_id])

    for cell, ids in members.items():
        if len(ids) == 2:
            grid[cell] = int(colors[random.choice(ids)])

    return grid


def _sample_paths(shape: tuple[int, int], mode: str) -> list[list[Coord]] | None:
    h, w = shape
    if mode == "single":
        count = random.choice([2, 3])
    elif mode == "cascade":
        count = random.choice([3, 4])
    else:
        count = random.choice([3, 4])

    paths: list[list[Coord]] = []
    for kind in _path_kinds(count):
        path = _sample_vertical_path(h, w) if kind == "vertical" else _sample_horizontal_path(h, w)
        if path is None:
            return None
        paths.append(path)
    return paths


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(["single", "cascade", "mixed"])
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    paths = _sample_paths(shape, mode)
    if paths is None:
        return None

    info = _validate_paths(paths, shape)
    if info is None or not _mode_requirements(mode, info):
        return None

    colors = random.sample(PALETTE, len(paths))
    inp = _render_input(shape, paths, colors)
    try:
        expected = _solve_from_paths(shape, paths, colors, BG, tie_grid=inp)
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
