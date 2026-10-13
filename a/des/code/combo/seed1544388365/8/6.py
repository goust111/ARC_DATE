from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
MAX_GEN_ATTEMPTS = 400
LOCAL_OBJECT_ATTEMPTS = 160

RING_OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)
CARDINAL_OFFSETS = {(-1, 0), (1, 0), (0, -1), (0, 1)}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _path_endpoints(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    degrees = {
        cell: sum((nb in cell_set) for nb in _neighbors4(*cell))
        for cell in cell_set
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "colored component must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "colored component must be a simple path"
    endpoints.sort()
    return endpoints


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2

    grey_components = _components(g == GREY, connectivity=8)
    assert grey_components, "grid must contain grey endpoint patterns"
    grey_owner: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(grey_components):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        assert max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2
        for cell in comp:
            grey_owner[cell] = idx

    path_components = _components((g != BG) & (g != GREY), connectivity=4)
    assert path_components, "grid must contain colored paths"

    records: list[dict[str, object]] = []
    used_grey: list[int] = []

    for comp in path_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each path component must be monochromatic"
        color = next(iter(colors))
        endpoints = _path_endpoints(comp)
        shapes: list[tuple[tuple[int, int], ...]] = []
        grey_ids: list[int] = []

        for er, ec in endpoints:
            adj = {
                grey_owner[(nr, nc)]
                for nr, nc in _neighbors4(er, ec)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in grey_owner
            }
            assert len(adj) == 1, "each path endpoint must touch exactly one grey pattern"
            grey_id = next(iter(adj))
            grey_ids.append(grey_id)
            shape = tuple(sorted((r - er, c - ec) for r, c in grey_components[grey_id]))
            assert shape
            shapes.append(shape)

        assert grey_ids[0] != grey_ids[1], "a path must connect two distinct patterns"
        used_grey.extend(grey_ids)
        records.append(
            {
                "color": color,
                "endpoints": endpoints,
                "shapes": shapes,
            }
        )

    assert len(used_grey) == len(grey_components), "every grey pattern must belong to one endpoint"
    assert len(set(used_grey)) == len(grey_components), "grey patterns must not be shared"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap grey endpoint patterns across each colored path, recoloring them with the path color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == GREY] = BG
    h, w = g.shape

    for record in _analyze_grid(g):
        color = int(record["color"])
        endpoints = record["endpoints"]
        shapes = record["shapes"]
        (a_r, a_c), (b_r, b_c) = endpoints
        shape_a, shape_b = shapes

        for dr, dc in shape_b:
            rr, cc = a_r + dr, a_c + dc
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = color
        for dr, dc in shape_a:
            rr, cc = b_r + dr, b_c + dc
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = color

    return out


def _shape_connected(shape: tuple[tuple[int, int], ...]) -> bool:
    cells = set(shape)
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _build_shape_library() -> list[tuple[tuple[int, int], ...]]:
    shapes: list[tuple[tuple[int, int], ...]] = []
    for size in range(1, 6):
        for combo in itertools.combinations(RING_OFFSETS, size):
            if not any(cell in CARDINAL_OFFSETS for cell in combo):
                continue
            shape = tuple(sorted(combo))
            if _shape_connected(shape):
                shapes.append(shape)
    shapes.sort(key=lambda s: (len(s), s))
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _canonical_path() -> list[tuple[int, int]]:
    kind = random.choice(("straight", "elbow", "zigzag"))
    if kind == "straight":
        length = random.randint(4, 8)
        return [(0, c) for c in range(length)]
    if kind == "elbow":
        width = random.randint(3, 6)
        height = random.randint(3, 6)
        return [(0, c) for c in range(width)] + [(r, width - 1) for r in range(1, height)]

    left = random.randint(2, 4)
    drop = random.randint(2, 4)
    right = random.randint(2, 4)
    return (
        [(0, c) for c in range(left)]
        + [(r, left - 1) for r in range(1, drop + 1)]
        + [(drop, c) for c in range(left, left + right)]
    )


def _transform_point(r: int, c: int, rot: int, flip: bool) -> tuple[int, int]:
    if flip:
        c = -c
    for _ in range(rot):
        r, c = c, -r
    return r, c


def _random_transform(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rot = random.randrange(4)
    flip = random.random() < 0.5
    transformed = [_transform_point(r, c, rot, flip) for r, c in coords]
    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    return sorted((r - min_r, c - min_c) for r, c in transformed)


def _valid_pattern_offset(
    endpoint: tuple[int, int],
    off: tuple[int, int],
    path_cells: set[tuple[int, int]],
) -> bool:
    cell = (endpoint[0] + off[0], endpoint[1] + off[1])
    if cell in path_cells:
        return False
    for nb in _neighbors4(*cell):
        if nb in path_cells and nb != endpoint:
            return False
    return True


def _place_shape(
    endpoint: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
) -> set[tuple[int, int]]:
    return {(endpoint[0] + dr, endpoint[1] + dc) for dr, dc in shape}


def _separated8(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return False
    return True


def _make_local_object() -> dict[str, object]:
    for _ in range(LOCAL_OBJECT_ATTEMPTS):
        path = _random_transform(_canonical_path())
        path_set = set(path)
        endpoints = _path_endpoints(path)

        valid_shapes = [
            shape
            for shape in SHAPE_LIBRARY
            if all(_valid_pattern_offset(endpoints[0], off, path_set) for off in shape)
            and all(_valid_pattern_offset(endpoints[1], off, path_set) for off in shape)
        ]
        if len(valid_shapes) < 2:
            continue

        for _ in range(40):
            shape_a = random.choice(valid_shapes)
            if random.random() < 0.75:
                diff_pool = [shape for shape in valid_shapes if shape != shape_a]
                shape_b = random.choice(diff_pool or valid_shapes)
            else:
                shape_b = random.choice(valid_shapes)

            a_now = _place_shape(endpoints[0], shape_a)
            b_now = _place_shape(endpoints[1], shape_b)
            a_swap = _place_shape(endpoints[0], shape_b)
            b_swap = _place_shape(endpoints[1], shape_a)
            if a_now & b_now:
                continue
            if a_swap & b_swap:
                continue
            if not _separated8(a_now, b_now):
                continue
            if not _separated8(a_swap, b_swap):
                continue

            reserve = path_set | a_now | b_now | a_swap | b_swap
            return {
                "path": path,
                "shape_a": shape_a,
                "shape_b": shape_b,
                "reserve": reserve,
            }

    raise RuntimeError("failed to build a valid local object")


def _expand_chebyshev(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_object(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    color: int,
    obj: dict[str, object],
) -> bool:
    h, w = grid.shape
    path = obj["path"]
    shape_a = obj["shape_a"]
    shape_b = obj["shape_b"]
    reserve = obj["reserve"]

    reserve_box = _bbox(reserve)
    min_r, min_c, max_r, max_c = reserve_box
    row_offsets = list(range(-min_r, h - max_r))
    col_offsets = list(range(-min_c, w - max_c))
    offsets = [(dr, dc) for dr in row_offsets for dc in col_offsets]
    random.shuffle(offsets)

    endpoints = _path_endpoints(path)
    for off_r, off_c in offsets:
        shifted_reserve = {(r + off_r, c + off_c) for r, c in reserve}
        halo = _expand_chebyshev(shifted_reserve, h, w)
        if halo & blocked:
            continue

        shifted_path = {(r + off_r, c + off_c) for r, c in path}
        a_cells = {
            (endpoints[0][0] + off_r + dr, endpoints[0][1] + off_c + dc)
            for dr, dc in shape_a
        }
        b_cells = {
            (endpoints[1][0] + off_r + dr, endpoints[1][1] + off_c + dc)
            for dr, dc in shape_b
        }

        for r, c in shifted_path:
            grid[r, c] = color
        for r, c in a_cells | b_cells:
            grid[r, c] = GREY
        blocked.update(halo)
        return True

    return False


def _target_path_count(h: int, w: int) -> int:
    area = h * w
    if area < 420:
        return random.choice((1, 2, 2))
    if area < 560:
        return random.choice((2, 2, 3))
    return random.choice((2, 3, 3, 4))


def _validate_instance(grid: np.ndarray) -> None:
    records = _analyze_grid(grid)
    colors = [int(record["color"]) for record in records]
    assert len(colors) == len(set(colors))
    assert all(int(np.count_nonzero(grid == color)) >= 2 for color in colors)
    for record in records:
        for shape in record["shapes"]:
            rows = [dr for dr, _ in shape]
            cols = [dc for _, dc in shape]
            assert max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == GREY)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        path_count = _target_path_count(h, w)
        colors = random.sample(PATH_COLORS, path_count)

        grid = np.full((h, w), BG, dtype=int)
        blocked: set[tuple[int, int]] = set()
        ok = True

        for color in colors:
            placed = False
            for _ in range(50):
                obj = _make_local_object()
                if _paint_object(grid, blocked, color, obj):
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        _validate_instance(grid)
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
