from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
POINTER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
POINTER_SIZES = (5, 7)
GRID_SIDE_RANGE = (20, 29)
HIT_POINTER_RANGE = (2, 3)
MISS_POINTER_PROB = 0.5
UNHIT_TARGET_RANGE = (1, 2)
GENERATION_RETRIES = 400
PLACEMENT_TRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _components_by_color(grid: np.ndarray, colors: list[int]) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(int(c) for c in colors)):
        for coords in _components_mask(grid == color):
            bbox = _bbox_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "coord_set": {tuple(map(int, rc)) for rc in coords.tolist()},
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _defect_and_direction(comp: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    cells = comp["coord_set"]
    defect: tuple[int, int] | None = None
    for r, c in cells:
        rr = r0 + r1 - r
        cc = c0 + c1 - c
        if (rr, cc) not in cells:
            if defect is not None:
                raise AssertionError("pointer tile must have exactly one defect pixel")
            defect = (r, c)
    if defect is None:
        raise AssertionError("pointer tile defect not found")

    dr2 = 2 * defect[0] - (r0 + r1)
    dc2 = 2 * defect[1] - (c0 + c1)
    if dr2 == 0 and dc2 == 0:
        raise AssertionError("defect cannot be at the center")
    if dr2 != 0 and dc2 != 0 and abs(dr2) != abs(dc2):
        raise AssertionError("defect must encode one of 8 compass directions")

    dr = 0 if dr2 == 0 else (1 if dr2 > 0 else -1)
    dc = 0 if dc2 == 0 else (1 if dc2 > 0 else -1)
    return defect, (dr, dc)


def _trace_first_target(
    grid: np.ndarray,
    cell_to_target: dict[tuple[int, int], int],
    defect: tuple[int, int],
    direction: tuple[int, int],
) -> int | None:
    h, w = grid.shape
    dr, dc = direction
    r = defect[0] + dr
    c = defect[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == GREY:
            return cell_to_target[(r, c)]
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the first grey target hit by each defect-encoded symmetric pointer and erase all unclaimed grey targets.

    Input structure:
    - The background is black.
    - Several non-grey monochrome pointer tiles are each almost 180-degree symmetric about their own bbox center, except for one extra defect pixel.
    - Several grey 4-connected target objects are separated from the pointer tiles.

    Rule:
    - For every pointer tile, find the unique foreground pixel whose 180-degree rotated partner is absent.
    - Use the direction from the tile center to that defect pixel as an 8-direction ray, and follow the ray outward until it meets the first grey target.
    - Recolor that whole hit target to the pointer tile's color; grey targets hit by no ray are erased to background; the pointer tiles themselves stay unchanged.

    Key insight:
    - The tile shape is not an arrow by outline: the lone symmetry-breaking pixel is the only direction cue, and target survival depends entirely on whether some ray claims it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)))

    targets = _components_by_color(g, [GREY])
    cell_to_target: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for cell in target["coord_set"]:
            cell_to_target[cell] = idx

    pointer_colors = [color for color in colors if color not in (bg, GREY)]
    pointers = _components_by_color(g, pointer_colors)

    claims: dict[int, int] = {}
    for pointer in pointers:
        defect, direction = _defect_and_direction(pointer)
        hit = _trace_first_target(g, cell_to_target, defect, direction)
        if hit is None:
            continue
        color = int(pointer["color"])
        prev = claims.get(hit)
        if prev is not None and prev != color:
            raise AssertionError("a grey target was claimed by multiple pointer colors")
        claims[hit] = color

    out = g.copy()
    for idx, target in enumerate(targets):
        fill = claims.get(idx, bg)
        coords = target["coords"]
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _pointer_anchor(size: int, direction: tuple[int, int]) -> tuple[int, int]:
    mid = size // 2
    mapping = {
        (-1, 0): (0, mid),
        (-1, 1): (0, size - 1),
        (0, 1): (mid, size - 1),
        (1, 1): (size - 1, size - 1),
        (1, 0): (size - 1, mid),
        (1, -1): (size - 1, 0),
        (0, -1): (mid, 0),
        (-1, -1): (0, 0),
    }
    return mapping[direction]


def _make_pointer_tile(size: int, color: int, direction: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    tile = np.zeros((size, size), dtype=int)
    tile[0, :] = color
    tile[-1, :] = color
    tile[:, 0] = color
    tile[:, -1] = color

    defect = _pointer_anchor(size, direction)
    missing = (size - 1 - defect[0], size - 1 - defect[1])
    tile[missing] = BG
    return tile, defect


def _normalize_shape(
    cells: set[tuple[int, int]], anchor: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_cells = sorted((r - min_r, c - min_c) for r, c in cells)
    norm_anchor = (anchor[0] - min_r, anchor[1] - min_c)
    return norm_cells, norm_anchor


def _transform_shape(
    cells: list[tuple[int, int]], anchor: tuple[int, int], rotate_k: int, flip: bool
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cur_cells = [(int(r), int(c)) for r, c in cells]
    cur_anchor = (int(anchor[0]), int(anchor[1]))

    for _ in range(int(rotate_k) % 4):
        cur_cells = [(-c, r) for r, c in cur_cells]
        cur_anchor = (-cur_anchor[1], cur_anchor[0])
    if flip:
        cur_cells = [(r, -c) for r, c in cur_cells]
        cur_anchor = (cur_anchor[0], -cur_anchor[1])
    return _normalize_shape(set(cur_cells), cur_anchor)


def _random_target_shape() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    area = random.randint(3, 6)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if abs(nr) > 2 or abs(nc) > 2:
                    continue
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_target_shape()
        cells.add(random.choice(frontier))
    base_cells = sorted(cells)
    return _transform_shape(base_cells, (0, 0), random.randrange(4), random.random() < 0.5)


def _in_bounds_all(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _padded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                padded.add((rr, cc))
    return padded


def _can_place(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    return _in_bounds_all(cells, shape) and not any(cell in blocked for cell in cells)


def _ray_cells_from(defect: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r = defect[0] + dr
    c = defect[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _path_reaches_target_first(
    occupied: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
    defect: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    for cell in _ray_cells_from(defect, direction, shape):
        if cell in target_cells:
            return True
        if cell in occupied:
            return False
    return False


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _build_expected(grid: np.ndarray, targets: list[dict], claims: dict[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for idx, target in enumerate(targets):
        fill = claims.get(idx, BG)
        for r, c in target["cells"]:
            out[r, c] = fill
    return out


def _validate_scene(grid: np.ndarray, pointers: list[dict], targets: list[dict]) -> bool:
    actual_claims: dict[int, int] = {}
    cell_to_target: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for cell in target["cells"]:
            cell_to_target[cell] = idx

    for pointer in pointers:
        hit = _trace_first_target(grid, cell_to_target, pointer["defect"], pointer["direction"])
        if hit != pointer["target_idx"]:
            return False
        if hit is not None:
            if hit in actual_claims:
                return False
            actual_claims[hit] = int(pointer["color"])

    if not actual_claims or len(actual_claims) == len(targets):
        return False
    expected = _build_expected(grid, targets, actual_claims)
    return np.array_equal(main(grid), expected)


def _try_place_hit_pair(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    targets: list[dict],
    pointers: list[dict],
) -> bool:
    h, w = grid.shape
    size = random.choice(POINTER_SIZES)
    tile, defect_local = _make_pointer_tile(size, color, direction)
    tile_cells_local = {(int(r), int(c)) for r, c in np.argwhere(tile == color)}
    shape_cells_local, anchor_local = _random_target_shape()

    for _ in range(PLACEMENT_TRIES):
        anchor = (random.randrange(h), random.randrange(w))
        gap = random.randint(3, 7)
        defect = (anchor[0] - direction[0] * gap, anchor[1] - direction[1] * gap)
        tile_origin = (defect[0] - defect_local[0], defect[1] - defect_local[1])
        pointer_cells = {(tile_origin[0] + r, tile_origin[1] + c) for r, c in tile_cells_local}
        target_origin = (anchor[0] - anchor_local[0], anchor[1] - anchor_local[1])
        target_cells = {(target_origin[0] + r, target_origin[1] + c) for r, c in shape_cells_local}

        if pointer_cells & ray_reserved or target_cells & ray_reserved:
            continue
        if not _can_place(pointer_cells, blocked, grid.shape):
            continue
        if not _can_place(target_cells, blocked, grid.shape):
            continue
        if target_cells & _padded_cells(pointer_cells, grid.shape):
            continue
        if not _path_reaches_target_first(occupied, target_cells, defect, direction, grid.shape):
            continue

        _paint_cells(grid, pointer_cells, color)
        _paint_cells(grid, target_cells, GREY)
        occupied.update(pointer_cells)
        occupied.update(target_cells)
        blocked.update(_padded_cells(pointer_cells, grid.shape))
        blocked.update(_padded_cells(target_cells, grid.shape))

        target_idx = len(targets)
        targets.append({"cells": sorted(target_cells)})
        pointers.append(
            {
                "color": int(color),
                "defect": defect,
                "direction": direction,
                "target_idx": target_idx,
            }
        )
        ray_reserved.update(_ray_cells_from(defect, direction, grid.shape))
        return True
    return False


def _try_place_miss_pointer(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    pointers: list[dict],
) -> bool:
    h, w = grid.shape
    size = random.choice(POINTER_SIZES)
    tile, defect_local = _make_pointer_tile(size, color, direction)
    tile_cells_local = {(int(r), int(c)) for r, c in np.argwhere(tile == color)}

    for _ in range(PLACEMENT_TRIES):
        tile_origin = (random.randrange(h - size + 1), random.randrange(w - size + 1))
        pointer_cells = {(tile_origin[0] + r, tile_origin[1] + c) for r, c in tile_cells_local}
        if pointer_cells & ray_reserved:
            continue
        if not _can_place(pointer_cells, blocked, grid.shape):
            continue

        defect = (tile_origin[0] + defect_local[0], tile_origin[1] + defect_local[1])
        ray_cells = _ray_cells_from(defect, direction, grid.shape)
        if len(ray_cells) < 4:
            continue
        if any(cell in occupied for cell in ray_cells):
            continue

        _paint_cells(grid, pointer_cells, color)
        occupied.update(pointer_cells)
        blocked.update(_padded_cells(pointer_cells, grid.shape))
        ray_reserved.update(ray_cells)
        pointers.append(
            {
                "color": int(color),
                "defect": defect,
                "direction": direction,
                "target_idx": None,
            }
        )
        return True
    return False


def _try_place_unhit_target(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    targets: list[dict],
) -> bool:
    h, w = grid.shape
    shape_cells_local, anchor_local = _random_target_shape()
    max_r = max(r for r, _ in shape_cells_local)
    max_c = max(c for _, c in shape_cells_local)

    for _ in range(PLACEMENT_TRIES):
        origin = (random.randrange(h - max_r), random.randrange(w - max_c))
        target_cells = {(origin[0] + r, origin[1] + c) for r, c in shape_cells_local}
        if target_cells & ray_reserved:
            continue
        if not _can_place(target_cells, blocked, grid.shape):
            continue

        _paint_cells(grid, target_cells, GREY)
        occupied.update(target_cells)
        blocked.update(_padded_cells(target_cells, grid.shape))
        targets.append({"cells": sorted(target_cells)})
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    ray_reserved: set[tuple[int, int]] = set()
    pointers: list[dict] = []
    targets: list[dict] = []

    hit_count = random.randint(*HIT_POINTER_RANGE)
    miss_pointer_count = 1 if random.random() < MISS_POINTER_PROB else 0
    unhit_target_count = random.randint(*UNHIT_TARGET_RANGE)
    total_pointers = hit_count + miss_pointer_count
    if total_pointers > len(POINTER_COLORS):
        return None

    colors = random.sample(POINTER_COLORS, total_pointers)
    directions = random.sample(DIRECTIONS, total_pointers)

    for idx in range(hit_count):
        if not _try_place_hit_pair(
            grid,
            occupied,
            blocked,
            ray_reserved,
            colors[idx],
            directions[idx],
            targets,
            pointers,
        ):
            return None

    for idx in range(hit_count, total_pointers):
        if not _try_place_miss_pointer(
            grid,
            occupied,
            blocked,
            ray_reserved,
            colors[idx],
            directions[idx],
            pointers,
        ):
            return None

    for _ in range(unhit_target_count):
        if not _try_place_unhit_target(grid, occupied, blocked, ray_reserved, targets):
            return None

    if len(targets) < 2:
        return None
    if not _validate_scene(grid, pointers, targets):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid defect-pointer ARC input")
