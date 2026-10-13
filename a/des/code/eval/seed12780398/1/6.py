from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
KEY_COLOR = 4
CONTAINER_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GENERATE_MAX_RETRIES = 400
PLACEMENT_TRIES = 250

STRAIGHT_SHAPES = [
    {"name": "h2", "branch": "straight", "norm": ((0, 0), (0, 1))},
    {"name": "v2", "branch": "straight", "norm": ((0, 0), (1, 0))},
    {"name": "h3", "branch": "straight", "norm": ((0, 0), (0, 1), (0, 2))},
    {"name": "v3", "branch": "straight", "norm": ((0, 0), (1, 0), (2, 0))},
]

CORNER_SHAPES = [
    {"name": "L3_tl", "branch": "corner", "norm": ((0, 0), (0, 1), (1, 0))},
    {"name": "L3_tr", "branch": "corner", "norm": ((0, 0), (0, 1), (1, 1))},
    {"name": "L3_bl", "branch": "corner", "norm": ((0, 0), (1, 0), (1, 1))},
    {"name": "L3_br", "branch": "corner", "norm": ((0, 1), (1, 0), (1, 1))},
    {"name": "L4_tl_flat", "branch": "corner", "norm": ((0, 0), (0, 1), (0, 2), (1, 0))},
    {"name": "L4_tl_tall", "branch": "corner", "norm": ((0, 0), (0, 1), (1, 0), (2, 0))},
    {"name": "L4_tr_flat", "branch": "corner", "norm": ((0, 0), (0, 1), (0, 2), (1, 2))},
    {"name": "L4_tr_tall", "branch": "corner", "norm": ((0, 0), (0, 1), (1, 1), (2, 1))},
    {"name": "L4_bl_flat", "branch": "corner", "norm": ((0, 0), (1, 0), (1, 1), (1, 2))},
    {"name": "L4_bl_tall", "branch": "corner", "norm": ((0, 0), (1, 0), (2, 0), (2, 1))},
    {"name": "L4_br_flat", "branch": "corner", "norm": ((0, 2), (1, 0), (1, 1), (1, 2))},
    {"name": "L4_br_tall", "branch": "corner", "norm": ((0, 1), (1, 1), (2, 0), (2, 1))},
]

SHAPE_SPECS = {spec["name"]: spec for spec in STRAIGHT_SHAPES + CORNER_SHAPES}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _normalize_shape(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr[:, 0].min(), arr[:, 1].min()
    rel = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(rel)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_to_cells(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + dr), int(c0 + dc)) for dr, dc in shape]


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((int(r0), int(c)))
        if r1 != r0:
            cells.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.append((int(r), int(c0)))
        if c1 != c0:
            cells.append((int(r), int(c1)))
    return cells


def _is_connected(cells: list[tuple[int, int]]) -> bool:
    target = set((int(r), int(c)) for r, c in cells)
    start = next(iter(target))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in target and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(target)


def _extract_scene(grid: np.ndarray) -> tuple[dict, dict]:
    g = np.asarray(grid, dtype=int)
    key_comps = _components(g == KEY_COLOR)
    if not key_comps:
        raise ValueError("expected at least one key")

    keys_by_shape: dict[tuple[tuple[int, int], ...], list[np.ndarray]] = defaultdict(list)
    for comp in key_comps:
        area = int(comp.shape[0])
        if not 2 <= area <= 4:
            raise ValueError("key area must be 2..4")
        keys_by_shape[_normalize_shape(comp)].append(comp)

    containers_by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, KEY_COLOR)]
    for color in colors:
        comps = _components(g == color)
        if len(comps) != 1:
            raise ValueError("each container color must form one component")
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            raise ValueError("container bbox too small")
        full_perimeter = set(_perimeter_cells(r0, c0, r1, c1))
        comp_set = set((int(r), int(c)) for r, c in comp)
        if not comp_set <= full_perimeter:
            raise ValueError("container pixels must lie on the rectangle perimeter")
        gap = sorted(full_perimeter - comp_set)
        if not 2 <= len(gap) <= 4:
            raise ValueError("gap area must be 2..4")
        if not _is_connected(gap):
            raise ValueError("gap must be 4-connected")
        containers_by_shape[_normalize_shape(gap)].append(
            {"color": color, "gap": np.array(gap, dtype=int)}
        )

    if set(keys_by_shape) != set(containers_by_shape):
        raise ValueError("key shapes and gap shapes do not match")
    for shape in keys_by_shape:
        if len(keys_by_shape[shape]) != len(containers_by_shape[shape]):
            raise ValueError("shape multiplicities do not match")
    return keys_by_shape, containers_by_shape


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    keys_by_shape, containers_by_shape = _extract_scene(g)
    out = g.copy()
    out[out == KEY_COLOR] = BG
    for shape in keys_by_shape:
        for container in containers_by_shape[shape]:
            gap = container["gap"]
            out[gap[:, 0], gap[:, 1]] = int(container["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each yellow key to the container whose missing perimeter cells have the same shape.
    Remove the key from its source position, fill the matching gap, and recolor the filled cells
    to the container's border color.
    """
    return _solve(np.asarray(grid, dtype=int))


def _grid_size_for_count(n_objects: int) -> tuple[int, int]:
    if n_objects == 2:
        return random.randint(18, 23), random.randint(18, 23)
    if n_objects == 3:
        return random.randint(21, 27), random.randint(21, 27)
    return random.randint(24, 30), random.randint(24, 30)


def _choose_shape_names(n_objects: int) -> list[str]:
    if n_objects == 2:
        n_corner = 1
    else:
        n_corner = random.choice([n_objects // 2, (n_objects + 1) // 2])
        n_corner = max(1, min(n_objects - 1, n_corner))
    n_straight = n_objects - n_corner
    chosen = random.sample([spec["name"] for spec in STRAIGHT_SHAPES], n_straight)
    chosen += random.sample([spec["name"] for spec in CORNER_SHAPES], n_corner)
    random.shuffle(chosen)
    return chosen


def _instantiate_gap(shape_name: str, r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    if shape_name == "h2":
        row = r0 if random.random() < 0.5 else r0 + h - 1
        start = random.randint(0, w - 2)
        return [(row, c0 + start), (row, c0 + start + 1)]
    if shape_name == "h3":
        row = r0 if random.random() < 0.5 else r0 + h - 1
        start = random.randint(0, w - 3)
        return [(row, c0 + start + i) for i in range(3)]
    if shape_name == "v2":
        col = c0 if random.random() < 0.5 else c0 + w - 1
        start = random.randint(0, h - 2)
        return [(r0 + start, col), (r0 + start + 1, col)]
    if shape_name == "v3":
        col = c0 if random.random() < 0.5 else c0 + w - 1
        start = random.randint(0, h - 3)
        return [(r0 + start + i, col) for i in range(3)]
    if shape_name == "L3_tl":
        return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0)]
    if shape_name == "L3_tr":
        return [(r0, c0 + w - 2), (r0, c0 + w - 1), (r0 + 1, c0 + w - 1)]
    if shape_name == "L3_bl":
        return [(r0 + h - 2, c0), (r0 + h - 1, c0), (r0 + h - 1, c0 + 1)]
    if shape_name == "L3_br":
        return [(r0 + h - 2, c0 + w - 1), (r0 + h - 1, c0 + w - 2), (r0 + h - 1, c0 + w - 1)]
    if shape_name == "L4_tl_flat":
        return [(r0, c0), (r0, c0 + 1), (r0, c0 + 2), (r0 + 1, c0)]
    if shape_name == "L4_tl_tall":
        return [(r0, c0), (r0 + 1, c0), (r0 + 2, c0), (r0, c0 + 1)]
    if shape_name == "L4_tr_flat":
        return [(r0, c0 + w - 3), (r0, c0 + w - 2), (r0, c0 + w - 1), (r0 + 1, c0 + w - 1)]
    if shape_name == "L4_tr_tall":
        return [(r0, c0 + w - 2), (r0, c0 + w - 1), (r0 + 1, c0 + w - 1), (r0 + 2, c0 + w - 1)]
    if shape_name == "L4_bl_flat":
        return [(r0 + h - 2, c0), (r0 + h - 1, c0), (r0 + h - 1, c0 + 1), (r0 + h - 1, c0 + 2)]
    if shape_name == "L4_bl_tall":
        return [(r0 + h - 3, c0), (r0 + h - 2, c0), (r0 + h - 1, c0), (r0 + h - 1, c0 + 1)]
    if shape_name == "L4_br_flat":
        return [(r0 + h - 2, c0 + w - 1), (r0 + h - 1, c0 + w - 3), (r0 + h - 1, c0 + w - 2), (r0 + h - 1, c0 + w - 1)]
    if shape_name == "L4_br_tall":
        return [(r0 + h - 3, c0 + w - 1), (r0 + h - 2, c0 + w - 1), (r0 + h - 1, c0 + w - 2), (r0 + h - 1, c0 + w - 1)]
    raise ValueError(f"unknown shape {shape_name}")


def _bbox_is_free(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0] - 1, r1 + margin)
    cc1 = min(blocked.shape[1] - 1, c1 + margin)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_bbox(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0] - 1, r1 + margin)
    cc1 = min(blocked.shape[1] - 1, c1 + margin)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_container(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    shape_name: str,
) -> dict | None:
    h_lim, w_lim = grid.shape
    for _ in range(PLACEMENT_TRIES):
        h = random.randint(5, 9)
        w = random.randint(5, 9)
        if h_lim - h - 2 < 1 or w_lim - w - 2 < 1:
            return None
        r0 = random.randint(1, h_lim - h - 2)
        c0 = random.randint(1, w_lim - w - 2)
        r1, c1 = r0 + h - 1, c0 + w - 1
        if not _bbox_is_free(blocked, r0, c0, r1, c1, margin=1):
            continue
        gap = _instantiate_gap(shape_name, r0, c0, h, w)
        gap_set = set(gap)
        perim = _perimeter_cells(r0, c0, r1, c1)
        border = [cell for cell in perim if cell not in gap_set]
        for r, c in border:
            grid[r, c] = int(color)
        _reserve_bbox(blocked, r0, c0, r1, c1, margin=1)
        return {
            "color": int(color),
            "gap": np.array(gap, dtype=int),
            "shape": SHAPE_SPECS[shape_name]["norm"],
        }
    return None


def _place_key(grid: np.ndarray, blocked: np.ndarray, shape: tuple[tuple[int, int], ...]) -> np.ndarray | None:
    h_lim, w_lim = grid.shape
    kh, kw = _shape_size(shape)
    for _ in range(PLACEMENT_TRIES):
        if h_lim - kh - 2 < 1 or w_lim - kw - 2 < 1:
            return None
        r0 = random.randint(1, h_lim - kh - 2)
        c0 = random.randint(1, w_lim - kw - 2)
        r1, c1 = r0 + kh - 1, c0 + kw - 1
        if not _bbox_is_free(blocked, r0, c0, r1, c1, margin=1):
            continue
        coords = np.array(_shape_to_cells(shape, r0, c0), dtype=int)
        grid[coords[:, 0], coords[:, 1]] = KEY_COLOR
        _reserve_bbox(blocked, r0, c0, r1, c1, margin=1)
        return coords
    return None


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    h, w = _grid_size_for_count(n_objects)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = random.sample(CONTAINER_COLORS, n_objects)
    shape_names = _choose_shape_names(n_objects)

    containers: list[dict] = []
    for color, shape_name in zip(colors, shape_names):
        placed = _place_container(grid, blocked, color, shape_name)
        if placed is None:
            return None
        containers.append(placed)

    random.shuffle(containers)
    key_cells: list[np.ndarray] = []
    for container in containers:
        coords = _place_key(grid, blocked, container["shape"])
        if coords is None:
            return None
        key_cells.append(coords)

    expected = grid.copy()
    for coords in key_cells:
        expected[coords[:, 0], coords[:, 1]] = BG
    for container in containers:
        gap = container["gap"]
        expected[gap[:, 0], gap[:, 1]] = int(container["color"])

    try:
        parsed_keys, parsed_containers = _extract_scene(grid)
    except ValueError:
        return None
    if sum(len(v) for v in parsed_keys.values()) != n_objects:
        return None
    if sum(len(v) for v in parsed_containers.values()) != n_objects:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
