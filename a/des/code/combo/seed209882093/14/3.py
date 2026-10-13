from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PATH_COLORS = {GREY, AZURE}
PATTERN_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON]
COMPLEMENT = {
    BLUE: YELLOW,
    YELLOW: BLUE,
    RED: GREEN,
    GREEN: RED,
    ORANGE: AZURE,
    AZURE: ORANGE,
    MAGENTA: MAROON,
    MAROON: MAGENTA,
}

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ATTACHMENT_POS = {
    "top": (2, 1),
    "bottom": (0, 1),
    "left": (1, 2),
    "right": (1, 0),
}

# --------------- solver ---------------


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, mask.shape):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(comp)
    return components


def _block_slice(
    endpoint: tuple[int, int],
    outward: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[slice, slice] | None:
    r, c = endpoint
    dr, dc = outward
    if (dr, dc) == (-1, 0):
        r0, c0 = r - 3, c - 1
    elif (dr, dc) == (1, 0):
        r0, c0 = r + 1, c - 1
    elif (dr, dc) == (0, -1):
        r0, c0 = r - 1, c - 3
    elif (dr, dc) == (0, 1):
        r0, c0 = r - 1, c + 1
    else:
        raise AssertionError("invalid outward direction")

    r1 = r0 + 3
    c1 = c0 + 3
    h, w = shape
    if not (0 <= r0 and 0 <= c0 and r1 <= h and c1 <= w):
        return None
    return slice(r0, r1), slice(c0, c1)


def _classify_endpoint_patch(
    grid: np.ndarray,
    endpoint: tuple[int, int],
    neighbor: tuple[int, int],
) -> dict | None:
    outward = (endpoint[0] - neighbor[0], endpoint[1] - neighbor[1])
    block = _block_slice(endpoint, outward, grid.shape)
    if block is None:
        return None

    patch = grid[block].copy()
    if np.any(patch == GREY):
        return None
    kind = "blank" if np.all(patch == BLACK) else "pattern"
    return {"kind": kind, "slice": block, "patch": patch}


def _path_candidate_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    coord_set = set(coords)
    if len(coord_set) < 2:
        return None

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coord_set:
        neighbors = [nbr for nbr in _neighbors4(cell[0], cell[1], grid.shape) if nbr in coord_set]
        degree = len(neighbors)
        if degree not in (1, 2):
            return None
        neighbor_map[cell] = neighbors

    endpoints = [cell for cell, neighbors in neighbor_map.items() if len(neighbors) == 1]
    if len(endpoints) != 2:
        return None

    for cell in coord_set:
        if int(grid[cell]) != AZURE:
            continue
        neighbors = neighbor_map[cell]
        if len(neighbors) != 2:
            return None
        straight = neighbors[0][0] == neighbors[1][0] or neighbors[0][1] == neighbors[1][1]
        if not straight:
            return None

    endpoint_infos = []
    for endpoint in endpoints:
        info = _classify_endpoint_patch(grid, endpoint, neighbor_map[endpoint][0])
        if info is None:
            return None
        endpoint_infos.append(info)

    pattern_infos = [info for info in endpoint_infos if info["kind"] == "pattern"]
    blank_infos = [info for info in endpoint_infos if info["kind"] == "blank"]
    if len(pattern_infos) != 1 or len(blank_infos) != 1:
        return None

    source_patch = pattern_infos[0]["patch"]
    source_colors = set(map(int, np.unique(source_patch)))
    if not source_colors <= ({BLACK} | set(COMPLEMENT)):
        return None

    return {
        "source_patch": source_patch,
        "target_slice": blank_infos[0]["slice"],
        "relay_count": sum(int(grid[cell]) == AZURE for cell in coord_set),
        "path_size": len(coord_set),
    }


def _find_task_path(grid: np.ndarray) -> dict:
    mask = np.isin(grid, tuple(PATH_COLORS))
    candidates = []
    for coords in _mask_components(mask):
        info = _path_candidate_info(grid, coords)
        if info is not None:
            candidates.append(info)

    assert candidates, "expected one valid grey/azure path"
    candidates.sort(key=lambda item: item["path_size"], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best["path_size"] > candidates[1]["path_size"], "ambiguous path candidate"
    return best


def _complement_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    colors = set(map(int, np.unique(patch)))
    assert colors <= ({BLACK} | set(COMPLEMENT)), "unexpected pattern color"
    for src, dst in COMPLEMENT.items():
        out[patch == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a 3x3 signal along a grey path, rotating once at each azure relay and
    complementing colors before drawing the result beside the far endpoint.
    """
    g = np.asarray(grid, dtype=int)
    info = _find_task_path(g)

    turns = info["relay_count"] % 4
    rotated = np.rot90(info["source_patch"], -turns)
    transformed = _complement_patch(rotated)

    out = g.copy()
    out[info["target_slice"]] = transformed
    return out


# --------------- generator ---------------


MIN_COARSE_SIDE = 4
MAX_COARSE_SIDE = 7
MIN_COARSE_PATH_LEN = 6
MAX_WALK_STEPS = 5000
MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_ATTEMPTS = 200


def _boundary_cells(side: str, rows: int, cols: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(cols)]
    if side == "bottom":
        return [(rows - 1, c) for c in range(cols)]
    if side == "left":
        return [(r, 0) for r in range(rows)]
    if side == "right":
        return [(r, cols - 1) for r in range(rows)]
    raise AssertionError("invalid side")


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            bends += 1
    return bends


def _coarse_to_fine(cell: tuple[int, int]) -> tuple[int, int]:
    return 2 * cell[0] + 4, 2 * cell[1] + 4


def _fine_block_slice(
    endpoint: tuple[int, int],
    side: str,
    shape: tuple[int, int],
) -> tuple[slice, slice] | None:
    return _block_slice(endpoint, SIDE_TO_OUTWARD[side], shape)


def _rectangles_overlap(a: tuple[slice, slice], b: tuple[slice, slice]) -> bool:
    ar0, ar1 = a[0].start, a[0].stop
    ac0, ac1 = a[1].start, a[1].stop
    br0, br1 = b[0].start, b[0].stop
    bc0, bc1 = b[1].start, b[1].stop
    return ar0 < br1 and br0 < ar1 and ac0 < bc1 and bc0 < ac1


def _cell_in_slice(cell: tuple[int, int], block: tuple[slice, slice]) -> bool:
    r, c = cell
    return block[0].start <= r < block[0].stop and block[1].start <= c < block[1].stop


def _loop_erased_random_path(
    rows: int,
    cols: int,
    start: tuple[int, int],
    target: tuple[int, int],
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    total = rows * cols
    upper_len = min(total, 14)
    lower_len = min(MIN_COARSE_PATH_LEN, upper_len)
    if lower_len > upper_len:
        return None
    blocked = set() if forbidden is None else set(forbidden)

    for _ in range(MAX_PATH_ATTEMPTS):
        desired_len = random.randint(lower_len, upper_len)
        walk = [start]
        index_of = {start: 0}
        cur = start

        for _step in range(MAX_WALK_STEPS):
            if cur == target and len(walk) >= desired_len and _count_bends(walk) >= 1:
                return walk

            neighbors = [nbr for nbr in _coarse_neighbors(cur, rows, cols) if nbr not in blocked or nbr == target]
            if not neighbors:
                break
            weights: list[float] = []
            for nbr in neighbors:
                dist = _manhattan(nbr, target)
                if len(walk) < desired_len:
                    weight = 1.0 + 0.35 * dist
                    if nbr == target and len(walk) + 1 < desired_len:
                        weight = 0.05
                else:
                    weight = 1.0 / (1.0 + dist)
                weights.append(weight)

            nxt = random.choices(neighbors, weights=weights, k=1)[0]
            if nxt in index_of:
                keep = index_of[nxt] + 1
                for removed in walk[keep:]:
                    index_of.pop(removed, None)
                walk = walk[:keep]
            else:
                walk.append(nxt)
                index_of[nxt] = len(walk) - 1
            cur = walk[-1]

    return None


def _expand_fine_path(coarse_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    fine_path = [_coarse_to_fine(coarse_path[0])]
    for prev, cur in zip(coarse_path, coarse_path[1:], strict=False):
        pr, pc = _coarse_to_fine(prev)
        cr, cc = _coarse_to_fine(cur)
        fine_path.append(((pr + cr) // 2, (pc + cc) // 2))
        fine_path.append((cr, cc))
    return fine_path


def _straight_indices(path: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) == (dr2, dc2):
            indices.append(i)
    return indices


def _choose_endpoints(
    coarse_rows: int,
    coarse_cols: int,
    shape: tuple[int, int],
) -> tuple[str, tuple[int, int], str, tuple[int, int]] | None:
    sides = list(SIDE_TO_OUTWARD)
    for _ in range(120):
        start_side, target_side = random.sample(sides, 2)
        start = random.choice(_boundary_cells(start_side, coarse_rows, coarse_cols))
        target = random.choice(_boundary_cells(target_side, coarse_rows, coarse_cols))
        if start == target or _manhattan(start, target) < 3:
            continue

        start_slice = _fine_block_slice(_coarse_to_fine(start), start_side, shape)
        target_slice = _fine_block_slice(_coarse_to_fine(target), target_side, shape)
        if start_slice is None or target_slice is None:
            continue
        if _rectangles_overlap(start_slice, target_slice):
            continue
        return start_side, start, target_side, target
    return None


def _random_pattern(side: str) -> np.ndarray:
    patch = np.full((3, 3), BLACK, dtype=int)
    attachment = ATTACHMENT_POS[side]

    cells = [(r, c) for r in range(3) for c in range(3)]
    fill_count = random.randint(4, 9)
    filled = {attachment}
    remaining = [cell for cell in cells if cell != attachment]
    random.shuffle(remaining)
    filled.update(remaining[: fill_count - 1])

    non_azure_pool = [color for color in PATTERN_COLORS if color != AZURE]
    use_azure = random.random() < 0.35 and attachment != (1, 1)
    azure_cell = None
    if use_azure:
        azure_candidates = [cell for cell in filled if cell != attachment]
        if azure_candidates:
            azure_cell = random.choice(azure_candidates)

    palette_size = random.randint(1, min(4, len(PATTERN_COLORS) - (1 if azure_cell else 0)))
    palette = random.sample(non_azure_pool, palette_size)

    for cell in filled:
        if cell == azure_cell:
            patch[cell] = AZURE
        elif cell == attachment:
            attachment_choices = [color for color in palette if color != AZURE]
            patch[cell] = random.choice(attachment_choices or non_azure_pool)
        else:
            patch[cell] = random.choice(palette)

    return patch


def _expected_output(
    inp: np.ndarray,
    source_slice: tuple[slice, slice],
    target_slice: tuple[slice, slice],
    relay_count: int,
) -> np.ndarray:
    source_patch = inp[source_slice].copy()
    rotated = np.rot90(source_patch, -(relay_count % 4))
    out = inp.copy()
    out[target_slice] = _complement_patch(rotated)
    return out


def _generate_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(MIN_COARSE_SIDE, MAX_COARSE_SIDE)
    coarse_cols = random.randint(MIN_COARSE_SIDE, MAX_COARSE_SIDE)
    shape = (2 * coarse_rows + 6, 2 * coarse_cols + 6)

    endpoint_info = _choose_endpoints(coarse_rows, coarse_cols, shape)
    if endpoint_info is None:
        return None
    start_side, start, target_side, target = endpoint_info

    start_outward = SIDE_TO_OUTWARD[start_side]
    target_outward = SIDE_TO_OUTWARD[target_side]
    start_inner = (start[0] - start_outward[0], start[1] - start_outward[1])
    target_inner = (target[0] - target_outward[0], target[1] - target_outward[1])
    if start_inner == target_inner:
        return None
    core_path = _loop_erased_random_path(
        coarse_rows,
        coarse_cols,
        start_inner,
        target_inner,
        forbidden={start, target},
    )
    if core_path is None:
        return None
    coarse_path = [start] + core_path + [target]

    fine_path = _expand_fine_path(coarse_path)
    straight_positions = _straight_indices(fine_path)
    if len(straight_positions) < 4:
        return None

    grid = np.full(shape, BLACK, dtype=int)
    for r, c in fine_path:
        grid[r, c] = GREY

    relay_options = list(range(1, min(6, len(straight_positions)) + 1))
    desired_mod = random.randrange(4)
    matching = [count for count in relay_options if count % 4 == desired_mod]
    relay_count = random.choice(matching or relay_options)
    for idx in random.sample(straight_positions, relay_count):
        r, c = fine_path[idx]
        grid[r, c] = AZURE

    start_endpoint = _coarse_to_fine(start)
    target_endpoint = _coarse_to_fine(target)
    source_slice = _fine_block_slice(start_endpoint, start_side, shape)
    target_slice = _fine_block_slice(target_endpoint, target_side, shape)
    assert source_slice is not None and target_slice is not None
    if any(_cell_in_slice(cell, source_slice) or _cell_in_slice(cell, target_slice) for cell in fine_path):
        return None

    grid[source_slice] = _random_pattern(start_side)
    assert grid[target_slice].shape == (3, 3)
    grid[target_slice] = BLACK

    expected = _expected_output(grid, source_slice, target_slice, relay_count)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
