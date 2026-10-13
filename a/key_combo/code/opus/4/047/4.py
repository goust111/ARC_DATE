from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
WALL = 8
TASK_COLORS = (BLUE, GREEN)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _frame_like(coords: np.ndarray) -> bool:
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    coord_set = {tuple(map(int, rc)) for rc in coords}
    perim = _perimeter_cells(bbox)
    if not coord_set <= perim:
        return False
    missing = perim - coord_set
    return len(missing) <= 1


def _locate_role(grid: np.ndarray, color: int) -> dict:
    comps = _components_mask(grid == color)
    seeds = [comp for comp in comps if comp.shape[0] == 1]
    frames = [comp for comp in comps if comp.shape[0] > 1 and _frame_like(comp)]
    matches = []
    for frame in frames:
        bbox = _bbox_coords(frame)
        r0, c0, r1, c1 = bbox
        for seed in seeds:
            sr, sc = map(int, seed[0])
            if r0 < sr < r1 and c0 < sc < c1:
                matches.append(
                    {
                        "color": int(color),
                        "seed": (sr, sc),
                        "frame_coords": frame,
                        "bbox": bbox,
                        "inner_black_area": int(np.sum(grid[r0 + 1 : r1, c0 + 1 : c1] == BG)),
                    }
                )
    assert len(matches) == 1, f"expected one seed/frame pair for color {color}"
    return matches[0]


def _distance_map(passable: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    q = deque([seed])
    dist[seed] = 0
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _render_output(
    grid: np.ndarray,
    roles: list[dict],
    primary_color: int,
    secondary_color: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    out = np.array(grid, dtype=int, copy=True)
    passable = grid == BG
    for role in roles:
        passable[role["seed"]] = True

    dists = [_distance_map(passable, role["seed"]) for role in roles]
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            d0 = int(dists[0][r, c])
            d1 = int(dists[1][r, c])
            if d0 < 0 and d1 < 0:
                continue
            if d0 >= 0 and d1 >= 0 and d0 == d1:
                out[r, c] = int(primary_color)
                continue
            best = d1 if d0 < 0 else d0
            if d1 >= 0 and (d0 < 0 or d1 < d0):
                best = d1
            out[r, c] = int(primary_color if best % 2 == 0 else secondary_color)
    return out, dists


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the larger seed-framing rectangle to decide which color starts the maze flood fill.

    Input structure:
    - Teal (8) cells form maze walls, and black (0) cells form the walkable corridor network.
    - There are exactly two non-teal, non-black colors: blue (1) and green (3).
    - Each color appears as a one-pixel seed plus a same-colored hollow rectangle-like frame surrounding that seed.
    - The frame interior contains black corridor cells, and the two seed regions connect into a shared corridor system.

    Rule:
    - Detect the two seed/frame pairs and compute each frame's internal black area, counting only black cells strictly inside the frame's bounding box.
    - The color whose frame has the larger internal black area becomes `C_primary`; the other becomes `C_secondary`.
    - Run simultaneous 4-neighbor BFS from the two seeds over black corridor cells, also treating the seed cells themselves as distance 0.
    - If one wave reaches a cell first, color that cell by parity of the winning distance: even distances become `C_primary`, odd distances become `C_secondary`.
    - If both waves reach a cell at the same distance, `C_primary` wins that cell regardless of parity. Keep teal walls and colored frame lines unchanged.

    Key insight:
    - The rectangle comparison does not directly produce the answer; it only assigns the primary and secondary roles that control the later alternating flood fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(int(c) for c in np.unique(g) if int(c) not in {BG, WALL})
    assert len(colors) == 2, "expected exactly two seed/frame colors"

    roles = [_locate_role(g, color) for color in colors]
    areas = [role["inner_black_area"] for role in roles]
    assert areas[0] != areas[1], "rectangle inner areas must differ"

    if areas[0] > areas[1]:
        primary, secondary = roles[0]["color"], roles[1]["color"]
    else:
        primary, secondary = roles[1]["color"], roles[0]["color"]

    out, _ = _render_output(g, roles, primary, secondary)
    return out


def _draw_line(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo : c_hi + 1] = BG
        return
    if c0 == c1:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo : r_hi + 1, c0] = BG
        return
    raise ValueError("only axis-aligned segments are supported")


def _carve_box_area(grid: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(1, r0 - pad)
    cc0 = max(1, c0 - pad)
    rr1 = min(grid.shape[0] - 2, r1 + pad)
    cc1 = min(grid.shape[1] - 2, c1 + pad)
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG


def _paint_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gate_side: str,
    gate_row: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    for r, c in _perimeter_cells(bbox):
        grid[r, c] = int(color)

    if gate_side == "right":
        gate = (gate_row, c1)
        seed = (gate_row, c1 - 1)
    else:
        assert gate_side == "left"
        gate = (gate_row, c0)
        seed = (gate_row, c0 + 1)

    grid[gate] = BG
    grid[seed] = int(color)
    return seed


def _owner_color_output(grid: np.ndarray, roles: list[dict], primary_color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    passable = grid == BG
    for role in roles:
        passable[role["seed"]] = True
    dists = [_distance_map(passable, role["seed"]) for role in roles]
    seed_colors = [role["color"] for role in roles]
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            d0 = int(dists[0][r, c])
            d1 = int(dists[1][r, c])
            if d0 < 0 and d1 < 0:
                continue
            if d0 >= 0 and d1 >= 0 and d0 == d1:
                out[r, c] = int(primary_color)
            elif d1 < 0 or (d0 >= 0 and d0 < d1):
                out[r, c] = int(seed_colors[0])
            else:
                out[r, c] = int(seed_colors[1])
    return out


def _validate_grid(grid: np.ndarray) -> bool:
    try:
        roles = [_locate_role(grid, color) for color in TASK_COLORS]
    except AssertionError:
        return False

    areas = [role["inner_black_area"] for role in roles]
    if areas[0] == areas[1]:
        return False

    if areas[0] > areas[1]:
        primary, secondary = roles[0]["color"], roles[1]["color"]
    else:
        primary, secondary = roles[1]["color"], roles[0]["color"]

    out, dists = _render_output(grid, roles, primary, secondary)
    swapped, _ = _render_output(grid, roles, secondary, primary)
    if np.array_equal(out, swapped):
        return False

    both_reachable = (dists[0] >= 0) & (dists[1] >= 0)
    if not np.any(both_reachable):
        return False

    odd_tie = both_reachable & (dists[0] == dists[1]) & ((dists[0] % 2) == 1)
    if not np.any(odd_tie):
        return False

    earlier0 = (dists[0] >= 0) & ((dists[1] < 0) | (dists[0] < dists[1]))
    earlier1 = (dists[1] >= 0) & ((dists[0] < 0) | (dists[1] < dists[0]))
    if not np.any(earlier0) or not np.any(earlier1):
        return False

    if len(set(np.unique(out)) - {BG, WALL}) < 2:
        return False

    owner_out = _owner_color_output(grid, roles, primary)
    if np.array_equal(out, owner_out):
        return False

    if not np.array_equal(out, main(grid)):
        return False

    return True


def _canonical_attempt() -> np.ndarray | None:
    n = random.choice((21, 23))
    hub_r = n // 2
    hub_c = n // 2

    size_choices = [(5, 5), (5, 7), (7, 5), (7, 7)]
    left_hw, right_hw = random.sample(size_choices, 2)
    left_area = (left_hw[0] - 2) * (left_hw[1] - 2) - 1
    right_area = (right_hw[0] - 2) * (right_hw[1] - 2) - 1
    if left_area == right_area:
        return None

    max_gate = min(hub_c - left_hw[1] - 1, hub_c - right_hw[1] - 1, 6)
    gate_dists = [d for d in (2, 4, 6) if d <= max_gate]
    if not gate_dists:
        return None
    gate_dist = random.choice(gate_dists)

    left_gate = (hub_r, hub_c - gate_dist)
    right_gate = (hub_r, hub_c + gate_dist)
    left_bbox = (
        hub_r - left_hw[0] // 2,
        left_gate[1] - left_hw[1] + 1,
        hub_r + left_hw[0] // 2,
        left_gate[1],
    )
    right_bbox = (
        hub_r - right_hw[0] // 2,
        right_gate[1],
        hub_r + right_hw[0] // 2,
        right_gate[1] + right_hw[1] - 1,
    )

    if min(left_bbox[0], left_bbox[1], right_bbox[0], right_bbox[1]) < 2:
        return None
    if max(left_bbox[2], right_bbox[2]) > n - 3 or max(left_bbox[3], right_bbox[3]) > n - 3:
        return None

    grid = np.full((n, n), WALL, dtype=int)

    _draw_line(grid, left_gate, right_gate)

    common_dir = random.choice((-1, 1))
    common_len = random.randint(2, 5)
    common_end_r = hub_r + common_dir * common_len
    if not (1 <= common_end_r <= n - 2):
        return None
    _draw_line(grid, (hub_r, hub_c), (common_end_r, hub_c))

    for bbox in (left_bbox, right_bbox):
        _carve_box_area(grid, bbox, pad=1)

    left_seed = _paint_frame(grid, left_bbox, BLUE, "right", hub_r)
    right_seed = _paint_frame(grid, right_bbox, GREEN, "left", hub_r)

    left_dead_col = random.randint(left_gate[1] + 1, hub_c - 1)
    left_dead_dir = random.choice((-1, 1))
    left_dead_len = random.randint(2, 4)
    end_r = hub_r + left_dead_dir * left_dead_len
    if 1 <= end_r <= n - 2:
        _draw_line(grid, (hub_r, left_dead_col), (end_r, left_dead_col))

    right_dead_col = random.randint(hub_c + 1, right_gate[1] - 1)
    right_dead_dir = random.choice((-1, 1))
    right_dead_len = random.randint(2, 4)
    end_r = hub_r + right_dead_dir * right_dead_len
    if 1 <= end_r <= n - 2:
        _draw_line(grid, (hub_r, right_dead_col), (end_r, right_dead_col))

    branch_row = hub_r + common_dir * random.randint(1, common_len)
    side_len = random.randint(2, 4)
    side_dir = random.choice((-1, 1))
    side_end_c = hub_c + side_dir * side_len
    if 1 <= side_end_c <= n - 2:
        _draw_line(grid, (branch_row, hub_c), (branch_row, side_end_c))

    if grid[left_seed] != BLUE or grid[right_seed] != GREEN:
        return None

    return np.rot90(grid, k=random.randint(0, 3)).copy()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _canonical_attempt()
        if grid is not None and _validate_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid maze-fill ARC input")
