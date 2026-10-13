from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sorted_components(mask: np.ndarray) -> list[np.ndarray]:
    comps = _components(mask)
    comps.sort(key=lambda pts: (-int(pts[:, 0].max()), int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _drop_distance(coords: np.ndarray, blockers: np.ndarray) -> tuple[int, str]:
    h, _ = blockers.shape
    drop = 0
    while True:
        moved = coords.copy()
        moved[:, 0] += drop + 1
        if int(moved[:, 0].max()) >= h:
            return drop, "boundary"
        if np.any(blockers[moved[:, 0], moved[:, 1]] != BLACK):
            touched = blockers[moved[:, 0], moved[:, 1]]
            if np.any(touched == GREY):
                return drop, "ground"
            if np.any(touched == BLUE):
                return drop, "blue"
            if np.any(touched == GREEN):
                return drop, "green"
            return drop, "blocked"
        drop += 1


def _settle_green_sprites(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    blockers = g.copy()
    blockers[blockers == GREEN] = BLACK
    infos: list[dict] = []
    for coords in _sorted_components(g == GREEN):
        drop, reason = _drop_distance(coords, blockers)
        final_coords = coords.copy()
        final_coords[:, 0] += drop
        blockers[final_coords[:, 0], final_coords[:, 1]] = GREEN
        infos.append(
            {
                "initial": coords.copy(),
                "final": final_coords,
                "drop": int(drop),
                "reason": reason,
            }
        )
    return blockers, infos


def _recolor_by_area(settled: np.ndarray) -> np.ndarray:
    out = settled.copy()
    for comp in _components(out == GREEN):
        color = RED if len(comp) >= 5 else YELLOW
        out[comp[:, 0], comp[:, 1]] = color
    return out


def _green_areas(grid: np.ndarray) -> list[int]:
    return sorted(len(comp) for comp in _components(np.asarray(grid) == GREEN))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop green sprites around fixed blue blockers, then recolor merged landings by area.

    Input structure:
    - Black background fills the canvas and the bottom row is a solid grey ground line.
    - Blue cells form 1x2 or 2x1 fixed obstacles in the lower-middle area.
    - Green cells form several separate 4-connected sprites, each a small rigid shape suspended above the blockers and ground.

    Rule:
    - Keep the blue obstacles and grey ground fixed.
    - Let all green sprites fall downward as rigid bodies, resolving them from lower sprites to higher sprites; each sprite stops when moving one more row would hit the ground, a blue obstacle, or already-settled green cells.
    - After all falling stops, recompute the 4-connected green components because previously separate sprites may now merge.
    - Recolor each merged green component red if its area is at least 5, otherwise recolor it yellow.

    Key insight:
    - The final color depends on post-fall merged area, so blue obstacles matter indirectly by changing where sprites stop and which sprites become connected.
    """
    settled, _infos = _settle_green_sprites(np.asarray(grid, dtype=int))
    return _recolor_by_area(settled)


MOTIF_SHAPES = [
    np.asarray([(0, 0), (1, 0), (1, 1), (1, 2)], dtype=int),
    np.asarray([(0, 1), (1, 0), (1, 1), (1, 2)], dtype=int),
]
DOMINO_V = np.asarray([(0, 0), (1, 0)], dtype=int)


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _place_shape(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    placed = shape.copy()
    placed[:, 0] += int(top)
    placed[:, 1] += int(left)
    return placed


def _random_connected_shape(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        cells.add(random.choice(sorted(frontier)))
    coords = np.asarray(sorted(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    h, w = _shape_dims(coords)
    if h > 3 or w > 4:
        return _random_connected_shape(area)
    return coords


def _chebyshev_clear(cells: np.ndarray, occupied: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> bool:
    cell_set = {tuple(map(int, cell)) for cell in cells}
    for r, c in cell_set:
        if not (0 <= r < h - 1 and 0 <= c < w):
            return False
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if (rr, cc) in occupied and (rr, cc) not in cell_set:
                    return False
    return True


def _paint(grid: np.ndarray, cells: np.ndarray, color: int) -> None:
    grid[cells[:, 0], cells[:, 1]] = int(color)


def _sample_gap_rows(row_limit: int) -> tuple[int, int] | None:
    pairs: list[tuple[int, int]] = []
    for a_top in range(1, row_limit - 1):
        for b_top in range(1, row_limit - 1):
            a_bottom = a_top + 1
            b_bottom = b_top + 1
            if a_bottom + 2 <= b_top or b_bottom + 2 <= a_top:
                pairs.append((a_top, b_top))
    if not pairs:
        return None
    return random.choice(pairs)


def _free_columns(w: int, banned: set[int]) -> list[int]:
    return [c for c in range(1, w - 1) if c not in banned]


def _candidate_obstacle(shape: str, row: int, col: int) -> np.ndarray:
    if shape == "h":
        return np.asarray([(row, col), (row, col + 1)], dtype=int)
    return np.asarray([(row, col), (row + 1, col)], dtype=int)


def _find_third_obstacle(h: int, w: int, occupied: set[tuple[int, int]], banned_cols: set[int]) -> np.ndarray | None:
    for _ in range(120):
        shape = random.choice(["h", "v"])
        row = random.randint(h - 8, h - 4)
        if shape == "h":
            col = random.randint(1, w - 3)
            cells = _candidate_obstacle("h", row, col)
        else:
            row = min(row, h - 3)
            col = random.randint(1, w - 2)
            cells = _candidate_obstacle("v", row, col)
        if any(int(c) in banned_cols for _r, c in cells):
            continue
        if _chebyshev_clear(cells, occupied, h, w, pad=1):
            return cells
    return None


def _remove_cells(grid: np.ndarray, cells: np.ndarray) -> np.ndarray:
    g2 = np.asarray(grid, dtype=int).copy()
    g2[cells[:, 0], cells[:, 1]] = BLACK
    return g2


def _obstacle_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components(np.asarray(grid) == BLUE)


def _semantic_checks(grid: np.ndarray, key_obstacle: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if not np.all(g[-1, :] == GREY):
        return False

    obstacle_comps = _obstacle_components(g)
    if len(obstacle_comps) not in (2, 3):
        return False
    for comp in obstacle_comps:
        if len(comp) != 2:
            return False
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) not in {(1, 2), (2, 1)}:
            return False

    green_comps = _components(g == GREEN)
    if len(green_comps) < 4 or len(green_comps) > 6:
        return False
    for comp in green_comps:
        if not (2 <= len(comp) <= 5):
            return False

    settled, infos = _settle_green_sprites(g)
    if any(info["drop"] < 1 for info in infos):
        return False
    reasons = {info["reason"] for info in infos}
    if not {"ground", "blue", "green"}.issubset(reasons):
        return False

    areas = _green_areas(settled)
    if not areas or max(areas) < 5 or min(areas) >= 5:
        return False
    if len(areas) >= len(green_comps):
        return False

    output = main(g)
    if RED not in output or YELLOW not in output:
        return False

    if _green_areas(_settle_green_sprites(_remove_cells(g, key_obstacle))[0]) == areas:
        return False

    grounded_blue = False
    for comp in _components(settled == GREEN):
        for r, c in comp:
            if r + 1 < h and settled[r + 1, c] == BLUE:
                grounded_blue = True
                break
        if grounded_blue:
            break
    if not grounded_blue:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = GREY

    occupied: set[tuple[int, int]] = set()
    obstacle_objs: list[np.ndarray] = []
    sprite_objs: list[np.ndarray] = []

    support_row = random.randint(h - 8, h - 5)
    motif_col = random.randint(1, w - 8)

    o1 = np.asarray([(support_row, motif_col), (support_row, motif_col + 1)], dtype=int)
    o2 = np.asarray([(support_row, motif_col + 3), (support_row + 1, motif_col + 3)], dtype=int)
    if not _chebyshev_clear(o1, occupied, h, w, pad=1):
        return None
    occupied.update(map(tuple, o1))
    obstacle_objs.append(o1)
    if not _chebyshev_clear(o2, occupied, h, w, pad=1):
        return None
    occupied.update(map(tuple, o2))
    obstacle_objs.append(o2)

    a_shape = random.choice(MOTIF_SHAPES)
    rows = _sample_gap_rows(support_row - 2)
    if rows is None:
        return None
    a_top, b_top = rows
    a_cells = _place_shape(a_shape, a_top, motif_col)
    b_cells = _place_shape(DOMINO_V, b_top, motif_col + 3)
    for sprite in (a_cells, b_cells):
        if not _chebyshev_clear(sprite, occupied, h, w, pad=1):
            return None
        occupied.update(map(tuple, sprite))
        sprite_objs.append(sprite)

    stack_col = w - 3
    lower_top = random.randint(h - 8, h - 6)
    upper_top = random.randint(1, lower_top - 4)
    c_cells = _place_shape(DOMINO_V, lower_top, stack_col)
    d_cells = _place_shape(DOMINO_V, upper_top, stack_col)
    for sprite in (c_cells, d_cells):
        if not _chebyshev_clear(sprite, occupied, h, w, pad=1):
            return None
        occupied.update(map(tuple, sprite))
        sprite_objs.append(sprite)

    obstacle_count = random.choice([2, 3])
    extra_target = random.choice([0, 1, 2]) if obstacle_count == 2 else random.choice([1, 2])
    banned_cols = set(range(motif_col - 1, motif_col + 6)) | set(range(stack_col - 1, stack_col + 2))
    key_obstacle = o1

    if obstacle_count == 3:
        o3 = _find_third_obstacle(h, w, occupied, banned_cols)
        if o3 is None:
            return None
        occupied.update(map(tuple, o3))
        obstacle_objs.append(o3)

        area = random.choice([2, 3, 4])
        e_shape = _random_connected_shape(area)
        o3_r0, o3_c0, _o3_r1, _o3_c1 = _bbox(o3)
        max_top = o3_r0 - _shape_dims(e_shape)[0] - 1
        if max_top < 1:
            return None
        e_top = random.randint(1, max_top)
        e_left = max(1, min(o3_c0, w - _shape_dims(e_shape)[1] - 1))
        e_cells = _place_shape(e_shape, e_top, e_left)
        if not _chebyshev_clear(e_cells, occupied, h, w, pad=1):
            return None
        occupied.update(map(tuple, e_cells))
        sprite_objs.append(e_cells)
        extra_target -= 1

    for _ in range(max(0, extra_target)):
        area = random.choice([2, 3, 4])
        shape = _random_connected_shape(area)
        sh, sw = _shape_dims(shape)
        placed = None
        candidate_cols = _free_columns(w, banned_cols)
        random.shuffle(candidate_cols)
        tops = list(range(1, h - sh - 2))
        random.shuffle(tops)
        for left in candidate_cols:
            if left + sw > w - 1:
                continue
            for top in tops:
                cells = _place_shape(shape, top, left)
                if _chebyshev_clear(cells, occupied, h, w, pad=1):
                    placed = cells
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        occupied.update(map(tuple, placed))
        sprite_objs.append(placed)

    for obj in obstacle_objs:
        _paint(grid, obj, BLUE)
    for obj in sprite_objs:
        _paint(grid, obj, GREEN)

    if not _semantic_checks(grid, key_obstacle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-obstacle ARC input")
