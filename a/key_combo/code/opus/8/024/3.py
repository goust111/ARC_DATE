from __future__ import annotations

from itertools import combinations
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CELL_SIZE = 5
PITCH = CELL_SIZE + 1
GEN_ATTEMPTS = 300
MAX_SHAPE_PLACEMENTS = 72


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid).ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _regular_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int, int]:
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        if len(rows) < 2 or len(cols) < 2:
            continue
        if int(coords.shape[0]) != len(rows) * len(cols):
            continue
        if any(int(grid[r, c]) != color for r in rows for c in cols):
            continue
        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        row_pitch = int(row_diffs[0])
        col_pitch = int(col_diffs[0])
        if row_pitch >= 4 and col_pitch >= 4:
            candidates.append((int(color), rows, cols, row_pitch, col_pitch))
    assert len(candidates) == 1, "expected exactly one regular marker lattice"
    return candidates[0]


def _visible_obstacle_data(local: np.ndarray) -> dict:
    sprite = {tuple(map(int, rc)) for rc in np.argwhere(local == Color.GREY)}
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(local == Color.RED)}
    visible: set[tuple[int, int]] = set()
    occluded: set[tuple[int, int]] = set()

    for r, c in sprite:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            first_hit = None
            while 0 <= rr < local.shape[0] and 0 <= cc < local.shape[1]:
                pos = (rr, cc)
                if pos in sprite:
                    break
                if pos in obstacles:
                    if first_hit is None:
                        first_hit = pos
                        visible.add(pos)
                    else:
                        occluded.add(pos)
                rr += dr
                cc += dc

    def clear_to_edge(start: tuple[int, int], dr: int, dc: int) -> bool:
        rr, cc = start[0] + dr, start[1] + dc
        while 0 <= rr < local.shape[0] and 0 <= cc < local.shape[1]:
            pos = (rr, cc)
            if pos in sprite or pos in obstacles:
                return False
            rr += dr
            cc += dc
        return True

    open_right_rows = {r for r, c in sprite if clear_to_edge((r, c), 0, 1)}
    open_left_rows = {r for r, c in sprite if clear_to_edge((r, c), 0, -1)}
    open_down_cols = {c for r, c in sprite if clear_to_edge((r, c), 1, 0)}
    open_up_cols = {c for r, c in sprite if clear_to_edge((r, c), -1, 0)}

    def first_non_bg_on_row(r: int, forward: bool) -> tuple[int, int] | None:
        cols = range(local.shape[1]) if forward else range(local.shape[1] - 1, -1, -1)
        for c in cols:
            val = int(local[r, c])
            if val != Color.BLACK:
                return val, c
        return None

    def first_non_bg_on_col(c: int, forward: bool) -> tuple[int, int] | None:
        rows = range(local.shape[0]) if forward else range(local.shape[0] - 1, -1, -1)
        for r in rows:
            val = int(local[r, c])
            if val != Color.BLACK:
                return val, r
        return None

    enterable_from_left_rows = {
        r for r in range(local.shape[0]) if (first := first_non_bg_on_row(r, True)) is not None and first[0] == Color.RED
    }
    enterable_from_right_rows = {
        r for r in range(local.shape[0]) if (first := first_non_bg_on_row(r, False)) is not None and first[0] == Color.RED
    }
    enterable_from_top_cols = {
        c for c in range(local.shape[1]) if (first := first_non_bg_on_col(c, True)) is not None and first[0] == Color.RED
    }
    enterable_from_bottom_cols = {
        c for c in range(local.shape[1]) if (first := first_non_bg_on_col(c, False)) is not None and first[0] == Color.RED
    }

    return {
        "visible": tuple(sorted(visible)),
        "visible_count": len(visible),
        "occluded_by_obstacle": bool(occluded),
        "total_obstacles": len(obstacles),
        "sprite": tuple(sorted(sprite)),
        "obstacles": tuple(sorted(obstacles)),
        "sprite_rows": {r for r, _ in sprite},
        "sprite_cols": {c for _, c in sprite},
        "obstacle_rows": {r for r, _ in obstacles},
        "obstacle_cols": {c for _, c in obstacles},
        "open_right_rows": open_right_rows,
        "open_left_rows": open_left_rows,
        "open_down_cols": open_down_cols,
        "open_up_cols": open_up_cols,
        "enterable_from_left_rows": enterable_from_left_rows,
        "enterable_from_right_rows": enterable_from_right_rows,
        "enterable_from_top_cols": enterable_from_top_cols,
        "enterable_from_bottom_cols": enterable_from_bottom_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey sprite by how many red obstacles it can hit with cell-bounded cardinal rays.

    Input structure:
    - The background is black, one color forms a complete regular Cartesian lattice of isolated yellow markers,
      and each lattice cell interior contains one 4-connected grey sprite plus zero to three red singleton obstacles.
    - Sprites and obstacles stay strictly inside the open rectangle between adjacent lattice markers.

    Rule:
    - Infer the lattice marker rows and columns, then treat every rectangle between consecutive marker rows and columns
      as one cell interior.
    - In each cell, cast rays up, down, left, and right from every grey sprite pixel, stopping a ray when it would leave
      the cell, when it meets another grey sprite pixel, or when it hits a red obstacle.
    - Count the distinct red obstacles hit by at least one such ray and recolor the whole sprite by that count:
      0 -> blue, 1 -> green, 2 -> orange, 3 or more -> magenta. Keep yellow markers and red obstacles unchanged.

    Key insight:
    - The answer depends on the marker-defined cell boundaries and the sprite geometry together: only obstacles visible
      along unobstructed within-cell rays count, so total obstacle count alone is not enough.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker_color, marker_rows, marker_cols, _row_pitch, _col_pitch = _regular_lattice(g, bg)
    assert marker_color == Color.YELLOW, "expected yellow lattice markers"

    out = g.copy()
    hit_to_color = {
        0: Color.BLUE,
        1: Color.GREEN,
        2: Color.ORANGE,
    }

    for ri in range(len(marker_rows) - 1):
        for ci in range(len(marker_cols) - 1):
            r0, r1 = marker_rows[ri] + 1, marker_rows[ri + 1] - 1
            c0, c1 = marker_cols[ci] + 1, marker_cols[ci + 1] - 1
            local = g[r0 : r1 + 1, c0 : c1 + 1]
            sprite_comps = _component_coords(local == Color.GREY)
            assert len(sprite_comps) == 1, "expected exactly one grey sprite per cell"
            assert all(local[r, c] in (Color.BLACK, Color.GREY, Color.RED) for r in range(local.shape[0]) for c in range(local.shape[1]))
            hits = _visible_obstacle_data(local)["visible_count"]
            new_color = hit_to_color.get(hits, Color.MAGENTA)
            for rr, cc in sprite_comps[0]:
                out[r0 + int(rr), c0 + int(cc)] = new_color

    return out


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _transform_shape(coords: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    pts = np.array(coords, dtype=int)
    if k & 1:
        pts = pts[:, ::-1]
    for _ in range((k >> 1) % 4):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    return _normalize(tuple(map(tuple, pts.tolist())))


def _shape_placements() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    ]
    placements: set[tuple[tuple[int, int], ...]] = set()
    for base in base_shapes:
        for k in range(8):
            shape = _transform_shape(base, k)
            height = max(r for r, _ in shape) + 1
            width = max(c for _, c in shape) + 1
            for dr in range(CELL_SIZE - height + 1):
                for dc in range(CELL_SIZE - width + 1):
                    placements.add(tuple(sorted((r + dr, c + dc) for r, c in shape)))
    return sorted(placements)


def _obstacles_separated(obstacles: tuple[tuple[int, int], ...]) -> bool:
    for i, (r0, c0) in enumerate(obstacles):
        for r1, c1 in obstacles[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _candidate_from_layout(sprite: tuple[tuple[int, int], ...], obstacles: tuple[tuple[int, int], ...], cid: int) -> dict:
    local = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    for r, c in sprite:
        local[r, c] = Color.GREY
    for r, c in obstacles:
        local[r, c] = Color.RED
    meta = _visible_obstacle_data(local)
    meta["id"] = cid
    meta["local"] = local
    return meta


def _build_candidate_pool() -> list[dict]:
    all_cells = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)]
    placements = _shape_placements()
    if len(placements) > MAX_SHAPE_PLACEMENTS:
        placements = random.Random(0).sample(placements, MAX_SHAPE_PLACEMENTS)
    seen: set[bytes] = set()
    candidates: list[dict] = []
    cid = 0
    for sprite in placements:
        sprite_set = set(sprite)
        blocked = set(sprite_set)
        for r, c in sprite:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < CELL_SIZE and 0 <= nc < CELL_SIZE:
                    blocked.add((nr, nc))
        free = [cell for cell in all_cells if cell not in blocked]
        for k in range(4):
            for obstacles in combinations(free, k):
                if not _obstacles_separated(obstacles):
                    continue
                candidate = _candidate_from_layout(sprite, tuple(sorted(obstacles)), cid)
                raw = candidate["local"].tobytes()
                if raw in seen:
                    continue
                seen.add(raw)
                cid += 1
                candidates.append(candidate)
    return candidates


_CANDIDATES = _build_candidate_pool()
_BY_VISIBLE = {count: [cand for cand in _CANDIDATES if cand["visible_count"] == count] for count in range(4)}
_OCCLUSION_CANDS = [cand for cand in _CANDIDATES if cand["occluded_by_obstacle"]]


def _obstacle_groups() -> dict[tuple[tuple[int, int], ...], list[dict]]:
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for cand in _CANDIDATES:
        if cand["total_obstacles"] >= 1:
            groups.setdefault(cand["obstacles"], []).append(cand)
    return groups


def _boundary_row_indices_h() -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    left_by_row = {r: [] for r in range(CELL_SIZE)}
    right_by_row = {r: [] for r in range(CELL_SIZE)}
    for left in _CANDIDATES:
        for row in left["open_right_rows"]:
            left_by_row[row].append(left)
    for right in _CANDIDATES:
        for row in right["enterable_from_left_rows"]:
            right_by_row[row].append(right)
    return left_by_row, right_by_row


_OBSTACLE_GROUPS = _obstacle_groups()
_SAME_OBSTACLE_KEYS = [
    key
    for key, items in _OBSTACLE_GROUPS.items()
    if len({cand["visible_count"] for cand in items}) >= 2
]
_BOUNDARY_LEFT_BY_ROW, _BOUNDARY_RIGHT_BY_ROW = _boundary_row_indices_h()
_BOUNDARY_ROWS_H = [
    row for row in range(CELL_SIZE) if _BOUNDARY_LEFT_BY_ROW[row] and _BOUNDARY_RIGHT_BY_ROW[row]
]


def _random_same_obstacle_pair() -> tuple[dict, dict]:
    key = random.choice(_SAME_OBSTACLE_KEYS)
    items = _OBSTACLE_GROUPS[key][:]
    random.shuffle(items)
    for a in items:
        for b in items:
            if a["visible_count"] != b["visible_count"]:
                return a, b
    raise AssertionError("expected a same-obstacle pair with different visibility")


def _random_boundary_pair_h() -> tuple[dict, dict]:
    row = random.choice(_BOUNDARY_ROWS_H)
    return random.choice(_BOUNDARY_LEFT_BY_ROW[row]), random.choice(_BOUNDARY_RIGHT_BY_ROW[row])


def _cell_output_color(hit_count: int) -> int:
    if hit_count == 0:
        return Color.BLUE
    if hit_count == 1:
        return Color.GREEN
    if hit_count == 2:
        return Color.ORANGE
    return Color.MAGENTA


def _render_grid(rows_cells: int, cols_cells: int, assignment: dict[tuple[int, int], dict]) -> np.ndarray:
    h = rows_cells * PITCH + 1
    w = cols_cells * PITCH + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, PITCH):
        for c in range(0, w, PITCH):
            grid[r, c] = Color.YELLOW
    for (cell_r, cell_c), cand in assignment.items():
        r0 = cell_r * PITCH + 1
        c0 = cell_c * PITCH + 1
        grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cand["local"]
    return grid


def _analyze_cells(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    marker_color, marker_rows, marker_cols, row_pitch, col_pitch = _regular_lattice(grid, bg)
    assert marker_color == Color.YELLOW
    assert row_pitch == PITCH and col_pitch == PITCH
    info = []
    for ri in range(len(marker_rows) - 1):
        for ci in range(len(marker_cols) - 1):
            r0, r1 = marker_rows[ri] + 1, marker_rows[ri + 1] - 1
            c0, c1 = marker_cols[ci] + 1, marker_cols[ci + 1] - 1
            local = grid[r0 : r1 + 1, c0 : c1 + 1]
            sprite = _component_coords(local == Color.GREY)
            assert len(sprite) == 1
            meta = _visible_obstacle_data(local)
            meta["cell"] = (ri, ci)
            meta["local"] = local
            info.append(meta)
    return info


def _semantic_checks(grid: np.ndarray) -> bool:
    info = _analyze_cells(grid)
    counts = {cell["visible_count"] for cell in info}
    if counts != {0, 1, 2, 3}:
        return False
    if not any(cell["occluded_by_obstacle"] for cell in info):
        return False

    same_obstacles = False
    for a, b in combinations(info, 2):
        if a["obstacles"] == b["obstacles"] and a["visible_count"] != b["visible_count"]:
            same_obstacles = True
            break
    if not same_obstacles:
        return False

    trap = False
    by_cell = {cell["cell"]: cell for cell in info}
    max_r = max(r for r, _ in by_cell)
    max_c = max(c for _, c in by_cell)
    for r in range(max_r + 1):
        for c in range(max_c):
            left = by_cell[(r, c)]
            right = by_cell[(r, c + 1)]
            if left["open_right_rows"] & right["enterable_from_left_rows"]:
                trap = True
                break
        if trap:
            break
    if not trap:
        return False

    out = main(grid)
    colors = set(int(v) for v in out.ravel())
    for color in (Color.BLUE, Color.GREEN, Color.ORANGE, Color.MAGENTA):
        if color not in colors:
            return False
    return True


def _choose_layout() -> tuple[int, int]:
    return random.choice([(3, 3), (3, 4), (4, 3), (4, 4)])


def _disjoint_horizontal_pairs(rows_cells: int, cols_cells: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for r in range(rows_cells):
        for c in range(cols_cells - 1):
            pairs.append(((r, c), (r, c + 1)))
    return pairs


def _random_candidate(count: int) -> dict:
    return random.choice(_BY_VISIBLE[count])


def _fill_remaining(slots: list[tuple[int, int]], assignment: dict[tuple[int, int], dict]) -> None:
    used_ids = {cand["id"] for cand in assignment.values()}
    pool = _CANDIDATES[:]
    random.shuffle(pool)
    for slot in slots:
        for cand in pool:
            if cand["id"] not in used_ids or random.random() < 0.25:
                assignment[slot] = cand
                used_ids.add(cand["id"])
                break


def _gen_attempt() -> np.ndarray | None:
    rows_cells, cols_cells = _choose_layout()
    h_pairs = _disjoint_horizontal_pairs(rows_cells, cols_cells)
    for _ in range(200):
        trap_pair = _random_boundary_pair_h()
        shape_pair = _random_same_obstacle_pair()
        trap_slots = random.choice(h_pairs)
        disjoint_pairs = [pair for pair in h_pairs if set(pair).isdisjoint(set(trap_slots))]
        if not disjoint_pairs:
            continue
        shape_slots = random.choice(disjoint_pairs)

        assignment: dict[tuple[int, int], dict] = {
            trap_slots[0]: trap_pair[0],
            trap_slots[1]: trap_pair[1],
            shape_slots[0]: shape_pair[0],
            shape_slots[1]: shape_pair[1],
        }
        counts = {cand["visible_count"] for cand in assignment.values()}
        remaining = [slot for slot in [(r, c) for r in range(rows_cells) for c in range(cols_cells)] if slot not in assignment]

        extras: list[dict] = []
        if not any(cand["occluded_by_obstacle"] for cand in assignment.values()):
            extras.append(random.choice(_OCCLUSION_CANDS))
        for count in range(4):
            if count not in counts:
                extras.append(_random_candidate(count))
        if len(extras) > len(remaining):
            continue

        random.shuffle(remaining)
        for slot, cand in zip(remaining, extras):
            assignment[slot] = cand
        leftover = [slot for slot in remaining if slot not in assignment]
        _fill_remaining(leftover, assignment)

        grid = _render_grid(rows_cells, cols_cells, assignment)
        try:
            if _semantic_checks(grid):
                return grid
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    assert _SAME_OBSTACLE_KEYS, "expected shape-sensitive local examples"
    assert _BOUNDARY_ROWS_H, "expected cross-cell boundary traps"
    assert _OCCLUSION_CANDS, "expected obstacle-occlusion examples"
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visibility-recolor input")
