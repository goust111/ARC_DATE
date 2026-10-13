from __future__ import annotations

from collections import deque
import itertools
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIR4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
SAFE_OTHER_SIZES = (2, 4)
GEN_RETRIES = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _nonblack_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cross_info(grid: np.ndarray) -> dict:
    candidates = []
    for coords in _nonblack_components(grid):
        vals = {int(grid[r, c]) for r, c in coords}
        if vals <= {BLUE, RED} and BLUE in vals and RED in vals:
            candidates.append(coords)
    assert len(candidates) == 1, "expected exactly one blue/red cross component"
    coords = candidates[0]
    r0, c0, r1, c1 = _bbox(coords)
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in coords:
        crop[r - r0, c - c0] = int(grid[r, c])

    full_rows = [r for r in range(crop.shape[0]) if np.all(crop[r, :] != BLACK)]
    full_cols = [c for c in range(crop.shape[1]) if np.all(crop[:, c] != BLACK)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "cross must have one full row and one full column"

    row_idx = full_rows[0]
    col_idx = full_cols[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "crop": crop,
        "step_h": crop.shape[0],
        "step_w": crop.shape[1],
        "seed_row": r0 + row_idx,
        "seed_col": c0 + col_idx,
    }


def _periodic_positions(seed: int, step: int, limit: int) -> list[int]:
    vals = []
    x = int(seed)
    while x >= 0:
        vals.append(x)
        x -= int(step)
    vals.reverse()
    x = int(seed) + int(step)
    while x < limit:
        vals.append(x)
        x += int(step)
    return vals


def _frame_is_hollow_rect(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return False
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == YELLOW) and np.all(patch[1:-1, 1:-1] != YELLOW))


def _nearest_unique(value: float, choices: Iterable[int]) -> int:
    ranked = sorted((abs(float(value) - float(choice)), int(choice)) for choice in choices)
    assert len(ranked) >= 1
    if len(ranked) > 1:
        assert ranked[0][0] != ranked[1][0], "nearest wall tie is not allowed"
    return int(ranked[0][1])


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    cross = _cross_info(g)
    row_walls = _periodic_positions(cross["seed_row"], cross["step_h"], g.shape[0])
    col_walls = _periodic_positions(cross["seed_col"], cross["step_w"], g.shape[1])
    assert len(row_walls) >= 2 and len(col_walls) >= 2, "need at least two walls in each direction"

    gap_hs = sorted({b - a - 1 for a, b in zip(row_walls, row_walls[1:])})
    gap_ws = sorted({b - a - 1 for a, b in zip(col_walls, col_walls[1:])})
    assert len(gap_hs) == 1 and len(gap_ws) == 1, "wall spacing must be periodic"
    gap_h = gap_hs[0]
    gap_w = gap_ws[0]

    vertical_doors = []
    horizontal_doors = []
    for coords in _components_of_color(g, YELLOW):
        box = _bbox(coords)
        assert _frame_is_hollow_rect(g, box), "yellow objects must be hollow rectangular frames"
        r0, c0, r1, c1 = box
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        v_match = inner_h == gap_h
        h_match = inner_w == gap_w
        assert v_match ^ h_match, "each frame must encode exactly one door type"

        if v_match:
            wall_col = _nearest_unique((c0 + c1) / 2.0, col_walls)
            vertical_doors.append(
                {
                    "wall": wall_col,
                    "start": r0 + 1,
                    "end": r1 - 1,
                    "center": ((r0 + 1 + r1 - 1) // 2, wall_col),
                    "box": box,
                }
            )
        else:
            wall_row = _nearest_unique((r0 + r1) / 2.0, row_walls)
            horizontal_doors.append(
                {
                    "wall": wall_row,
                    "start": c0 + 1,
                    "end": c1 - 1,
                    "center": (wall_row, (c0 + 1 + c1 - 1) // 2),
                    "box": box,
                }
            )

    markers = np.argwhere(g == GREEN)
    assert markers.shape[0] == 2, "need exactly two green markers"
    markers = sorted((int(r), int(c)) for r, c in markers)

    return {
        "grid": g,
        "cross": cross,
        "row_walls": row_walls,
        "col_walls": col_walls,
        "gap_h": gap_h,
        "gap_w": gap_w,
        "vertical_doors": vertical_doors,
        "horizontal_doors": horizontal_doors,
        "markers": markers,
    }


def _build_wall_grid(shape: tuple[int, int], row_walls: list[int], col_walls: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for c in col_walls:
        out[:, c] = RED
    for r in row_walls:
        out[r, :] = BLUE
    return out


def _apply_doors(base: np.ndarray, vertical_doors: list[dict], horizontal_doors: list[dict]) -> np.ndarray:
    out = base.copy()
    for door in vertical_doors:
        out[door["start"] : door["end"] + 1, door["wall"]] = BLACK
    for door in horizontal_doors:
        out[door["wall"], door["start"] : door["end"] + 1] = BLACK
    return out


def _preferred_dirs(start: tuple[int, int], goal: tuple[int, int], prefer: str) -> list[tuple[int, int]]:
    sr, sc = start
    gr, gc = goal
    h = (0, 1) if gc >= sc else (0, -1)
    v = (1, 0) if gr >= sr else (-1, 0)
    if prefer == "horizontal":
        primary = [h, v]
    else:
        primary = [v, h]
    rest = [d for d in DIR4 if d not in primary]
    return primary + rest


def _shortest_path(
    open_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_cells: set[tuple[int, int]] | None = None,
    prefer: str = "horizontal",
) -> list[tuple[int, int]] | None:
    blocked_cells = blocked_cells or set()
    if start == goal:
        return [start]
    h, w = open_mask.shape
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        for dr, dc in _preferred_dirs(cur, goal, prefer):
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in prev or nxt in blocked_cells:
                continue
            if not open_mask[nr, nc]:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def _count_shortest_paths(
    open_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_cells: set[tuple[int, int]] | None = None,
) -> int:
    blocked_cells = blocked_cells or set()
    h, w = open_mask.shape
    dist = {start: 0}
    ways = {start: 1}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in DIR4:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked_cells:
                continue
            if not open_mask[nr, nc]:
                continue
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[cur]
                q.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(2, ways[nxt] + ways[cur])
    return int(ways.get(goal, 0))


def _ordered_waypoints(structure: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    doors = [{"kind": "V", **d} for d in structure["vertical_doors"]] + [{"kind": "H", **d} for d in structure["horizontal_doors"]]
    doors = sorted(doors, key=lambda d: (d["center"][0], d["center"][1], d["kind"]))
    markers = sorted(structure["markers"])
    start = markers[0]
    end = markers[1]
    return doors, [start] + [door["center"] for door in doors] + [end]


def _route_through_doors(structure: dict) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    base = _build_wall_grid(structure["grid"].shape, structure["row_walls"], structure["col_walls"])
    scene = _apply_doors(base, structure["vertical_doors"], structure["horizontal_doors"])
    open_mask = scene == BLACK
    doors, waypoints = _ordered_waypoints(structure)

    used: set[tuple[int, int]] = set()
    segments: list[list[tuple[int, int]]] = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]
        prefer = "horizontal"
        if i + 1 <= len(doors):
            prefer = "horizontal" if doors[i]["kind"] == "V" else "vertical"
        blocked = set(used)
        blocked.discard(start)
        future_centers = {wp for wp in waypoints[i + 2 : -1]}
        blocked |= future_centers
        path = _shortest_path(open_mask, start, goal, blocked, prefer=prefer)
        assert path is not None, "ordered door path must exist"
        segments.append(path)
        used.update(path[1:])

    out = scene.copy()
    for path in segments:
        for r, c in path[1:]:
            out[r, c] = ORANGE
    start, end = waypoints[0], waypoints[-1]
    out[start] = GREEN
    out[end] = GREEN
    return out, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue/red cross into a fence, open size-matched wall gaps, and draw the route through them.

    Input structure:
    - The background is black.
    - One blue/red cross component supplies a full colored row and a full colored column inside its tight crop.
    - Several yellow one-cell-thick hollow frames and two green marker pixels sit elsewhere on the same canvas.

    Rule:
    - Crop the cross and use its full-row and full-column positions as the periods for horizontal blue walls and vertical red walls across the whole grid.
    - For each yellow frame, compare its inner height and inner width with the repeated wall gaps: a height match opens a vertical-wall door on the nearest red wall, and a width match opens a horizontal-wall door on the nearest blue wall, using the frame's inner row or column span as the erased segment.
    - Remove the yellow frames, keep the two green markers, then draw the orthogonal orange route from the upper-left marker to the lower-right marker through the door centers in wall order.

    Key insight:
    - The cross does not directly show the final maze; it only encodes the two repetition periods, while the frame interior sizes decide which future wall segments turn into doors.
    """
    structure = _parse_structure(np.asarray(grid, dtype=int))
    out, _segments = _route_through_doors(structure)
    return out


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _stamp_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = YELLOW
    grid[r1, c0 : c1 + 1] = YELLOW
    grid[r0 : r1 + 1, c0] = YELLOW
    grid[r0 : r1 + 1, c1] = YELLOW


def _cross_sprite(step_h: int, step_w: int) -> tuple[np.ndarray, int, int]:
    sprite = np.zeros((step_h, step_w), dtype=int)
    rr = step_h // 2
    cc = step_w // 2
    sprite[rr, :] = BLUE
    sprite[:, cc] = RED
    sprite[rr, cc] = BLUE
    return sprite, rr, cc


def _candidate_vertical_boxes(
    wall_col: int,
    row_start: int,
    gap_h: int,
    grid_shape: tuple[int, int],
    target_cols: list[int],
) -> list[tuple[int, int, int, int]]:
    h, w = grid_shape
    boxes = []
    for inner_w in SAFE_OTHER_SIZES:
        outer_w = inner_w + 2
        for c0 in range(0, w - outer_w + 1):
            center = c0 + (outer_w - 1) / 2.0
            if _nearest_unique(center, target_cols) != wall_col:
                continue
            boxes.append((row_start - 1, c0, row_start + gap_h, c0 + outer_w - 1))
    random.shuffle(boxes)
    return boxes


def _candidate_horizontal_boxes(
    wall_row: int,
    col_start: int,
    gap_w: int,
    grid_shape: tuple[int, int],
    target_rows: list[int],
) -> list[tuple[int, int, int, int]]:
    h, w = grid_shape
    boxes = []
    for inner_h in SAFE_OTHER_SIZES:
        outer_h = inner_h + 2
        for r0 in range(0, h - outer_h + 1):
            center = r0 + (outer_h - 1) / 2.0
            if _nearest_unique(center, target_rows) != wall_row:
                continue
            boxes.append((r0, col_start - 1, r0 + outer_h - 1, col_start + gap_w))
    random.shuffle(boxes)
    return boxes


FAMILIES = (
    {
        "pair": (4, 6),
        "seed": (7, 5),
        "start": (0, 0),
        "end": (8, 16),
        "doors": (("V", 5, 1), ("H", 3, 7), ("V", 11, 6)),
    },
    {
        "pair": (4, 6),
        "seed": (3, 11),
        "start": (0, 0),
        "end": (8, 16),
        "doors": (("H", 3, 1), ("H", 7, 2), ("V", 11, 7)),
    },
)

_FAMILY_CACHE: dict[int, list[tuple[tuple[int, int, int, int], ...]]] = {}


def _build_family_grid(
    family: dict,
    boxes: tuple[tuple[int, int, int, int], ...],
    shift: tuple[int, int] = (0, 0),
) -> np.ndarray:
    step_h, step_w = family["pair"]
    dr, dc = shift
    height = 3 * step_h - 1 + dr
    width = 3 * step_w - 1 + dc
    sprite, rr, cc = _cross_sprite(step_h, step_w)
    seed_row, seed_col = family["seed"][0] + dr, family["seed"][1] + dc
    grid = np.zeros((height, width), dtype=int)
    grid[seed_row - rr : seed_row - rr + step_h, seed_col - cc : seed_col - cc + step_w] = sprite
    for box in boxes:
        shifted = (box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc)
        _stamp_frame(grid, shifted)
    start = (family["start"][0] + dr, family["start"][1] + dc)
    end = (family["end"][0] + dr, family["end"][1] + dc)
    grid[start] = GREEN
    grid[end] = GREEN
    return grid


def _family_variants(family_index: int) -> list[tuple[tuple[int, int, int, int], ...]]:
    if family_index in _FAMILY_CACHE:
        return _FAMILY_CACHE[family_index]

    family = FAMILIES[family_index]
    step_h, step_w = family["pair"]
    gap_h = step_h - 1
    gap_w = step_w - 1
    height = 3 * step_h - 1
    width = 3 * step_w - 1
    row_walls = [gap_h, 2 * step_h - 1]
    col_walls = [gap_w, 2 * step_w - 1]
    sprite, rr, cc = _cross_sprite(step_h, step_w)
    seed_row, seed_col = family["seed"]
    cross_box = (seed_row - rr, seed_col - cc, seed_row - rr + step_h - 1, seed_col - cc + step_w - 1)
    marker_boxes = [
        (family["start"][0], family["start"][1], family["start"][0], family["start"][1]),
        (family["end"][0], family["end"][1], family["end"][0], family["end"][1]),
    ]

    candidate_lists = []
    for kind, wall, start in family["doors"]:
        if kind == "V":
            candidate_lists.append(_candidate_vertical_boxes(wall, start, gap_h, (height, width), col_walls))
        else:
            candidate_lists.append(_candidate_horizontal_boxes(wall, start, gap_w, (height, width), row_walls))

    valid: list[tuple[tuple[int, int, int, int], ...]] = []
    for b0 in candidate_lists[0]:
        for b1 in candidate_lists[1]:
            for b2 in candidate_lists[2]:
                boxes = (b0, b1, b2)
                if len(set(boxes)) < 3:
                    continue
                if any(_rect_overlap(box, cross_box, pad=0) for box in boxes):
                    continue
                bad = False
                for box in boxes:
                    if any(_rect_overlap(box, marker, pad=0) for marker in marker_boxes):
                        bad = True
                        break
                if bad:
                    continue
                for i in range(3):
                    for j in range(i + 1, 3):
                        if _rect_overlap(boxes[i], boxes[j], pad=0):
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue

                grid = np.zeros((height, width), dtype=int)
                grid[seed_row - rr : seed_row - rr + step_h, seed_col - cc : seed_col - cc + step_w] = sprite
                for box in boxes:
                    _stamp_frame(grid, box)
                grid[family["start"]] = GREEN
                grid[family["end"]] = GREEN
                if _validate_generated_case(grid):
                    valid.append(boxes)
    assert valid, f"family {family_index} has no valid variants"
    _FAMILY_CACHE[family_index] = valid
    return valid


def _generate_case() -> np.ndarray | None:
    family_index = random.randrange(len(FAMILIES))
    variants = _family_variants(family_index)
    boxes = random.choice(variants)
    step_h, step_w = FAMILIES[family_index]["pair"]
    shift = (random.randint(0, step_h - 1), random.randint(0, step_w - 1))
    grid = _build_family_grid(FAMILIES[family_index], boxes, shift=shift)
    if _validate_generated_case(grid):
        return grid
    return None


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        structure = _parse_structure(grid)
        total_doors = len(structure["vertical_doors"]) + len(structure["horizontal_doors"])
        if total_doors != 3:
            return False
        if not structure["vertical_doors"] or not structure["horizontal_doors"]:
            return False

        if max(structure["grid"].shape) > 30:
            return False

        expected, segments = _route_through_doors(structure)
        if not np.array_equal(main(grid), expected):
            return False

        base = _build_wall_grid(grid.shape, structure["row_walls"], structure["col_walls"])
        open_scene = _apply_doors(base, structure["vertical_doors"], structure["horizontal_doors"])
        open_mask = open_scene == BLACK
        doors, waypoints = _ordered_waypoints(structure)

        viable_orders = 0
        for perm in itertools.permutations(doors):
            points = [waypoints[0]] + [door["center"] for door in perm] + [waypoints[-1]]
            used: set[tuple[int, int]] = set()
            ok = True
            for i in range(len(points) - 1):
                start = points[i]
                goal = points[i + 1]
                blocked = set(used)
                blocked.discard(start)
                blocked |= set(points[i + 2 : -1])
                prefer = "horizontal" if (i < len(perm) and perm[i]["kind"] == "V") else "vertical"
                path = _shortest_path(open_mask, start, goal, blocked, prefer)
                if path is None:
                    ok = False
                    break
                used.update(path[1:])
            if ok:
                viable_orders += 1
        if viable_orders != 1:
            return False

        for idx, door in enumerate(doors):
            closed = open_scene.copy()
            if door["kind"] == "V":
                closed[door["start"] : door["end"] + 1, door["wall"]] = RED
            else:
                closed[door["wall"], door["start"] : door["end"] + 1] = BLUE
            closed_mask = closed == BLACK
            used = set()
            ok = True
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                goal = waypoints[i + 1]
                blocked = set(used)
                blocked.discard(start)
                blocked |= set(waypoints[i + 2 : -1])
                prefer = "horizontal" if (i < len(doors) and doors[i]["kind"] == "V") else "vertical"
                path = _shortest_path(closed_mask, start, goal, blocked, prefer)
                if path is None:
                    ok = False
                    break
                used.update(path[1:])
            if ok:
                return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-door puzzle")
