from __future__ import annotations

import random
import numpy as np

BG = 0
WALL_COLOR = 5
MAROON = 9
TOWER_PALETTE = (1, 2, 3, 4, 6, 7, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = {
    "up": UP,
    "down": DOWN,
    "left": LEFT,
    "right": RIGHT,
}

GEN_RETRIES = 500


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
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _segment_from_coords(coords: np.ndarray) -> dict:
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    if len(rows) == 1:
        c0 = int(min(cols))
        c1 = int(max(cols))
        assert coords.shape[0] == c1 - c0 + 1 and coords.shape[0] >= 2
        return {
            "orientation": "horizontal",
            "row": int(rows[0]),
            "c0": c0,
            "c1": c1,
            "length": int(coords.shape[0]),
            "cells": [(int(rows[0]), c) for c in range(c0, c1 + 1)],
        }
    if len(cols) == 1:
        r0 = int(min(rows))
        r1 = int(max(rows))
        assert coords.shape[0] == r1 - r0 + 1 and coords.shape[0] >= 2
        return {
            "orientation": "vertical",
            "col": int(cols[0]),
            "r0": r0,
            "r1": r1,
            "length": int(coords.shape[0]),
            "cells": [(r, int(cols[0])) for r in range(r0, r1 + 1)],
        }
    raise AssertionError("wall components must be straight line segments")


def _infer_wall_color(grid: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    candidates: list[tuple[int, int]] = []
    for color in colors:
        if int(np.count_nonzero(grid == color)) <= 1:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        try:
            for comp in comps:
                _segment_from_coords(comp)
        except AssertionError:
            continue
        candidates.append((int(color), int(np.count_nonzero(grid == color))))
    if candidates:
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return int(candidates[0][0])
    raise AssertionError("failed to infer wall color")


def _extract_segments(grid: np.ndarray, wall_color: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    segments: list[dict] = []
    cell_to_segment: dict[tuple[int, int], int] = {}
    for comp in _components(grid == wall_color):
        seg = _segment_from_coords(comp)
        seg_idx = len(segments)
        segments.append(seg)
        for cell in seg["cells"]:
            cell_to_segment[(int(cell[0]), int(cell[1]))] = seg_idx
    assert segments, "expected at least one wall"
    return segments, cell_to_segment


def _infer_towers(grid: np.ndarray, wall_color: int) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BG) & (grid != wall_color)):
        towers.append((int(r), int(c), int(grid[int(r), int(c)])))
    assert 2 <= len(towers) <= 4, "expected 2-4 single-pixel towers"
    colors = [color for _r, _c, color in towers]
    assert len(set(colors)) == len(colors), "tower colors must be distinct"
    for _r, _c, color in towers:
        assert int(np.count_nonzero(grid == color)) == 1, "tower colors must appear exactly once"
    return towers


def _turn_from_hit(seg: dict, hit_r: int, hit_c: int) -> tuple[tuple[int, int], bool]:
    if seg["orientation"] == "vertical":
        center = (int(seg["r0"]) + int(seg["r1"])) / 2.0
        if hit_r < center:
            return UP, False
        if hit_r > center:
            return DOWN, False
        return UP, True
    center = (int(seg["c0"]) + int(seg["c1"])) / 2.0
    if hit_c < center:
        return LEFT, False
    if hit_c > center:
        return RIGHT, False
    return LEFT, True


def _trace_ray(
    grid: np.ndarray,
    wall_color: int,
    tower_mask: np.ndarray,
    segments: list[dict],
    cell_to_segment: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], int, dict[str, int]]:
    h, w = grid.shape
    path: list[tuple[int, int]] = []
    pos = (int(start[0]), int(start[1]))
    dr, dc = map(int, direction)
    turns = 0
    seen_states: set[tuple[int, int, int, int, int]] = set()
    stats = {
        "side_hits": 0,
        "endpoint_hits": 0,
        "boundary_hits": 0,
        "turn_cap_hits": 0,
        "center_ties": 0,
        "loops": 0,
    }

    while True:
        state = (int(pos[0]), int(pos[1]), int(dr), int(dc), int(turns))
        if state in seen_states:
            stats["loops"] += 1
            break
        seen_states.add(state)

        nr = int(pos[0] + dr)
        nc = int(pos[1] + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            stats["boundary_hits"] += 1
            break

        if int(grid[nr, nc]) == int(wall_color):
            seg = segments[cell_to_segment[(nr, nc)]]
            side_hit = (seg["orientation"] == "vertical" and dc != 0) or (
                seg["orientation"] == "horizontal" and dr != 0
            )
            if side_hit:
                if turns >= 2:
                    stats["turn_cap_hits"] += 1
                    break
                new_dir, tie = _turn_from_hit(seg, nr, nc)
                if tie:
                    stats["center_ties"] += 1
                stats["side_hits"] += 1
                dr, dc = map(int, new_dir)
                turns += 1
                continue
            stats["endpoint_hits"] += 1
            break

        pos = (nr, nc)
        if not bool(tower_mask[nr, nc]):
            path.append(pos)

    return path, turns, stats


def _solve_and_trace(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    wall_color = _infer_wall_color(g)
    segments, cell_to_segment = _extract_segments(g, wall_color)
    towers = _infer_towers(g, wall_color)
    tower_mask = np.zeros_like(g, dtype=bool)
    for r, c, _color in towers:
        tower_mask[int(r), int(c)] = True

    path_owner = np.zeros_like(g, dtype=int)
    stats = {
        "tower_count": len(towers),
        "wall_count": len(segments),
        "total_rays": 0,
        "side_hits": 0,
        "endpoint_hits": 0,
        "boundary_hits": 0,
        "turn_cap_hits": 0,
        "center_ties": 0,
        "loops": 0,
        "turn_twice_rays": 0,
        "painted_cells": 0,
        "cross_cells": 0,
    }

    for r, c, color in towers:
        for direction in (UP, DOWN, LEFT, RIGHT):
            path, turns, ray_stats = _trace_ray(
                g,
                wall_color,
                tower_mask,
                segments,
                cell_to_segment,
                (int(r), int(c)),
                direction,
            )
            stats["total_rays"] += 1
            if turns == 2:
                stats["turn_twice_rays"] += 1
            for key, value in ray_stats.items():
                stats[key] += int(value)
            for pr, pc in path:
                prev = int(path_owner[pr, pc])
                if prev == 0:
                    path_owner[pr, pc] = int(color)
                    stats["painted_cells"] += 1
                elif prev != int(color) and prev != MAROON:
                    path_owner[pr, pc] = MAROON

    out = g.copy()
    ray_mask = path_owner != 0
    out[ray_mask] = path_owner[ray_mask]
    stats["cross_cells"] = int(np.count_nonzero(path_owner == MAROON))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored tower's four rays through grey reflector segments.

    Rays turn only when they strike a wall from the side, terminate on endpoint
    impacts, keep their tower color on empty path cells, and mix to maroon when
    different colors overlap.
    """
    out, _stats = _solve_and_trace(grid)
    return out


def _sample_spaced_values(start: int, stop: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(start, stop))
    if len(values) < count:
        return None
    for _ in range(200):
        pool = values[:]
        random.shuffle(pool)
        chosen: list[int] = []
        for value in pool:
            if all(abs(int(value) - int(other)) >= min_gap for other in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return None


def _make_wall(orientation: str, fixed: int, start: int, length: int) -> dict:
    if orientation == "horizontal":
        c0 = int(start)
        c1 = int(start + length - 1)
        return {
            "orientation": "horizontal",
            "row": int(fixed),
            "c0": c0,
            "c1": c1,
            "length": int(length),
            "cells": [(int(fixed), c) for c in range(c0, c1 + 1)],
        }
    r0 = int(start)
    r1 = int(start + length - 1)
    return {
        "orientation": "vertical",
        "col": int(fixed),
        "r0": r0,
        "r1": r1,
        "length": int(length),
        "cells": [(r, int(fixed)) for r in range(r0, r1 + 1)],
    }


def _build_forbidden(h: int, w: int, towers: list[dict], walls: list[dict]) -> np.ndarray:
    forbidden = np.zeros((h, w), dtype=bool)
    cells: list[tuple[int, int]] = []
    for tower in towers:
        cells.append((int(tower["r"]), int(tower["c"])))
    for wall in walls:
        cells.extend((int(r), int(c)) for r, c in wall["cells"])
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True
    return forbidden


def _can_place_wall(h: int, w: int, towers: list[dict], walls: list[dict], wall: dict) -> bool:
    forbidden = _build_forbidden(h, w, towers, walls)
    for r, c in wall["cells"]:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        if bool(forbidden[int(r), int(c)]):
            return False
    return True


def _turn_from_offset(orientation: str, length: int, offset: int) -> str:
    center = (int(length) - 1) / 2.0
    if orientation == "vertical":
        return "up" if offset < center else "down"
    return "left" if offset < center else "right"


def _side_wall_candidates(h: int, w: int, towers: list[dict], walls: list[dict], tower: dict, direction: str) -> list[tuple[dict, str]]:
    r = int(tower["r"])
    c = int(tower["c"])
    candidates: list[tuple[dict, str]] = []
    lengths = list(range(3, 7))
    dists = list(range(2, 6))
    random.shuffle(lengths)
    random.shuffle(dists)

    for length in lengths:
        offsets = list(range(length))
        center = (length - 1) / 2.0
        offsets = [offset for offset in offsets if offset != center]
        random.shuffle(offsets)
        for dist in dists:
            if direction == "right":
                fixed = c + dist
                if not (1 <= fixed <= w - 2):
                    continue
                for offset in offsets:
                    start = r - offset
                    if start < 1 or start + length - 1 > h - 2:
                        continue
                    wall = _make_wall("vertical", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append((wall, _turn_from_offset("vertical", length, offset)))
            elif direction == "left":
                fixed = c - dist
                if not (1 <= fixed <= w - 2):
                    continue
                for offset in offsets:
                    start = r - offset
                    if start < 1 or start + length - 1 > h - 2:
                        continue
                    wall = _make_wall("vertical", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append((wall, _turn_from_offset("vertical", length, offset)))
            elif direction == "down":
                fixed = r + dist
                if not (1 <= fixed <= h - 2):
                    continue
                for offset in offsets:
                    start = c - offset
                    if start < 1 or start + length - 1 > w - 2:
                        continue
                    wall = _make_wall("horizontal", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append((wall, _turn_from_offset("horizontal", length, offset)))
            else:
                fixed = r - dist
                if not (1 <= fixed <= h - 2):
                    continue
                for offset in offsets:
                    start = c - offset
                    if start < 1 or start + length - 1 > w - 2:
                        continue
                    wall = _make_wall("horizontal", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append((wall, _turn_from_offset("horizontal", length, offset)))
    random.shuffle(candidates)
    return candidates


def _pre_collision_cell(tower: dict, direction: str, wall: dict) -> tuple[int, int]:
    r = int(tower["r"])
    c = int(tower["c"])
    if direction == "right":
        return (r, int(wall["col"]) - 1)
    if direction == "left":
        return (r, int(wall["col"]) + 1)
    if direction == "down":
        return (int(wall["row"]) - 1, c)
    return (int(wall["row"]) + 1, c)


def _desired_second_turn(first_wall: dict, start_cell: tuple[int, int]) -> str:
    sr, sc = map(int, start_cell)
    if first_wall["orientation"] == "vertical":
        return "left" if sc < int(first_wall["col"]) else "right"
    return "up" if sr < int(first_wall["row"]) else "down"


def _second_side_wall_candidates(
    h: int,
    w: int,
    towers: list[dict],
    walls: list[dict],
    start_cell: tuple[int, int],
    travel_dir: str,
    desired_turn: str,
) -> list[dict]:
    sr, sc = map(int, start_cell)
    candidates: list[dict] = []
    lengths = list(range(3, 7))
    dists = list(range(2, 6))
    random.shuffle(lengths)
    random.shuffle(dists)

    for length in lengths:
        offsets = list(range(length))
        center = (length - 1) / 2.0
        offsets = [offset for offset in offsets if offset != center]
        random.shuffle(offsets)
        for dist in dists:
            if travel_dir == "up":
                fixed = sr - dist
                if not (1 <= fixed <= h - 2):
                    continue
                for offset in offsets:
                    if _turn_from_offset("horizontal", length, offset) != desired_turn:
                        continue
                    start = sc - offset
                    if start < 1 or start + length - 1 > w - 2:
                        continue
                    wall = _make_wall("horizontal", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append(wall)
            elif travel_dir == "down":
                fixed = sr + dist
                if not (1 <= fixed <= h - 2):
                    continue
                for offset in offsets:
                    if _turn_from_offset("horizontal", length, offset) != desired_turn:
                        continue
                    start = sc - offset
                    if start < 1 or start + length - 1 > w - 2:
                        continue
                    wall = _make_wall("horizontal", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append(wall)
            elif travel_dir == "left":
                fixed = sc - dist
                if not (1 <= fixed <= w - 2):
                    continue
                for offset in offsets:
                    if _turn_from_offset("vertical", length, offset) != desired_turn:
                        continue
                    start = sr - offset
                    if start < 1 or start + length - 1 > h - 2:
                        continue
                    wall = _make_wall("vertical", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append(wall)
            else:
                fixed = sc + dist
                if not (1 <= fixed <= w - 2):
                    continue
                for offset in offsets:
                    if _turn_from_offset("vertical", length, offset) != desired_turn:
                        continue
                    start = sr - offset
                    if start < 1 or start + length - 1 > h - 2:
                        continue
                    wall = _make_wall("vertical", fixed, start, length)
                    if _can_place_wall(h, w, towers, walls, wall):
                        candidates.append(wall)
    random.shuffle(candidates)
    return candidates


def _endpoint_wall_candidates(h: int, w: int, towers: list[dict], walls: list[dict], tower: dict, direction: str) -> list[dict]:
    r = int(tower["r"])
    c = int(tower["c"])
    candidates: list[dict] = []
    lengths = list(range(3, 7))
    dists = list(range(2, 6))
    random.shuffle(lengths)
    random.shuffle(dists)

    for length in lengths:
        for dist in dists:
            if direction == "right":
                start = c + dist
                if start + length - 1 > w - 2:
                    continue
                wall = _make_wall("horizontal", r, start, length)
            elif direction == "left":
                end = c - dist
                start = end - length + 1
                if start < 1:
                    continue
                wall = _make_wall("horizontal", r, start, length)
            elif direction == "down":
                start = r + dist
                if start + length - 1 > h - 2:
                    continue
                wall = _make_wall("vertical", c, start, length)
            else:
                end = r - dist
                start = end - length + 1
                if start < 1:
                    continue
                wall = _make_wall("vertical", c, start, length)
            if _can_place_wall(h, w, towers, walls, wall):
                candidates.append(wall)
    random.shuffle(candidates)
    return candidates


def _random_wall(h: int, w: int, towers: list[dict], walls: list[dict]) -> dict | None:
    for _ in range(80):
        orientation = random.choice(("horizontal", "vertical"))
        length = random.randint(3, 6)
        if orientation == "horizontal":
            row = random.randint(1, h - 2)
            start = random.randint(1, w - length - 1)
            wall = _make_wall("horizontal", row, start, length)
        else:
            col = random.randint(1, w - 2)
            start = random.randint(1, h - length - 1)
            wall = _make_wall("vertical", col, start, length)
        if _can_place_wall(h, w, towers, walls, wall):
            return wall
    return None


def _render_grid(h: int, w: int, towers: list[dict], walls: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for wall in walls:
        for r, c in wall["cells"]:
            grid[int(r), int(c)] = WALL_COLOR
    for tower in towers:
        grid[int(tower["r"]), int(tower["c"])] = int(tower["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    tower_count = random.choices([2, 3, 4], weights=[4, 4, 2])[0]

    rows = _sample_spaced_values(2, h - 2, tower_count, 2)
    cols = _sample_spaced_values(2, w - 2, tower_count, 2)
    if rows is None or cols is None:
        return None

    colors = random.sample(TOWER_PALETTE, tower_count)
    random.shuffle(cols)
    towers = [
        {"r": int(rows[i]), "c": int(cols[i]), "color": int(colors[i])}
        for i in range(tower_count)
    ]
    random.shuffle(towers)

    want_double = random.random() < 0.5
    need_cross = random.random() < 0.65

    walls: list[dict] = []

    side_options = [(idx, direction) for idx in range(tower_count) for direction in DIRS]
    random.shuffle(side_options)
    side_setup_ok = False
    for idx, direction in side_options:
        tower = towers[idx]
        for first_wall, turn1 in _side_wall_candidates(h, w, towers, walls, tower, direction):
            if want_double:
                start_cell = _pre_collision_cell(tower, direction, first_wall)
                desired_turn = _desired_second_turn(first_wall, start_cell)
                second_candidates = _second_side_wall_candidates(
                    h,
                    w,
                    towers,
                    walls + [first_wall],
                    start_cell,
                    turn1,
                    desired_turn,
                )
                if not second_candidates:
                    continue
                walls.extend([first_wall, random.choice(second_candidates)])
            else:
                walls.append(first_wall)
            side_setup_ok = True
            break
        if side_setup_ok:
            break
    if not side_setup_ok:
        return None

    endpoint_indices = list(range(tower_count))
    random.shuffle(endpoint_indices)
    endpoint_ok = False
    for idx in endpoint_indices:
        tower = towers[idx]
        directions = list(DIRS)
        random.shuffle(directions)
        for direction in directions:
            candidates = _endpoint_wall_candidates(h, w, towers, walls, tower, direction)
            if candidates:
                walls.append(random.choice(candidates))
                endpoint_ok = True
                break
        if endpoint_ok:
            break
    if not endpoint_ok:
        return None

    extra_walls = random.choices([0, 1], weights=[3, 2])[0]
    for _ in range(extra_walls):
        extra = _random_wall(h, w, towers, walls)
        if extra is not None:
            walls.append(extra)

    if not (2 <= len(walls) <= 4):
        return None

    grid = _render_grid(h, w, towers, walls)
    out, stats = _solve_and_trace(grid)

    if stats["center_ties"] != 0 or stats["turn_cap_hits"] != 0 or stats["loops"] != 0:
        return None
    if stats["side_hits"] < 1 or stats["endpoint_hits"] < 1 or stats["boundary_hits"] < 1:
        return None
    if want_double and stats["turn_twice_rays"] < 1:
        return None
    if need_cross and stats["cross_cells"] < 1:
        return None
    if stats["painted_cells"] < 12:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflector-ray input")
