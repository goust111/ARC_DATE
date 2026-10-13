from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400
SCENARIO_RETRIES = 180
PAIR_RETRIES = 220
EDGE_RETRIES = 220
RANDOM_TOWER_RETRIES = 180
MIN_RAY_LEN = 2
HALO = 1


def _body_anchor(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    if len(cells) != 4:
        return None
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) != 2 or len(cols) != 2:
        return None
    if rows[1] != rows[0] + 1 or cols[1] != cols[0] + 1:
        return None
    expected = {
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    }
    return (rows[0], cols[0]) if cells == expected else None


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    H, W = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, grid.shape):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _parse_tower(color: int, coords: list[tuple[int, int]]) -> dict:
    cells = {tuple(map(int, cell)) for cell in coords}
    assert len(cells) == 5, "each tower component must contain a 2x2 body plus one protrusion"
    candidates: list[dict] = []
    for pr, pc in cells:
        body_cells = cells - {(pr, pc)}
        anchor = _body_anchor(body_cells)
        if anchor is None:
            continue
        r0, c0 = anchor
        direction = None
        offset = None
        if pr == r0 - 1 and pc in (c0, c0 + 1):
            direction = "up"
            offset = pc - c0
        elif pr == r0 + 2 and pc in (c0, c0 + 1):
            direction = "down"
            offset = pc - c0
        elif pc == c0 - 1 and pr in (r0, r0 + 1):
            direction = "left"
            offset = pr - r0
        elif pc == c0 + 2 and pr in (r0, r0 + 1):
            direction = "right"
            offset = pr - r0
        if direction is None:
            continue
        candidates.append(
            {
                "color": int(color),
                "body_r": int(r0),
                "body_c": int(c0),
                "direction": direction,
                "offset": int(offset),
                "protrusion": (int(pr), int(pc)),
                "vector": DIR_VECS[direction],
                "cells": sorted(cells),
            }
        )
    assert len(candidates) == 1, "each foreground component must be one directional tower"
    return candidates[0]


def _parse_towers(grid: np.ndarray) -> list[dict]:
    towers = [_parse_tower(color, coords) for color, coords in _components(grid)]
    assert towers, "expected at least one tower"
    towers.sort(key=lambda tower: (tower["protrusion"][0], tower["protrusion"][1], tower["color"]))
    return towers


def _trace_rays(grid: np.ndarray, towers: list[dict]) -> tuple[dict[tuple[int, int], list[int]], list[dict]]:
    H, W = grid.shape
    occupied = grid != BLACK
    ray_hits: dict[tuple[int, int], list[int]] = {}
    ray_infos: list[dict] = []
    for tower in towers:
        pr, pc = tower["protrusion"]
        dr, dc = tower["vector"]
        color = int(tower["color"])
        r = pr + dr
        c = pc + dc
        path: list[tuple[int, int]] = []
        while 0 <= r < H and 0 <= c < W:
            if bool(occupied[r, c]):
                ray_infos.append(
                    {
                        "color": color,
                        "path": path,
                        "stop": "tower",
                    }
                )
                break
            path.append((r, c))
            ray_hits.setdefault((r, c), []).append(color)
            r += dr
            c += dc
        else:
            ray_infos.append(
                {
                    "color": color,
                    "path": path,
                    "stop": "edge",
                }
            )
    return ray_hits, ray_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each tower's colored ray in the direction marked by its protrusion, turning different-color crossings green.

    Input structure:
    - The background is black.
    - Every non-black 4-connected component is a same-color 2x2 solid tower body with one adjacent same-color protrusion pixel on exactly one cardinal side.
    - Different towers may reuse colors, and the protrusion pixel indicates whether the tower emits upward, downward, leftward, or rightward.

    Rule:
    - Keep every tower body and protrusion pixel unchanged.
    - From each protrusion, cast a same-color ray one cell at a time in the protrusion's direction through black cells only.
    - Stop a ray just before it would leave the grid or hit any pixel belonging to another tower.
    - Color each traversed cell with the ray color unless multiple different ray colors reach that cell, in which case color that cell green; overlapping same-color rays stay in that color.

    Key insight:
    - The lone extra pixel on each 2x2 block is not decoration: it uniquely encodes a cardinal launch direction, so the task reduces to parsing each tower shape and simulating all rays simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    towers = _parse_towers(g)
    ray_hits, _ = _trace_rays(g, towers)
    out = g.copy()
    for (r, c), colors in ray_hits.items():
        unique = set(colors)
        out[r, c] = GREEN if len(unique) > 1 else next(iter(unique))
    return out


def _tower_from_body(color: int, body_r: int, body_c: int, direction: str, offset: int) -> dict:
    body = [
        (body_r, body_c),
        (body_r, body_c + 1),
        (body_r + 1, body_c),
        (body_r + 1, body_c + 1),
    ]
    if direction == "up":
        protrusion = (body_r - 1, body_c + offset)
    elif direction == "down":
        protrusion = (body_r + 2, body_c + offset)
    elif direction == "left":
        protrusion = (body_r + offset, body_c - 1)
    else:
        assert direction == "right"
        protrusion = (body_r + offset, body_c + 2)
    cells = body + [protrusion]
    return {
        "color": int(color),
        "body_r": int(body_r),
        "body_c": int(body_c),
        "direction": direction,
        "offset": int(offset),
        "protrusion": (int(protrusion[0]), int(protrusion[1])),
        "vector": DIR_VECS[direction],
        "cells": [(int(r), int(c)) for r, c in cells],
    }


def _tower_from_protrusion(color: int, pr: int, pc: int, direction: str, offset: int) -> dict:
    if direction == "up":
        body_r = pr + 1
        body_c = pc - offset
    elif direction == "down":
        body_r = pr - 2
        body_c = pc - offset
    elif direction == "left":
        body_r = pr - offset
        body_c = pc + 1
    else:
        assert direction == "right"
        body_r = pr - offset
        body_c = pc - 2
    return _tower_from_body(color, body_r, body_c, direction, offset)


def _random_color(exclude: set[int] | None = None) -> int:
    banned = set() if exclude is None else set(map(int, exclude))
    choices = [color for color in PALETTE if color not in banned]
    return int(random.choice(choices))


def _in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    H, W = shape
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _can_place(shape: tuple[int, int], occupied: set[tuple[int, int]], cells: list[tuple[int, int]]) -> bool:
    if len(set(cells)) != len(cells) or not _in_bounds(shape, cells):
        return False
    H, W = shape
    for r, c in cells:
        for rr in range(r - HALO, r + HALO + 1):
            for cc in range(c - HALO, c + HALO + 1):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in occupied:
                    return False
    return True


def _ray_preview(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    tower: dict,
) -> tuple[int, str]:
    H, W = shape
    blocked = set(occupied)
    for cell in tower["cells"]:
        blocked.discard(cell)
    pr, pc = tower["protrusion"]
    dr, dc = tower["vector"]
    r = pr + dr
    c = pc + dc
    steps = 0
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in blocked:
            return steps, "tower"
        steps += 1
        r += dr
        c += dc
    return steps, "edge"


def _commit_candidates(
    shape: tuple[int, int],
    towers: list[dict],
    occupied: set[tuple[int, int]],
    candidates: list[dict],
) -> bool:
    trial = set(occupied)
    for tower in candidates:
        if not _can_place(shape, trial, tower["cells"]):
            return False
        trial.update(tower["cells"])
    for tower in candidates:
        steps, _ = _ray_preview(shape, trial, tower)
        if steps < MIN_RAY_LEN:
            return False
    towers.extend(candidates)
    occupied.clear()
    occupied.update(trial)
    return True


def _sample_range(lo: int, hi: int) -> int | None:
    return None if lo > hi else random.randint(lo, hi)


def _add_cross_pair(
    shape: tuple[int, int],
    towers: list[dict],
    occupied: set[tuple[int, int]],
    colors: tuple[int, int],
    row_bounds: tuple[int, int] | None = None,
    col_bounds: tuple[int, int] | None = None,
) -> bool:
    H, W = shape
    r_lo = 4 if row_bounds is None else max(4, int(row_bounds[0]))
    r_hi = H - 5 if row_bounds is None else min(H - 5, int(row_bounds[1]))
    c_lo = 4 if col_bounds is None else max(4, int(col_bounds[0]))
    c_hi = W - 5 if col_bounds is None else min(W - 5, int(col_bounds[1]))
    if r_lo > r_hi or c_lo > c_hi:
        return False

    for _ in range(PAIR_RETRIES):
        ir = random.randint(r_lo, r_hi)
        ic = random.randint(c_lo, c_hi)
        dir_h = random.choice(("left", "right"))
        dir_v = random.choice(("up", "down"))
        max_dh = ic - 2 if dir_h == "right" else W - 3 - ic
        max_dv = ir - 2 if dir_v == "down" else H - 3 - ir
        if max_dh < MIN_RAY_LEN or max_dv < MIN_RAY_LEN:
            continue
        dh = random.randint(MIN_RAY_LEN, min(7, max_dh))
        dv = random.randint(MIN_RAY_LEN, min(7, max_dv))
        if dir_h == "right":
            pr_h, pc_h = ir, ic - dh
        else:
            pr_h, pc_h = ir, ic + dh
        if dir_v == "down":
            pr_v, pc_v = ir - dv, ic
        else:
            pr_v, pc_v = ir + dv, ic
        tower_h = _tower_from_protrusion(colors[0], pr_h, pc_h, dir_h, random.randint(0, 1))
        tower_v = _tower_from_protrusion(colors[1], pr_v, pc_v, dir_v, random.randint(0, 1))
        if _commit_candidates(shape, towers, occupied, [tower_h, tower_v]):
            return True
    return False


def _add_blocker_pair(
    shape: tuple[int, int],
    towers: list[dict],
    occupied: set[tuple[int, int]],
    colors: tuple[int, int],
) -> bool:
    H, W = shape
    for _ in range(PAIR_RETRIES):
        emitter_dir = random.choice(tuple(DIR_VECS))
        emitter_color, blocker_color = colors
        offset_e = random.randint(0, 1)

        if emitter_dir == "right":
            pr = _sample_range(2, H - 3)
            pc = _sample_range(2, W - 6)
            if pr is None or pc is None:
                continue
            dist = random.randint(3, min(7, W - 2 - pc))
            align = random.randint(0, 1)
            body_r_b = pr - align
            body_c_b = pc + dist
            blocker_dirs = ("up", "down")
        elif emitter_dir == "left":
            pr = _sample_range(2, H - 3)
            pc = _sample_range(4, W - 3)
            if pr is None or pc is None:
                continue
            dist = random.randint(3, min(7, pc - 1))
            align = random.randint(0, 1)
            body_r_b = pr - align
            body_c_b = pc - dist - 1
            blocker_dirs = ("up", "down")
        elif emitter_dir == "down":
            pr = _sample_range(2, H - 6)
            pc = _sample_range(2, W - 3)
            if pr is None or pc is None:
                continue
            dist = random.randint(3, min(7, H - 2 - pr))
            align = random.randint(0, 1)
            body_r_b = pr + dist
            body_c_b = pc - align
            blocker_dirs = ("left", "right")
        else:
            assert emitter_dir == "up"
            pr = _sample_range(4, H - 3)
            pc = _sample_range(2, W - 3)
            if pr is None or pc is None:
                continue
            dist = random.randint(3, min(7, pr - 1))
            align = random.randint(0, 1)
            body_r_b = pr - dist - 1
            body_c_b = pc - align
            blocker_dirs = ("left", "right")

        emitter = _tower_from_protrusion(emitter_color, pr, pc, emitter_dir, offset_e)
        random_dirs = list(blocker_dirs)
        random.shuffle(random_dirs)
        candidate_blockers = [
            _tower_from_body(blocker_color, body_r_b, body_c_b, blocker_dir, random.randint(0, 1))
            for blocker_dir in random_dirs
        ]
        for blocker in candidate_blockers:
            trial = set(occupied)
            if not _can_place(shape, trial, emitter["cells"]):
                continue
            trial.update(emitter["cells"])
            if not _can_place(shape, trial, blocker["cells"]):
                continue
            trial.update(blocker["cells"])
            emitter_steps, emitter_stop = _ray_preview(shape, trial, emitter)
            blocker_steps, _ = _ray_preview(shape, trial, blocker)
            if emitter_steps < MIN_RAY_LEN or blocker_steps < MIN_RAY_LEN:
                continue
            if emitter_stop != "tower":
                continue
            towers.extend([emitter, blocker])
            occupied.clear()
            occupied.update(trial)
            return True
    return False


def _add_random_tower(
    shape: tuple[int, int],
    towers: list[dict],
    occupied: set[tuple[int, int]],
    color: int | None = None,
) -> bool:
    H, W = shape
    for _ in range(RANDOM_TOWER_RETRIES):
        direction = random.choice(tuple(DIR_VECS))
        offset = random.randint(0, 1)
        tower_color = _random_color() if color is None else int(color)
        if direction == "up":
            body_r = random.randint(1, H - 3)
            body_c = random.randint(0, W - 2)
        elif direction == "down":
            body_r = random.randint(0, H - 4)
            body_c = random.randint(0, W - 2)
        elif direction == "left":
            body_r = random.randint(0, H - 2)
            body_c = random.randint(1, W - 3)
        else:
            body_r = random.randint(0, H - 2)
            body_c = random.randint(0, W - 4)
        tower = _tower_from_body(tower_color, body_r, body_c, direction, offset)
        if not _can_place(shape, occupied, tower["cells"]):
            continue
        trial = set(occupied)
        trial.update(tower["cells"])
        steps, _ = _ray_preview(shape, trial, tower)
        if steps < MIN_RAY_LEN:
            continue
        towers.append(tower)
        occupied.clear()
        occupied.update(trial)
        return True
    return False


def _add_edge_tower(
    shape: tuple[int, int],
    towers: list[dict],
    occupied: set[tuple[int, int]],
    color: int | None = None,
) -> bool:
    H, W = shape
    for _ in range(EDGE_RETRIES):
        direction = random.choice(tuple(DIR_VECS))
        tower_color = _random_color() if color is None else int(color)
        offset = random.randint(0, 1)
        if direction == "up":
            pr = random.randint(2, H - 2)
            pc = random.randint(0, W - 1)
        elif direction == "down":
            pr = random.randint(0, H - 3)
            pc = random.randint(0, W - 1)
        elif direction == "left":
            pr = random.randint(0, H - 1)
            pc = random.randint(2, W - 2)
        else:
            pr = random.randint(0, H - 1)
            pc = random.randint(0, W - 3)
        tower = _tower_from_protrusion(tower_color, pr, pc, direction, offset)
        if not _can_place(shape, occupied, tower["cells"]):
            continue
        trial = set(occupied)
        trial.update(tower["cells"])
        steps, stop = _ray_preview(shape, trial, tower)
        if steps < MIN_RAY_LEN or stop != "edge":
            continue
        towers.append(tower)
        occupied.clear()
        occupied.update(trial)
        return True
    return False


def _paint_input(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for tower in towers:
        for r, c in tower["cells"]:
            grid[r, c] = int(tower["color"])
    return grid


def _mode_shape(mode: str) -> tuple[int, int]:
    if mode == "both_cross":
        return random.randint(21, 28), random.randint(21, 28)
    if mode == "no_cross":
        return random.randint(14, 22), random.randint(14, 22)
    return random.randint(16, 26), random.randint(16, 26)


def _build_input(mode: str) -> np.ndarray | None:
    H, W = _mode_shape(mode)
    shape = (H, W)
    towers: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    mid_r = H // 2
    mid_c = W // 2

    if mode == "diff_cross":
        a = _random_color()
        b = _random_color({a})
        if not _add_cross_pair(shape, towers, occupied, (a, b), (4, mid_r), (4, mid_c)):
            return None
        if not _add_blocker_pair(shape, towers, occupied, (_random_color(), _random_color())):
            return None
        if not _add_edge_tower(shape, towers, occupied):
            return None
    elif mode == "same_cross":
        color = _random_color()
        if not _add_cross_pair(shape, towers, occupied, (color, color), (4, mid_r), (4, mid_c)):
            return None
        if not _add_blocker_pair(shape, towers, occupied, (color, color)):
            return None
        if not _add_edge_tower(shape, towers, occupied, color=color):
            return None
    elif mode == "both_cross":
        a = _random_color()
        b = _random_color({a})
        c = _random_color({a, b})
        if not _add_cross_pair(shape, towers, occupied, (a, b), (4, mid_r - 1), (4, mid_c - 1)):
            return None
        if not _add_cross_pair(shape, towers, occupied, (c, c), (mid_r, H - 5), (mid_c, W - 5)):
            return None
        if not _add_blocker_pair(shape, towers, occupied, (_random_color(), _random_color())):
            return None
        if not _add_edge_tower(shape, towers, occupied):
            return None
    else:
        assert mode == "no_cross"
        if not _add_blocker_pair(shape, towers, occupied, (_random_color(), _random_color())):
            return None
        if random.random() < 0.5 and not _add_random_tower(shape, towers, occupied):
            return None
        if not _add_edge_tower(shape, towers, occupied):
            return None

    return _paint_input(shape, towers)


def _validate_mode(mode: str, grid: np.ndarray, out: np.ndarray) -> bool:
    towers = _parse_towers(grid)
    ray_hits, ray_infos = _trace_rays(grid, towers)
    if any(len(info["path"]) < MIN_RAY_LEN for info in ray_infos):
        return False
    if not any(info["stop"] == "edge" for info in ray_infos):
        return False
    if not any(info["stop"] == "tower" for info in ray_infos):
        return False

    diff_overlap = any(len(set(colors)) > 1 for colors in ray_hits.values())
    same_overlap = any(len(colors) > 1 and len(set(colors)) == 1 for colors in ray_hits.values())

    if mode == "diff_cross" and not diff_overlap:
        return False
    if mode == "same_cross" and (not same_overlap or diff_overlap):
        return False
    if mode == "both_cross" and not (diff_overlap and same_overlap):
        return False
    if mode == "no_cross" and (diff_overlap or same_overlap):
        return False

    if GREEN in set(np.unique(grid).tolist()):
        return False

    expected = grid.copy()
    for (r, c), colors in ray_hits.items():
        unique = set(colors)
        expected[r, c] = GREEN if len(unique) > 1 else next(iter(unique))
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    modes = ("diff_cross", "same_cross", "both_cross", "no_cross")
    for _ in range(GEN_RETRIES):
        mode = random.choice(modes)
        for _ in range(SCENARIO_RETRIES):
            candidate = _build_input(mode)
            if candidate is None:
                continue
            try:
                output = main(candidate)
            except AssertionError:
                continue
            if _validate_mode(mode, candidate, output):
                return candidate
    raise RuntimeError("generate_input: failed to build a valid tower-ray puzzle")
