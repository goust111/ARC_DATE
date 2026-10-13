import random
from collections import Counter

import numpy as np


BG = 8
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
START_FROM_SIDE = {
    "top": "D",
    "bottom": "U",
    "left": "R",
    "right": "L",
}
TURN_OPTIONS = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 80


def _dominant_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _edge_cells(shape: tuple[int, int]) -> list[tuple[int, int, str]]:
    h, w = shape
    cells: list[tuple[int, int, str]] = []
    for c in range(w):
        cells.append((0, c, "top"))
        cells.append((h - 1, c, "bottom"))
    for r in range(1, h - 1):
        cells.append((r, 0, "left"))
        cells.append((r, w - 1, "right"))
    return cells


def _light_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            vals, cnts = np.unique(block, return_counts=True)
            if len(vals) != 2 or sorted(map(int, cnts)) != [1, 3]:
                continue
            if int(vals[np.argmax(cnts)]) == bg:
                continue
            signal_idx = np.argwhere(block != int(vals[np.argmax(cnts)]))
            sr, sc = map(int, signal_idx[0])
            candidates.append(
                {
                    "top": r,
                    "left": c,
                    "body": int(vals[np.argmax(cnts)]),
                    "signal": int(vals[np.argmin(cnts)]),
                    "signal_rc": (r + sr, c + sc),
                    "cells": {(r + dr, c + dc) for dr in range(2) for dc in range(2)},
                }
            )
    used: set[tuple[int, int]] = set()
    lights: list[dict] = []
    for cand in candidates:
        if cand["cells"] & used:
            continue
        lights.append(cand)
        used |= cand["cells"]
    return lights


def _turn_from_light(light: dict, direction: str) -> tuple[tuple[int, int], str]:
    r = int(light["top"])
    c = int(light["left"])
    sr, sc = light["signal_rc"]
    if direction == "R":
        assert sc == c + 1
        if sr == r:
            return (r - 1, c + 1), "U"
        return (r + 2, c + 1), "D"
    if direction == "L":
        assert sc == c
        if sr == r:
            return (r - 1, c), "U"
        return (r + 2, c), "D"
    if direction == "D":
        assert sr == r + 1
        if sc == c:
            return (r + 1, c - 1), "L"
        return (r + 1, c + 2), "R"
    assert direction == "U"
    assert sr == r
    if sc == c:
        return (r, c - 1), "L"
    return (r, c + 2), "R"


def main(grid: np.ndarray) -> np.ndarray:
    """Trace edge-launched paths that turn at 2x2 traffic lights."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _dominant_color(g)

    lights = _light_candidates(g, bg)
    light_cells = {}
    body_colors = set()
    for idx, light in enumerate(lights):
        body_colors.add(int(light["body"]))
        for cell in light["cells"]:
            light_cells[cell] = idx

    starts: list[tuple[int, int, int, str]] = []
    for r, c, side in _edge_cells(g.shape):
        if (r, c) in light_cells:
            continue
        color = int(g[r, c])
        if color == bg or color in body_colors:
            continue
        starts.append((r, c, color, side))

    start_colors = [color for _r, _c, color, _side in starts]
    assert len(start_colors) == len(set(start_colors))

    out = g.copy()
    for r0, c0, color, side in starts:
        direction = START_FROM_SIDE[side]
        r, c = r0, c0
        seen: set[tuple[int, int, str]] = set()
        while True:
            state = (r, c, direction)
            assert state not in seen, "path must not cycle"
            seen.add(state)

            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if (nr, nc) in light_cells:
                next_pos, direction = _turn_from_light(lights[light_cells[(nr, nc)]], direction)
                r, c = next_pos
                if not (0 <= r < h and 0 <= c < w):
                    break
                if int(out[r, c]) == bg:
                    out[r, c] = color
                continue

            r, c = nr, nc
            if int(out[r, c]) == bg:
                out[r, c] = color

    return out


def _boundary_distance(h: int, w: int, pos: tuple[int, int], direction: str) -> int:
    r, c = pos
    if direction == "U":
        return r
    if direction == "D":
        return h - 1 - r
    if direction == "L":
        return c
    return w - 1 - c


def _line_cells(start: tuple[int, int], direction: str, steps: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _turn_geometry(
    stop: tuple[int, int], direction: str, turn_to: str, align: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    r, c = stop
    if direction == "R":
        top = r - align
        left = c + 1
        if turn_to == "U":
            signal = (top, left + 1)
            exit_pos = (top - 1, left + 1)
        else:
            signal = (top + 1, left + 1)
            exit_pos = (top + 2, left + 1)
        return (top, left), signal, exit_pos
    if direction == "L":
        top = r - align
        left = c - 2
        if turn_to == "U":
            signal = (top, left)
            exit_pos = (top - 1, left)
        else:
            signal = (top + 1, left)
            exit_pos = (top + 2, left)
        return (top, left), signal, exit_pos
    if direction == "D":
        top = r + 1
        left = c - align
        if turn_to == "L":
            signal = (top + 1, left)
            exit_pos = (top + 1, left - 1)
        else:
            signal = (top + 1, left + 1)
            exit_pos = (top + 1, left + 2)
        return (top, left), signal, exit_pos
    top = r - 2
    left = c - align
    if turn_to == "L":
        signal = (top, left)
        exit_pos = (top, left - 1)
    else:
        signal = (top, left + 1)
        exit_pos = (top, left + 2)
    return (top, left), signal, exit_pos


def _halo_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in range(-1, 3) for dc in range(-1, 3)}


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _build_path(
    h: int,
    w: int,
    start: tuple[int, int],
    direction: str,
    turns_target: int,
    path_occ: set[tuple[int, int]],
    input_occ: set[tuple[int, int]],
    light_halo_occ: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    current = start
    current_dir = direction
    path_cells = [start]
    lights: list[dict] = []

    for turn_idx in range(turns_target):
        max_steps = _boundary_distance(h, w, current, current_dir)
        candidates = []
        if max_steps < 3:
            return None
        distances = list(range(1, max_steps))
        rng.shuffle(distances)
        for steps in distances:
            seg = _line_cells(current, current_dir, steps)
            if any(cell in path_occ or cell in input_occ for cell in seg):
                continue
            stop = seg[-1]
            turns = list(TURN_OPTIONS[current_dir])
            rng.shuffle(turns)
            for turn_to in turns:
                aligns = [0, 1]
                rng.shuffle(aligns)
                for align in aligns:
                    geom = _turn_geometry(stop, current_dir, turn_to, align)
                    if geom is None:
                        continue
                    (top, left), signal, exit_pos = geom
                    light_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
                    halo = _halo_cells(top, left)
                    if not all(_in_bounds((h, w), cell) for cell in light_cells):
                        continue
                    if not _in_bounds((h, w), exit_pos):
                        continue
                    if any(cell in path_occ for cell in light_cells):
                        continue
                    if any(cell in input_occ for cell in light_cells):
                        continue
                    if exit_pos in path_occ or exit_pos in input_occ or exit_pos in light_cells:
                        continue
                    if any(_in_bounds((h, w), cell) and cell in light_halo_occ for cell in halo):
                        continue
                    if turn_idx < turns_target - 1 and _boundary_distance(h, w, exit_pos, turn_to) < 3:
                        continue
                    candidates.append(
                        {
                            "seg": seg,
                            "top_left": (top, left),
                            "signal": signal,
                            "exit_pos": exit_pos,
                            "next_dir": turn_to,
                            "light_cells": light_cells,
                            "halo": {cell for cell in halo if _in_bounds((h, w), cell)},
                        }
                    )
        if not candidates:
            return None
        chosen = rng.choice(candidates)
        path_cells.extend(chosen["seg"])
        path_cells.append(chosen["exit_pos"])
        path_occ.update(chosen["seg"])
        path_occ.add(chosen["exit_pos"])
        lights.append(chosen)
        current = chosen["exit_pos"]
        current_dir = chosen["next_dir"]
        input_occ.update(chosen["light_cells"])
        light_halo_occ.update(chosen["halo"])

    final_steps = _boundary_distance(h, w, current, current_dir)
    if final_steps < 1:
        return None
    final_seg = _line_cells(current, current_dir, final_steps)
    if any(cell in path_occ or cell in input_occ for cell in final_seg):
        return None
    path_cells.extend(final_seg)
    path_occ.update(final_seg)

    return {
        "start": start,
        "direction": direction,
        "path_cells": path_cells,
        "lights": lights,
        "end": final_seg[-1],
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 19)
    w = rng.randint(12, 19)
    grid = np.full((h, w), BG, dtype=int)

    num_paths = rng.randint(2, 4)
    turns_plan = [0, 1]
    while len(turns_plan) < num_paths:
        turns_plan.append(rng.randint(0, 3))
    rng.shuffle(turns_plan)

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    side_cycle = [sides[i % 4] for i in range(num_paths)]
    rng.shuffle(side_cycle)

    if num_paths * 2 > len(COLOR_POOL):
        return None
    chosen_colors = rng.sample(COLOR_POOL, num_paths * 2)
    start_colors = chosen_colors[:num_paths]
    body_colors = chosen_colors[num_paths:]

    path_occ: set[tuple[int, int]] = set()
    input_occ: set[tuple[int, int]] = set()
    light_halo_occ: set[tuple[int, int]] = set()
    built_paths = []

    for idx in range(num_paths):
        side = side_cycle[idx]
        direction = START_FROM_SIDE[side]
        starts = []
        if side in {"top", "bottom"}:
            row = 0 if side == "top" else h - 1
            cols = list(range(1, w - 1))
            rng.shuffle(cols)
            starts = [(row, c) for c in cols]
        else:
            col = 0 if side == "left" else w - 1
            rows = list(range(1, h - 1))
            rng.shuffle(rows)
            starts = [(r, col) for r in rows]

        built = None
        for start in starts[:PATH_BUILD_ATTEMPTS]:
            if start in path_occ or start in input_occ:
                continue
            local_path_occ = set(path_occ)
            local_input_occ = set(input_occ)
            local_light_halo = set(light_halo_occ)
            local_path_occ.add(start)
            local_input_occ.add(start)
            candidate = _build_path(
                h,
                w,
                start,
                direction,
                turns_plan[idx],
                local_path_occ,
                local_input_occ,
                local_light_halo,
                rng,
            )
            if candidate is None:
                continue
            path_occ = local_path_occ
            input_occ = local_input_occ
            light_halo_occ = local_light_halo
            built = candidate
            break
        if built is None:
            return None
        built_paths.append(
            {
                "color": start_colors[idx],
                "body": body_colors[idx],
                "side": side,
                **built,
            }
        )

    for spec in built_paths:
        sr, sc = spec["start"]
        grid[sr, sc] = spec["color"]
        for light_idx, light in enumerate(spec["lights"]):
            top, left = light["top_left"]
            grid[top : top + 2, left : left + 2] = spec["body"]
            sig_r, sig_c = light["signal"]
            signal_color = start_colors[(light_idx + 1 + chosen_colors.index(spec["color"])) % len(start_colors)]
            if signal_color == spec["body"]:
                signal_color = spec["color"]
            grid[sig_r, sig_c] = signal_color

    if Counter(grid.flatten())[BG] <= (h * w) // 2:
        return None

    inferred = main(grid)
    if inferred.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for seed in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid traffic-light input")
