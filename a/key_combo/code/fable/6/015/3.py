from __future__ import annotations

import random
import sys
from typing import Iterable

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg, neighbors4, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, find_components, get_bg, neighbors4, paste


GREY = Color.GREY
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
NON_GREY_COLORS = [c for c in Color.ALL_COLORS if c != GREY]
MAX_GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _parse_arrow_component(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    cells = [tuple(map(int, rc)) for rc in coords]
    if len(cells) < 2:
        return None

    values = [int(grid[r, c]) for r, c in cells]
    non_grey = [v for v in values if v != GREY]
    if len(non_grey) != 1:
        return None

    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1:
        r = rows[0]
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        left = (r, cols[0])
        right = (r, cols[-1])
        if int(grid[left]) != GREY and int(grid[right]) == GREY:
            head = left
            dr, dc = 0, -1
        elif int(grid[right]) != GREY and int(grid[left]) == GREY:
            head = right
            dr, dc = 0, 1
        else:
            return None
    elif len(cols) == 1:
        c = cols[0]
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        top = (rows[0], c)
        bottom = (rows[-1], c)
        if int(grid[top]) != GREY and int(grid[bottom]) == GREY:
            head = top
            dr, dc = -1, 0
        elif int(grid[bottom]) != GREY and int(grid[top]) == GREY:
            head = bottom
            dr, dc = 1, 0
        else:
            return None
    else:
        return None

    head_color = int(grid[head])
    body = set(cells) - {head}
    if not body or any(int(grid[r, c]) != GREY for r, c in body):
        return None

    return {
        "head": head,
        "color": head_color,
        "dr": dr,
        "dc": dc,
        "tail_len": len(body),
        "cells": set(cells),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)

    arrows: list[dict] = []
    arrow_cells: set[tuple[int, int]] = set()
    for comp in comps:
        arrow = _parse_arrow_component(grid, comp["coords"])
        if arrow is None:
            continue
        arrows.append(arrow)
        arrow_cells |= arrow["cells"]

    blockers = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) != bg and (r, c) not in arrow_cells
    }
    arrows.sort(key=lambda a: (a["head"][0], a["head"][1], a["color"]))
    return bg, arrows, blockers


def _trace_ray(
    shape: tuple[int, int],
    head: tuple[int, int],
    dr: int,
    dc: int,
    wrap_budget: int,
    blockers: set[tuple[int, int]],
    self_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    h, w = shape
    r, c = head
    wraps = 0
    path: list[tuple[int, int]] = []
    seen = {head}

    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            if wraps >= wrap_budget:
                break
            wraps += 1
            nr %= h
            nc %= w

        if (nr, nc) in blockers or (nr, nc) in self_cells or (nr, nc) in seen:
            break

        path.append((nr, nc))
        seen.add((nr, nc))
        r, c = nr, nc

    return path, wraps


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored arrow head into a toroidal ray.

    The ray starts one cell ahead of the head, wraps across the opposite edge
    at most tail_len-1 times, and stops before any wall or arrow cell.
    """
    grid = np.asarray(input_grid, dtype=int)
    _bg, arrows, blockers = _extract_scene(grid)
    out = grid.copy()

    for arrow in arrows:
        path, _ = _trace_ray(
            grid.shape,
            arrow["head"],
            arrow["dr"],
            arrow["dc"],
            arrow["tail_len"] - 1,
            blockers,
            arrow["cells"],
        )
        for r, c in path:
            out[r, c] = arrow["color"]

    return out


def _cells_4nbr(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        out.add((r, c))
        for nr, nc in neighbors4(r, c):
            out.add((int(nr), int(nc)))
    return out


def _arrow_cells(head: tuple[int, int], dr: int, dc: int, tail_len: int) -> list[tuple[int, int]]:
    hr, hc = head
    return [(hr, hc)] + [(hr - dr * i, hc - dc * i) for i in range(1, tail_len + 1)]


def _path_before_edge(shape: tuple[int, int], head: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    r, c = head
    cells: list[tuple[int, int]] = []
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(shape, nr, nc):
            break
        cells.append((nr, nc))
        r, c = nr, nc
    return cells


def _sample_spaced_lanes(length: int, count: int) -> list[int] | None:
    for _ in range(200):
        lanes = sorted(random.sample(range(length), count))
        if all(b - a >= 2 for a, b in zip(lanes, lanes[1:])):
            return lanes
    return None


def _try_build_arrow(
    shape: tuple[int, int],
    orientation: str,
    lane: int,
    behavior: str,
) -> dict | None:
    h, w = shape
    for _ in range(200):
        if orientation == "h":
            dr, dc = random.choice([(0, 1), (0, -1)])
            tail_len = random.randint(2, 5) if behavior.startswith("wrap") else random.randint(1, 5)
            if behavior == "straight_boundary":
                tail_len = 1
            if behavior.startswith("wrap"):
                head = (lane, w - 1 if dc == 1 else 0)
            else:
                lo = tail_len if dc == 1 else 1
                hi = w - 2 if dc == 1 else w - 1 - tail_len
                if lo > hi:
                    continue
                head = (lane, random.randint(lo, hi))
        else:
            dr, dc = random.choice([(1, 0), (-1, 0)])
            tail_len = random.randint(2, 5) if behavior.startswith("wrap") else random.randint(1, 5)
            if behavior == "straight_boundary":
                tail_len = 1
            if behavior.startswith("wrap"):
                head = (h - 1 if dr == 1 else 0, lane)
            else:
                lo = tail_len if dr == 1 else 1
                hi = h - 2 if dr == 1 else h - 1 - tail_len
                if lo > hi:
                    continue
                head = (random.randint(lo, hi), lane)

        cells = _arrow_cells(head, dr, dc, tail_len)
        if not all(_in_bounds(shape, r, c) for r, c in cells):
            continue
        self_cells = set(cells)
        wrap_budget = tail_len - 1

        if behavior == "straight_boundary":
            path = _path_before_edge(shape, head, dr, dc)
            wall = None
            wrapped = 0
        elif behavior == "straight_wall":
            forward = _path_before_edge(shape, head, dr, dc)
            if len(forward) < 2:
                continue
            wall = random.choice(forward[1:])
            path = []
            for cell in forward:
                if cell == wall:
                    break
                path.append(cell)
            wrapped = 0
        else:
            natural_path, wrapped = _trace_ray(shape, head, dr, dc, wrap_budget, set(), self_cells)
            if wrapped == 0 or not natural_path:
                continue
            if behavior == "wrap_self":
                path = natural_path
                wall = None
            else:
                if len(natural_path) < 2:
                    continue
                wall = random.choice(natural_path[1:])
                path = []
                for cell in natural_path:
                    if cell == wall:
                        break
                    path.append(cell)

        if not path:
            continue

        return {
            "head": head,
            "dr": dr,
            "dc": dc,
            "tail_len": tail_len,
            "path": path,
            "wall": wall,
            "wrapped": wrapped,
            "cells": self_cells,
        }
    return None


def _place_extra_wall(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked_lanes: set[int],
    orientation: str,
    wall_colors: list[int],
) -> bool:
    h, w = grid.shape
    for _ in range(200):
        ph = random.randint(1, 2)
        pw = random.randint(1, 2)
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        if orientation == "h" and any((r0 + dr) in blocked_lanes for dr in range(ph)):
            continue
        if orientation == "v" and any((c0 + dc) in blocked_lanes for dc in range(pw)):
            continue

        patch_cells = {(r0 + dr, c0 + dc) for dr in range(ph) for dc in range(pw)}
        halo = _cells_4nbr(patch_cells)
        if any((r, c) in occupied for r, c in halo if _in_bounds(grid.shape, r, c)):
            continue

        color = random.choice(wall_colors)
        patch = np.full((ph, pw), color, dtype=int)
        paste(grid, patch, r0, c0)
        occupied |= patch_cells
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        orientation = random.choice(["h", "v"])
        h = random.randint(11, 18)
        w = random.randint(11, 18)
        n_arrows = random.randint(2, 4)

        lanes = _sample_spaced_lanes(h if orientation == "h" else w, n_arrows)
        if lanes is None:
            continue

        bg = random.choice([c for c in NON_GREY_COLORS if c != Color.BLACK])
        grid = np.full((h, w), bg, dtype=int)

        behaviors = [random.choice(["wrap_self", "wrap_wall"])]
        while len(behaviors) < n_arrows:
            behaviors.append(random.choice(["straight_boundary", "straight_wall", "wrap_self", "wrap_wall"]))
        random.shuffle(behaviors)

        head_colors = random.sample([c for c in NON_GREY_COLORS if c != bg], n_arrows)
        wall_colors = [c for c in NON_GREY_COLORS if c not in {bg, GREY}]
        occupied: set[tuple[int, int]] = set()
        arrow_rows = set()
        arrow_cols = set()
        dedicated_walls = 0
        specs: list[dict] = []
        ok = True

        for lane, behavior, head_color in zip(lanes, behaviors, head_colors):
            spec = _try_build_arrow((h, w), orientation, lane, behavior)
            if spec is None:
                ok = False
                break

            halo = _cells_4nbr(spec["cells"])
            wall = spec["wall"]
            if wall is not None:
                halo |= _cells_4nbr({wall})
            if any((r, c) in occupied for r, c in halo if _in_bounds(grid.shape, r, c)):
                ok = False
                break

            hr, hc = spec["head"]
            grid[hr, hc] = head_color
            for tr, tc in spec["cells"] - {spec["head"]}:
                grid[tr, tc] = GREY
            occupied |= spec["cells"]
            arrow_rows.add(hr)
            arrow_cols.add(hc)

            if wall is not None:
                wall_color = random.choice([c for c in wall_colors if c != head_color] or wall_colors)
                grid[wall] = wall_color
                occupied.add(wall)
                dedicated_walls += 1

            specs.append({**spec, "color": head_color})

        if not ok:
            continue

        extra_walls = random.randint(0, 3)
        if dedicated_walls == 0:
            extra_walls = max(extra_walls, 1)
        for _wall_idx in range(extra_walls):
            _place_extra_wall(
                grid,
                occupied,
                arrow_rows if orientation == "h" else arrow_cols,
                orientation,
                wall_colors,
            )

        bg2, parsed_arrows, blockers = _extract_scene(grid)
        if bg2 != bg or len(parsed_arrows) != n_arrows:
            continue

        used_paths: set[tuple[int, int]] = set()
        valid = True
        for arrow in parsed_arrows:
            path, _ = _trace_ray(
                grid.shape,
                arrow["head"],
                arrow["dr"],
                arrow["dc"],
                arrow["tail_len"] - 1,
                blockers,
                arrow["cells"],
            )
            if not path:
                valid = False
                break
            if any(cell in used_paths for cell in path):
                valid = False
                break
            used_paths |= set(path)

        if not valid:
            continue

        assert grid.ndim == 2 and grid.shape[0] <= 30 and grid.shape[1] <= 30
        return grid

    raise RuntimeError("generate_input failed to build a valid instance")
