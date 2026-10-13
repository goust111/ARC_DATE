from __future__ import annotations

import random

import numpy as np


# ARC palette.
BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SOURCE_COLORS = (RED, BLUE, YELLOW)
PAIR_MODES = (
    (RED, BLUE, MAGENTA),
    (RED, YELLOW, ORANGE),
    (BLUE, YELLOW, GREEN),
)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

GENERATE_RETRIES = 400
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
TARGET_MARGIN = 3
TARGET_RAY_DISTANCE_MAX = 4
FREE_RAY_RETRIES = 140
EXTRA_RAY_PROB = 0.45
EXTRA_STRAIGHT_PROB = 0.35


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _mix_color(colors: set[int]) -> int:
    key = frozenset(int(c) for c in colors)
    table = {
        frozenset({RED}): RED,
        frozenset({BLUE}): BLUE,
        frozenset({YELLOW}): YELLOW,
        frozenset({RED, BLUE}): MAGENTA,
        frozenset({RED, YELLOW}): ORANGE,
        frozenset({BLUE, YELLOW}): GREEN,
        frozenset({RED, BLUE, YELLOW}): GREY,
    }
    assert key in table, f"unsupported color mix: {sorted(key)}"
    return int(table[key])


def _border_entries(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    entries: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color not in SOURCE_COLORS:
                continue
            edges: list[str] = []
            if r == 0:
                edges.append("top")
            if r == h - 1:
                edges.append("bottom")
            if c == 0:
                edges.append("left")
            if c == w - 1:
                edges.append("right")
            assert len(edges) == 1, "entrances must avoid corners"
            edge = edges[0]
            direction = {
                "top": DOWN,
                "bottom": UP,
                "left": RIGHT,
                "right": LEFT,
            }[edge]
            entries.append(
                {
                    "pos": (int(r), int(c)),
                    "edge": edge,
                    "direction": direction,
                    "color": color,
                }
            )
    assert entries, "expected at least one border entrance"
    return entries


def _mirror_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != GREY:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != GREY:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            rows = {rr for rr, _cc in coords}
            cols = {cc for _rr, cc in coords}
            assert len(coords) in (2, 3), "each mirror must be length 2 or 3"
            if len(rows) == 1:
                ordered = sorted(coords, key=lambda rc: rc[1])
                assert ordered[-1][1] - ordered[0][1] + 1 == len(ordered), "horizontal mirror must be contiguous"
                orientation = "h"
            else:
                assert len(cols) == 1, "mirror must be horizontal or vertical"
                ordered = sorted(coords, key=lambda rc: rc[0])
                assert ordered[-1][0] - ordered[0][0] + 1 == len(ordered), "vertical mirror must be contiguous"
                orientation = "v"
            comps.append(
                {
                    "orientation": orientation,
                    "cells": ordered,
                    "index": {cell: i for i, cell in enumerate(ordered)},
                }
            )
    return comps


def _reflect(direction: tuple[int, int], mirror: dict[str, object], hit: tuple[int, int]) -> tuple[int, int] | None:
    orientation = str(mirror["orientation"])
    cells = mirror["cells"]
    idx = int(mirror["index"][hit])
    length = int(len(cells))

    if orientation == "h":
        if direction[0] == 0:
            return None
        return RIGHT if 2 * idx < length else LEFT

    if direction[1] == 0:
        return None
    return DOWN if 2 * idx < length else UP


def _trace_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    mirror_lookup: dict[tuple[int, int], dict[str, object]],
) -> tuple[list[tuple[int, int]], int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    visited_states: set[tuple[int, int, int, int]] = set()
    path: list[tuple[int, int]] = []
    hits = 0

    while True:
        state = (int(r), int(c), int(dr), int(dc))
        if state in visited_states:
            break
        visited_states.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        hit = (int(nr), int(nc))
        mirror = mirror_lookup.get(hit)
        if mirror is not None:
            new_direction = _reflect((dr, dc), mirror, hit)
            if new_direction is None:
                break
            hits += 1
            r, c = hit
            dr, dc = new_direction
            # After a reflection, move across any remaining cells of the same
            # mirror segment so the ray exits from the mirror's far side.
            while True:
                sr, sc = r + dr, c + dc
                next_mirror = mirror_lookup.get((int(sr), int(sc)))
                if next_mirror is not mirror:
                    break
                r, c = int(sr), int(sc)
            continue

        path.append(hit)
        r, c = hit

    return path, int(hits)


def _solve(grid: np.ndarray, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    _ = _most_common_color(g)
    mirrors = _mirror_components(g)
    mirror_lookup: dict[tuple[int, int], dict[str, object]] = {}
    for mirror in mirrors:
        for cell in mirror["cells"]:
            mirror_lookup[tuple(cell)] = mirror

    entries = _border_entries(g)
    color_sets: dict[tuple[int, int], set[int]] = {}
    hit_counts: list[int] = []

    for entry in entries:
        pos = tuple(entry["pos"])
        color_sets.setdefault(pos, set()).add(int(entry["color"]))

    for entry in entries:
        path, hits = _trace_ray(
            g.shape,
            tuple(entry["pos"]),
            tuple(entry["direction"]),
            mirror_lookup,
        )
        hit_counts.append(int(hits))
        color = int(entry["color"])
        for cell in path:
            if cell in mirror_lookup:
                continue
            color_sets.setdefault(cell, set()).add(color)

    for cell, colors in color_sets.items():
        if cell in mirror_lookup:
            continue
        out[cell] = _mix_color(colors)

    if with_meta:
        return out, {
            "entries": entries,
            "mirrors": mirrors,
            "hits": hit_counts,
            "color_sets": color_sets,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast colored rays inward from edge sources, reflect them on short grey mirrors, and mix colors where different rays cross.

    Input structure:
    - bg is the dominant black background.
    - Red, blue, and yellow single-pixel entrances sit on the outer border; each
      one launches a ray straight inward, perpendicular to that border.
    - Grey foreground components inside the grid are 1-pixel-thick horizontal or
      vertical mirror segments of length 2 or 3.

    Rule:
    - Simulate every border entrance independently. The ray colors every visited
      non-mirror cell with its source color as it travels in a straight line.
    - When a vertical ray hits a horizontal mirror, it turns horizontal: hitting
      the mirror's left half reflects it rightward, while hitting the right half
      reflects it leftward. Horizontal rays hitting vertical mirrors behave
      analogously using top-half -> down and bottom-half -> up.
    - Rays keep traveling after each reflection until they either leave the grid
      or reach a mirror that does not support a perpendicular reflection.
    - After all rays are traced, each non-mirror cell shows the mix of all source
      colors that passed through it: red+blue=magenta, red+yellow=orange,
      blue+yellow=green, and all three together=grey.

    Key insight:
    - The task combines three independent deductions: border position determines
      launch direction, mirror-hit position determines the reflected direction,
      and crossings must be resolved by color-set mixing rather than by last-write
      overwriting.
    """
    return _solve(grid, with_meta=False)


def _build_ray(
    h: int,
    w: int,
    color: int,
    entry_edge: str,
    fixed: int,
    mirror_pos: int | None = None,
    turn_dir: str | None = None,
    mirror_len: int | None = None,
) -> dict[str, object] | None:
    assert color in SOURCE_COLORS
    if entry_edge == "top":
        start = (0, int(fixed))
        inward = DOWN
        axis = "v"
    elif entry_edge == "bottom":
        start = (h - 1, int(fixed))
        inward = UP
        axis = "v"
    elif entry_edge == "left":
        start = (int(fixed), 0)
        inward = RIGHT
        axis = "h"
    elif entry_edge == "right":
        start = (int(fixed), w - 1)
        inward = LEFT
        axis = "h"
    else:
        raise AssertionError(f"bad edge: {entry_edge}")

    if axis == "v" and not (1 <= fixed <= w - 2):
        return None
    if axis == "h" and not (1 <= fixed <= h - 2):
        return None

    if mirror_pos is None:
        path: list[tuple[int, int]] = []
        r, c = start
        while 0 <= r < h and 0 <= c < w:
            path.append((int(r), int(c)))
            r += inward[0]
            c += inward[1]
        return {
            "color": int(color),
            "entrance": tuple(start),
            "entry_edge": entry_edge,
            "path_cells": path,
            "mirror_cells": [],
        }

    assert turn_dir is not None and mirror_len is not None
    assert mirror_len in (2, 3)

    if axis == "v":
        hit = (int(mirror_pos), int(fixed))
        if not (0 < hit[0] < h - 1):
            return None
        path1 = (
            [(r, int(fixed)) for r in range(0, int(mirror_pos))]
            if entry_edge == "top"
            else [(r, int(fixed)) for r in range(h - 1, int(mirror_pos), -1)]
        )
        if turn_dir == "left":
            mirror_cells = [(hit[0], hit[1] - k) for k in range(mirror_len)]
            out_dir = LEFT
        elif turn_dir == "right":
            mirror_cells = [(hit[0], hit[1] + k) for k in range(mirror_len)]
            out_dir = RIGHT
        else:
            return None
    else:
        hit = (int(fixed), int(mirror_pos))
        if not (0 < hit[1] < w - 1):
            return None
        path1 = (
            [(int(fixed), c) for c in range(0, int(mirror_pos))]
            if entry_edge == "left"
            else [(int(fixed), c) for c in range(w - 1, int(mirror_pos), -1)]
        )
        if turn_dir == "up":
            mirror_cells = [(hit[0] - k, hit[1]) for k in range(mirror_len)]
            out_dir = UP
        elif turn_dir == "down":
            mirror_cells = [(hit[0] + k, hit[1]) for k in range(mirror_len)]
            out_dir = DOWN
        else:
            return None

    if not path1:
        return None
    if any(not (0 < r < h - 1 and 0 < c < w - 1) for r, c in mirror_cells):
        return None
    if len(set(mirror_cells)) != mirror_len:
        return None

    mirror_set = set(mirror_cells)
    path2: list[tuple[int, int]] = []
    rr, cc = hit[0] + out_dir[0], hit[1] + out_dir[1]
    while (int(rr), int(cc)) in mirror_set:
        rr += out_dir[0]
        cc += out_dir[1]
    while 0 <= rr < h and 0 <= cc < w:
        path2.append((int(rr), int(cc)))
        rr += out_dir[0]
        cc += out_dir[1]
    if not path2:
        return None

    path = path1 + path2
    if len(set(path)) != len(path):
        return None
    if set(path) & mirror_set:
        return None

    return {
        "color": int(color),
        "entrance": tuple(start),
        "entry_edge": entry_edge,
        "path_cells": path,
        "mirror_cells": mirror_cells,
    }


def _existing_sets(rays: list[dict[str, object]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    entrances = {tuple(ray["entrance"]) for ray in rays}
    paths = {cell for ray in rays for cell in ray["path_cells"]}
    mirrors = {cell for ray in rays for cell in ray["mirror_cells"]}
    return entrances, paths, mirrors


def _ray_fits(ray: dict[str, object] | None, rays: list[dict[str, object]], required_overlap: set[tuple[int, int]]) -> bool:
    if ray is None:
        return False

    entrances, paths, mirrors = _existing_sets(rays)
    new_entrance = tuple(ray["entrance"])
    new_paths = set(ray["path_cells"])
    new_mirrors = set(ray["mirror_cells"])

    if new_entrance in entrances or new_entrance in mirrors or new_entrance in paths:
        return False
    if (new_paths - {new_entrance}) & entrances:
        return False
    if new_paths & mirrors:
        return False
    if new_mirrors & entrances:
        return False
    if new_mirrors & paths:
        return False

    overlap = new_paths & paths
    if overlap != required_overlap:
        return False

    for mr, mc in new_mirrors:
        if max(abs(mr - new_entrance[0]), abs(mc - new_entrance[1])) <= 1:
            return False
        for er, ec in entrances:
            if max(abs(mr - er), abs(mc - ec)) <= 1:
                return False
        for orr, occ in mirrors:
            if max(abs(mr - orr), abs(mc - occ)) <= 1:
                return False

    return True


def _sample_target() -> tuple[int, int, int, int]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    tr = random.randint(TARGET_MARGIN, h - TARGET_MARGIN - 1)
    tc = random.randint(TARGET_MARGIN, w - TARGET_MARGIN - 1)
    return int(h), int(w), int(tr), int(tc)


def _sample_horizontal_target_ray(
    h: int,
    w: int,
    target: tuple[int, int],
    color: int,
    distance: int | None = None,
    turn_dir: str | None = None,
) -> dict[str, object] | None:
    tr, tc = target
    entry = random.choice(["left", "right"])
    side = "right" if entry == "left" else "left"
    max_delta = min(TARGET_RAY_DISTANCE_MAX, (w - 2 - tc) if side == "right" else (tc - 1))
    if max_delta < 1:
        return None
    delta = int(distance if distance is not None else random.randint(1, max_delta))
    mirror_col = tc + delta if side == "right" else tc - delta
    if turn_dir is None:
        turn_dir = random.choice(["up", "down"])
    mirror_len = random.choice([2, 3])
    return _build_ray(h, w, color, entry, tr, mirror_col, turn_dir, mirror_len)


def _sample_vertical_target_ray(
    h: int,
    w: int,
    target: tuple[int, int],
    color: int,
    horizontal_side: str,
    forced_entry: str | None = None,
    fixed_distance: int | None = None,
) -> dict[str, object] | None:
    tr, tc = target
    entry = forced_entry if forced_entry is not None else random.choice(["top", "bottom"])
    turn_dir = "left" if horizontal_side == "right" else "right"

    if entry == "top":
        max_delta = min(TARGET_RAY_DISTANCE_MAX, h - 2 - tr)
        if max_delta < 1:
            return None
        delta = int(fixed_distance if fixed_distance is not None else random.randint(1, max_delta))
        mirror_row = tr + delta
    else:
        max_delta = min(TARGET_RAY_DISTANCE_MAX, tr - 1)
        if max_delta < 1:
            return None
        delta = int(fixed_distance if fixed_distance is not None else random.randint(1, max_delta))
        mirror_row = tr - delta

    mirror_len = random.choice([2, 3])
    return _build_ray(h, w, color, entry, tc, mirror_row, turn_dir, mirror_len)


def _sample_free_ray(h: int, w: int, color: int, rays: list[dict[str, object]]) -> dict[str, object] | None:
    for _ in range(FREE_RAY_RETRIES):
        axis = random.choice(["v", "h"])
        straight = random.random() < EXTRA_STRAIGHT_PROB
        if axis == "v":
            entry = random.choice(["top", "bottom"])
            fixed = random.randint(1, w - 2)
            if straight:
                ray = _build_ray(h, w, color, entry, fixed)
            else:
                mirror_pos = random.randint(1, h - 2)
                turn = random.choice(["left", "right"])
                ray = _build_ray(h, w, color, entry, fixed, mirror_pos, turn, random.choice([2, 3]))
        else:
            entry = random.choice(["left", "right"])
            fixed = random.randint(1, h - 2)
            if straight:
                ray = _build_ray(h, w, color, entry, fixed)
            else:
                mirror_pos = random.randint(1, w - 2)
                turn = random.choice(["up", "down"])
                ray = _build_ray(h, w, color, entry, fixed, mirror_pos, turn, random.choice([2, 3]))

        if ray is None or len(ray["path_cells"]) < 6:
            continue
        if _ray_fits(ray, rays, set()):
            return ray
    return None


def _render_input(h: int, w: int, rays: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for ray in rays:
        er, ec = ray["entrance"]
        grid[er, ec] = int(ray["color"])
    for ray in rays:
        for r, c in ray["mirror_cells"]:
            grid[r, c] = GREY
    return grid


def _build_core_rays() -> tuple[np.ndarray, tuple[int, int], int] | None:
    h, w, tr, tc = _sample_target()
    target = (int(tr), int(tc))
    rays: list[dict[str, object]] = []

    if random.random() < 0.25:
        horizontal_color = random.choice(SOURCE_COLORS)
        vertical_colors = [c for c in SOURCE_COLORS if c != horizontal_color]

        for _ in range(40):
            hray = _sample_horizontal_target_ray(h, w, target, horizontal_color)
            if hray is None or not _ray_fits(hray, rays, set()):
                continue
            rays = [hray]
            h_side = "right" if hray["entrance"][1] == 0 else "left"

            top_ray = _sample_vertical_target_ray(
                h,
                w,
                target,
                vertical_colors[0],
                h_side,
                forced_entry="top",
                fixed_distance=1,
            )
            if not _ray_fits(top_ray, rays, {target}):
                rays = []
                continue
            rays.append(top_ray)

            bottom_ray = _sample_vertical_target_ray(
                h,
                w,
                target,
                vertical_colors[1],
                h_side,
                forced_entry="bottom",
                fixed_distance=1,
            )
            if not _ray_fits(bottom_ray, rays, {target}):
                rays = []
                continue
            rays.append(bottom_ray)
            expected = GREY
            break
        else:
            return None
    else:
        pair_a, pair_b, expected = random.choice(PAIR_MODES)
        remaining = next(color for color in SOURCE_COLORS if color not in (pair_a, pair_b))
        if random.random() < 0.5:
            horizontal_color, vertical_color = pair_a, pair_b
        else:
            horizontal_color, vertical_color = pair_b, pair_a

        for _ in range(40):
            hray = _sample_horizontal_target_ray(h, w, target, horizontal_color)
            if hray is None or not _ray_fits(hray, rays, set()):
                continue
            rays = [hray]
            h_side = "right" if hray["entrance"][1] == 0 else "left"

            vray = _sample_vertical_target_ray(h, w, target, vertical_color, h_side)
            if not _ray_fits(vray, rays, {target}):
                rays = []
                continue
            rays.append(vray)

            free_ray = _sample_free_ray(h, w, remaining, rays)
            if free_ray is None:
                rays = []
                continue
            rays.append(free_ray)
            break
        else:
            return None

    if random.random() < EXTRA_RAY_PROB:
        extra = _sample_free_ray(h, w, random.choice(SOURCE_COLORS), rays)
        if extra is not None:
            rays.append(extra)

    grid = _render_input(h, w, rays)
    return grid, target, int(expected)


def _generated_sample_ok(grid: np.ndarray, target: tuple[int, int], expected_color: int) -> bool:
    if int(grid[target]) != BG:
        return False

    try:
        out, meta = _solve(grid, with_meta=True)
    except Exception:
        return False

    if int(out[target]) != int(expected_color):
        return False
    if {int(entry["color"]) for entry in meta["entries"]} != set(SOURCE_COLORS):
        return False
    if len(meta["entries"]) < 3:
        return False
    if len(meta["mirrors"]) < 2:
        return False
    if {mirror["orientation"] for mirror in meta["mirrors"]} != {"h", "v"}:
        return False
    if sum(int(h) for h in meta["hits"]) < 2:
        return False

    changed = int(np.sum(out != grid))
    if changed < 14:
        return False

    mixed_mask = (
        (out == MAGENTA)
        | (out == ORANGE)
        | (out == GREEN)
        | ((out == GREY) & (grid != GREY))
    )
    if int(mixed_mask.sum()) < 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_core_rays()
    if built is None:
        return None
    grid, target, expected_color = built
    if not _generated_sample_ok(grid, target, expected_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-mirror ARC input")
