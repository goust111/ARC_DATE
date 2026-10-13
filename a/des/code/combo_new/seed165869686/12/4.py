from __future__ import annotations

import numpy as np


RED = 2
BLUE = 1
GREY = 5

BG_CHOICES = (0, 3, 4, 6, 7, 8, 9)
DIR_KEYS = ("R", "D", "L", "U")
DIRS = {
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
    "U": (-1, 0),
}
CW = {"R": "D", "D": "L", "L": "U", "U": "R"}

ARROW_COUNT_CHOICES = (1, 2, 2, 3, 3)
BOUNCE_CHOICES = (0, 1, 2, 3)
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
SEGMENT_LEN_RANGE = (2, 6)
FINAL_SEGMENT_LEN_RANGE = (3, 8)
WALL_LEN_RANGE = (2, 5)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _arrow_cells(tip_r: int, tip_c: int, dir_key: str) -> tuple[tuple[int, int], ...]:
    dr, dc = DIRS[dir_key]
    pr, pc = dc, -dr
    cells = {
        (int(tip_r), int(tip_c)),
        (int(tip_r - dr), int(tip_c - dc)),
        (int(tip_r - 2 * dr), int(tip_c - 2 * dc)),
        (int(tip_r + pr), int(tip_c + pc)),
        (int(tip_r - pr), int(tip_c - pc)),
    }
    return tuple(sorted(cells))


def _decode_arrow_component(coords: list[tuple[int, int]]) -> tuple[int, int, str] | None:
    if len(coords) != 5:
        return None
    coord_set = {(int(r), int(c)) for r, c in coords}
    degrees = {
        p: sum((nr, nc) in coord_set for nr, nc in _neighbors4(*p))
        for p in coord_set
    }
    tips = [p for p, degree in degrees.items() if degree == 3]
    mids = [p for p, degree in degrees.items() if degree == 2]
    leaves = [p for p, degree in degrees.items() if degree == 1]
    if len(tips) != 1 or len(mids) != 1 or len(leaves) != 3:
        return None

    tip_r, tip_c = tips[0]
    mid_r, mid_c = mids[0]
    back = (int(mid_r - tip_r), int(mid_c - tip_c))
    dir_vec = (-back[0], -back[1])
    dir_key = next((key for key, vec in DIRS.items() if vec == dir_vec), None)
    if dir_key is None:
        return None

    tail = (int(mid_r + back[0]), int(mid_c + back[1]))
    if tail not in coord_set or degrees[tail] != 1:
        return None

    dr, dc = dir_vec
    pr, pc = dc, -dr
    wing_a = (int(tip_r + pr), int(tip_c + pc))
    wing_b = (int(tip_r - pr), int(tip_c - pc))
    if wing_a not in coord_set or wing_b not in coord_set:
        return None
    if tuple(sorted(coord_set)) != _arrow_cells(tip_r, tip_c, dir_key):
        return None
    return int(tip_r), int(tip_c), dir_key


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    colors = sorted(int(color) for color in np.unique(grid) if int(color) not in {bg, GREY})
    assert RED not in colors and BLUE not in colors, "signal colors must be absent from the input"
    for color in colors:
        comps = _components(grid == int(color))
        assert len(comps) == 1, "each arrow color must form exactly one component"
        decoded = _decode_arrow_component(comps[0])
        assert decoded is not None, "foreground component must be a 5-cell arrow"
        tip_r, tip_c, dir_key = decoded
        arrows.append(
            {
                "color": int(color),
                "tip": (int(tip_r), int(tip_c)),
                "dir": str(dir_key),
                "cells": _arrow_cells(int(tip_r), int(tip_c), str(dir_key)),
            }
        )
    assert arrows, "expected at least one arrow"
    arrows.sort(key=lambda arrow: (arrow["tip"][0], arrow["tip"][1], arrow["color"]))  # type: ignore[index]
    return arrows


def _detect_walls(grid: np.ndarray) -> dict[tuple[int, int], str]:
    wall_map: dict[tuple[int, int], str] = {}
    for comp in _components(grid == GREY):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(comp) >= 2, "each wall must be a line segment of length at least 2"
        if len(rows) == 1:
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert c1 - c0 + 1 == len(comp), "horizontal wall must be contiguous"
            for c in range(c0, c1 + 1):
                wall_map[(int(row), int(c))] = "H"
        else:
            assert len(cols) == 1, "wall component must be horizontal or vertical"
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert r1 - r0 + 1 == len(comp), "vertical wall must be contiguous"
            for r in range(r0, r1 + 1):
                wall_map[(int(r), int(col))] = "V"
    return wall_map


def _trace_arrow(
    shape: tuple[int, int],
    wall_map: dict[tuple[int, int], str],
    arrow_cells: set[tuple[int, int]],
    tip: tuple[int, int],
    dir_key: str,
) -> tuple[list[tuple[tuple[int, int], int]], list[tuple[int, int]]]:
    cur = (int(tip[0]), int(tip[1]))
    cur_dir = str(dir_key)
    color = RED
    paint: list[tuple[tuple[int, int], int]] = []
    hits: list[tuple[int, int]] = []
    seen: set[tuple[int, int, str, int]] = set()

    while True:
        state = (int(cur[0]), int(cur[1]), cur_dir, int(color))
        assert state not in seen, "signal path must not loop"
        seen.add(state)

        dr, dc = DIRS[cur_dir]
        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not _in_bounds(shape, nr, nc):
            break

        if (nr, nc) in wall_map:
            orientation = wall_map[(nr, nc)]
            if dr == 0:
                assert orientation == "V", "horizontal motion must hit a vertical wall"
            else:
                assert orientation == "H", "vertical motion must hit a horizontal wall"
            hits.append((int(nr), int(nc)))
            cur_dir = CW[cur_dir]
            color = BLUE if color == RED else RED
            continue

        cur = (int(nr), int(nc))
        if cur not in arrow_cells:
            paint.append((cur, int(color)))

    return paint, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored signals from each arrow tip and bend them at grey walls.

    Input structure:
    - The grid has one dominant background color.
    - Grey pixels form separate one-cell-thick horizontal or vertical wall segments.
    - Every other foreground object is a monochrome 5-cell arrow: a 3-cell shaft ending in a 3-cell head, pointing up, down, left, or right.

    Rule:
    - Keep every arrow and every grey wall unchanged.
    - For each arrow, start just beyond its tip and paint a red signal cell-by-cell in the arrow's direction until the next step would leave the grid or enter a grey wall cell.
    - When the next step is a wall, do not paint the wall; instead the signal makes a right-angle reflection at the current cell, rotating clockwise, and its color toggles red/blue.
    - Continue tracing the reflected path, toggling color after every wall hit, until the signal leaves the grid.

    Key insight:
    - Solving the task requires three linked inferences: decode each arrow's orientation from its geometry, identify wall orientation from straight grey segments, and then simulate the bouncing path while tracking the parity of wall hits for the alternating output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    walls = _detect_walls(g)
    arrows = _find_arrows(g, bg)
    arrow_cells = {
        (int(r), int(c))
        for arrow in arrows
        for r, c in arrow["cells"]  # type: ignore[index]
    }

    out = g.copy()
    for arrow in arrows:
        tip = arrow["tip"]  # type: ignore[assignment]
        dir_key = arrow["dir"]  # type: ignore[assignment]
        paint, _hits = _trace_arrow(g.shape, walls, arrow_cells, tip, dir_key)
        for (r, c), color in paint:
            out[r, c] = int(color)
    return out


def _translate(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], tr: int, tc: int) -> list[tuple[int, int]]:
    return [(int(r + tr), int(c + tc)) for r, c in cells]


def _build_route(dir_key: str, lengths: list[int]) -> dict[str, object] | None:
    assert lengths
    cur_r, cur_c = 0, 0
    cur_dir = str(dir_key)
    path: list[tuple[int, int]] = []
    path_set: set[tuple[int, int]] = set()
    touches: list[tuple[tuple[int, int], str]] = []
    dirs: list[str] = []

    for idx, seg_len in enumerate(lengths):
        dirs.append(cur_dir)
        dr, dc = DIRS[cur_dir]
        for step in range(1, int(seg_len) + 1):
            cell = (int(cur_r + dr * step), int(cur_c + dc * step))
            if cell in path_set:
                return None
            path.append(cell)
            path_set.add(cell)
        cur_r += dr * int(seg_len)
        cur_c += dc * int(seg_len)
        if idx + 1 < len(lengths):
            touch = (int(cur_r + dr), int(cur_c + dc))
            if touch in path_set:
                return None
            touches.append((touch, cur_dir))
            cur_dir = CW[cur_dir]

    arrow = set(_arrow_cells(0, 0, dir_key))
    if path_set & arrow:
        return None
    if any(touch in arrow or touch in path_set for touch, _ in touches):
        return None

    return {
        "path": tuple(path),
        "touches": tuple(touches),
        "arrow": tuple(sorted(arrow)),
        "final_endpoint": (int(cur_r), int(cur_c)),
        "final_dir": dirs[-1],
        "dir": str(dir_key),
    }


def _translation_options(layout: dict[str, object], h: int, w: int) -> list[tuple[int, int]]:
    cells = list(layout["path"]) + list(layout["arrow"]) + [touch for touch, _ in layout["touches"]]  # type: ignore[index]
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    end_r, end_c = layout["final_endpoint"]  # type: ignore[assignment]
    final_dir = str(layout["final_dir"])

    options: list[tuple[int, int]] = []
    if final_dir == "R":
        tc = int(w - 1 - end_c)
        if not (-min_c <= tc <= w - 1 - max_c):
            return []
        tr_lo = int(-min_r)
        tr_hi = int(h - 1 - max_r)
        for tr in range(tr_lo, tr_hi + 1):
            options.append((int(tr), int(tc)))
    elif final_dir == "L":
        tc = int(-end_c)
        if not (-min_c <= tc <= w - 1 - max_c):
            return []
        tr_lo = int(-min_r)
        tr_hi = int(h - 1 - max_r)
        for tr in range(tr_lo, tr_hi + 1):
            options.append((int(tr), int(tc)))
    elif final_dir == "D":
        tr = int(h - 1 - end_r)
        if not (-min_r <= tr <= h - 1 - max_r):
            return []
        tc_lo = int(-min_c)
        tc_hi = int(w - 1 - max_c)
        for tc in range(tc_lo, tc_hi + 1):
            options.append((int(tr), int(tc)))
    else:
        tr = int(-end_r)
        if not (-min_r <= tr <= h - 1 - max_r):
            return []
        tc_lo = int(-min_c)
        tc_hi = int(w - 1 - max_c)
        for tc in range(tc_lo, tc_hi + 1):
            options.append((int(tr), int(tc)))
    return options


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if _in_bounds(shape, nr, nc):
                    expanded.add((int(nr), int(nc)))
    return expanded


def _segment_touches_other_walls(
    segment: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> bool:
    seg_set = set(segment)
    for r, c in segment:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in wall_cells and (nr, nc) not in seg_set:
                return True
    return False


def _sample_wall_segment(
    touch: tuple[int, int],
    incoming_dir: str,
    shape: tuple[int, int],
    rng: np.random.Generator,
    global_moat: set[tuple[int, int]],
    local_blocked: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        total_len = int(rng.integers(WALL_LEN_RANGE[0], WALL_LEN_RANGE[1] + 1))
        before = int(rng.integers(0, total_len))
        after = int(total_len - 1 - before)

        if incoming_dir in {"R", "L"}:
            segment = [(int(touch[0] + k), int(touch[1])) for k in range(-before, after + 1)]
        else:
            segment = [(int(touch[0]), int(touch[1] + k)) for k in range(-before, after + 1)]

        if any(not _in_bounds(shape, r, c) for r, c in segment):
            continue
        if any((r, c) in global_moat or (r, c) in local_blocked for r, c in segment):
            continue
        if _segment_touches_other_walls(segment, wall_cells):
            continue
        return segment
    return None


def _route_signature(
    shape: tuple[int, int],
    wall_map: dict[tuple[int, int], str],
    arrow_cells: set[tuple[int, int]],
    tip: tuple[int, int],
    dir_key: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    paint, hits = _trace_arrow(shape, wall_map, arrow_cells, tip, dir_key)
    return [cell for cell, _color in paint], hits


def _plan_arrow(
    grid: np.ndarray,
    bg: int,
    color: int,
    bounce_count: int,
    dir_key: str,
    rng: np.random.Generator,
    global_structures: set[tuple[int, int]],
    global_paths: set[tuple[int, int]],
    global_walls: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = grid.shape
    global_moat = _expand8(global_structures | global_paths, grid.shape, radius=1)

    for _ in range(220):
        lengths = [int(rng.integers(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1)) for _ in range(bounce_count)]
        lengths.append(int(rng.integers(FINAL_SEGMENT_LEN_RANGE[0], FINAL_SEGMENT_LEN_RANGE[1] + 1)))
        layout = _build_route(dir_key, lengths)
        if layout is None:
            continue

        options = _translation_options(layout, h, w)
        if not options:
            continue
        rng.shuffle(options)

        for tr, tc in options[:30]:
            arrow_cells = _translate(layout["arrow"], tr, tc)  # type: ignore[arg-type]
            path_cells = _translate(layout["path"], tr, tc)  # type: ignore[arg-type]
            touches = [((int(r + tr), int(c + tc)), incoming) for (r, c), incoming in layout["touches"]]  # type: ignore[index]
            tip = (int(tr), int(tc))

            local_structure = set(arrow_cells)
            local_blocked = set(arrow_cells) | set(path_cells)
            if any(cell in global_moat for cell in local_structure | set(path_cells)):
                continue
            if any(cell in global_structures or cell in global_paths or cell in global_walls for cell in local_structure | set(path_cells)):
                continue

            wall_segments: list[list[tuple[int, int]]] = []
            wall_cells: set[tuple[int, int]] = set()
            ok = True
            for touch, incoming in touches:
                segment = _sample_wall_segment(
                    touch,
                    incoming,
                    grid.shape,
                    rng,
                    global_moat,
                    local_blocked,
                    global_walls | wall_cells,
                )
                if segment is None:
                    ok = False
                    break
                wall_segments.append(segment)
                wall_cells.update(segment)
                local_blocked.update(segment)
            if not ok:
                continue

            temp = grid.copy()
            for r, c in arrow_cells:
                temp[r, c] = int(color)
            for r, c in wall_cells:
                temp[r, c] = GREY

            wall_map = _detect_walls(temp)
            arrow_lookup = set(global_structures - global_walls) | set(arrow_cells)
            actual_path, actual_hits = _route_signature(temp.shape, wall_map, arrow_lookup, tip, dir_key)
            if actual_path != path_cells:
                continue
            if actual_hits != [touch for touch, _ in touches]:
                continue

            return {
                "color": int(color),
                "tip": tip,
                "dir": str(dir_key),
                "arrow_cells": tuple(arrow_cells),
                "path_cells": tuple(path_cells),
                "wall_cells": tuple(sorted(wall_cells)),
            }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    bg = int(rng.choice(BG_CHOICES))

    grid = np.full((h, w), int(bg), dtype=int)
    arrow_pool = [color for color in BG_CHOICES if int(color) != bg]
    if len(arrow_pool) < 3:
        return None

    n_arrows = int(rng.choice(ARROW_COUNT_CHOICES))
    n_arrows = min(n_arrows, len(arrow_pool))
    arrow_colors = [int(x) for x in rng.choice(np.array(arrow_pool, dtype=int), size=n_arrows, replace=False)]

    bounce_counts = [int(rng.choice(BOUNCE_CHOICES)) for _ in range(n_arrows)]
    if max(bounce_counts) == 0:
        bounce_counts[int(rng.integers(0, n_arrows))] = int(rng.choice((1, 2, 3)))
    if max(bounce_counts) < 2 and bool(rng.integers(0, 2)):
        bounce_counts[int(rng.integers(0, n_arrows))] = int(rng.choice((2, 3)))

    dir_keys = [str(rng.choice(DIR_KEYS)) for _ in range(n_arrows)]
    order = sorted(range(n_arrows), key=lambda idx: (-bounce_counts[idx], idx))

    global_structures: set[tuple[int, int]] = set()
    global_paths: set[tuple[int, int]] = set()
    global_walls: set[tuple[int, int]] = set()

    placed: list[dict[str, object]] = []
    for idx in order:
        plan = _plan_arrow(
            grid,
            bg,
            arrow_colors[idx],
            bounce_counts[idx],
            dir_keys[idx],
            rng,
            global_structures,
            global_paths,
            global_walls,
        )
        if plan is None:
            return None

        for r, c in plan["arrow_cells"]:  # type: ignore[index]
            grid[r, c] = int(plan["color"])
        for r, c in plan["wall_cells"]:  # type: ignore[index]
            grid[r, c] = GREY

        global_structures.update(plan["arrow_cells"])  # type: ignore[arg-type]
        global_structures.update(plan["wall_cells"])  # type: ignore[arg-type]
        global_paths.update(plan["path_cells"])  # type: ignore[arg-type]
        global_walls.update(plan["wall_cells"])  # type: ignore[arg-type]
        placed.append(plan)

    if not placed or not global_walls:
        return None
    if np.count_nonzero(grid == bg) <= (h * w) // 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == RED) or not np.any(out == BLUE):
        return None
    if len(_find_arrows(grid, bg)) != len(placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-bounce input")
