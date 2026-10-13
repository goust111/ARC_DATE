from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


RED = 2
MIRROR = 5
PINK = 6

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS4 = CARDINALS
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
TRAIL_LEN_RANGE = (2, 5)
SEGMENT_BG_RANGE = (1, 6)
REFLECTION_COUNTS = [0, 1, 1, 2, 2, 3, 3]
BLOCK_SIZE_RANGE = (2, 4)
EXTRA_MIRROR_COUNT = (0, 2)
EXTRA_BLOCK_COUNT = (0, 2)
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, dirs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in dirs]


def _same_color_component(grid: np.ndarray, start: tuple[int, int], *, connectivity: int) -> set[tuple[int, int]]:
    color = int(grid[start])
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen = {start}
    q = deque([start])
    h, w = grid.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, dirs):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _components_of_color(mask: np.ndarray, *, connectivity: int) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or (r, c) in seen:
                continue
            comp = {(r, c)}
            seen.add((r, c))
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors(rr, cc, dirs):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _trail_and_ray(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    red_cells = np.argwhere(grid == RED)
    assert len(red_cells) == 1, "expected a unique red source"
    sr, sc = map(int, red_cells[0])

    pink_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == PINK)}
    assert pink_cells, "expected a pink trail"

    adj_dirs = []
    for dr, dc in CARDINALS:
        if (sr + dr, sc + dc) in pink_cells:
            adj_dirs.append((dr, dc))
    assert len(adj_dirs) == 1, "trail must touch source in exactly one cardinal direction"

    trail_dir = adj_dirs[0]
    dr, dc = trail_dir
    expected: set[tuple[int, int]] = set()
    rr, cc = sr + dr, sc + dc
    while (rr, cc) in pink_cells:
        expected.add((rr, cc))
        rr += dr
        cc += dc
    assert expected == pink_cells, "pink trail must be one contiguous straight ray"
    ray_dir = (-dr, -dc)
    return (sr, sc), trail_dir, ray_dir, pink_cells


def _mirror_orientation_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirror_comps = _components_of_color(grid == MIRROR, connectivity=8)
    orient: dict[tuple[int, int], str] = {}
    for comp in mirror_comps:
        assert len(comp) >= 2, "mirror segment must be at least two cells long"
        diffs = {r - c for r, c in comp}
        sums = {r + c for r, c in comp}
        if len(diffs) == 1 and len(sums) > 1:
            kind = "\\"
        elif len(sums) == 1 and len(diffs) > 1:
            kind = "/"
        else:
            raise AssertionError("mirror component must be a straight diagonal segment")
        for cell in comp:
            orient[cell] = kind
    return orient


def _reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    dr, dc = direction
    if mirror_kind == "/":
        return -dc, -dr
    if mirror_kind == "\\":
        return dc, dr
    raise AssertionError("unknown mirror kind")


def _trace_hit(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    *,
    mirror_mode: str,
) -> dict | None:
    bg = _get_bg(grid)
    mirrors = _mirror_orientation_map(grid)
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    seen_states: set[tuple[int, int, int, int]] = set()
    path_cells: list[tuple[int, int]] = []
    reflection_count = 0

    while True:
        state = (r, c, dr, dc)
        if state in seen_states:
            raise AssertionError("beam entered a loop")
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return None

        val = int(grid[nr, nc])
        if val == bg:
            path_cells.append((nr, nc))
            r, c = nr, nc
            continue

        if (nr, nc) in mirrors:
            if mirror_mode == "reflect":
                dr, dc = _reflect((dr, dc), mirrors[(nr, nc)])
                reflection_count += 1
            elif mirror_mode != "straight":
                raise AssertionError("unknown mirror mode")
            r, c = nr, nc
            continue

        target = _same_color_component(grid, (nr, nc), connectivity=4)
        return {
            "hit": (nr, nc),
            "hit_color": val,
            "target": target,
            "path_cells": path_cells,
            "reflections": reflection_count,
        }


def _apply_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    source, _trail_dir, ray_dir, _pink = _trail_and_ray(grid)
    trace = _trace_hit(grid, source, ray_dir, mirror_mode="reflect")
    assert trace is not None, "beam must hit a target block"
    out = np.array(grid, dtype=int, copy=True)
    for r, c in trace["path_cells"]:
        out[r, c] = RED
    for r, c in trace["target"]:
        out[r, c] = RED
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """Cast the red beam opposite the pink tail, reflect it on gray diagonal mirrors, and repaint the first hit block red.

    Input structure:
    - A dominant background contains one red source pixel and a straight pink tail that touches the source horizontally or vertically.
    - Gray mirror pixels form one or more diagonal 8-connected line segments, each oriented as either "/" or "\\".
    - Several other monochrome 4-connected colored blocks are present, and one of them lies at the end of the reflected beam path.

    Rule:
    - Infer the initial beam direction as the direction opposite the pink tail.
    - Advance the beam cell by cell: every visited background cell becomes red.
    - When the beam enters a gray mirror cell, keep that mirror unchanged and rotate the beam by 90 degrees according to the mirror segment's diagonal orientation.
    - The beam stops at the first non-background, non-mirror pixel it reaches, and the entire 4-connected monochrome block containing that pixel is recolored red.

    Key insight:
    - The pink tail shows where the source came from rather than where it should go, so the correct target is found only by reversing that cue and tracing the full reflected polyline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out, _trace = _apply_trace(g)
    return out


def _step(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], bg_len: int) -> list[tuple[int, int]]:
    return [_step(start, direction, k) for k in range(1, bg_len + 1)]


def _turn_options(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _turn_to_mirror(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    for kind in ("/", "\\"):
        if _reflect(in_dir, kind) == out_dir:
            return kind
    raise AssertionError("turn is not a 90-degree mirror reflection")


def _mirror_diag_dirs(kind: str) -> list[tuple[int, int]]:
    return [(1, -1), (-1, 1)] if kind == "/" else [(1, 1), (-1, -1)]


def _object_clear(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    occupied_nonbg: set[tuple[int, int]],
    blocked_bg: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied_nonbg or (r, c) in blocked_bg:
            return False
    for r, c in cells:
        for nr, nc in _neighbors(r, c, DIRS8):
            if (nr, nc) in occupied_nonbg and (nr, nc) not in cells:
                return False
    return True


def _rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _edge_hit_rectangles(
    hit: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    h, w = shape
    hr, hc = hit
    candidates: list[set[tuple[int, int]]] = []
    for height in range(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1] + 1):
        for width in range(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1] + 1):
            if direction == (0, 1):
                for off in range(height):
                    top = hr - off
                    left = hc
                    if 0 <= top and top + height <= h and left + width <= w:
                        candidates.append(_rect_cells(top, left, height, width))
            elif direction == (0, -1):
                for off in range(height):
                    top = hr - off
                    left = hc - width + 1
                    if 0 <= top and top + height <= h and 0 <= left:
                        candidates.append(_rect_cells(top, left, height, width))
            elif direction == (1, 0):
                for off in range(width):
                    top = hr
                    left = hc - off
                    if top + height <= h and 0 <= left and left + width <= w:
                        candidates.append(_rect_cells(top, left, height, width))
            else:
                for off in range(width):
                    top = hr - height + 1
                    left = hc - off
                    if 0 <= top and 0 <= left and left + width <= w:
                        candidates.append(_rect_cells(top, left, height, width))
    random.shuffle(candidates)
    return candidates


def _place_block_at_hit(
    hit: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    occupied_nonbg: set[tuple[int, int]],
    blocked_bg: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for cells in _edge_hit_rectangles(hit, direction, shape):
        if _object_clear(cells, shape, occupied_nonbg, blocked_bg):
            return cells
    return None


def _random_free_rect(
    shape: tuple[int, int],
    occupied_nonbg: set[tuple[int, int]],
    blocked_bg: set[tuple[int, int]],
    *,
    center_limit: tuple[int, int] | None = None,
    max_manhattan: int | None = None,
) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        height = random.randint(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1])
        width = random.randint(BLOCK_SIZE_RANGE[0], BLOCK_SIZE_RANGE[1])
        top = random.randint(0, h - height)
        left = random.randint(0, w - width)
        cells = _rect_cells(top, left, height, width)
        if center_limit is not None and max_manhattan is not None:
            sr, sc = center_limit
            dist = min(abs(r - sr) + abs(c - sc) for r, c in cells)
            if dist >= max_manhattan:
                continue
        if _object_clear(cells, shape, occupied_nonbg, blocked_bg):
            return cells
    return None


def _place_extra_mirror(
    shape: tuple[int, int],
    occupied_nonbg: set[tuple[int, int]],
    blocked_bg: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(PLACE_ATTEMPTS):
        kind = random.choice(["/", "\\"])
        base_r = random.randint(1, h - 2)
        base_c = random.randint(1, w - 2)
        dr, dc = random.choice(_mirror_diag_dirs(kind))
        length = random.randint(2, 4)
        cells = {(_step((base_r, base_c), (dr, dc), k)) for k in range(length)}
        if _object_clear(cells, shape, occupied_nonbg, blocked_bg):
            return cells
    return None


def _plan_scene() -> dict | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([0, 1, 3, 4, 7, 8, 9])
    shape = (h, w)

    reflections_goal = random.choice(REFLECTION_COUNTS)
    trail_len = random.randint(*TRAIL_LEN_RANGE)
    ray_dir = random.choice(CARDINALS)
    trail_dir = (-ray_dir[0], -ray_dir[1])

    for _ in range(PLACE_ATTEMPTS):
        sr = random.randint(3, h - 4)
        sc = random.randint(3, w - 4)
        trail_cells = {_step((sr, sc), trail_dir, k) for k in range(1, trail_len + 1)}
        if not all(_in_bounds(shape, r, c) for r, c in trail_cells):
            continue

        source = (sr, sc)
        occupied_nonbg = {source} | trail_cells
        blocked_bg = trail_cells.copy()
        path_cells: set[tuple[int, int]] = set()
        mirrors: list[dict] = []

        current = source
        current_dir = ray_dir
        ok = True

        for _mirror_idx in range(reflections_goal):
            placed = False
            turn_dirs = _turn_options(current_dir)
            random.shuffle(turn_dirs)
            bg_lens = list(range(SEGMENT_BG_RANGE[0], SEGMENT_BG_RANGE[1] + 1))
            random.shuffle(bg_lens)

            for out_dir in turn_dirs:
                mirror_kind = _turn_to_mirror(current_dir, out_dir)
                diag_dirs = _mirror_diag_dirs(mirror_kind)
                random.shuffle(diag_dirs)
                for bg_len in bg_lens:
                    bg_segment = _segment_cells(current, current_dir, bg_len)
                    mirror_cell = _step(current, current_dir, bg_len + 1)
                    if not _in_bounds(shape, *mirror_cell):
                        continue
                    if any(cell in occupied_nonbg or cell in path_cells for cell in bg_segment):
                        continue
                    if mirror_cell in occupied_nonbg or mirror_cell in path_cells:
                        continue

                    comp_cells = None
                    for diag in diag_dirs:
                        cells = {mirror_cell, _step(mirror_cell, diag, 1)}
                        if random.random() < 0.45:
                            cells.add(_step(mirror_cell, diag, 2))
                        if _object_clear(cells, shape, occupied_nonbg, path_cells | blocked_bg):
                            comp_cells = cells
                            break
                    if comp_cells is None:
                        continue

                    next_probe = _step(mirror_cell, out_dir, SEGMENT_BG_RANGE[0] + 1)
                    if not _in_bounds(shape, *next_probe):
                        continue
                    if next_probe in occupied_nonbg or next_probe in path_cells:
                        continue

                    path_cells.update(bg_segment)
                    occupied_nonbg.update(comp_cells)
                    mirrors.append(
                        {
                            "cells": comp_cells,
                            "kind": mirror_kind,
                            "cell": mirror_cell,
                            "in_dir": current_dir,
                            "out_dir": out_dir,
                        }
                    )
                    current = mirror_cell
                    current_dir = out_dir
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        target_hit = None
        target_cells = None
        bg_lens = list(range(SEGMENT_BG_RANGE[0], SEGMENT_BG_RANGE[1] + 1))
        random.shuffle(bg_lens)
        for bg_len in bg_lens:
            bg_segment = _segment_cells(current, current_dir, bg_len)
            hit = _step(current, current_dir, bg_len + 1)
            if not _in_bounds(shape, *hit):
                continue
            if any(cell in occupied_nonbg or cell in path_cells for cell in bg_segment):
                continue
            if hit in occupied_nonbg or hit in path_cells:
                continue
            block = _place_block_at_hit(hit, current_dir, shape, occupied_nonbg, path_cells | blocked_bg)
            if block is None:
                continue
            path_cells.update(bg_segment)
            occupied_nonbg.update(block)
            target_hit = hit
            target_cells = block
            break

        if target_cells is None or target_hit is None:
            continue

        blocks: list[set[tuple[int, int]]] = [target_cells]

        if mirrors:
            first = mirrors[0]
            for bg_len in range(1, SEGMENT_BG_RANGE[1] + 1):
                hit = _step(first["cell"], first["in_dir"], bg_len + 1)
                if not _in_bounds(shape, *hit):
                    continue
                straight_cells = {_step(first["cell"], first["in_dir"], k) for k in range(1, bg_len + 1)}
                if straight_cells & occupied_nonbg or straight_cells & path_cells:
                    continue
                decoy = _place_block_at_hit(hit, first["in_dir"], shape, occupied_nonbg, path_cells | blocked_bg)
                if decoy is None:
                    continue
                occupied_nonbg.update(decoy)
                blocks.append(decoy)
                break

        target_dist = abs(target_hit[0] - sr) + abs(target_hit[1] - sc)
        close_decoy = _random_free_rect(
            shape,
            occupied_nonbg,
            path_cells | blocked_bg,
            center_limit=source,
            max_manhattan=max(3, target_dist),
        )
        if close_decoy is None:
            continue
        occupied_nonbg.update(close_decoy)
        blocks.append(close_decoy)

        for _ in range(random.randint(*EXTRA_BLOCK_COUNT)):
            extra = _random_free_rect(shape, occupied_nonbg, path_cells | blocked_bg)
            if extra is not None:
                occupied_nonbg.update(extra)
                blocks.append(extra)

        extra_mirrors: list[set[tuple[int, int]]] = []
        for _ in range(random.randint(*EXTRA_MIRROR_COUNT)):
            extra = _place_extra_mirror(shape, occupied_nonbg, path_cells | blocked_bg)
            if extra is not None:
                occupied_nonbg.update(extra)
                extra_mirrors.append(extra)

        if len(blocks) < 3:
            continue

        palette = [c for c in range(10) if c not in {bg, RED, PINK, MIRROR}]
        if len(blocks) > len(palette):
            continue
        random.shuffle(palette)
        block_colors = palette[: len(blocks)]

        grid = np.full(shape, bg, dtype=int)
        grid[source] = RED
        for cell in trail_cells:
            grid[cell] = PINK
        for mirror in mirrors:
            for cell in mirror["cells"]:
                grid[cell] = MIRROR
        for cells in extra_mirrors:
            for cell in cells:
                grid[cell] = MIRROR
        for cells, color in zip(blocks, block_colors):
            for cell in cells:
                grid[cell] = color

        return {
            "grid": grid,
            "source": source,
            "trail_dir": trail_dir,
            "ray_dir": ray_dir,
            "path_cells": path_cells,
            "target_hit": target_hit,
            "target_cells": target_cells,
            "block_sets": blocks,
        }
    return None


def _foreground_blocks(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    bg = _get_bg(grid)
    reserved = {bg, RED, PINK, MIRROR}
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    blocks: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or int(grid[r, c]) in reserved:
                continue
            comp = _same_color_component(grid, (r, c), connectivity=4)
            seen |= comp
            blocks.append(comp)
    return blocks


def _semantic_ok(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    if int(np.count_nonzero(grid == bg)) <= (grid.size // 2):
        return False

    try:
        source, trail_dir, ray_dir, pink = _trail_and_ray(grid)
        trace = _trace_hit(grid, source, ray_dir, mirror_mode="reflect")
    except AssertionError:
        return False
    if trace is None:
        return False
    if len(trace["path_cells"]) < 2:
        return False
    if trace["hit_color"] in {bg, RED, PINK, MIRROR}:
        return False

    blocks = _foreground_blocks(grid)
    if len(blocks) < 3:
        return False

    target = trace["target"]
    source_dist = min(abs(r - source[0]) + abs(c - source[1]) for r, c in target)
    closer_exists = False
    for comp in blocks:
        if comp == target:
            continue
        dist = min(abs(r - source[0]) + abs(c - source[1]) for r, c in comp)
        if dist < source_dist:
            closer_exists = True
            break
    if not closer_exists:
        return False

    wrong = _trace_hit(grid, source, trail_dir, mirror_mode="reflect")
    if wrong is None or not wrong["target"].issubset(pink):
        return False

    if trace["reflections"] > 0:
        straight = _trace_hit(grid, source, ray_dir, mirror_mode="straight")
        if straight is not None and straight["target"] == target:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    diff = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
    if not target.issubset(diff):
        return False
    if not set(trace["path_cells"]).issubset(diff):
        return False
    if any(grid[r, c] != bg for r, c in trace["path_cells"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        planned = _plan_scene()
        if planned is None:
            continue
        grid = planned["grid"]
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirror-beam ARC input")
