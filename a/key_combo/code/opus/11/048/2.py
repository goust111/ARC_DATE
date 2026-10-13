from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
)

GRID_ROWS = (20, 28)
GRID_COLS = (20, 28)
START_SEG_H = (2, 3)
START_GAP = (4, 7)
CHAIN_GAP = (2, 5)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _side_offsets(h: int, w: int, side: tuple[int, int]) -> list[tuple[int, int]]:
    if side == UP:
        return [(0, c) for c in range(w)]
    if side == DOWN:
        return [(h - 1, c) for c in range(w)]
    if side == LEFT:
        return [(r, 0) for r in range(h)]
    if side == RIGHT:
        return [(r, w - 1) for r in range(h)]
    raise ValueError("bad side")


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _outward_dir_from_port(blob_cells: set[tuple[int, int]], port: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(blob_cells)
    pr, pc = port
    if pr < r0:
        return UP
    if pr > r1:
        return DOWN
    if pc < c0:
        return LEFT
    if pc > c1:
        return RIGHT
    raise AssertionError("port must lie outside the blob bbox")


def _find_start(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != YELLOW:
                continue
            rr = r + 1
            seg: list[tuple[int, int]] = []
            while rr < h and int(grid[rr, c]) == RED:
                seg.append((rr, c))
                rr += 1
            if rr == h and 2 <= len(seg) <= 3:
                candidates.append(((r, c), seg))
    assert len(candidates) == 1, "expected one yellow start cap above a bottom red segment"
    return candidates[0]


def _extract_blob_infos(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], list[set[tuple[int, int]]]]:
    blue_comps = _components(grid == BLUE)
    grey_comps = _components(grid == GREY)
    cell_to_blob: dict[tuple[int, int], int] = {}
    infos: list[dict] = []
    green_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GREEN)}

    for blob_id, cells in enumerate(blue_comps):
        for cell in cells:
            cell_to_blob[cell] = blob_id
        touching_green = [g for g in green_cells if any(n in cells for n in _neighbors4(*g))]
        assert len(touching_green) == 1, "each blob must have exactly one adjacent green port"
        port = touching_green[0]
        gray_count = 0
        for comp in grey_comps:
            if any(any(n in cells for n in _neighbors4(r, c)) for r, c in comp):
                gray_count += 1
        infos.append(
            {
                "id": blob_id,
                "cells": cells,
                "bbox": _bbox(cells),
                "port": port,
                "port_dir": _outward_dir_from_port(cells, port),
                "gray_count": gray_count,
            }
        )
    return infos, cell_to_blob, grey_comps


def _cast_until_blue(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    cell_to_blob: dict[tuple[int, int], int],
    max_steps: int | None,
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    traced: list[tuple[int, int]] = []
    steps = 0
    while True:
        r += dr
        c += dc
        steps += 1
        if not _in_bounds((h, w), r, c):
            return traced, None
        if max_steps is not None and steps > max_steps:
            return traced, None
        if (r, c) in cell_to_blob:
            return traced, cell_to_blob[(r, c)]
        traced.append((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate blue blobs in a gated cascade where each blob's nearby grey-object count sets the next ray length.

    Input structure:
    - Black background with 4-6 compact blue 4-connected blobs and exactly one adjacent green port for each blob.
    - Grey objects are small 4-connected components placed in the one-step 4-neighborhood ring around blobs.
    - A yellow cap sits directly above a short red vertical segment at the bottom, giving the unique upward start ray.

    Rule:
    - Recolor the yellow cap red and cast a red ray upward until it first touches a blue blob, then recolor that whole blob red.
    - For each newly activated blob, count the number of distinct grey 4-connected objects that touch the blob by 4-neighborhood.
    - From the blob's green port, cast a red ray in the port direction for exactly that many cells; if the ray reaches another blue blob within that limit, activate it and continue the cascade.
    - Grey objects and green ports stay unchanged; blobs never reached by the cascade remain blue.

    Key insight:
    - The hidden gate is not geometry alone: each blob must first be identified, then its nearby grey components counted, because that count is the precise length cap that decides whether the next blob is reachable.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    start, start_seg = _find_start(g)
    infos, cell_to_blob, _ = _extract_blob_infos(g)
    by_id = {info["id"]: info for info in infos}

    for r, c in start_seg:
        out[r, c] = RED
    out[start] = RED

    traced, first_blob = _cast_until_blue(g, start, UP, cell_to_blob, None)
    assert first_blob is not None, "start ray must hit a blob"
    for r, c in traced:
        if int(out[r, c]) == BG:
            out[r, c] = RED

    activated: set[int] = set()
    current = first_blob
    while current is not None and current not in activated:
        activated.add(current)
        info = by_id[current]
        for r, c in info["cells"]:
            out[r, c] = RED
        traced, next_blob = _cast_until_blue(g, info["port"], info["port_dir"], cell_to_blob, info["gray_count"])
        for r, c in traced:
            if int(out[r, c]) == BG:
                out[r, c] = RED
        current = next_blob

    return out


def _allowed_out_dirs(incoming: tuple[int, int]) -> list[tuple[int, int]]:
    if incoming == UP:
        return [UP, LEFT, RIGHT]
    if incoming == LEFT:
        return [LEFT, UP]
    if incoming == RIGHT:
        return [RIGHT, UP]
    return [d for d in (UP, LEFT, RIGHT) if d != OPPOSITE[incoming]]


def _blob_valid(
    blob_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> bool:
    h, w = grid_shape
    r0, c0, r1, c1 = _bbox(blob_cells)
    if r0 < 1 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
        return False
    if blob_cells & occupied:
        return False
    for r, c in blob_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _port_cell(blob: dict, out_dir: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    br, bc = blob["r0"] + offset[0], blob["c0"] + offset[1]
    return br + out_dir[0], bc + out_dir[1]


def _port_valid(
    blob_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    port: tuple[int, int],
    grid_shape: tuple[int, int],
) -> bool:
    if not _in_bounds(grid_shape, *port) or port in occupied:
        return False
    occupied_neighbors = {(nr, nc) for nr, nc in _neighbors4(*port) if (nr, nc) in occupied}
    return bool(occupied_neighbors) and occupied_neighbors <= blob_cells


def _path_clear(
    occupied: set[tuple[int, int]],
    port: tuple[int, int],
    direction: tuple[int, int],
    gap: int,
) -> bool:
    dr, dc = direction
    r, c = port
    for step in range(1, gap):
        rr = r + dr * step
        cc = c + dc * step
        if (rr, cc) in occupied:
            return False
    return True


def _place_chain(
    h: int,
    w: int,
    num_blobs: int,
    start_col: int,
    yellow_row: int,
) -> list[dict] | None:
    occupied: set[tuple[int, int]] = {(yellow_row, start_col)}
    occupied.update((r, start_col) for r in range(yellow_row + 1, h))
    blobs: list[dict] = []

    for _ in range(120):
        bh, bw = random.choice(SHAPES)
        hit_off = random.choice(_side_offsets(bh, bw, DOWN))
        hit_row = yellow_row - random.randint(*START_GAP)
        r0 = hit_row - hit_off[0]
        c0 = start_col - hit_off[1]
        cells = _rect_cells(r0, c0, bh, bw)
        if not _blob_valid(cells, occupied, (bh, bw), (h, w)):
            continue
        blob = {"r0": r0, "c0": c0, "h": bh, "w": bw, "cells": cells, "incoming": UP}
        blobs.append(blob)
        occupied.update(cells)
        break
    if not blobs:
        return None

    for _index in range(num_blobs - 1):
        current = blobs[-1]
        placed = False
        dirs = _allowed_out_dirs(current["incoming"])
        dir_pool = [d for d in dirs for _ in range(2 if d == UP else 1)]
        for _ in range(240):
            out_dir = random.choice(dir_pool)
            edge_off = random.choice(_side_offsets(current["h"], current["w"], out_dir))
            port = _port_cell(current, out_dir, edge_off)
            if not _port_valid(current["cells"], occupied, port, (h, w)):
                continue
            gap = random.randint(*CHAIN_GAP)
            next_h, next_w = random.choice(SHAPES)
            hit_side = OPPOSITE[out_dir]
            target = (port[0] + out_dir[0] * gap, port[1] + out_dir[1] * gap)
            hit_off = random.choice(_side_offsets(next_h, next_w, hit_side))
            nr0 = target[0] - hit_off[0]
            nc0 = target[1] - hit_off[1]
            next_cells = _rect_cells(nr0, nc0, next_h, next_w)
            if not _blob_valid(next_cells, occupied | {port}, (next_h, next_w), (h, w)):
                continue
            if not _path_clear(occupied, port, out_dir, gap):
                continue
            current["out_dir"] = out_dir
            current["port"] = port
            current["gap_to_next"] = gap
            occupied.add(port)
            blobs.append(
                {
                    "r0": nr0,
                    "c0": nc0,
                    "h": next_h,
                    "w": next_w,
                    "cells": next_cells,
                    "incoming": out_dir,
                }
            )
            occupied.update(next_cells)
            placed = True
            break
        if not placed:
            return None

    last = blobs[-1]
    for _ in range(120):
        out_dir = random.choice(_allowed_out_dirs(last["incoming"]))
        edge_off = random.choice(_side_offsets(last["h"], last["w"], out_dir))
        port = _port_cell(last, out_dir, edge_off)
        if not _port_valid(last["cells"], occupied, port, (h, w)):
            continue
        last["out_dir"] = out_dir
        last["port"] = port
        last["gap_to_next"] = None
        occupied.add(port)
        return blobs
    return None


def _paint_base(grid: np.ndarray, blobs: list[dict], yellow_row: int, start_col: int) -> None:
    grid[yellow_row, start_col] = YELLOW
    grid[yellow_row + 1 :, start_col] = RED
    for blob in blobs:
        for r, c in blob["cells"]:
            grid[r, c] = BLUE
        pr, pc = blob["port"]
        grid[pr, pc] = GREEN


def _gray_candidates(blob: dict, grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    r0, c0, h, w = blob["r0"], blob["c0"], blob["h"], blob["w"]
    sides = {
        UP: [(r0 - 1, c) for c in range(c0, c0 + w)],
        DOWN: [(r0 + h, c) for c in range(c0, c0 + w)],
        LEFT: [(r, c0 - 1) for r in range(r0, r0 + h)],
        RIGHT: [(r, c0 + w) for r in range(r0, r0 + h)],
    }
    candidates: set[tuple[tuple[int, int], ...]] = set()
    for coords in sides.values():
        free = [cell for cell in coords if int(grid[cell]) == BG]
        for i in range(len(free)):
            candidates.add((free[i],))
        for length in (2, 3):
            for i in range(len(coords) - length + 1):
                run = coords[i : i + length]
                if all(int(grid[cell]) == BG for cell in run):
                    candidates.add(tuple(run))
    return list(candidates)


def _can_add_object(
    obj: tuple[tuple[int, int], ...],
    gray_cells: set[tuple[int, int]],
) -> bool:
    obj_set = set(obj)
    if obj_set & gray_cells:
        return False
    for r, c in obj_set:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in gray_cells:
                return False
    return True


def _choose_gray_objects(
    candidates: list[tuple[tuple[int, int], ...]],
    count: int,
    gray_cells: set[tuple[int, int]],
    require_large: bool,
) -> list[tuple[tuple[int, int], ...]] | None:
    singles = [obj[0] for obj in candidates if len(obj) == 1 and _can_add_object(obj, gray_cells)]
    multis = [obj for obj in candidates if len(obj) > 1 and _can_add_object(obj, gray_cells)]
    random.shuffle(singles)
    random.shuffle(multis)

    def pick_singletons(cells: list[tuple[int, int]], needed: int, blocked: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]] | None:
        def dfs(start_idx: int, chosen: list[tuple[tuple[int, int], ...]], forbidden: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]] | None:
            if len(chosen) == needed:
                return [tuple(obj) for obj in chosen]
            if len(cells) - start_idx < needed - len(chosen):
                return None
            for idx in range(start_idx, len(cells)):
                cell = cells[idx]
                if cell in forbidden:
                    continue
                next_forbidden = set(forbidden)
                next_forbidden.add(cell)
                next_forbidden.update(_neighbors4(*cell))
                chosen.append((cell,))
                found = dfs(idx + 1, chosen, next_forbidden)
                if found is not None:
                    return found
                chosen.pop()
            return None

        return dfs(0, [], set(blocked))

    if require_large:
        for obj in multis:
            blocked = set(gray_cells)
            blocked.update(obj)
            for r, c in obj:
                blocked.update(_neighbors4(r, c))
            singles_chosen = pick_singletons(singles, count - 1, blocked)
            if singles_chosen is not None:
                return [obj] + singles_chosen
        return None

    return pick_singletons(singles, count, set(gray_cells))


def _desired_counts(blobs: list[dict], stop_idx: int, grid_shape: tuple[int, int]) -> list[int] | None:
    h, w = grid_shape
    counts: list[int] = []
    for idx, blob in enumerate(blobs):
        if idx < stop_idx:
            counts.append(int(blob["gap_to_next"]))
            continue
        if idx == stop_idx:
            if idx < len(blobs) - 1:
                counts.append(int(blob["gap_to_next"]) - 1)
            else:
                pr, pc = blob["port"]
                dr, dc = blob["out_dir"]
                steps = 0
                rr, cc = pr, pc
                while True:
                    rr += dr
                    cc += dc
                    if not _in_bounds((h, w), rr, cc):
                        break
                    steps += 1
                if steps == 0:
                    return None
                counts.append(random.randint(1, min(5, steps)))
            continue
        counts.append(random.randint(1, 5))
    if not all(1 <= c <= 5 for c in counts):
        return None
    return counts


def _render_expected(grid: np.ndarray, blobs: list[dict], counts: list[int], stop_idx: int, yellow_row: int, start_col: int) -> np.ndarray:
    out = grid.copy()
    out[yellow_row:, start_col] = RED
    for r in range(yellow_row - 1, -1, -1):
        out[r, start_col] = RED
        if (r, start_col) in blobs[0]["cells"]:
            break
    for idx, blob in enumerate(blobs):
        if idx > stop_idx:
            break
        for r, c in blob["cells"]:
            out[r, c] = RED
        pr, pc = blob["port"]
        dr, dc = blob["out_dir"]
        limit = counts[idx]
        if idx == stop_idx and idx < len(blobs) - 1:
            for step in range(1, limit + 1):
                rr = pr + dr * step
                cc = pc + dc * step
                if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BG:
                    out[rr, cc] = RED
            break
        if idx < len(blobs) - 1:
            for step in range(1, blobs[idx]["gap_to_next"]):
                rr = pr + dr * step
                cc = pc + dc * step
                if int(out[rr, cc]) == BG:
                    out[rr, cc] = RED
        else:
            for step in range(1, limit + 1):
                rr = pr + dr * step
                cc = pc + dc * step
                if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BG:
                    out[rr, cc] = RED
    return out


def _validate_case(
    grid: np.ndarray,
    blobs: list[dict],
    counts: list[int],
    stop_idx: int,
    yellow_row: int,
    start_col: int,
) -> bool:
    infos, cell_to_blob, _ = _extract_blob_infos(grid)
    if len(infos) != len(blobs):
        return False
    expected_counts = {frozenset(blob["cells"]): count for blob, count in zip(blobs, counts)}
    expected_ports = {frozenset(blob["cells"]): blob["port"] for blob in blobs}
    component_to_gen: dict[int, int] = {}
    gen_info: dict[int, dict] = {}
    for info in infos:
        key = frozenset(info["cells"])
        if key not in expected_counts:
            return False
        if info["gray_count"] != expected_counts[key]:
            return False
        if info["port"] != expected_ports[key]:
            return False
        gen_idx = next(idx for idx, blob in enumerate(blobs) if frozenset(blob["cells"]) == key)
        component_to_gen[info["id"]] = gen_idx
        gen_info[gen_idx] = info

    _traced, first_blob = _cast_until_blue(grid, (yellow_row, start_col), UP, cell_to_blob, None)
    if first_blob is None:
        return False
    if component_to_gen[first_blob] != 0:
        return False

    activated_order: list[int] = []
    cur = first_blob
    seen: set[int] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        activated_order.append(component_to_gen[cur])
        info = infos[cur]
        _, cur = _cast_until_blue(grid, info["port"], info["port_dir"], cell_to_blob, info["gray_count"])

    if activated_order != list(range(stop_idx + 1)):
        return False

    for idx in range(stop_idx):
        if counts[idx] != blobs[idx]["gap_to_next"]:
            return False
        _, next_blob = _cast_until_blue(grid, gen_info[idx]["port"], gen_info[idx]["port_dir"], cell_to_blob, counts[idx])
        if next_blob is None or component_to_gen[next_blob] != idx + 1:
            return False
    if stop_idx < len(blobs) - 1 and counts[stop_idx] + 1 != blobs[stop_idx]["gap_to_next"]:
        return False
    _, stop_hit = _cast_until_blue(grid, gen_info[stop_idx]["port"], gen_info[stop_idx]["port_dir"], cell_to_blob, counts[stop_idx])
    if stop_hit is not None:
        return False

    expected = _render_expected(grid, blobs, counts, stop_idx, yellow_row, start_col)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.count_nonzero(actual == RED) <= np.count_nonzero(grid == RED):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    num_blobs = random.randint(4, 6)
    seg_h = random.randint(*START_SEG_H)
    yellow_row = h - seg_h - 1
    start_col = random.randint(3, w - 4)
    blobs = _place_chain(h, w, num_blobs, start_col, yellow_row)
    if blobs is None:
        return None

    stop_idx = random.randrange(num_blobs)
    counts = _desired_counts(blobs, stop_idx, (h, w))
    if counts is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_base(grid, blobs, yellow_row, start_col)

    gray_cells: set[tuple[int, int]] = set()
    must_have_large = random.random() < 0.65
    placed_large = False
    all_candidates = [_gray_candidates(blob, grid) for blob in blobs]
    eligible_large = [idx for idx, (cand, count) in enumerate(zip(all_candidates, counts)) if count <= 4 and any(len(obj) > 1 for obj in cand)]
    if must_have_large and not eligible_large:
        must_have_large = False
    large_blob_idx = random.choice(eligible_large) if must_have_large else -1
    for idx, (blob, count) in enumerate(zip(blobs, counts)):
        candidates = all_candidates[idx]
        require_large = must_have_large and not placed_large and idx == large_blob_idx
        chosen = _choose_gray_objects(candidates, count, gray_cells, require_large)
        if chosen is None and require_large:
            chosen = _choose_gray_objects(candidates, count, gray_cells, False)
        if chosen is None:
            return None
        for obj in chosen:
            if len(obj) > 1:
                placed_large = True
            for cell in obj:
                grid[cell] = GREY
                gray_cells.add(cell)

    if must_have_large and not placed_large:
        return None
    if not _validate_case(grid, blobs, counts, stop_idx, yellow_row, start_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-activation ARC input")
