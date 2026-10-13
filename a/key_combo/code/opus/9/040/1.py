from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_at(grid: np.ndarray, start: tuple[int, int], color: int) -> list[tuple[int, int]]:
    sr, sc = start
    if int(grid[sr, sc]) != int(color):
        return []
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in seen or not _in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) != int(color):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return comp


def _all_blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == BLUE):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        comp = _component_at(grid, cell, BLUE)
        seen.update(comp)
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _shortest_random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    out = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        out.append((r, c))
    return out


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _grow_mask(
    h: int,
    w: int,
    required: set[tuple[int, int]],
    *,
    area_lo: int = 6,
    area_hi: int = 12,
) -> np.ndarray | None:
    max_area = min(area_hi, h * w - 1)
    if max_area < area_lo:
        return None

    req_list = list(required)
    for _ in range(120):
        random.shuffle(req_list)
        cells = {req_list[0]}
        for target in req_list[1:]:
            anchor = min(cells, key=lambda rc: abs(rc[0] - target[0]) + abs(rc[1] - target[1]))
            cells.update(_shortest_random_path(anchor, target))

        low = max(area_lo, len(cells))
        if low > max_area:
            continue
        target_area = random.randint(low, max_area)

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue
        if not _is_connected_cells(cells):
            continue
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if np.all(mask):
            continue
        if not any(mask[r].sum() < w for r in range(h)):
            continue
        if not any(mask[:, c].sum() < h for c in range(w)):
            continue
        return mask
    return None


def _simulate(
    grid: np.ndarray,
    *,
    order: str = "ltr",
    ids: np.ndarray | None = None,
    return_trace: bool = False,
):
    g = np.array(grid, dtype=int, copy=True)
    id_grid = np.zeros_like(g, dtype=int) if ids is None else np.array(ids, dtype=int, copy=True)

    seeds = sorted(int(c) for c in np.flatnonzero(g[-1] == RED))
    process_cols = seeds if order == "ltr" else list(reversed(seeds))
    trace: list[dict] = []

    for seed_order, seed_col in enumerate(process_cols):
        r = g.shape[0] - 1
        c = int(seed_col)
        while r > 0:
            above = int(g[r - 1, c])
            if above == BLACK:
                r -= 1
                g[r, c] = RED
                continue
            if above == RED:
                break
            if above != BLUE:
                raise AssertionError(f"unexpected color {above} above a path")

            comp = _component_at(g, (r - 1, c), BLUE)
            comp_data = [(rr, cc, int(id_grid[rr, cc])) for rr, cc in comp]
            comp_ids = {cell_id for _rr, _cc, cell_id in comp_data if cell_id != 0}

            split_cols_before = sorted({cc for _rr, cc in comp})
            for rr, cc, _cell_id in comp_data:
                g[rr, cc] = BLACK
                id_grid[rr, cc] = 0

            shifted_positions: list[tuple[int, int]] = []
            for rr, cc, cell_id in comp_data:
                if cc == c:
                    continue
                nc = cc - 1 if cc < c else cc + 1
                if not _in_bounds(g.shape, rr, nc):
                    continue
                if int(g[rr, nc]) != BLACK:
                    raise AssertionError("shifted fragment overlapped an existing non-background cell")
                g[rr, nc] = BLUE
                id_grid[rr, nc] = cell_id
                shifted_positions.append((rr, nc))

            if return_trace:
                trace.append(
                    {
                        "seed_col": c,
                        "seed_order": seed_order,
                        "row_below": r,
                        "obj_ids": sorted(comp_ids),
                        "pre_split_cols": split_cols_before,
                        "created_seed_cols": sorted({nc for _rr, nc in shifted_positions if nc in seeds and nc != c}),
                        "shifted_positions": shifted_positions,
                    }
                )
            # Stay in place; the next loop step climbs through the cleared cut column.

    if return_trace:
        return g, trace, id_grid
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow red paths upward and split any crossed blue object into left and right fragments.

    Input structure:
    - The background is black.
    - The bottom row contains 2-3 red seed pixels, read from left to right.
    - Blue pixels form 2-3 separated 4-connected objects away from the top and bottom edges.

    Rule:
    - Process the red seeds from left to right, growing each path straight upward through black cells.
    - If the cell directly above the current path head is blue, take that whole 4-connected blue object, erase every blue pixel on the path column, shift the object's strictly-left pixels one cell left, and shift its strictly-right pixels one cell right.
    - After the cut, keep growing upward through the cleared column; if the next cell is already red, stop that path; reaching row 0 also stops it.

    Key insight:
    - A path is both a vertical painter and a destructive cutter, so earlier cuts can move blue fragments into later path columns and change how those later paths interact with the remaining blue layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _simulate(g, order="ltr")


def _choose_seed_columns(seed_count: int, width: int) -> list[int] | None:
    gaps = [random.randint(5, 7) for _ in range(seed_count - 1)]
    span = sum(gaps)
    if width < span + 3:
        return None
    c1 = random.randint(1, width - span - 2)
    cols = [c1]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    if cols[-1] >= width - 1:
        return None
    return cols


def _make_coupled_descriptor(source_col: int, target_col: int) -> dict | None:
    gap = target_col - source_col
    width = random.randint(gap + 1, gap + 2)
    left = target_col - width
    cut = source_col - left
    if left < 0 or cut <= 0 or cut >= width - 1:
        return None
    height = random.choice([2, 3, 4])
    required = {
        (0, 0),
        (height - 1, width - 1),
        (height - 1, cut),
        (0, cut - 1),
        (0, cut + 1),
        (0, width - 1),
    }
    mask = _grow_mask(height, width, required)
    if mask is None or not mask[:, cut].any() or not mask[:, width - 1].any():
        return None
    return {
        "kind": "coupled",
        "left": left,
        "width": width,
        "height": height,
        "mask": mask,
    }


def _make_simple_descriptor(path_col: int, total_width: int) -> dict | None:
    options: list[tuple[str, int, int]] = []
    for orientation in ["wide", "tall"]:
        width_choices = range(3, 6) if orientation == "wide" else range(3, 5)
        for width in width_choices:
            for cut in range(1, width - 1):
                left = path_col - cut
                right = left + width - 1
                if 0 <= left and right < total_width:
                    options.append((orientation, width, cut))
    if not options:
        return None

    orientation, width, cut = random.choice(options)
    if orientation == "wide":
        height = random.choice([2, 3, 4])
    else:
        height = random.choice([5, 6])
    if width >= height and orientation == "tall":
        height = width + random.choice([1, 2])
    if width == height:
        height += 1
    left = path_col - cut
    required = {
        (0, 0),
        (height - 1, width - 1),
        (height - 1, cut),
        (0, cut - 1),
        (0, cut + 1),
    }
    mask = _grow_mask(height, width, required)
    if mask is None or not mask[:, cut].any():
        return None
    return {
        "kind": "simple",
        "left": left,
        "width": width,
        "height": height,
        "mask": mask,
    }


def _make_uncrossed_descriptor(seed_cols: list[int], width: int, side: str | None = None) -> dict | None:
    orientation = random.choice(["wide", "tall"])
    obj_w = random.randint(3, 5) if orientation == "wide" else random.randint(3, 4)
    obj_h = random.choice([2, 3, 4]) if orientation == "wide" else random.choice([5, 6])
    if obj_w == obj_h:
        obj_h += 1

    windows = []
    leftmost = seed_cols[0]
    rightmost = seed_cols[-1]
    if side in (None, "left") and leftmost >= obj_w + 2:
        windows.append((1, leftmost - obj_w - 1))
    if side in (None, "right") and width - rightmost - 1 >= obj_w + 1:
        windows.append((rightmost + 2, width - obj_w))
    if not windows:
        return None
    lo, hi = random.choice(windows)
    if lo > hi:
        return None
    left = random.randint(lo, hi)
    required = {
        (0, 0),
        (0, obj_w - 1),
        (obj_h - 1, 0),
        (obj_h - 1, obj_w - 1),
    }
    mask = _grow_mask(obj_h, obj_w, required)
    if mask is None:
        return None
    return {
        "kind": "uncrossed",
        "left": left,
        "width": obj_w,
        "height": obj_h,
        "mask": mask,
    }


def _assign_tops(height: int, descs: list[dict], *, path2_variant: str | None = None) -> bool:
    if len(descs) == 2:
        prefs = ["top", "low"]
    elif len(descs) == 3:
        if path2_variant == "own_before_chain":
            prefs = ["mid", "low", "top"]
        else:
            prefs = ["mid", "top", "low"]
    else:
        prefs = ["mid"] * len(descs)

    def slot_top(label: str, obj_h: int) -> int | None:
        if label == "top":
            top = 1
        elif label == "mid":
            top = max(3, height // 2 - obj_h // 2)
        else:
            top = height - obj_h - 2
        if not (1 <= top <= height - obj_h - 1):
            return None
        return top

    used: list[tuple[int, int, int, int]] = []
    for i, desc in enumerate(descs):
        h = desc["height"]
        w = desc["width"]
        left = desc["left"]
        label = prefs[i] if i < len(prefs) else "mid"
        top = slot_top(label, h)
        if top is None:
            return False
        bbox = (top, left, top + h - 1, left + w - 1)
        if any(_expanded_overlap(bbox, old) for old in used):
            return False
        used.append(bbox)
        desc["top"] = top
    return True


def _paint_objects(shape: tuple[int, int], descs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros(shape, dtype=int)
    ids = np.zeros(shape, dtype=int)
    for obj_id, desc in enumerate(descs, start=1):
        top = desc["top"]
        left = desc["left"]
        mask = desc["mask"]
        h, w = mask.shape
        grid[top : top + h, left : left + w][mask] = BLUE
        ids[top : top + h, left : left + w][mask] = obj_id
    return grid, ids


def _input_constraints_ok(grid: np.ndarray, seed_cols: list[int]) -> bool:
    comps = _all_blue_components(grid)
    if len(comps) < 2 or len(comps) > 3:
        return False
    if int(np.count_nonzero(grid[-1] == RED)) != len(seed_cols):
        return False
    if any(int(grid[0, c]) != BLACK for c in range(grid.shape[1])):
        return False

    bboxes = []
    for comp in comps:
        area = len(comp)
        r0, c0, r1, c1 = _bbox(comp)
        if not 6 <= area <= 12:
            return False
        if r0 == 0 or r1 == grid.shape[0] - 1:
            return False
        if c1 - c0 + 1 < 3:
            return False
        if (r1 - r0) == (c1 - c0):
            return False
        crossings = sum(c0 <= c <= c1 for c in seed_cols)
        if crossings > 1:
            return False
        bboxes.append((r0, c0, r1, c1))

    for i, a in enumerate(bboxes):
        for b in bboxes[i + 1 :]:
            if _expanded_overlap(a, b):
                return False
    return True


def _has_chain_interaction(trace: list[dict]) -> bool:
    for i, event in enumerate(trace):
        if not event["created_seed_cols"]:
            continue
        for later in trace[i + 1 :]:
            if later["seed_col"] not in event["created_seed_cols"]:
                continue
            if set(event["obj_ids"]) & set(later["obj_ids"]):
                return True
    return False


def _semantic_checks(inp: np.ndarray, ids: np.ndarray) -> bool:
    seed_cols = sorted(int(c) for c in np.flatnonzero(inp[-1] == RED))
    if not (2 <= len(seed_cols) <= 3):
        return False
    if any((b - a) < 5 for a, b in zip(seed_cols, seed_cols[1:])):
        return False
    if not _input_constraints_ok(inp, seed_cols):
        return False

    try:
        forward, trace, _ = _simulate(inp, ids=ids, return_trace=True)
    except AssertionError:
        return False

    if len(trace) < 2:
        return False
    if not any(event["created_seed_cols"] for event in trace):
        return False
    if not _has_chain_interaction(trace):
        return False
    try:
        reverse = _simulate(inp, order="rtl")
        if np.array_equal(forward, reverse):
            return False
    except AssertionError:
        pass
    if not np.any(forward == RED):
        return False

    outputsig = tuple(tuple(map(int, row)) for row in forward)
    if outputsig == tuple(tuple(map(int, row)) for row in inp):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    scenario = random.choice(
        [
            "two_paths_two_objs",
            "two_paths_three_objs",
            "three_paths_two_objs",
            "three_paths_three_objs",
        ]
    )
    seed_count = 2 if scenario.startswith("two_") else 3
    height = random.randint(15, 18)
    width = random.randint(18, 20)
    seed_cols = _choose_seed_columns(seed_count, width)
    if seed_cols is None:
        return None

    object_count = 2 if scenario in {"two_paths_two_objs", "three_paths_two_objs"} else 3
    descs: list[dict] = []
    for _ in range(object_count):
        desc = _make_coupled_descriptor(seed_cols[0], seed_cols[1])
        if desc is None:
            return None
        descs.append(desc)

    path2_variant = "own_before_chain" if object_count == 2 else "chain_before_own"
    if not _assign_tops(height, descs, path2_variant=path2_variant):
        return None

    blue_grid, ids = _paint_objects((height, width), descs)
    inp = blue_grid.copy()
    for c in seed_cols:
        if int(inp[-1, c]) != BLACK:
            return None
        inp[-1, c] = RED

    if not _semantic_checks(inp, ids):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _build_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid path-split ARC input")
