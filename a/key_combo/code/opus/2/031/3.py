from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
FG_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _collect_anchors(grid: np.ndarray) -> dict[int, list[Coord]]:
    anchors: dict[int, list[Coord]] = {}
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, WALL):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        anchors[color] = sorted(coords)
    return anchors


def _block_cells(anchor: Coord, shape: tuple[int, int]) -> list[Coord]:
    r0, c0 = map(int, anchor)
    h, w = map(int, shape)
    cells: list[Coord] = []
    for r in (r0, r0 + 1):
        for c in (c0, c0 + 1):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    return cells


def _paint_block(out: np.ndarray, anchor: Coord, color: int) -> None:
    for r, c in _block_cells(anchor, out.shape):
        existing = int(out[r, c])
        assert existing in (BG, color), "expanded blocks must not overlap other colors or walls"
        out[r, c] = int(color)


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment endpoints must share a row or column")


def _path_cells(start: Coord, end: Coord, first_axis: str) -> list[Coord]:
    if start[0] == end[0] or start[1] == end[1]:
        return _segment_cells(start, end)
    if first_axis == "h":
        turn = (start[0], end[1])
    elif first_axis == "v":
        turn = (end[0], start[1])
    else:
        raise ValueError("first_axis must be 'h' or 'v'")
    return _segment_cells(start, turn) + _segment_cells(turn, end)[1:]


def _preferred_axis(anchor_a: Coord, anchor_b: Coord) -> str | None:
    dr = abs(int(anchor_b[0]) - int(anchor_a[0]))
    dc = abs(int(anchor_b[1]) - int(anchor_a[1]))
    if dc > dr:
        return "h"
    if dr > dc:
        return "v"
    return None


def _path_clear(scene: np.ndarray, path: list[Coord], color: int) -> bool:
    for r, c in path:
        value = int(scene[r, c])
        if value not in (BG, int(color)):
            return False
    return True


def _candidate_sort_key(candidate: dict, preferred_axis: str | None) -> tuple:
    penalty = 0
    if candidate["turns"] == 1 and preferred_axis is not None and candidate["first_axis"] != preferred_axis:
        penalty = 1
    return (
        int(candidate["length"]),
        penalty,
        int(candidate["turns"]),
        tuple(candidate["path"]),
    )


def _enumerate_candidates(
    scene: np.ndarray,
    anchor_a: Coord,
    anchor_b: Coord,
    color: int,
) -> list[dict]:
    candidates: dict[tuple[Coord, ...], dict] = {}
    for start in _block_cells(anchor_a, scene.shape):
        for end in _block_cells(anchor_b, scene.shape):
            if start == end:
                continue
            axes = ["h", "v"]
            if start[0] == end[0]:
                axes = ["h"]
            elif start[1] == end[1]:
                axes = ["v"]
            for axis in axes:
                path = _path_cells(start, end, axis)
                if not _path_clear(scene, path, color):
                    continue
                key = tuple(path)
                candidate = {
                    "start": start,
                    "end": end,
                    "first_axis": axis,
                    "turns": 0 if start[0] == end[0] or start[1] == end[1] else 1,
                    "length": len(path) - 1,
                    "path": path,
                }
                preferred = _preferred_axis(anchor_a, anchor_b)
                current = candidates.get(key)
                if current is None or _candidate_sort_key(candidate, preferred) < _candidate_sort_key(current, preferred):
                    candidates[key] = candidate
    return list(candidates.values())


def _paint_path(out: np.ndarray, path: list[Coord], color: int) -> None:
    for r, c in path:
        out[r, c] = int(color)


def _solve(grid: np.ndarray, capture: bool = False) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchors_by_color = _collect_anchors(g)
    out = g.copy()
    for color, anchors in anchors_by_color.items():
        for anchor in anchors:
            _paint_block(out, anchor, color)

    records: list[dict] = []
    color_order = sorted(anchors_by_color, key=lambda color: anchors_by_color[color][0])
    for color in color_order:
        anchors = anchors_by_color[color]
        for pair_index in range(len(anchors) - 1):
            anchor_a = anchors[pair_index]
            anchor_b = anchors[pair_index + 1]
            candidates = _enumerate_candidates(out, anchor_a, anchor_b, color)
            assert candidates, f"no valid L route for color {color}"
            preferred = _preferred_axis(anchor_a, anchor_b)
            chosen = min(candidates, key=lambda cand: _candidate_sort_key(cand, preferred))
            _paint_path(out, chosen["path"], color)
            if capture:
                min_len = min(int(cand["length"]) for cand in candidates)
                minimal = [cand for cand in candidates if int(cand["length"]) == min_len]
                classes = {cand["first_axis"] for cand in minimal if int(cand["turns"]) == 1}
                if any(int(cand["turns"]) == 0 for cand in minimal):
                    branch = "straight"
                elif len(classes) >= 2:
                    branch = "tie"
                else:
                    branch = "forced"
                records.append(
                    {
                        "color": int(color),
                        "pair_index": int(pair_index),
                        "anchors": (anchor_a, anchor_b),
                        "preferred_axis": preferred,
                        "chosen": chosen,
                        "minimal_candidates": minimal,
                        "all_candidates": candidates,
                        "class": branch,
                    }
                )

    if capture:
        return out, records
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each colored singleton into a 2x2 block, then connect same-color blocks with shortest obstacle-avoiding orthogonal links.

    Input structure:
    - The background is black, and grey pixels form fixed horizontal or vertical wall segments.
    - Each other foreground color appears as 2-3 isolated single anchor pixels.
    - Different anchor colors are separated so their 2x2 expansions do not overlap.

    Rule:
    - Expand every non-black, non-grey anchor pixel into a same-color 2x2 block with the original pixel as the block's top-left corner, clipping at the grid border if needed.
    - For each foreground color, sort its block anchors by top-left coordinate and connect each adjacent pair with the shortest orthogonal path from one block edge pixel to the next, using at most one right-angle turn.
    - Paths may pass through background or the same color, but they may not pass through grey walls or other colors; when two equally short bend directions work, move first along the larger anchor-to-anchor span axis.

    Key insight:
    - The 2x2 expansion is both geometry and routing state: the new blocks become the connection endpoints for their own color and fresh obstacles for every other color.
    """
    return _solve(np.asarray(grid, dtype=int), capture=False)


def _expand_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for color, anchors in _collect_anchors(g).items():
        for anchor in anchors:
            _paint_block(out, anchor, color)
    return out


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    choices = list(range(low, high + 1))
    if len(choices) < count:
        return None
    for _ in range(120):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _split_axis(length: int, parts: int, gap: int, margin: int, min_size: int) -> list[tuple[int, int]] | None:
    usable = length - 2 * margin - gap * (parts - 1)
    if usable < parts * min_size:
        return None
    sizes = [min_size] * parts
    for _ in range(usable - parts * min_size):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    intervals: list[tuple[int, int]] = []
    cursor = margin
    for size in sizes:
        intervals.append((cursor, cursor + size - 1))
        cursor += size + gap
    return intervals


def _sample_chain_anchors(
    h: int,
    w: int,
    region: tuple[int, int],
    layout: str,
    count: int,
) -> list[Coord] | None:
    lo, hi = map(int, region)
    if layout == "vertical":
        rows = _sample_spaced_positions(1, h - 2, count, min_gap=5)
        if rows is None:
            return None
        mid = (lo + hi) // 2
        left_pool = [c for c in range(lo, hi + 1) if c <= mid]
        right_pool = [c for c in range(lo, hi + 1) if c > mid]
        if not left_pool or not right_pool:
            return None
        for _ in range(120):
            cols: list[int] = []
            use_left = random.choice([True, False])
            for _idx in range(count):
                pool = left_pool if use_left else right_pool
                cols.append(random.choice(pool))
                use_left = not use_left
            if all(abs(b - a) >= 2 for a, b in zip(cols, cols[1:])):
                return list(zip(rows, cols))
        return None

    cols = _sample_spaced_positions(1, w - 2, count, min_gap=5)
    if cols is None:
        return None
    mid = (lo + hi) // 2
    upper_pool = [r for r in range(lo, hi + 1) if r <= mid]
    lower_pool = [r for r in range(lo, hi + 1) if r > mid]
    if not upper_pool or not lower_pool:
        return None
    for _ in range(120):
        rows = []
        use_upper = random.choice([True, False])
        for _idx in range(count):
            pool = upper_pool if use_upper else lower_pool
            rows.append(random.choice(pool))
            use_upper = not use_upper
        if all(abs(b - a) >= 2 for a, b in zip(rows, rows[1:])):
            return list(zip(rows, cols))
    return None


def _maybe_push_to_border(
    anchors: list[Coord],
    region: tuple[int, int],
    layout: str,
    h: int,
    w: int,
) -> list[Coord]:
    out = list(anchors)
    lo, hi = map(int, region)
    if random.random() >= 0.25:
        return out
    if layout == "vertical":
        if random.random() < 0.5 and out[-1][0] <= h - 6:
            out[-1] = (h - 1, out[-1][1])
        elif hi == w - 1:
            idx = random.randrange(len(out))
            out[idx] = (out[idx][0], w - 1)
    else:
        if random.random() < 0.5 and out[-1][1] <= w - 6:
            out[-1] = (out[-1][0], w - 1)
        elif hi == h - 1:
            idx = random.randrange(len(out))
            out[idx] = (h - 1, out[idx][1])
    return sorted(out)


def _anchors_are_isolated(anchors_by_color: dict[int, list[Coord]]) -> bool:
    all_anchors = [anchor for anchors in anchors_by_color.values() for anchor in anchors]
    for idx, (r, c) in enumerate(all_anchors):
        for rr, cc in all_anchors[idx + 1 :]:
            if max(abs(r - rr), abs(c - cc)) < 2:
                return False
    return True


def _build_input(h: int, w: int, anchors_by_color: dict[int, list[Coord]]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    for color, anchors in anchors_by_color.items():
        for r, c in anchors:
            if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
                return None
            grid[r, c] = int(color)
    return grid


def _wall_components(grid: np.ndarray) -> list[list[Coord]]:
    seen: set[Coord] = set()
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(grid == WALL):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp: list[Coord] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                nxt = (nr, nc)
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def _wall_segment_ok(grid: np.ndarray, coords: list[Coord], forbidden: set[Coord], anchor_cells: set[Coord]) -> bool:
    coords_set = set(coords)
    for r, c in coords:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BG:
            return False
        if (r, c) in forbidden:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in anchor_cells:
                return False
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL and (nr, nc) not in coords_set:
                return False
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) != 1 and len(cols) != 1:
        return False
    if not (2 <= len(coords) <= 5):
        return False
    return True


def _pick_segment_through_cell(
    grid: np.ndarray,
    cell: Coord,
    forbidden: set[Coord],
    anchor_cells: set[Coord],
) -> list[Coord] | None:
    r, c = map(int, cell)
    for _ in range(80):
        axis = random.choice(["h", "v"])
        length = random.randint(2, 5)
        offset = random.randint(0, length - 1)
        if axis == "h":
            start_c = c - offset
            coords = [(r, start_c + i) for i in range(length)]
        else:
            start_r = r - offset
            coords = [(start_r + i, c) for i in range(length)]
        if _wall_segment_ok(grid, coords, forbidden, anchor_cells):
            return coords
    return None


def _paint_wall_segment(grid: np.ndarray, coords: list[Coord]) -> None:
    for r, c in coords:
        grid[r, c] = WALL


def _route_feature_ok(records: list[dict], target_branch: str) -> bool:
    nonstraight = [rec for rec in records if int(rec["chosen"]["turns"]) == 1]
    if not nonstraight:
        return False
    if target_branch == "tie":
        return any(rec["class"] == "tie" and int(rec["chosen"]["turns"]) == 1 for rec in records)
    return any(rec["class"] == "forced" and int(rec["chosen"]["turns"]) == 1 for rec in records)


def _try_add_forcing_wall(grid: np.ndarray, records: list[dict], anchor_cells: set[Coord]) -> np.ndarray | None:
    for record in records:
        if record["class"] != "tie" or int(record["chosen"]["turns"]) != 1:
            continue
        chosen_path = set(record["chosen"]["path"])
        alternatives = [
            cand
            for cand in record["minimal_candidates"]
            if cand["first_axis"] != record["chosen"]["first_axis"]
        ]
        random.shuffle(alternatives)
        for alt in alternatives:
            unique = [cell for cell in alt["path"] if cell not in chosen_path and int(grid[cell]) == BG]
            random.shuffle(unique)
            for cell in unique:
                trial = grid.copy()
                segment = _pick_segment_through_cell(trial, cell, chosen_path, anchor_cells)
                if segment is None:
                    continue
                _paint_wall_segment(trial, segment)
                try:
                    _out, new_records = _solve(trial, capture=True)
                except AssertionError:
                    continue
                if _route_feature_ok(new_records, "forced"):
                    return trial
    return None


def _add_decorative_walls(
    grid: np.ndarray,
    protected: set[Coord],
    anchor_cells: set[Coord],
    target_count: int,
) -> np.ndarray | None:
    out = grid.copy()
    attempts = 0
    while len(_wall_components(out)) < target_count and attempts < 500:
        attempts += 1
        free = [tuple(map(int, rc)) for rc in np.argwhere(out == BG) if tuple(map(int, rc)) not in protected]
        if not free:
            break
        cell = random.choice(free)
        segment = _pick_segment_through_cell(out, cell, protected, anchor_cells)
        if segment is None:
            continue
        _paint_wall_segment(out, segment)
    return out if len(_wall_components(out)) >= target_count else None


def _validate_structure(grid: np.ndarray, anchors_by_color: dict[int, list[Coord]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not (2 <= len(anchors_by_color) <= 3):
        return False
    if any(not (2 <= len(anchors) <= 3) for anchors in anchors_by_color.values()):
        return False
    if not _anchors_are_isolated(anchors_by_color):
        return False
    for color, anchors in anchors_by_color.items():
        for r, c in anchors:
            if int(grid[r, c]) != int(color):
                return False
            if any(_in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG for nr, nc in _neighbors8(r, c)):
                return False
    walls = _wall_components(grid)
    if not (3 <= len(walls) <= 6):
        return False
    for comp in walls:
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        if len(comp) < 2 or len(comp) > 5:
            return False
        if len(rows) != 1 and len(cols) != 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(21, 26)
    w = random.randint(21, 26)
    n_colors = random.choice([2, 3])
    target_branch = random.choice(["tie", "forced"])
    colors = random.sample(list(FG_POOL), n_colors)

    split_ok = None
    for _ in range(20):
        layout = random.choice(["vertical", "horizontal"])
        gap = random.choice([1, 2])
        margin = random.choice([0, 1])
        if layout == "vertical":
            regions = _split_axis(w, n_colors, gap=gap, margin=margin, min_size=5)
        else:
            regions = _split_axis(h, n_colors, gap=gap, margin=margin, min_size=5)
        if regions is not None:
            split_ok = (layout, regions)
            break
    if split_ok is None:
        return None
    layout, regions = split_ok

    anchors_by_color: dict[int, list[Coord]] = {}
    for color, region in zip(colors, regions):
        count = random.choice([2, 3])
        anchors = _sample_chain_anchors(h, w, region, layout, count)
        if anchors is None:
            return None
        anchors = _maybe_push_to_border(anchors, region, layout, h, w)
        anchors_by_color[color] = anchors

    if not _anchors_are_isolated(anchors_by_color):
        return None
    inp = _build_input(h, w, anchors_by_color)
    if inp is None:
        return None

    anchor_cells = {anchor for anchors in anchors_by_color.values() for anchor in anchors}
    try:
        expanded = _expand_only(inp)
        out, records = _solve(inp, capture=True)
    except AssertionError:
        return None
    if np.array_equal(expanded, out):
        return None
    if not _route_feature_ok(records, "tie"):
        return None

    if target_branch == "forced":
        forced_inp = _try_add_forcing_wall(inp, records, anchor_cells)
        if forced_inp is None:
            return None
        inp = forced_inp
        try:
            out, records = _solve(inp, capture=True)
        except AssertionError:
            return None
        if not _route_feature_ok(records, "forced"):
            return None
    else:
        if not _route_feature_ok(records, "tie"):
            return None

    desired_walls = random.randint(3, 6)
    protected = {tuple(map(int, rc)) for rc in np.argwhere(out != BG)}
    inp = _add_decorative_walls(inp, protected, anchor_cells, desired_walls)
    if inp is None:
        return None

    try:
        out, records = _solve(inp, capture=True)
    except AssertionError:
        return None
    if not _route_feature_ok(records, target_branch):
        return None
    if np.count_nonzero(out == WALL) != np.count_nonzero(inp == WALL):
        return None
    if not _validate_structure(inp, anchors_by_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routing-with-expansion ARC input")
