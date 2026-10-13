import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
PAIR_RETRIES = 300


def _is_boundary(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_interior(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return 0 < r < h - 1 and 0 < c < w - 1


def _perimeter_size(h: int, w: int) -> int:
    return 2 * (h + w) - 4


def _perimeter_index(cell: tuple[int, int], h: int, w: int) -> int:
    r, c = cell
    assert _is_boundary(cell, h, w)
    if r == 0:
        return c
    if c == w - 1:
        return (w - 1) + r
    if r == h - 1:
        return (w + h - 2) + (w - 1 - c)
    return (2 * w + h - 3) + (h - 1 - r)


def _canonical_endpoints(
    p: tuple[int, int], q: tuple[int, int], h: int, w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if _perimeter_index(p, h, w) <= _perimeter_index(q, h, w):
        return p, q
    return q, p


def _row_segment(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    step = 1 if c1 >= c0 else -1
    return [(r, c) for c in range(c0, c1 + step, step)]


def _col_segment(c: int, r0: int, r1: int) -> list[tuple[int, int]]:
    step = 1 if r1 >= r0 else -1
    return [(r, c) for r in range(r0, r1 + step, step)]


def _candidate_paths(
    p: tuple[int, int], q: tuple[int, int], h: int, w: int
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    start, end = _canonical_endpoints(p, q, h, w)
    r0, c0 = start
    r1, c1 = end
    hv = _row_segment(r0, c0, c1) + _col_segment(c1, r0, r1)[1:]
    vh = _col_segment(c0, r0, r1) + _row_segment(r1, c0, c1)[1:]
    return start, end, hv, vh


def _path_clear(path: Iterable[tuple[int, int]], obstacle_cells: set[tuple[int, int]]) -> bool:
    return all(cell not in obstacle_cells for cell in path)


def _path_priority(
    start: tuple[int, int], end: tuple[int, int], h: int, w: int
) -> int:
    i0 = _perimeter_index(start, h, w)
    i1 = _perimeter_index(end, h, w)
    clockwise = (i1 - i0) % _perimeter_size(h, w)
    return max(clockwise, _perimeter_size(h, w) - clockwise)


def _build_pair_record(
    color: int,
    p: tuple[int, int],
    q: tuple[int, int],
    branch: str,
    h: int,
    w: int,
    block_cell: tuple[int, int] | None = None,
) -> dict:
    start, end, hv, vh = _candidate_paths(p, q, h, w)
    assert branch in {"hv", "vh"}
    path = hv if branch == "hv" else vh
    return {
        "color": int(color),
        "p": tuple(map(int, p)),
        "q": tuple(map(int, q)),
        "start": tuple(map(int, start)),
        "end": tuple(map(int, end)),
        "hv": [tuple(map(int, cell)) for cell in hv],
        "vh": [tuple(map(int, cell)) for cell in vh],
        "path": [tuple(map(int, cell)) for cell in path],
        "branch": branch,
        "priority": _path_priority(start, end, h, w),
        "path_key": (
            _perimeter_index(start, h, w),
            _perimeter_index(end, h, w),
        ),
        "block_cell": None if block_cell is None else tuple(map(int, block_cell)),
    }


def _overlay_paths(
    base: np.ndarray,
    pair_records: list[dict],
    obstacle_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    best: dict[tuple[int, int], tuple[tuple[int, tuple[int, int]], int]] = {}
    for record in pair_records:
        rank = (int(record["priority"]), tuple(record["path_key"]))
        for cell in record["path"]:
            if cell in obstacle_cells:
                continue
            prev = best.get(cell)
            if prev is None or rank > prev[0]:
                best[cell] = (rank, int(record["color"]))
    for (r, c), (_rank, color) in best.items():
        out[r, c] = color
    return out


def _extract_pairs_and_obstacles(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pairs = []
    obstacle_cells: set[tuple[int, int]] = set()
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        pts = [tuple(map(int, cell)) for cell in np.argwhere(g == int(color))]
        boundary_pts = [cell for cell in pts if _is_boundary(cell, h, w)]
        if len(pts) == 2 and len(boundary_pts) == 2:
            pairs.append({"color": int(color), "p": boundary_pts[0], "q": boundary_pts[1]})
        else:
            obstacle_cells.update(pts)
    return pairs, obstacle_cells


def _selected_branch(
    p: tuple[int, int],
    q: tuple[int, int],
    obstacle_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> str:
    _start, _end, hv, vh = _candidate_paths(p, q, h, w)
    hv_clear = _path_clear(hv, obstacle_cells)
    vh_clear = _path_clear(vh, obstacle_cells)
    if hv_clear and vh_clear:
        return "hv"
    if hv_clear:
        return "hv"
    if vh_clear:
        return "vh"
    return "hv"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color edge pair with an L-shaped pipe and resolve crossings by larger clockwise border span.

    Input structure:
    - The background is black.
    - Several non-black colors appear exactly twice each on the grid boundary, with each pair occupying two different edges.
    - Grey interior cells act as obstacles; no path may use them when an obstacle-free alternative exists.

    Rule:
    - For every matched edge-color pair, consider the two L paths between the endpoints.
    - If exactly one candidate avoids obstacles, use it; if both avoid obstacles, use the horizontal-then-vertical candidate when traversing from the clockwise-earlier endpoint to the later one.
    - Paint the chosen path, including endpoints, in the pair color.
    - When different paths overlap, keep the color from the pair whose endpoints span the longer clockwise border arc.

    Key insight:
    - The transformation is fully determined by pairing edge endpoints, choosing between the two possible L corners via obstacle checks, and using the perimeter-span priority to break path conflicts.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    raw_pairs, obstacle_cells = _extract_pairs_and_obstacles(g)
    pair_records = []
    for pair in raw_pairs:
        start, end, hv, vh = _candidate_paths(pair["p"], pair["q"], h, w)
        hv_clear = _path_clear(hv, obstacle_cells)
        vh_clear = _path_clear(vh, obstacle_cells)
        if hv_clear and vh_clear:
            branch = "hv"
        elif hv_clear:
            branch = "hv"
        elif vh_clear:
            branch = "vh"
        else:
            branch = "hv"
        pair_records.append(
            {
                "color": int(pair["color"]),
                "start": start,
                "end": end,
                "path": hv if branch == "hv" else vh,
                "priority": _path_priority(start, end, h, w),
                "path_key": (
                    _perimeter_index(start, h, w),
                    _perimeter_index(end, h, w),
                ),
            }
        )
    return _overlay_paths(g, pair_records, obstacle_cells)


def _side_positions(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    raise ValueError(side)


def _visible_path_conflict(
    cells: set[tuple[int, int]],
    used_boundary: set[tuple[int, int]],
    own_endpoints: set[tuple[int, int]],
) -> bool:
    return bool((cells - own_endpoints) & used_boundary)


def _render_input(
    h: int,
    w: int,
    pair_records: list[dict],
    obstacle_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in obstacle_cells:
        grid[r, c] = GREY
    for record in pair_records:
        pr, pc = record["p"]
        qr, qc = record["q"]
        grid[pr, pc] = int(record["color"])
        grid[qr, qc] = int(record["color"])
    return grid


def _core_template_one(
    h: int,
    w: int,
    colors: list[int],
    dominant_target: str,
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PAIR_RETRIES):
        c_top, c_bottom = random.sample(range(1, w - 1), 2)
        r_left, r_right = random.sample(range(1, h - 1), 2)
        block_choices = [c for c in range(1, w - 1) if c not in (c_top, c_bottom)]
        if not block_choices:
            continue
        block_cell = (r_right, random.choice(block_choices))

        pair_a = _build_pair_record(colors[0], (0, c_top), (h - 1, c_bottom), "hv", h, w)
        pair_b = _build_pair_record(colors[1], (r_left, 0), (r_right, w - 1), "vh", h, w, block_cell)

        protected_a = set(pair_a["hv"]) | set(pair_a["vh"])
        if block_cell in protected_a or block_cell in set(pair_b["vh"]):
            continue
        overlap = set(pair_a["path"]) & set(pair_b["path"])
        expected_overlap = {(r_left, c_bottom)}
        if overlap != expected_overlap:
            continue
        if block_cell not in set(pair_b["hv"]) - set(pair_b["vh"]):
            continue

        if dominant_target == "a" and not (pair_a["priority"] > pair_b["priority"]):
            continue
        if dominant_target == "b" and not (pair_b["priority"] > pair_a["priority"]):
            continue

        obstacles = {block_cell}
        protected = protected_a | set(pair_b["vh"])
        used_boundary = {pair_a["p"], pair_a["q"], pair_b["p"], pair_b["q"]}
        occupied_paths = set(pair_a["path"]) | set(pair_b["path"])
        return [pair_a, pair_b], obstacles, protected, used_boundary | occupied_paths
    return None


def _core_template_two(
    h: int,
    w: int,
    colors: list[int],
    dominant_target: str,
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PAIR_RETRIES):
        r_left, r_right = random.sample(range(1, h - 1), 2)
        c_top, c_bottom = random.sample(range(1, w - 1), 2)
        block_choices = [r for r in range(1, h - 1) if r not in (r_left, r_right)]
        if not block_choices:
            continue
        block_cell = (random.choice(block_choices), c_bottom)

        pair_a = _build_pair_record(colors[0], (r_left, 0), (r_right, w - 1), "hv", h, w)
        pair_b = _build_pair_record(colors[1], (0, c_top), (h - 1, c_bottom), "vh", h, w, block_cell)

        protected_a = set(pair_a["hv"]) | set(pair_a["vh"])
        if block_cell in protected_a or block_cell in set(pair_b["vh"]):
            continue
        overlap = set(pair_a["path"]) & set(pair_b["path"])
        expected_overlap = {(r_right, c_top)}
        if overlap != expected_overlap:
            continue
        if block_cell not in set(pair_b["hv"]) - set(pair_b["vh"]):
            continue

        if dominant_target == "a" and not (pair_a["priority"] > pair_b["priority"]):
            continue
        if dominant_target == "b" and not (pair_b["priority"] > pair_a["priority"]):
            continue

        obstacles = {block_cell}
        protected = protected_a | set(pair_b["vh"])
        used_boundary = {pair_a["p"], pair_a["q"], pair_b["p"], pair_b["q"]}
        occupied_paths = set(pair_a["path"]) | set(pair_b["path"])
        return [pair_a, pair_b], obstacles, protected, used_boundary | occupied_paths
    return None


def _try_add_extra_pair(
    h: int,
    w: int,
    color: int,
    branch_kind: str,
    obstacle_cells: set[tuple[int, int]],
    protected_cells: set[tuple[int, int]],
    endpoint_cells: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int] | None, set[tuple[int, int]]] | None:
    sides = ["top", "right", "bottom", "left"]
    for _ in range(PAIR_RETRIES):
        side_a, side_b = random.sample(sides, 2)
        choices_a = [cell for cell in _side_positions(side_a, h, w) if cell not in endpoint_cells]
        choices_b = [cell for cell in _side_positions(side_b, h, w) if cell not in endpoint_cells]
        if not choices_a or not choices_b:
            continue
        p = random.choice(choices_a)
        q = random.choice(choices_b)
        if p[0] == q[0] or p[1] == q[1]:
            continue

        record_hv = _build_pair_record(color, p, q, "hv", h, w)
        own_endpoints = {record_hv["p"], record_hv["q"]}
        hv_cells = set(record_hv["hv"])
        vh_cells = set(record_hv["vh"])

        if not _path_clear(record_hv["hv"], obstacle_cells):
            continue
        if not _path_clear(record_hv["vh"], obstacle_cells):
            continue
        if _visible_path_conflict(hv_cells | vh_cells, endpoint_cells, own_endpoints):
            continue

        if branch_kind == "both_clear_hv":
            if hv_cells & occupied_paths:
                continue
            record = _build_pair_record(color, p, q, "hv", h, w)
            return record, None, hv_cells | vh_cells

        if branch_kind != "forced_vh":
            raise ValueError(branch_kind)

        if vh_cells & occupied_paths:
            continue
        block_choices = [
            cell
            for cell in hv_cells - vh_cells
            if _is_interior(cell, h, w)
            and cell not in obstacle_cells
            and cell not in protected_cells
            and cell not in occupied_paths
        ]
        if not block_choices:
            continue
        block_cell = random.choice(block_choices)
        record = _build_pair_record(color, p, q, "vh", h, w, block_cell)
        return record, block_cell, set(record["vh"])
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    pair_target = random.randint(2, 4)
    colors = random.sample(PALETTE, pair_target)

    dominant_target = random.choice(["a", "b"])
    if random.random() < 0.5:
        core = _core_template_one(h, w, colors[:2], dominant_target)
    else:
        core = _core_template_two(h, w, colors[:2], dominant_target)
    if core is None:
        return None

    pair_records, obstacle_cells, protected_cells, occupied_union = core
    endpoint_cells = {record["p"] for record in pair_records} | {record["q"] for record in pair_records}
    occupied_paths = set()
    for record in pair_records:
        occupied_paths.update(record["path"])

    branch_pool = ["both_clear_hv", "forced_vh"]
    random.shuffle(branch_pool)
    for color in colors[2:]:
        branch_kind = random.choice(branch_pool)
        extra = _try_add_extra_pair(
            h,
            w,
            color,
            branch_kind,
            obstacle_cells,
            protected_cells,
            endpoint_cells,
            occupied_paths,
        )
        if extra is None:
            branch_kind = "forced_vh" if branch_kind == "both_clear_hv" else "both_clear_hv"
            extra = _try_add_extra_pair(
                h,
                w,
                color,
                branch_kind,
                obstacle_cells,
                protected_cells,
                endpoint_cells,
                occupied_paths,
            )
        if extra is None:
            continue
        record, block_cell, protected_add = extra
        pair_records.append(record)
        protected_cells.update(protected_add)
        endpoint_cells.update({record["p"], record["q"]})
        occupied_paths.update(record["path"])
        occupied_union.update(record["path"])
        occupied_union.update({record["p"], record["q"]})
        if block_cell is not None:
            obstacle_cells.add(block_cell)

    free_interior = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in obstacle_cells
        and (r, c) not in protected_cells
        and (r, c) not in occupied_paths
    ]
    random.shuffle(free_interior)
    for cell in free_interior[: random.randint(0, min(3, len(free_interior)))]:
        obstacle_cells.add(cell)

    inp = _render_input(h, w, pair_records, obstacle_cells)
    expected = _overlay_paths(inp, pair_records, obstacle_cells)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    raw_pairs, actual_obstacles = _extract_pairs_and_obstacles(inp)
    if len(raw_pairs) < 2:
        return None
    if actual_obstacles != obstacle_cells:
        return None

    for record in pair_records:
        if _selected_branch(record["p"], record["q"], obstacle_cells, h, w) != record["branch"]:
            return None

    overlap_cells = {}
    for record in pair_records:
        for cell in record["path"]:
            overlap_cells[cell] = overlap_cells.get(cell, 0) + 1
    if max(overlap_cells.values()) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
