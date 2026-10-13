from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 7
GREY = 5
CONTROL_COLORS = (1, 2, 3, 4)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    return DIR4 if connectivity == 4 else DIR8


def _component_masks(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    steps = _neighbors(0, 0, connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diagonal_contact_dirs(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[int, int]]:
    dirs = set()
    for r, c in a:
        for dr, dc in DIAGONALS:
            if (r + dr, c + dc) in b:
                dirs.add((dr, dc))
    return dirs


def _classify_region(mask: np.ndarray) -> bool:
    """Return True iff the grey 8-component is a non-branching diagonal chain or ring."""
    lobes = _component_masks(mask, connectivity=4)
    if len(lobes) < 2:
        return False

    lobe_sets = [{(int(r), int(c)) for r, c in lobe} for lobe in lobes]
    graph = [set() for _ in lobes]

    for i in range(len(lobes)):
        for j in range(i + 1, len(lobes)):
            dirs = _diagonal_contact_dirs(lobe_sets[i], lobe_sets[j])
            if not dirs:
                continue
            if len(dirs) != 1:
                return False
            graph[i].add(j)
            graph[j].add(i)

    if any(len(neis) == 0 for neis in graph):
        return False
    if any(len(neis) > 2 for neis in graph):
        return False

    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != len(graph):
        return False

    edge_count = sum(len(neis) for neis in graph) // 2
    return edge_count in {len(graph) - 1, len(graph)}


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    grey_components = _component_masks(grid == GREY, connectivity=8)
    region_index = np.full(grid.shape, -1, dtype=int)
    regions = []
    for idx, coords in enumerate(grey_components):
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        region_index[mask] = idx
        cols = coords[:, 1]
        rows = coords[:, 0]
        regions.append(
            {
                "coords": coords,
                "mask": mask,
                "passable": _classify_region(mask),
                "top": int(rows.min()),
                "bottom": int(rows.max()),
                "left": int(cols.min()),
                "right": int(cols.max()),
                "size": int(coords.shape[0]),
            }
        )
    return regions, region_index


def _control_info(grid: np.ndarray) -> list[tuple[int, int]]:
    ctrls = [(int(c), int(grid[0, c])) for c in np.flatnonzero(grid[0] != BACKGROUND)]
    ctrls.sort()
    return ctrls


def _solve(
    grid: np.ndarray,
    *,
    region_mode: str = "actual",
    order: str = "left_to_right",
    fill_whole_region: bool = True,
    trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    regions, region_index = _extract_regions(g)
    ctrls = _control_info(g)

    out = g.copy()
    owners: dict[int, int] = {}
    ray_logs: list[dict] = []

    if order == "right_to_left":
        ordered_ctrls = list(reversed(ctrls))
    else:
        ordered_ctrls = ctrls

    for col, color in ordered_ctrls:
        events: list[tuple[str, int | None]] = []
        r = 1
        while r < g.shape[0]:
            rid = int(region_index[r, col])
            if rid == -1:
                out[r, col] = color
                r += 1
                continue

            if region_mode == "all_pass":
                passable = True
            elif region_mode == "all_block":
                passable = False
            else:
                passable = bool(regions[rid]["passable"])

            if not passable:
                events.append(("block", rid))
                break

            if fill_whole_region and rid not in owners:
                out[regions[rid]["mask"]] = color
                owners[rid] = color
            elif not fill_whole_region:
                out[r, col] = color
            events.append(("pass", rid))
            r = int(regions[rid]["bottom"]) + 1

        if trace:
            ray_logs.append({"col": col, "color": color, "events": events})

    if trace:
        return out, {"regions": regions, "region_index": region_index, "controls": ctrls, "rays": ray_logs}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shoot top-row color rays downward, letting only diagonal-chain grey regions be permeable.

    Input structure:
    - The background is orange, and the top row contains 2-4 distinct colored control pixels.
    - Below the top row are several grey 8-connected regions, some spanning multiple columns.
    - Each grey region may decompose into one or more 4-connected lobes.

    Rule:
    - Classify every grey 8-connected region from its 4-connected lobe contact graph.
    - A region is permeable only when its lobes touch solely by diagonal corners and form a non-branching chain or ring; otherwise it is blocking.
    - Process control pixels from smaller column to larger column. Each control shoots a vertical ray downward: background cells become the control color, permeable grey regions are recolored as a whole and the ray resumes below that region, while blocking grey regions stay grey and stop the ray immediately. The top row remains unchanged.

    Key insight:
    - The grey regions are not transformed by color alone; their internal diagonal-chain topology decides whether a later ray operation can pass through them or must stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(set(CONTROL_COLORS) | {BACKGROUND, GREY})
    ctrls = _control_info(g)
    assert 2 <= len(ctrls) <= 4
    assert len({color for _, color in ctrls}) == len(ctrls)
    assert all(np.any(g[1:, c] == GREY) for c, _ in ctrls)
    return _solve(g)


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _diag_line(length: int, descending_right: bool) -> set[tuple[int, int]]:
    if descending_right:
        return {(i, i) for i in range(length)}
    return {(i, length - 1 - i) for i in range(length)}


def _diamond_cycle() -> set[tuple[int, int]]:
    return {(0, 1), (1, 0), (1, 2), (2, 1)}


def _branch_blocker() -> set[tuple[int, int]]:
    return {(0, 0), (1, 1), (0, 2), (2, 2)}


def _l_blocker(size: int) -> set[tuple[int, int]]:
    if size == 4:
        return {(0, 0), (1, 0), (2, 0), (2, 1)}
    return {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}


def _rect_blocker(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _shape_cols(shape: set[tuple[int, int]]) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in shape:
        cols[c].append(r)
    return cols


def _abs_cells(shape: set[tuple[int, int]], origin_r: int, origin_c: int) -> set[tuple[int, int]]:
    return {(origin_r + r, origin_c + c) for r, c in shape}


def _can_place(cells: set[tuple[int, int]], h: int, w: int, occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not (1 <= r < h and 0 <= c < w):
            return False
        for dr, dc in DIR8 + ((0, 0),):
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_shape(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    shape: set[tuple[int, int]],
    row_range: tuple[int, int],
    col_targets: tuple[int, ...],
    *,
    require_all_targets: bool,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    shape = _normalize(shape)
    cols = _shape_cols(shape)
    min_target_matches = len(col_targets) if require_all_targets else 1
    options: list[tuple[int, int]] = []

    for anchor_col in col_targets:
        for rel_col in cols:
            origin_c = anchor_col - rel_col
            width = max(c for _, c in shape) + 1
            height = max(r for r, _ in shape) + 1
            if not (0 <= origin_c and origin_c + width <= w):
                continue
            for origin_r in range(row_range[0], row_range[1] + 1):
                if origin_r + height > h:
                    continue
                cells = _abs_cells(shape, origin_r, origin_c)
                if not _can_place(cells, h, w, occupied):
                    continue
                hit_cols = {c for _, c in cells}
                matches = sum(1 for target in col_targets if target in hit_cols)
                if matches >= min_target_matches:
                    options.append((origin_r, origin_c))

    random.shuffle(options)
    for origin_r, origin_c in options:
        cells = _abs_cells(shape, origin_r, origin_c)
        hit_cols = {c for _, c in cells}
        if require_all_targets and any(target not in hit_cols for target in col_targets):
            continue
        for r, c in cells:
            grid[r, c] = GREY
        occupied.update(cells)
        return cells
    return None


def _choose_control_columns(width: int, count: int) -> list[int] | None:
    for _ in range(200):
        cols = []
        first_gap = random.choice((2, 3, 4))
        start_max = width - 1 - (first_gap + 2 * (count - 2) + 1)
        if start_max < 1:
            continue
        cols.append(random.randint(1, start_max))
        cols.append(cols[0] + first_gap)
        while len(cols) < count:
            prev = cols[-1]
            gap = random.randint(2, 5)
            nxt = prev + gap
            if nxt >= width - 1:
                break
            cols.append(nxt)
        if len(cols) == count:
            return cols
    return None


def _build_input() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(14, 19)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    control_count = random.randint(2, 4)
    control_cols = _choose_control_columns(w, control_count)
    if control_cols is None:
        return None
    control_colors = random.sample(list(CONTROL_COLORS), control_count)
    for c, color in zip(control_cols, control_colors):
        grid[0, c] = color

    occupied: set[tuple[int, int]] = set()
    placed: list[set[tuple[int, int]]] = []

    shared_gap = control_cols[1] - control_cols[0]
    if shared_gap == 2 and random.random() < 0.35:
        shared_shape = _diamond_cycle()
    else:
        shared_shape = _diag_line(shared_gap + 1, descending_right=random.random() < 0.5)
    shared = _place_shape(grid, occupied, shared_shape, (2, 4), (control_cols[0], control_cols[1]), require_all_targets=True)
    if shared is None:
        return None
    placed.append(shared)

    shared_size = len(shared_shape)
    same_size_blockers = []
    if shared_size == 4:
        same_size_blockers = [_rect_blocker(2, 2), _l_blocker(4), _branch_blocker()]
    elif shared_size == 5:
        same_size_blockers = [_l_blocker(5)]

    if control_count == 2:
        pass_targets = (control_cols[1],)
        blocker_first_targets = (control_cols[0],)
        blocker_second_targets = (control_cols[1],)
    elif control_count == 3:
        pass_targets = (control_cols[1],)
        blocker_first_targets = (control_cols[0],)
        blocker_second_targets = (control_cols[2],)
    else:
        pass_targets = (control_cols[2],)
        blocker_first_targets = (control_cols[0],)
        blocker_second_targets = (control_cols[3],)

    pass_shape = random.choice([_diag_line(random.randint(4, 5), bool(random.getrandbits(1))), _diamond_cycle()])
    pass_shape = _normalize(pass_shape)
    extra_pass = _place_shape(grid, occupied, pass_shape, (8, 11), pass_targets, require_all_targets=False)
    if extra_pass is None:
        return None
    placed.append(extra_pass)

    blocker_shape1 = random.choice(same_size_blockers or [_rect_blocker(2, 2), _l_blocker(4), _branch_blocker()])
    blocker1 = _place_shape(grid, occupied, blocker_shape1, (10, 13), blocker_first_targets, require_all_targets=False)
    if blocker1 is None:
        return None
    placed.append(blocker1)

    blocker_shape2 = random.choice([_branch_blocker(), _rect_blocker(2, 3), _rect_blocker(3, 2), _l_blocker(5)])
    blocker2 = _place_shape(grid, occupied, blocker_shape2, (4, 8), blocker_second_targets, require_all_targets=False)
    if blocker2 is None:
        return None
    placed.append(blocker2)

    optional_region_count = random.randint(0, 2)
    all_targets = tuple(control_cols)
    for _ in range(optional_region_count):
        if random.random() < 0.5:
            shape = _normalize(random.choice([_diag_line(random.randint(4, 6), True), _diag_line(random.randint(4, 6), False), _diamond_cycle()]))
        else:
            shape = _normalize(random.choice([_rect_blocker(2, 2), _rect_blocker(2, 3), _l_blocker(4), _branch_blocker()]))
        row_lo = random.randint(5, h - 6)
        row_hi = min(h - 3, row_lo + 2)
        region = _place_shape(grid, occupied, shape, (row_lo, row_hi), all_targets, require_all_targets=False)
        if region is not None:
            placed.append(region)

    if not (4 <= len(placed) <= 6):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    ctrls = _control_info(grid)
    if not (2 <= len(ctrls) <= 4):
        return False
    if len({c for c, _ in ctrls}) != len(ctrls):
        return False
    if len({color for _, color in ctrls}) != len(ctrls):
        return False
    if any(not np.any(grid[1:, c] == GREY) for c, _ in ctrls):
        return False

    out, info = _solve(grid, trace=True)
    regions = info["regions"]
    if not (4 <= len(regions) <= 6):
        return False

    passable = [i for i, region in enumerate(regions) if region["passable"]]
    blockers = [i for i, region in enumerate(regions) if not region["passable"]]
    if len(passable) < 2 or len(blockers) < 2:
        return False

    pass_sizes = {regions[i]["size"] for i in passable}
    block_sizes = {regions[i]["size"] for i in blockers}
    if not (pass_sizes & block_sizes):
        return False

    hit_by_region: dict[int, list[int]] = defaultdict(list)
    rays = info["rays"]
    pass_then_continue = False
    immediate_block = False
    for ray in rays:
        events = ray["events"]
        if events and events[0][0] == "block":
            immediate_block = True
        seen_pass = False
        for idx, (kind, rid) in enumerate(events):
            if rid is not None:
                hit_by_region[int(rid)].append(int(ray["col"]))
            if kind == "pass":
                seen_pass = True
                if idx == len(events) - 1:
                    pass_then_continue = True
            if seen_pass and kind == "block":
                pass_then_continue = True
    if not immediate_block or not pass_then_continue:
        return False

    shared_hits = [
        rid
        for rid, cols in hit_by_region.items()
        if regions[rid]["passable"] and len(set(cols)) >= 2 and regions[rid]["right"] > regions[rid]["left"]
    ]
    if not shared_hits:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _solve(grid, region_mode="all_block")):
        return False
    if np.array_equal(out, _solve(grid, region_mode="all_pass")):
        return False
    if np.array_equal(out, _solve(grid, fill_whole_region=False)):
        return False
    if np.array_equal(out, _solve(grid, order="right_to_left")):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _build_input()
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain ray input")
