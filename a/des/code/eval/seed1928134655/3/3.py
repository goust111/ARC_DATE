from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
TOWER_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_RETRIES = 400


def _range_inclusive(a: int, b: int):
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _hierarchy_colors(grid: np.ndarray) -> list[int]:
    top = grid[0]
    cols = [int(c) for c in np.flatnonzero((top != BG) & (top != GREY))]
    assert len(cols) >= 2, "expected at least two hierarchy colors on the top row"

    colors = [int(top[c]) for c in cols]
    assert len(colors) == len(set(colors)), "hierarchy colors must be unique"
    return colors


def _extract_towers(grid: np.ndarray, hierarchy: list[int]) -> list[dict[str, int]]:
    seen_colors = {int(v) for v in np.unique(grid)} - {BG, GREY}
    assert seen_colors == set(hierarchy), "non-grey foreground colors must match the hierarchy strip"

    towers = []
    for rank, color in enumerate(hierarchy):
        coords = np.argwhere(grid == color)
        top_coords = coords[coords[:, 0] == 0]
        body_coords = coords[coords[:, 0] > 0]
        assert top_coords.shape == (1, 2), f"color {color} must have exactly one strip pixel"
        assert body_coords.shape == (4, 2), f"color {color} must have exactly one 2x2 tower"

        rows = sorted({int(r) for r, _c in body_coords})
        cols = sorted({int(c) for _r, c in body_coords})
        assert len(rows) == 2 and len(cols) == 2, f"tower {color} must be 2x2"
        for r in rows:
            for c in cols:
                assert np.any((body_coords[:, 0] == r) & (body_coords[:, 1] == c)), (
                    f"tower {color} must fill its 2x2 box"
                )

        towers.append({"rank": rank, "color": color, "r0": rows[0], "c0": cols[0]})
    return towers


def _grey_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    bboxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue

            label = len(bboxes)
            stack = [(r, c)]
            labels[r, c] = label
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                rr, cc = stack.pop()
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        stack.append((nr, nc))

            bboxes.append((r0, c0, r1, c1))

    return labels, bboxes


def _clear_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> bool:
    if not (0 <= col < grid.shape[1]):
        return False
    return all(int(grid[r, col]) == BG for r in _range_inclusive(r0, r1))


def _clear_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> bool:
    if not (0 <= row < grid.shape[0]):
        return False
    return all(int(grid[row, c]) == BG for c in _range_inclusive(c0, c1))


def _choose_bypass_row(grid: np.ndarray, row: int, pre_c: int, after_c: int) -> int | None:
    candidates: list[tuple[int, int]] = []
    left = min(pre_c, after_c)
    right = max(pre_c, after_c)

    for br in range(row - 1, -1, -1):
        if (
            _clear_vertical(grid, pre_c, row, br)
            and _clear_horizontal(grid, br, left, right)
            and _clear_vertical(grid, after_c, br, row)
        ):
            candidates.append((abs(row - br), br))
            break

    for br in range(row + 1, grid.shape[0]):
        if (
            _clear_vertical(grid, pre_c, row, br)
            and _clear_horizontal(grid, br, left, right)
            and _clear_vertical(grid, after_c, br, row)
        ):
            candidates.append((abs(row - br), br))
            break

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _choose_bypass_col(grid: np.ndarray, col: int, pre_r: int, after_r: int) -> int | None:
    candidates: list[tuple[int, int]] = []
    top = min(pre_r, after_r)
    bottom = max(pre_r, after_r)

    for bc in range(col - 1, -1, -1):
        if (
            _clear_horizontal(grid, pre_r, col, bc)
            and _clear_vertical(grid, bc, top, bottom)
            and _clear_horizontal(grid, after_r, bc, col)
        ):
            candidates.append((abs(col - bc), bc))
            break

    for bc in range(col + 1, grid.shape[1]):
        if (
            _clear_horizontal(grid, pre_r, col, bc)
            and _clear_vertical(grid, bc, top, bottom)
            and _clear_horizontal(grid, after_r, bc, col)
        ):
            candidates.append((abs(col - bc), bc))
            break

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _horizontal_detour_cells(row: int, pre_c: int, after_c: int, bypass_row: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    step_r = 1 if bypass_row > row else -1
    for rr in range(row + step_r, bypass_row + step_r, step_r):
        cells.append((rr, pre_c))

    step_c = 1 if after_c > pre_c else -1
    for cc in range(pre_c + step_c, after_c + step_c, step_c):
        cells.append((bypass_row, cc))

    step_back = 1 if row > bypass_row else -1
    for rr in range(bypass_row + step_back, row + step_back, step_back):
        cells.append((rr, after_c))
    return cells


def _vertical_detour_cells(col: int, pre_r: int, after_r: int, bypass_col: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    step_c = 1 if bypass_col > col else -1
    for cc in range(col + step_c, bypass_col + step_c, step_c):
        cells.append((pre_r, cc))

    step_r = 1 if after_r > pre_r else -1
    for rr in range(pre_r + step_r, after_r + step_r, step_r):
        cells.append((rr, bypass_col))

    step_back = 1 if col > bypass_col else -1
    for cc in range(bypass_col + step_back, col + step_back, step_back):
        cells.append((after_r, cc))
    return cells


def _route_horizontal(
    grid: np.ndarray,
    obstacle_labels: np.ndarray,
    obstacle_boxes: list[tuple[int, int, int, int]],
    row: int,
    start_c: int,
    end_c: int,
    line_color: int,
    detour_color: int,
) -> list[tuple[int, int, int]] | None:
    if not (0 <= row < grid.shape[0]):
        return None
    if start_c == end_c:
        return [] if int(grid[row, start_c]) == BG else None
    if not (0 <= start_c < grid.shape[1] and 0 <= end_c < grid.shape[1]):
        return None
    if int(grid[row, start_c]) != BG or int(grid[row, end_c]) != BG:
        return None

    cells: list[tuple[int, int, int]] = []
    dc = 1 if end_c > start_c else -1
    current_c = start_c

    while True:
        if (dc == 1 and current_c > end_c) or (dc == -1 and current_c < end_c):
            break

        obstacle_c: int | None = None
        c = current_c
        while True:
            if int(grid[row, c]) == GREY:
                obstacle_c = c
                break
            if c == end_c:
                break
            c += dc

        if obstacle_c is None:
            for cc in _range_inclusive(current_c, end_c):
                cells.append((row, cc, line_color))
            break

        pre_c = obstacle_c - dc
        for cc in _range_inclusive(current_c, pre_c):
            cells.append((row, cc, line_color))

        label = int(obstacle_labels[row, obstacle_c])
        assert label >= 0, "grey obstacle cell must belong to a component"
        _r0, c0, _r1, c1 = obstacle_boxes[label]
        after_c = c1 + 1 if dc == 1 else c0 - 1
        if not (0 <= after_c < grid.shape[1]):
            return None

        bypass_row = _choose_bypass_row(grid, row, pre_c, after_c)
        if bypass_row is None:
            return None
        for rr, cc in _horizontal_detour_cells(row, pre_c, after_c, bypass_row):
            cells.append((rr, cc, detour_color))

        current_c = after_c + dc

    return cells


def _route_vertical(
    grid: np.ndarray,
    obstacle_labels: np.ndarray,
    obstacle_boxes: list[tuple[int, int, int, int]],
    col: int,
    start_r: int,
    end_r: int,
    line_color: int,
    detour_color: int,
) -> list[tuple[int, int, int]] | None:
    if not (0 <= col < grid.shape[1]):
        return None
    if start_r == end_r:
        return [] if int(grid[start_r, col]) == BG else None
    if not (0 <= start_r < grid.shape[0] and 0 <= end_r < grid.shape[0]):
        return None
    if int(grid[start_r, col]) != BG or int(grid[end_r, col]) != BG:
        return None

    cells: list[tuple[int, int, int]] = []
    dr = 1 if end_r > start_r else -1
    current_r = start_r

    while True:
        if (dr == 1 and current_r > end_r) or (dr == -1 and current_r < end_r):
            break

        obstacle_r: int | None = None
        r = current_r
        while True:
            if int(grid[r, col]) == GREY:
                obstacle_r = r
                break
            if r == end_r:
                break
            r += dr

        if obstacle_r is None:
            for rr in _range_inclusive(current_r, end_r):
                cells.append((rr, col, line_color))
            break

        pre_r = obstacle_r - dr
        for rr in _range_inclusive(current_r, pre_r):
            cells.append((rr, col, line_color))

        label = int(obstacle_labels[obstacle_r, col])
        assert label >= 0, "grey obstacle cell must belong to a component"
        r0, _c0, r1, _c1 = obstacle_boxes[label]
        after_r = r1 + 1 if dr == 1 else r0 - 1
        if not (0 <= after_r < grid.shape[0]):
            return None

        bypass_col = _choose_bypass_col(grid, col, pre_r, after_r)
        if bypass_col is None:
            return None
        for rr, cc in _vertical_detour_cells(col, pre_r, after_r, bypass_col):
            cells.append((rr, cc, detour_color))

        current_r = after_r + dr

    return cells


def _best_connection_path(
    grid: np.ndarray,
    obstacle_labels: np.ndarray,
    obstacle_boxes: list[tuple[int, int, int, int]],
    src: dict[str, int],
    dst: dict[str, int],
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, list[tuple[int, int, int]]]] = []

    if src["r0"] == dst["r0"]:
        rows = [src["r0"], src["r0"] + 1]
        if src["c0"] < dst["c0"]:
            endpoints = [(row, src["c0"] + 2, dst["c0"] - 1) for row in rows]
        else:
            endpoints = [(row, src["c0"] - 1, dst["c0"] + 2) for row in rows]
        for row, start_c, end_c in endpoints:
            if (start_c < end_c and src["c0"] > dst["c0"]) or (start_c > end_c and src["c0"] < dst["c0"]):
                continue
            path = _route_horizontal(
                grid,
                obstacle_labels,
                obstacle_boxes,
                row,
                start_c,
                end_c,
                src["color"],
                dst["color"],
            )
            if path is not None:
                candidates.append((len(path), row, path))

    if src["c0"] == dst["c0"]:
        cols = [src["c0"], src["c0"] + 1]
        if src["r0"] < dst["r0"]:
            endpoints = [(col, src["r0"] + 2, dst["r0"] - 1) for col in cols]
        else:
            endpoints = [(col, src["r0"] - 1, dst["r0"] + 2) for col in cols]
        for col, start_r, end_r in endpoints:
            if (start_r < end_r and src["r0"] > dst["r0"]) or (start_r > end_r and src["r0"] < dst["r0"]):
                continue
            path = _route_vertical(
                grid,
                obstacle_labels,
                obstacle_boxes,
                col,
                start_r,
                end_r,
                src["color"],
                dst["color"],
            )
            if path is not None:
                candidates.append((len(path), col, path))

    if not candidates:
        return []
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build directed tower-to-tower links using the top-row color hierarchy and reroute links around grey blockers.

    Input structure:
    - The background is black.
    - The top row contains a left-to-right hierarchy strip of single pixels, one color per tower.
    - Each tower is a distinct-color 2x2 block elsewhere in the grid.
    - Grey cells form small obstacle components.

    Rule:
    - Read the hierarchy strip from left to right: each tower connects only to the tower whose level is exactly one lower.
    - Connections are horizontal or vertical only, and they use the emitting tower's color.
    - When a connection hits a grey obstacle, it bends just before the obstacle, bypasses it on the shorter side,
      recolors the detour to the receiving tower's color, then returns to its original axis.
    - The lowest-level tower emits nothing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 6 and g.shape[1] >= 8, "grid is too small for strip, towers, and links"

    hierarchy = _hierarchy_colors(g)
    towers = _extract_towers(g, hierarchy)
    tower_by_color = {tower["color"]: tower for tower in towers}

    obstacle_labels, obstacle_boxes = _grey_components(g == GREY)
    out = g.copy()

    for tower in towers[:-1]:
        target = tower_by_color[hierarchy[tower["rank"] + 1]]
        path = _best_connection_path(g, obstacle_labels, obstacle_boxes, tower, target)
        for r, c, color in path:
            current = int(out[r, c])
            assert current in (BG, color), "connections must not overlap ambiguously"
            out[r, c] = color

    return out


def _slot_segment_points(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        left = min(a[1], b[1])
        right = max(a[1], b[1])
        return {(row, c) for c in range(left, right + 1)}
    if a[1] == b[1]:
        col = a[1]
        top = min(a[0], b[0])
        bottom = max(a[0], b[0])
        return {(r, col) for r in range(top, bottom + 1)}
    raise AssertionError("slot segments must be axis-aligned")


def _sample_chain_nodes(n_rows: int, n_cols: int, n_levels: int) -> list[tuple[int, int]] | None:
    nodes = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    random.shuffle(nodes)
    start = nodes[0]

    chain = [start]
    used = {start}
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(chain) < n_levels:
        prev = chain[-1]
        candidates = []
        for node in nodes:
            if node in used or (node[0] != prev[0] and node[1] != prev[1]):
                continue

            new_segment = _slot_segment_points(prev, node)
            if any(pt in used and pt not in {prev, node} for pt in new_segment):
                continue

            ok = True
            for old_a, old_b in segments:
                old_segment = _slot_segment_points(old_a, old_b)
                allowed = {prev} if prev in old_segment else set()
                if (new_segment & old_segment) - allowed:
                    ok = False
                    break
            if ok:
                candidates.append(node)

        if not candidates:
            return None

        next_node = random.choice(candidates)
        chain.append(next_node)
        used.add(next_node)
        segments.append((prev, next_node))

    return chain


def _block_flags(n_links: int) -> list[bool]:
    flags = [random.random() < 0.55 for _ in range(n_links)]
    if n_links >= 2:
        if not any(flags):
            flags[random.randrange(n_links)] = True
        if all(flags):
            flags[random.randrange(n_links)] = False
    return flags


def _reserve_mask(grid: np.ndarray, towers: list[dict[str, int]]) -> np.ndarray:
    reserve = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape

    for tower in towers:
        r0, c0 = tower["r0"], tower["c0"]
        reserve[max(0, r0 - 1) : min(h, r0 + 3), max(0, c0 - 1) : min(w, c0 + 3)] = True

    for src, dst in zip(towers, towers[1:]):
        if src["r0"] == dst["r0"]:
            r0 = src["r0"]
            c0 = min(src["c0"], dst["c0"])
            c1 = max(src["c0"], dst["c0"])
            reserve[max(1, r0 - 2) : min(h, r0 + 4), max(0, c0 - 1) : min(w, c1 + 3)] = True
        else:
            c0 = src["c0"]
            r0 = min(src["r0"], dst["r0"])
            r1 = max(src["r0"], dst["r0"])
            reserve[max(1, r0 - 1) : min(h, r1 + 3), max(0, c0 - 2) : min(w, c0 + 4)] = True

    reserve[0, :] = True
    return reserve


def _place_towers(grid: np.ndarray, hierarchy: list[int], chain: list[tuple[int, int]], row_slots: list[int], col_slots: list[int]) -> list[dict[str, int]]:
    towers = []
    for rank, (slot_r, slot_c) in enumerate(chain):
        r0 = row_slots[slot_r]
        c0 = col_slots[slot_c]
        color = hierarchy[rank]
        grid[r0 : r0 + 2, c0 : c0 + 2] = color
        towers.append({"rank": rank, "color": color, "r0": r0, "c0": c0})
    return towers


def _place_strip(grid: np.ndarray, hierarchy: list[int]) -> bool:
    gaps = [random.randint(0, 2) for _ in hierarchy[:-1]]
    span = len(hierarchy) + sum(gaps)
    if span > grid.shape[1]:
        return False
    start = random.randint(0, grid.shape[1] - span)
    c = start
    for i, color in enumerate(hierarchy):
        grid[0, c] = color
        if i + 1 < len(hierarchy):
            c += 1 + gaps[i]
    return True


def _place_blocking_obstacle(grid: np.ndarray, src: dict[str, int], dst: dict[str, int]) -> bool:
    h, w = grid.shape
    if src["r0"] == dst["r0"]:
        left, right = (src, dst) if src["c0"] < dst["c0"] else (dst, src)
        gap_cols = list(range(left["c0"] + 2, right["c0"]))
        if len(gap_cols) < 2:
            return False
        usable_cols = gap_cols[1:-1] or gap_cols
        c = random.choice(usable_cols)
        prefer_up = random.choice([True, False])

        shapes = []
        if prefer_up and left["r0"] + 2 < h:
            shapes.append([(left["r0"], c), (left["r0"] + 1, c), (left["r0"] + 2, c)])
        if not prefer_up and left["r0"] - 1 >= 1:
            shapes.append([(left["r0"] - 1, c), (left["r0"], c), (left["r0"] + 1, c)])
        if left["r0"] + 1 < h:
            shapes.append([(left["r0"], c), (left["r0"] + 1, c)])

    else:
        top, bottom = (src, dst) if src["r0"] < dst["r0"] else (dst, src)
        gap_rows = list(range(top["r0"] + 2, bottom["r0"]))
        if len(gap_rows) < 2:
            return False
        usable_rows = gap_rows[1:-1] or gap_rows
        r = random.choice(usable_rows)
        prefer_left = random.choice([True, False])

        shapes = []
        if prefer_left and top["c0"] + 2 < w:
            shapes.append([(r, top["c0"]), (r, top["c0"] + 1), (r, top["c0"] + 2)])
        if not prefer_left and top["c0"] - 1 >= 0:
            shapes.append([(r, top["c0"] - 1), (r, top["c0"]), (r, top["c0"] + 1)])
        if top["c0"] + 1 < w:
            shapes.append([(r, top["c0"]), (r, top["c0"] + 1)])

    for shape in shapes:
        if all(0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG for r, c in shape):
            for r, c in shape:
                grid[r, c] = GREY
            return True
    return False


_EXTRA_SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]


def _transform_shape(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = np.array(shape, dtype=int)
    for _ in range(random.randint(0, 3)):
        cells = np.stack([cells[:, 1], -cells[:, 0]], axis=1)
    if random.choice([True, False]):
        cells[:, 0] *= -1
    if random.choice([True, False]):
        cells[:, 1] *= -1
    cells[:, 0] -= cells[:, 0].min()
    cells[:, 1] -= cells[:, 1].min()
    return [tuple(map(int, cell)) for cell in cells]


def _place_extra_obstacles(grid: np.ndarray, reserve: np.ndarray, count: int) -> None:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _try in range(60):
            shape = _transform_shape(random.choice(_EXTRA_SHAPES))
            sh = 1 + max(r for r, _c in shape)
            sw = 1 + max(c for _r, c in shape)
            if sh >= h - 1 or sw >= w:
                continue
            r0 = random.randint(1, h - sh - 1)
            c0 = random.randint(0, w - sw)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if all(int(grid[r, c]) == BG and not reserve[r, c] for r, c in cells):
                halo_ok = True
                for r, c in cells:
                    for rr in range(max(0, r - 1), min(h, r + 2)):
                        for cc in range(max(0, c - 1), min(w, c + 2)):
                            if int(grid[rr, cc]) == GREY:
                                halo_ok = False
                                break
                        if not halo_ok:
                            break
                    if not halo_ok:
                        break
                if not halo_ok:
                    continue
                for r, c in cells:
                    grid[r, c] = GREY
                placed = True
                break
        if not placed:
            return


def _build_candidate() -> np.ndarray | None:
    n_levels = random.randint(3, 5)
    n_row_slots = random.randint(3, 4)
    n_col_slots = random.randint(4, 5)
    row_gap = random.randint(4, 5)
    col_gap = random.randint(5, 6)
    row_offset = random.randint(2, 4)
    col_offset = random.randint(1, 3)
    bottom_pad = random.randint(2, 4)
    right_pad = random.randint(2, 4)

    row_slots = [row_offset + row_gap * i for i in range(n_row_slots)]
    col_slots = [col_offset + col_gap * i for i in range(n_col_slots)]
    h = row_slots[-1] + 2 + bottom_pad
    w = col_slots[-1] + 2 + right_pad
    if h < 8 or w < 10:
        return None

    chain = _sample_chain_nodes(n_row_slots, n_col_slots, n_levels)
    if chain is None:
        return None

    hierarchy = random.sample(TOWER_PALETTE, n_levels)
    grid = np.full((h, w), BG, dtype=int)
    towers = _place_towers(grid, hierarchy, chain, row_slots, col_slots)
    if not _place_strip(grid, hierarchy):
        return None

    flags = _block_flags(n_levels - 1)
    blocked_any = False
    for flag, src, dst in zip(flags, towers, towers[1:]):
        if flag and _place_blocking_obstacle(grid, src, dst):
            blocked_any = True

    reserve = _reserve_mask(grid, towers)
    extra_count = random.randint(0, 2)
    if not blocked_any:
        extra_count = max(extra_count, 1)
    _place_extra_obstacles(grid, reserve, extra_count)

    try:
        out = main(grid)
    except AssertionError:
        return None

    new_line_cells = int(np.count_nonzero((grid == BG) & (out != BG)))
    if new_line_cells == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower routing instance")
