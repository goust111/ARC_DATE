import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
GEN_RETRIES = 600


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1, "expected exactly one full grey divider row"
    sep = int(rows[0])
    assert 0 < sep < grid.shape[0] - 1, "divider must have space on both sides"
    grey_coords = {tuple(rc) for rc in np.argwhere(grid == GREY)}
    expected = {(sep, c) for c in range(grid.shape[1])}
    assert grey_coords == expected, "grey must appear only on the divider row"
    return sep


def _find_portals(grid: np.ndarray, sep: int) -> list[int]:
    cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[sep - 1, c]) == MAROON and int(grid[sep + 1, c]) == MAROON
    ]
    assert cols, "expected at least one portal"
    maroon_coords = {tuple(rc) for rc in np.argwhere(grid == MAROON)}
    expected = {(sep - 1, c) for c in cols} | {(sep + 1, c) for c in cols}
    assert maroon_coords == expected, "maroon must appear only as portal pairs"
    return cols


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY, MAROON) or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    assert comps, "expected at least one foreground shape"
    return comps


def _transform_component(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int], sep: int, portal_col: int
) -> list[tuple[int, int]]:
    r0, _c0, r1, _c1 = bbox
    row_shift = 2 * sep - r0 - r1
    return [(r + row_shift, 2 * portal_col - c) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep = _find_divider_row(g)
    portal_cols = _find_portals(g, sep)
    comps = _components(g)

    out = np.full((h, w), BLACK, dtype=int)
    out[sep, :] = GREY
    blocked = {(sep, c) for c in range(w)}
    for c in portal_cols:
        out[sep - 1, c] = MAROON
        out[sep + 1, c] = MAROON
        blocked.add((sep - 1, c))
        blocked.add((sep + 1, c))

    placed: set[tuple[int, int]] = set()
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        matches = [c for c in portal_cols if c0 <= c <= c1]
        assert len(matches) == 1, "each shape bbox must cover exactly one portal column"
        portal_col = int(matches[0])

        if r1 < sep:
            assert r0 <= sep - 1
        else:
            assert r0 > sep

        transformed = _transform_component(comp["coords"], comp["bbox"], sep, portal_col)
        for rr, cc in transformed:
            assert 0 <= rr < h and 0 <= cc < w, "transformed shape must stay in bounds"
            assert (rr, cc) not in blocked, "transformed shape may not overwrite divider or portals"
            assert (rr, cc) not in placed, "transformed shapes may not overlap"
            placed.add((rr, cc))
            out[rr, cc] = int(comp["color"])

    return out


def _neighbors4(h: int, w: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    random.shuffle(result)
    return result


def _shortest_path(
    h: int, w: int, start: tuple[int, int], goal: tuple[int, int], blocked: tuple[int, int] | None
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    forbidden = set()
    if blocked is not None:
        forbidden.add(blocked)
    forbidden.discard(start)
    forbidden.discard(goal)

    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        for nxt in _neighbors4(h, w, cur):
            if nxt in forbidden or nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def _sample_cell(
    h: int, w: int, *, row: int | None = None, col: int | None = None, blocked: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    choices = []
    for r in range(h):
        if row is not None and r != row:
            continue
        for c in range(w):
            if col is not None and c != col:
                continue
            if blocked is not None and (r, c) == blocked:
                continue
            choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _trim_full_mask(mask: np.ndarray, anchors: set[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    if not mask.all() or h * w <= len(anchors):
        return mask
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in anchors]
    random.shuffle(candidates)
    for r, c in candidates:
        trimmed = mask.copy()
        trimmed[r, c] = False
        if _is_connected(trimmed):
            return trimmed
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_shape_mask(h: int, w: int, portal_x: int, blocked: tuple[int, int] | None) -> np.ndarray | None:
    portal = _sample_cell(h, w, col=portal_x, blocked=blocked)
    top = _sample_cell(h, w, row=0, blocked=blocked)
    bottom = _sample_cell(h, w, row=h - 1, blocked=blocked)
    left = _sample_cell(h, w, col=0, blocked=blocked)
    right = _sample_cell(h, w, col=w - 1, blocked=blocked)
    if None in (portal, top, bottom, left, right):
        return None

    anchors = [portal, top, bottom, left, right]
    cells = {portal}
    cur = portal
    targets = [top, bottom, left, right]
    random.shuffle(targets)
    for tgt in targets:
        path = _shortest_path(h, w, cur, tgt, blocked)
        if path is None:
            return None
        cells.update(path)
        cur = tgt

    anchors_set = set(anchors)
    frontier = set()
    for r, c in list(cells):
        for nr, nc in _neighbors4(h, w, (r, c)):
            if blocked is not None and (nr, nc) == blocked:
                continue
            if (nr, nc) not in cells:
                frontier.add((nr, nc))

    extra = random.randint(0, max(0, h * w - len(cells)))
    frontier = list(frontier)
    for _ in range(extra):
        if not frontier:
            break
        cell = random.choice(frontier)
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors4(h, w, cell):
            if blocked is not None and nxt == blocked:
                continue
            if nxt not in cells and nxt not in frontier:
                frontier.append(nxt)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_full_mask(mask, anchors_set)

    if not mask[:, portal_x].any():
        return None
    if not mask[0, :].any() or not mask[h - 1, :].any():
        return None
    if not mask[:, 0].any() or not mask[:, w - 1].any():
        return None
    if blocked is not None and mask[blocked]:
        return None
    if h * w > 1 and int(mask.sum()) < 2:
        return None
    if not _is_connected(mask):
        return None
    return mask


def _mark_buffer(buffer: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = buffer.shape
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                buffer[nr, nc] = True


def _sample_portal_columns(width: int, count: int, min_gap: int) -> list[int] | None:
    for _ in range(200):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= min_gap for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_shape_span(portal: int, left_bound: int, right_bound: int) -> tuple[int, int, int]:
    feasible = []
    for left_extent in range(portal - left_bound + 1):
        for right_extent in range(right_bound - portal + 1):
            width = left_extent + right_extent + 1
            if width <= 5:
                feasible.append((left_extent, right_extent, width))
    assert feasible
    nontrivial = [item for item in feasible if item[2] > 1]
    if nontrivial and random.random() < 0.85:
        left_extent, right_extent, width = random.choice(nontrivial)
    else:
        left_extent, right_extent, width = random.choice(feasible)
    return left_extent, right_extent, width


def _can_place(buffer: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    return all(not buffer[r, c] for r, c in coords)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 22)
    w = random.randint(11, 22)
    sep = random.randint(4, h - 5)
    common_depth = min(sep, h - sep - 1)
    if common_depth < 2:
        return None

    max_portals = min(5, w // 3)
    if max_portals < 2:
        return None
    portal_count = random.randint(2, max_portals)
    portal_cols = _sample_portal_columns(w, portal_count, min_gap=2)
    if portal_cols is None:
        return None

    side_flags = [SIDE_TOP] * portal_count
    bottom_count = random.randint(1, portal_count - 1)
    for idx in random.sample(range(portal_count), bottom_count):
        side_flags[idx] = SIDE_BOTTOM

    colors = random.sample(PALETTE, portal_count)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[sep, :] = GREY
    for c in portal_cols:
        grid[sep - 1, c] = MAROON
        grid[sep + 1, c] = MAROON

    input_buffer = np.zeros((h, w), dtype=bool)
    output_buffer = np.zeros((h, w), dtype=bool)
    blocked = {(sep, c) for c in range(w)}
    blocked |= {(sep - 1, c) for c in portal_cols}
    blocked |= {(sep + 1, c) for c in portal_cols}
    for r, c in blocked:
        input_buffer[r, c] = False
        output_buffer[r, c] = False

    order = list(range(portal_count))
    random.shuffle(order)
    for idx in order:
        portal = portal_cols[idx]
        side = side_flags[idx]
        color = colors[idx]
        left_bound = 0 if idx == 0 else portal_cols[idx - 1] + 1
        right_bound = w - 1 if idx == portal_count - 1 else portal_cols[idx + 1] - 1

        placed = False
        for _ in range(120):
            max_h = min(5, common_depth)
            height_choices = list(range(1, max_h + 1))
            if max_h >= 2 and random.random() < 0.85:
                height_choices = list(range(2, max_h + 1))
            shape_h = random.choice(height_choices)
            max_gap = common_depth - shape_h
            gap = random.randint(0, max_gap)

            left_extent, right_extent, shape_w = _sample_shape_span(portal, left_bound, right_bound)
            c0 = portal - left_extent
            r0 = sep - gap - shape_h if side == SIDE_TOP else sep + gap + 1
            if side == SIDE_TOP:
                r1 = sep - gap - 1
                blocked_rel = (shape_h - 1, left_extent) if gap == 0 else None
            else:
                r1 = r0 + shape_h - 1
                blocked_rel = (0, left_extent) if gap == 0 else None

            mask = _build_shape_mask(shape_h, shape_w, left_extent, blocked_rel)
            if mask is None:
                continue

            coords = [(r0 + r, c0 + c) for r, c in np.argwhere(mask)]
            if any((r, c) in blocked for r, c in coords):
                continue
            if not _can_place(input_buffer, coords):
                continue

            bbox = (r0, c0, r1, c0 + shape_w - 1)
            out_coords = _transform_component(coords, bbox, sep, portal)
            if any((r, c) in blocked for r, c in out_coords):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in out_coords):
                continue
            if side == SIDE_TOP and any(r <= sep for r, _c in out_coords):
                continue
            if side == SIDE_BOTTOM and any(r >= sep for r, _c in out_coords):
                continue
            if not _can_place(output_buffer, out_coords):
                continue

            for r, c in coords:
                grid[r, c] = color
            _mark_buffer(input_buffer, coords)
            _mark_buffer(output_buffer, out_coords)
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-portal ARC input")
