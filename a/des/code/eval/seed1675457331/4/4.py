from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
SOURCE_COLORS = [1, 2, 3, 6, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator bounds.
GRID_MIN = 18
GRID_MAX = 30
COMPONENT_COUNT_CHOICES = [2, 3]
COMPONENT_KINDS = ("single", "double", "triple", "silent")
LOCAL_CANVAS = 15
BASE_LEN_RANGE = (5, 7)
BRANCH_LEN_RANGE = (2, 4)
GENERATION_RETRIES = 300
COMPONENT_RETRIES = 200
PLACEMENT_RETRIES = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _degree(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIRS4)


def _endpoints(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(cell for cell in cells if _degree(cells, cell) == 1)


def _mixed_pipe_color(source_colors: set[int]) -> int:
    if len(source_colors) == 1:
        return next(iter(source_colors))
    if len(source_colors) == 2:
        return YELLOW
    return GREY


def _mixed_target_color(source_colors: set[int]) -> int:
    if len(source_colors) == 1:
        return next(iter(source_colors))
    if len(source_colors) == 2:
        return YELLOW
    return ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through its connected grey pipe network.

    Grey pipe pixels recolor by the number of distinct source colors that reach
    their connected component: one color keeps that color, two colors mix to
    yellow, and three or more colors leave the pipe grey. Orange targets next
    to pipe endpoints use the same one/two/other mixing rule, while source
    pixels themselves keep their original colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _component_cells(g == GREY):
        component = set(coords)
        endpoints = _endpoints(component)

        source_markers: dict[tuple[int, int], int] = {}
        targets: set[tuple[int, int]] = set()
        for r, c in endpoints:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(g.shape, nr, nc) or (nr, nc) in component:
                    continue
                color = int(g[nr, nc])
                if color in (BLACK, GREY):
                    continue
                if color == ORANGE:
                    targets.add((nr, nc))
                else:
                    source_markers[(nr, nc)] = color

        source_colors = set(source_markers.values())
        pipe_color = _mixed_pipe_color(source_colors)
        target_color = _mixed_target_color(source_colors)

        if pipe_color != GREY or len(source_colors) == 0:
            for r, c in component:
                out[r, c] = pipe_color
        for r, c in targets:
            out[r, c] = target_color

    return out


def _kind_source_count(kind: str) -> int:
    return {"silent": 0, "single": 1, "double": 2, "triple": 3}[kind]


def _branch_path(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    length: int,
    size: int,
) -> list[tuple[int, int]] | None:
    sr, sc = start
    path: list[tuple[int, int]] = []
    prev = start
    for step in range(1, length + 1):
        r = sr + dr * step
        c = sc + dc * step
        if not (1 <= r < size - 1 and 1 <= c < size - 1):
            return None
        if (r, c) in cells:
            return None
        for ndr, ndc in DIRS4:
            nr, nc = r + ndr, c + ndc
            if (nr, nc) in cells and (nr, nc) != prev:
                return None
        path.append((r, c))
        prev = (r, c)
    return path


def _candidate_branch_paths(cells: set[tuple[int, int]], size: int) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    for start in sorted(cells):
        deg = _degree(cells, start)
        if deg < 2 or deg >= 4:
            continue
        sr, sc = start
        for dr, dc in DIRS4:
            if (sr + dr, sc + dc) in cells:
                continue
            for length in range(BRANCH_LEN_RANGE[0], BRANCH_LEN_RANGE[1] + 1):
                path = _branch_path(cells, start, dr, dc, length, size)
                if path is not None:
                    candidates.append(path)
    return candidates


def _marker_directions(endpoint: tuple[int, int], neighbor: tuple[int, int]) -> list[tuple[int, int]]:
    dr = endpoint[0] - neighbor[0]
    dc = endpoint[1] - neighbor[1]
    if dr != 0:
        return [(dr, 0), (0, 1), (0, -1)]
    return [(0, dc), (1, 0), (-1, 0)]


def _place_marker(
    endpoint: tuple[int, int],
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    size: int,
) -> tuple[int, int] | None:
    neighbors = [cell for cell in ((endpoint[0] + dr, endpoint[1] + dc) for dr, dc in DIRS4) if cell in cells]
    assert len(neighbors) == 1
    for dr, dc in _marker_directions(endpoint, neighbors[0]):
        r = endpoint[0] + dr
        c = endpoint[1] + dc
        if not _in_bounds((size, size), r, c) or (r, c) in occupied:
            continue
        if any((r + ndr, c + ndc) in occupied and (r + ndr, c + ndc) != endpoint for ndr, ndc in DIRS4):
            continue
        return (r, c)
    return None


def _crop_non_black(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(grid != BLACK)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _generate_component(kind: str) -> tuple[np.ndarray, dict[str, object]] | None:
    source_count = _kind_source_count(kind)
    if kind == "silent":
        target_count = random.choice([2, 3])
    elif kind == "triple":
        target_count = random.choice([1, 2])
    else:
        target_count = random.choice([1, 2])

    endpoint_count = source_count + target_count
    endpoint_count = max(endpoint_count, 2)
    branch_count = endpoint_count - 2

    for _ in range(COMPONENT_RETRIES):
        size = LOCAL_CANVAS
        cells: set[tuple[int, int]] = set()
        orient = random.choice(["h", "v"])
        base_len = random.randint(*BASE_LEN_RANGE)
        center = size // 2
        if orient == "h":
            row = center + random.randint(-1, 1)
            col0 = center - base_len // 2
            cells = {(row, col0 + i) for i in range(base_len)}
        else:
            col = center + random.randint(-1, 1)
            row0 = center - base_len // 2
            cells = {(row0 + i, col) for i in range(base_len)}

        ok = True
        for _branch in range(branch_count):
            candidates = _candidate_branch_paths(cells, size)
            if not candidates:
                ok = False
                break
            cells.update(random.choice(candidates))
        if not ok:
            continue

        endpoints = _endpoints(cells)
        if len(endpoints) != endpoint_count:
            continue

        local = np.full((size, size), BLACK, dtype=int)
        for r, c in cells:
            local[r, c] = GREY

        occupied = set(cells)
        shuffled = endpoints[:]
        random.shuffle(shuffled)
        source_colors = random.sample(SOURCE_COLORS, source_count)
        source_idx = 0
        target_positions: list[tuple[int, int]] = []
        marker_sources: list[int] = []

        for idx, endpoint in enumerate(shuffled):
            marker = _place_marker(endpoint, cells, occupied, size)
            if marker is None:
                ok = False
                break
            if idx < source_count:
                color = source_colors[source_idx]
                source_idx += 1
                marker_sources.append(color)
            else:
                color = ORANGE
                target_positions.append(marker)
            local[marker] = color
            occupied.add(marker)
        if not ok:
            continue

        cropped, (r0, c0) = _crop_non_black(local)
        pipe_coords = [(r - r0, c - c0) for r, c in cells]
        target_coords = [(r - r0, c - c0) for r, c in target_positions]
        return cropped, {"pipe": pipe_coords, "targets": target_coords, "sources": marker_sources}

    return None


def _select_kinds() -> list[str]:
    count = random.choice(COMPONENT_COUNT_CHOICES)
    kinds = {random.choice(["double", "triple"])}
    while len(kinds) < count:
        kinds.add(random.choice(COMPONENT_KINDS))
    selected = list(kinds)
    random.shuffle(selected)
    return selected


def _fits_component(grid: np.ndarray, top: int, left: int, comp: np.ndarray) -> bool:
    h, w = comp.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c1 = min(grid.shape[1], left + w + 1)
    if np.any(grid[r0:r1, c0:c1] != BLACK):
        return False
    return True


def _stamp(grid: np.ndarray, top: int, left: int, comp: np.ndarray) -> None:
    h, w = comp.shape
    region = grid[top : top + h, left : left + w]
    mask = comp != BLACK
    region[mask] = comp[mask]


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _render_expected(inp: np.ndarray, metas: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for meta in metas:
        source_colors = set(int(c) for c in meta["sources"])
        pipe_color = _mixed_pipe_color(source_colors)
        target_color = _mixed_target_color(source_colors)
        for r, c in meta["pipe"]:
            out[r, c] = pipe_color
        for r, c in meta["targets"]:
            out[r, c] = target_color
    return out


def _generate_attempt() -> np.ndarray | None:
    kinds = _select_kinds()
    components: list[tuple[np.ndarray, dict[str, object]]] = []
    for kind in kinds:
        comp = _generate_component(kind)
        if comp is None:
            return None
        components.append(comp)

    max_h = max(comp.shape[0] for comp, _meta in components)
    max_w = max(comp.shape[1] for comp, _meta in components)
    min_size = max(GRID_MIN, max_h + 2, max_w + 2)
    if min_size > GRID_MAX:
        return None

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(min_size, GRID_MAX)
        w = random.randint(min_size, GRID_MAX)
        inp = np.full((h, w), BLACK, dtype=int)
        metas: list[dict[str, object]] = []
        ok = True

        shuffled = components[:]
        random.shuffle(shuffled)
        for comp, meta in shuffled:
            ch, cw = comp.shape
            placed = False
            for _place in range(PLACEMENT_RETRIES):
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                if not _fits_component(inp, top, left, comp):
                    continue
                _stamp(inp, top, left, comp)
                metas.append(
                    {
                        "pipe": _shift_coords(meta["pipe"], top, left),
                        "targets": _shift_coords(meta["targets"], top, left),
                        "sources": list(meta["sources"]),
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        expected = _render_expected(inp, metas)
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
