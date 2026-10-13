from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
_RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _normalize_points(points: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in points]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _mask_from_points(points: list[tuple[int, int]] | set[tuple[int, int]]) -> np.ndarray:
    pts = _normalize_points(points)
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _trace_cycle(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    cellset = {(int(r), int(c)) for r, c in cells}
    assert cellset, "loop must contain at least one pixel"
    for r, c in cellset:
        degree = sum((nr, nc) in cellset for nr, nc in _neighbors4(r, c))
        assert degree == 2, "foreground component must be a single-pixel-wide cycle"

    start = min(cellset)
    first = sorted((nbr for nbr in _neighbors4(*start) if nbr in cellset))[0]
    cycle = [start]
    prev, cur = start, first
    while cur != start:
        cycle.append(cur)
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in cellset and nbr != prev]
        assert len(nxts) == 1, "cycle tracing should have exactly one forward step"
        prev, cur = cur, nxts[0]
    return cycle


def _corner_indices(cycle: list[tuple[int, int]]) -> list[int]:
    corners: list[int] = []
    n = len(cycle)
    for i, (r, c) in enumerate(cycle):
        pr, pc = cycle[i - 1]
        nr, nc = cycle[(i + 1) % n]
        d_in = (r - pr, c - pc)
        d_out = (nr - r, nc - c)
        if d_in != d_out:
            corners.append(i)
    assert len(corners) >= 4, "orthogonal loop should have at least four corners"
    return corners


def _clockwise_corner_indices(cycle: list[tuple[int, int]]) -> list[int]:
    indices = _corner_indices(cycle)
    corners = [cycle[i] for i in indices]

    area2 = 0
    for (r1, c1), (r2, c2) in zip(corners, corners[1:] + corners[:1]):
        area2 += c1 * r2 - c2 * r1
    if area2 < 0:
        indices = list(reversed(indices))
        corners = [cycle[i] for i in indices]

    start = min(range(len(corners)), key=lambda i: corners[i])
    return indices[start:] + indices[:start]


def _count_inner_regions(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> int:
    pts = [(int(r), int(c)) for r, c in cells]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs) - 1, max(rs) + 1
    c0, c1 = min(cs) - 1, max(cs) + 1
    h, w = r1 - r0 + 1, c1 - c0 + 1

    blocked = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        blocked[r - r0, c - c0] = True

    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not bool(blocked[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(blocked[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not bool(blocked[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    count = 0
    for r in range(h):
        for c in range(w):
            if bool(blocked[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            seen[r, c] = True
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not bool(blocked[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _draw_closed_path(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(waypoints) >= 4
    cells: set[tuple[int, int]] = set()
    for (r1, c1), (r2, c2) in zip(waypoints, waypoints[1:] + waypoints[:1]):
        assert r1 == r2 or c1 == c2
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            for c in range(lo, hi + 1):
                cells.add((int(r1), int(c)))
        else:
            lo, hi = sorted((r1, r2))
            for r in range(lo, hi + 1):
                cells.add((int(r), int(c1)))
    return _trace_cycle(cells)


def _paint_loop(grid: np.ndarray, cycle: list[tuple[int, int]], colors: list[int]) -> None:
    corners = _corner_indices(cycle)
    assert len(colors) == len(corners)
    n = len(cycle)
    for seg_idx, start_idx in enumerate(corners):
        end_idx = corners[(seg_idx + 1) % len(corners)]
        idx = start_idx
        while True:
            r, c = cycle[idx]
            assert grid[r, c] == BG
            grid[r, c] = int(colors[seg_idx])
            idx = (idx + 1) % n
            if idx == end_idx:
                break


def _translate_cycle(cycle: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + int(dr), c + int(dc)) for r, c in cycle]


def _transform_points(points: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    mask = _mask_from_points(points)
    if int(_RNG.integers(2)):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(_RNG.integers(4)))
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _four_component_count(cells: set[tuple[int, int]]) -> int:
    seen: set[tuple[int, int]] = set()
    count = 0
    for start in list(cells):
        if start in seen:
            continue
        count += 1
        seen.add(start)
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nbr in _neighbors4(r, c):
                if nbr in cells and nbr not in seen:
                    seen.add(nbr)
                    q.append(nbr)
    return count


def _dilate8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS8 + [(0, 0)]:
            out.add((r + dr, c + dc))
    return out


def _expand_core(core: set[tuple[int, int]], scale: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in core:
        for rr in range(r * scale, r * scale + scale):
            for cc in range(c * scale, c * scale + scale):
                expanded.add((int(rr), int(cc)))
    return expanded


def _random_core(target_regions: int) -> set[tuple[int, int]]:
    for _ in range(200):
        size = int(_RNG.integers(target_regions, target_regions + 5))
        core = {(0, 0)}
        while len(core) < size:
            base = list(core)[int(_RNG.integers(len(core)))]
            dr, dc = DIRS8[int(_RNG.integers(len(DIRS8)))]
            core.add((base[0] + dr, base[1] + dc))
        if _four_component_count(core) == target_regions:
            return core
    raise RuntimeError("failed to sample a valid coarse core")


def _random_shell_loop(target_regions: int) -> list[tuple[int, int]]:
    for _ in range(200):
        scale = int(_RNG.integers(1, 5))
        core = _random_core(target_regions)
        expanded = _expand_core(core, scale)
        boundary = _dilate8(expanded) - expanded
        boundary = set(_transform_points(boundary))
        try:
            cycle = _trace_cycle(boundary)
        except AssertionError:
            continue
        if _count_inner_regions(cycle) != target_regions:
            continue
        if len(_corner_indices(cycle)) > len(COLOR_POOL):
            continue
        return _normalize_points(cycle)
    raise RuntimeError("failed to build a valid shell loop")


def _rectangle_loop() -> list[tuple[int, int]]:
    height = int(_RNG.integers(2, 7))
    width = int(_RNG.integers(2, 7))
    return _draw_closed_path([(0, 0), (0, width), (height, width), (height, 0)])


def _compose_cycles(cycles: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    norm_cycles = [_normalize_points(cycle) for cycle in cycles]
    dims = []
    for cycle in norm_cycles:
        h = max(r for r, _ in cycle) + 1
        w = max(c for _, c in cycle) + 1
        dims.append((h, w))

    if len(norm_cycles) == 1:
        h, w = dims[0]
        top = int(_RNG.integers(1, 5))
        bottom = int(_RNG.integers(1, 5))
        left = int(_RNG.integers(1, 5))
        right = int(_RNG.integers(1, 5))
        canvas_h = top + h + bottom
        canvas_w = left + w + right
        cycle = _translate_cycle(norm_cycles[0], top, left)
        assert max(r for r, _ in cycle) < canvas_h and max(c for _, c in cycle) < canvas_w
        return [cycle]

    assert len(norm_cycles) == 2
    order = list(_RNG.permutation(2))
    a = norm_cycles[order[0]]
    b = norm_cycles[order[1]]
    ha, wa = dims[order[0]]
    hb, wb = dims[order[1]]

    top = int(_RNG.integers(1, 5))
    bottom = int(_RNG.integers(1, 5))
    left = int(_RNG.integers(1, 5))
    right = int(_RNG.integers(1, 5))
    gap = int(_RNG.integers(3, 7))

    if int(_RNG.integers(2)) == 0:
        span_h = max(ha, hb)
        canvas_h = top + span_h + bottom
        canvas_w = left + wa + gap + wb + right
        off_a = (top + int(_RNG.integers(span_h - ha + 1)), left)
        off_b = (top + int(_RNG.integers(span_h - hb + 1)), left + wa + gap)
    else:
        span_w = max(wa, wb)
        canvas_h = top + ha + gap + hb + bottom
        canvas_w = left + span_w + right
        off_a = (top, left + int(_RNG.integers(span_w - wa + 1)))
        off_b = (top + ha + gap, left + int(_RNG.integers(span_w - wb + 1)))

    placed = [_translate_cycle(a, *off_a), _translate_cycle(b, *off_b)]
    for cycle in placed:
        assert max(r for r, _ in cycle) < canvas_h and max(c for _, c in cycle) < canvas_w
    return placed


def _is_straight_segment(points: np.ndarray) -> bool:
    rows = points[:, 0]
    cols = points[:, 1]
    if int(rows.min()) == int(rows.max()):
        uniq = sorted(set(int(c) for c in cols))
        return uniq == list(range(uniq[0], uniq[-1] + 1))
    if int(cols.min()) == int(cols.max()):
        uniq = sorted(set(int(r) for r in rows))
        return uniq == list(range(uniq[0], uniq[-1] + 1))
    return False


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if np.any(grid == YELLOW):
        return False

    used_colors = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    if len(used_colors) != len(set(used_colors)):
        return False
    for color in used_colors:
        pts = np.argwhere(grid == color)
        if pts.size == 0 or not _is_straight_segment(pts):
            return False

    comps = _components_from_mask(grid != BG)
    if not comps:
        return False
    total_corners = 0
    for comp in comps:
        try:
            cycle = _trace_cycle(comp)
        except AssertionError:
            return False
        if _count_inner_regions(cycle) not in (1, 2):
            return False
        total_corners += len(_corner_indices(cycle))
    if total_corners != len(used_colors) or total_corners > len(COLOR_POOL):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Mark the first N clockwise corners of each loop yellow, where N is its enclosed 4-region count."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for comp in _components_from_mask(g != BG):
        cycle = _trace_cycle(comp)
        region_count = _count_inner_regions(cycle)
        corner_order = _clockwise_corner_indices(cycle)
        for idx in corner_order[:region_count]:
            r, c = cycle[idx]
            out[r, c] = YELLOW

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = int(_RNG.integers(2))
        if mode == 0:
            loop_count = 1 + int(_RNG.integers(2))
            if loop_count == 1 and int(_RNG.integers(2)):
                cycles = [_random_shell_loop(target_regions=1)]
            else:
                cycles = [_rectangle_loop() for _ in range(loop_count)]
        else:
            cycles = [_random_shell_loop(target_regions=2)]

        cycles = _compose_cycles(cycles)
        segment_counts = [len(_corner_indices(cycle)) for cycle in cycles]
        if sum(segment_counts) > len(COLOR_POOL):
            continue

        h = max(r for cycle in cycles for r, _ in cycle) + 1
        w = max(c for cycle in cycles for _, c in cycle) + 1
        grid = np.full((h, w), BG, dtype=int)

        palette = list(np.asarray(COLOR_POOL)[_RNG.permutation(len(COLOR_POOL))])
        color_idx = 0
        for cycle, count in zip(cycles, segment_counts):
            seg_colors = [int(c) for c in palette[color_idx : color_idx + count]]
            color_idx += count
            _paint_loop(grid, cycle, seg_colors)

        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
