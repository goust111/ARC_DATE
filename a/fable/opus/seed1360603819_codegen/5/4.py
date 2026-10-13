from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EDGES = ("top", "bottom", "left", "right")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _road_endpoints(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    endpoints: list[tuple[int, int]] = []
    for r, c in cells:
        degree = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
        assert degree in (1, 2), "road must be a simple polyline"
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "road must have exactly two endpoints"
    return endpoints


def _frame_interior_cells(grid: np.ndarray, frame_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(frame_cells)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    black_comps = _components(sub == BLACK)
    interior: list[tuple[int, int]] = []
    for comp in black_comps:
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in comp):
            continue
        interior.extend((r0 + rr, c0 + cc) for rr, cc in comp)
    assert interior, "azure frame must enclose black interior"
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_components = _components(g == AZURE)
    assert frame_components, "expected at least one azure frame"

    frame_of_cell: dict[tuple[int, int], int] = {}
    frame_interiors: dict[int, list[tuple[int, int]]] = {}
    for idx, comp in enumerate(frame_components):
        for cell in comp:
            frame_of_cell[cell] = idx
        frame_interiors[idx] = _frame_interior_cells(g, comp)

    road_components = _components(g == GREY)
    assert road_components, "expected at least one grey road"

    road_recolors: dict[tuple[int, int], int] = {}
    frame_colors: dict[int, int] = {}

    for road in road_components:
        endpoints = _road_endpoints(road)
        seed_color = None
        frame_idx = None

        for ep in endpoints:
            r, c = ep
            seed_neighbors = set()
            frame_neighbors = set()
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                val = int(g[nr, nc])
                if val == AZURE:
                    frame_neighbors.add(frame_of_cell[(nr, nc)])
                elif val not in (BLACK, GREY):
                    seed_neighbors.add(val)

            if seed_neighbors:
                assert len(seed_neighbors) == 1, "seed endpoint must touch one seed color"
                assert not frame_neighbors, "seed endpoint cannot also touch a frame"
                found = next(iter(seed_neighbors))
                assert seed_color is None or seed_color == found, "road has conflicting seed colors"
                seed_color = found
            if frame_neighbors:
                assert len(frame_neighbors) == 1, "frame endpoint must touch one frame"
                assert not seed_neighbors, "frame endpoint cannot also touch a seed"
                found = next(iter(frame_neighbors))
                assert frame_idx is None or frame_idx == found, "road has conflicting frame targets"
                frame_idx = found

        assert seed_color is not None, "road must touch a seed at one endpoint"
        assert frame_idx is not None, "road must touch a frame at one endpoint"

        for cell in road:
            previous = road_recolors.get(cell)
            assert previous is None or previous == seed_color
            road_recolors[cell] = seed_color

        previous_frame_color = frame_colors.get(frame_idx)
        assert previous_frame_color is None or previous_frame_color == seed_color, "frame fill conflict"
        frame_colors[frame_idx] = seed_color

    out = g.copy()
    for (r, c), color in road_recolors.items():
        out[r, c] = color
    for idx, color in frame_colors.items():
        for r, c in frame_interiors[idx]:
            out[r, c] = color
    return out


GRID_MIN = 22
GRID_MAX = 30
OBJECT_COUNT_CHOICES = (2, 3, 4)
PATCH_H_RANGE = (10, 14)
PATCH_W_RANGE = (10, 14)
SEED_H_RANGE = (2, 4)
SEED_W_RANGE = (2, 4)
FRAME_INNER_RANGE = (2, 4)
MOTIF_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
GENERATE_ATTEMPTS = 300


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) not in cells:
                out.add((nr, nc))
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    return None


def _build_path(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    normalized = [points[0]]
    for pt in points[1:]:
        if pt != normalized[-1]:
            normalized.append(pt)

    path: list[tuple[int, int]] = []
    for i in range(len(normalized) - 1):
        seg = _segment_cells(normalized[i], normalized[i + 1])
        if seg is None:
            return None
        if i > 0:
            seg = seg[1:]
        path.extend(seg)

    if len(path) != len(set(path)):
        return None
    return path


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    cells = set(path)
    endpoints = 0
    for r, c in path:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints += 1
        elif degree != 2:
            return False
    return endpoints == 2


def _frame_cells(r0: int, c0: int, h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    border = set()
    full = set()
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            full.add((r, c))
            if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1):
                border.add((r, c))
    return border, full


def _place_seed(h: int, w: int, seed_h: int, seed_w: int, edge: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if edge == "left":
        r0 = random.randint(1, h - seed_h - 2)
        c0 = 1
        offset = random.randint(0, seed_h - 1)
        start = (r0 + offset, c0 + seed_w)
    elif edge == "right":
        r0 = random.randint(1, h - seed_h - 2)
        c0 = w - seed_w - 2
        offset = random.randint(0, seed_h - 1)
        start = (r0 + offset, c0 - 1)
    elif edge == "top":
        r0 = 1
        c0 = random.randint(1, w - seed_w - 2)
        offset = random.randint(0, seed_w - 1)
        start = (r0 + seed_h, c0 + offset)
    else:
        r0 = h - seed_h - 2
        c0 = random.randint(1, w - seed_w - 2)
        offset = random.randint(0, seed_w - 1)
        start = (r0 - 1, c0 + offset)
    cells = {(r, c) for r in range(r0, r0 + seed_h) for c in range(c0, c0 + seed_w)}
    return cells, start


def _place_frame(h: int, w: int, frame_h: int, frame_w: int, edge: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    if edge == "left":
        r0 = random.randint(1, h - frame_h - 2)
        c0 = 1
        offset = random.randint(1, frame_h - 2)
        end = (r0 + offset, c0 + frame_w)
    elif edge == "right":
        r0 = random.randint(1, h - frame_h - 2)
        c0 = w - frame_w - 2
        offset = random.randint(1, frame_h - 2)
        end = (r0 + offset, c0 - 1)
    elif edge == "top":
        r0 = 1
        c0 = random.randint(1, w - frame_w - 2)
        offset = random.randint(1, frame_w - 2)
        end = (r0 + frame_h, c0 + offset)
    else:
        r0 = h - frame_h - 2
        c0 = random.randint(1, w - frame_w - 2)
        offset = random.randint(1, frame_w - 2)
        end = (r0 - 1, c0 + offset)
    border, full = _frame_cells(r0, c0, frame_h, frame_w)
    return border, full, end


def _route_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = [[start, end], [start, (start[0], end[1]), end], [start, (end[0], start[1]), end]]
    mid_cols = list(range(1, shape[1] - 1))
    mid_rows = list(range(1, shape[0] - 1))
    random.shuffle(mid_cols)
    random.shuffle(mid_rows)
    for mid_c in mid_cols[:6]:
        candidates.append([start, (start[0], mid_c), (end[0], mid_c), end])
        candidates.append([start, (end[0], mid_c), (start[0], mid_c), end])
    for mid_r in mid_rows[:6]:
        candidates.append([start, (mid_r, start[1]), (mid_r, end[1]), end])
        candidates.append([start, (mid_r, end[1]), (mid_r, start[1]), end])
    random.shuffle(candidates)

    best: list[tuple[int, int]] | None = None
    for points in candidates:
        path = _build_path(points)
        if path is None or len(path) < 5:
            continue
        if not _is_simple_path(path):
            continue
        if any(not _in_bounds(shape, r, c) for r, c in path):
            continue
        if any(blocked[r, c] for r, c in path[1:-1]):
            continue
        if any(blocked[r, c] for r, c in (path[0], path[-1])):
            continue
        if best is None or len(path) > len(best):
            best = path
    return best


def _crop_with_margin(grid: np.ndarray, margin: int = 1) -> np.ndarray:
    pts = np.argwhere(grid != BLACK)
    r0 = max(0, int(pts[:, 0].min()) - margin)
    c0 = max(0, int(pts[:, 1].min()) - margin)
    r1 = min(grid.shape[0] - 1, int(pts[:, 0].max()) + margin)
    c1 = min(grid.shape[1] - 1, int(pts[:, 1].max()) + margin)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_motif(seed_color: int) -> np.ndarray | None:
    for _ in range(MOTIF_ATTEMPTS):
        h = random.randint(*PATCH_H_RANGE)
        w = random.randint(*PATCH_W_RANGE)
        seed_h = random.randint(*SEED_H_RANGE)
        seed_w = random.randint(*SEED_W_RANGE)

        if random.random() < 0.5:
            inner_h = inner_w = random.randint(*FRAME_INNER_RANGE)
        else:
            inner_h = random.randint(*FRAME_INNER_RANGE)
            inner_w = random.randint(*FRAME_INNER_RANGE)
        frame_h = inner_h + 2
        frame_w = inner_w + 2

        seed_edge, frame_edge = random.sample(EDGES, 2)
        seed_cells, start = _place_seed(h, w, seed_h, seed_w, seed_edge)
        frame_border, frame_full, end = _place_frame(h, w, frame_h, frame_w, frame_edge)

        if abs(start[0] - end[0]) + abs(start[1] - end[1]) < 5:
            continue

        seed_blocked = set(seed_cells) | _halo(seed_cells, (h, w))
        frame_blocked = set(frame_full) | _halo(frame_border, (h, w))
        boundary_blocked = np.zeros((h, w), dtype=bool)
        boundary_blocked[0, :] = True
        boundary_blocked[-1, :] = True
        boundary_blocked[:, 0] = True
        boundary_blocked[:, -1] = True

        if boundary_blocked[start] or boundary_blocked[end]:
            continue
        if start in frame_blocked or end in seed_blocked:
            continue

        blocked = boundary_blocked.copy()
        for r, c in seed_blocked | frame_blocked:
            blocked[r, c] = True
        blocked[start] = False
        blocked[end] = False

        path = _route_path((h, w), start, end, blocked)
        if path is None:
            continue

        patch = np.zeros((h, w), dtype=int)
        for r, c in seed_cells:
            patch[r, c] = seed_color
        for r, c in frame_border:
            patch[r, c] = AZURE
        for r, c in path:
            patch[r, c] = GREY

        out = main(patch)
        if np.array_equal(out, patch):
            continue
        if not np.any((patch == GREY) & (out == seed_color)):
            continue
        if not np.any((patch == BLACK) & (out == seed_color)):
            continue
        return _crop_with_margin(patch, margin=1)
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    area = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches)
    base = max(
        GRID_MIN,
        int(np.ceil(np.sqrt(area * 1.35))),
        max(p.shape[0] for p in patches) + 2,
        max(p.shape[1] for p in patches) + 2,
    )
    if base > GRID_MAX:
        return None

    h = random.randint(base, GRID_MAX)
    w = random.randint(base, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = patches[:]
    random.shuffle(order)
    for patch in order:
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paste(grid, patch, r0, c0)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        count = random.choice(OBJECT_COUNT_CHOICES)
        colors = random.sample(SEED_COLORS, count)
        patches: list[np.ndarray] = []
        for color in colors:
            patch = _make_motif(color)
            if patch is None:
                break
            patches.append(patch)
        if len(patches) != count:
            continue

        grid = _place_patches(patches)
        if grid is None:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        if not np.any((grid == GREY) & (out != GREY)):
            continue
        if not np.any((grid == BLACK) & (out != BLACK)):
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
