from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    dirs = DIRS8 if connectivity == 8 else DIRS4
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _median_cell(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int]:
    pts = list(cells)
    rows = sorted(r for r, _ in pts)
    cols = sorted(c for _, c in pts)
    idx = (len(pts) - 1) // 2
    return int(rows[idx]), int(cols[idx])


def _wall_component(g: np.ndarray, starts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = g.shape
    wall: set[tuple[int, int]] = set()
    q = deque(starts)
    while q:
        r, c = q.popleft()
        if (r, c) in wall or int(g[r, c]) != WALL:
            continue
        wall.add((r, c))
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in wall and int(g[nr, nc]) == WALL:
                q.append((nr, nc))
    return wall


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray closed border from its internal marker and move the marker to the region center.

    Input structure:
    - The background is black.
    - Several separate irregular closed regions have one-pixel-thick gray borders.
    - Each closed interior contains exactly one colored marker pixel; the other interior cells are black.

    Rule:
    - Find every black interior component enclosed by gray wall pixels, including its one colored marker.
    - Replace all gray border pixels around that interior with the marker color.
    - Erase the marker's original cell and redraw it at the median row and median column of the interior black cells, using the upper-left choice for even medians.

    Key insight: The gray wall is an uncolored container whose final color is supplied by the unique internal marker, and the marker itself must be recentered rather than left in place.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    passable = g != WALL
    updates: list[tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], int]] = []

    for comp in _components(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        markers: list[tuple[int, int, int]] = []
        black: list[tuple[int, int]] = []
        for r, c in comp:
            v = int(g[r, c])
            if v == BG:
                black.append((r, c))
            elif v != WALL:
                markers.append((r, c, v))

        if len(markers) != 1 or not black:
            continue

        marker_r, marker_c, color = markers[0]
        target = _median_cell(black)
        seeds: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == WALL:
                    seeds.add((nr, nc))
        boundary = _wall_component(g, seeds)
        if boundary:
            updates.append((boundary, (marker_r, marker_c), target, int(color)))

    for boundary, marker, target, color in updates:
        for r, c in boundary:
            out[r, c] = color
        out[marker] = BG
        out[target] = color

    return out


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _interior_has_hole(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    mask = np.ones((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = False
    for comp in _components(mask, connectivity=4):
        if not _touches_edge(comp, h, w):
            return True
    return False


def _transform_cells(
    interior: set[tuple[int, int]],
    boundary: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int, int]:
    def rot(pt: tuple[int, int], hh: int, ww: int, k: int) -> tuple[int, int]:
        r, c = pt
        if k == 0:
            return r, c
        if k == 1:
            return c, hh - 1 - r
        if k == 2:
            return hh - 1 - r, ww - 1 - c
        return ww - 1 - c, r

    k = rng.randrange(4)
    if k % 2:
        nh, nw = w, h
    else:
        nh, nw = h, w
    interior = {rot(p, h, w, k) for p in interior}
    boundary = {rot(p, h, w, k) for p in boundary}
    h, w = nh, nw

    if rng.random() < 0.5:
        interior = {(h - 1 - r, c) for r, c in interior}
        boundary = {(h - 1 - r, c) for r, c in boundary}
    if rng.random() < 0.5:
        interior = {(r, w - 1 - c) for r, c in interior}
        boundary = {(r, w - 1 - c) for r, c in boundary}
    return interior, boundary, h, w


def _crop_patch(
    interior: set[tuple[int, int]],
    boundary: set[tuple[int, int]],
) -> dict[str, object]:
    footprint = interior | boundary
    r0 = min(r for r, _ in footprint)
    c0 = min(c for _, c in footprint)
    r1 = max(r for r, _ in footprint)
    c1 = max(c for _, c in footprint)
    shifted_interior = {(r - r0, c - c0) for r, c in interior}
    shifted_boundary = {(r - r0, c - c0) for r, c in boundary}
    return {
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "interior": shifted_interior,
        "boundary": shifted_boundary,
    }


def _make_patch(rng: random.Random, even_black_count: bool) -> dict[str, object] | None:
    for _ in range(100):
        inner_h = rng.randint(3, 7)
        inner_w = rng.randint(3, 7)
        max_size = min(22, inner_h * inner_w)
        sizes = [s for s in range(5, max_size + 1) if ((s - 1) % 2 == 0) == even_black_count]
        if not sizes:
            continue
        target_size = rng.choice(sizes)

        start = (rng.randint(1, inner_h), rng.randint(1, inner_w))
        cells = {start}
        frontier: set[tuple[int, int]] = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 1 <= nr <= inner_h and 1 <= nc <= inner_w:
                frontier.add((nr, nc))

        while len(cells) < target_size and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIRS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 1 <= nr <= inner_h and 1 <= nc <= inner_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target_size:
            continue
        if _interior_has_hole(cells, inner_h + 2, inner_w + 2):
            continue

        boundary: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                p = (r + dr, c + dc)
                if p not in cells:
                    boundary.add(p)

        patch = _crop_patch(cells, boundary)
        interior = patch["interior"]
        boundary = patch["boundary"]
        ph = int(patch["h"])
        pw = int(patch["w"])
        interior, boundary, ph, pw = _transform_cells(interior, boundary, ph, pw, rng)
        patch = _crop_patch(interior, boundary)
        interior = patch["interior"]

        candidates = list(interior)
        rng.shuffle(candidates)
        for marker in candidates:
            black = [p for p in interior if p != marker]
            target = _median_cell(black)
            if target in interior and target != marker:
                patch["marker"] = marker
                patch["target"] = target
                return patch
    return None


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: dict[str, object],
    color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = grid.shape
    ph = int(patch["h"])
    pw = int(patch["w"])
    if ph + 2 > h or pw + 2 > w:
        return None

    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if not bool(reserved[r0 - 1:r0 + ph + 1, c0 - 1:c0 + pw + 1].any()):
                candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = rng.choice(candidates)
    boundary = {(r0 + r, c0 + c) for r, c in patch["boundary"]}
    interior = {(r0 + r, c0 + c) for r, c in patch["interior"]}
    marker_rel = patch["marker"]
    target_rel = patch["target"]
    marker = (r0 + marker_rel[0], c0 + marker_rel[1])
    target = (r0 + target_rel[0], c0 + target_rel[1])

    for r, c in boundary:
        grid[r, c] = WALL
    grid[marker] = int(color)
    reserved[r0 - 1:r0 + ph + 1, c0 - 1:c0 + pw + 1] = True
    return {
        "boundary": boundary,
        "interior": interior,
        "marker": marker,
        "target": target,
        "color": int(color),
    }


def _expected_output(inp: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for reg in regions:
        color = int(reg["color"])
        for r, c in reg["boundary"]:
            out[r, c] = color
        out[reg["marker"]] = BG
        out[reg["target"]] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 30)
    w = rng.randint(17, 30)
    area = h * w
    max_regions = 5 if area >= 520 else 4 if area >= 380 else 3
    n_regions = rng.randint(2, max_regions)

    colors = rng.sample(MARKER_COLORS, n_regions)
    parity_plan = [True, False]
    rng.shuffle(parity_plan)
    while len(parity_plan) < n_regions:
        parity_plan.append(bool(rng.getrandbits(1)))

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, object]] = []

    for color, even_black_count in zip(colors, parity_plan):
        placed = None
        for _ in range(80):
            patch = _make_patch(rng, even_black_count)
            if patch is None:
                continue
            placed = _place_patch(grid, reserved, patch, color, rng)
            if placed is not None:
                break
        if placed is None:
            return None
        regions.append(placed)

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG:
        return None

    expected = _expected_output(grid, regions)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid centered-marker container input")
