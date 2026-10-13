import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
SIDES = ("top", "right", "bottom", "left")
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(coords)
    return out


def _rotate_ccw(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return -dc, dr


def _find_gap(component: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    pts = set(component)
    endpoints = []
    for r, c in component:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c, shape))
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "outline should become a simple open chain after removing one gap"

    common = set(_neighbors4(*endpoints[0], shape)) & set(_neighbors4(*endpoints[1], shape))
    candidates = [pt for pt in common if pt not in pts]
    assert len(candidates) == 1, "expected one unique gap cell"
    return candidates[0]


def _closed_interior_mask(
    component: list[tuple[int, int]],
    gap: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    rows = []
    cols = []
    for r, c in component:
        wall[r, c] = True
        rows.append(r)
        cols.append(c)
    wall[gap] = True
    rows.append(gap[0])
    cols.append(gap[1])

    r0 = max(min(rows) - 1, 0)
    r1 = min(max(rows) + 1, h - 1)
    c0 = max(min(cols) - 1, 0)
    c1 = min(max(cols) + 1, w - 1)

    sub_wall = wall[r0 : r1 + 1, c0 : c1 + 1]
    outside = np.zeros_like(sub_wall, dtype=bool)
    q = deque()

    sh, sw = sub_wall.shape
    for rr in range(sh):
        for cc in (0, sw - 1):
            if not sub_wall[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(sw):
        for rr in (0, sh - 1):
            if not sub_wall[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc, sub_wall.shape):
            if not sub_wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    interior = (~sub_wall) & (~outside)
    assert interior.any(), "expected a non-empty enclosed interior"
    assert len(_components(interior)) == 1, "expected one enclosed region per outline"

    full_interior = np.zeros(shape, dtype=bool)
    full_interior[r0 : r1 + 1, c0 : c1 + 1] = interior
    return full_interior, wall


def _inner_cycle(
    interior_mask: np.ndarray,
    wall_mask: np.ndarray,
    gap: tuple[int, int],
) -> list[tuple[int, int]]:
    band = set()
    for r, c in map(tuple, np.argwhere(interior_mask)):
        if any(wall_mask[nr, nc] for nr, nc in _neighbors4(r, c, interior_mask.shape)):
            band.add((int(r), int(c)))
    assert band, "expected a non-empty inner boundary band"

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in band:
        r, c = cell
        neigh = [(nr, nc) for nr, nc in _neighbors4(r, c, interior_mask.shape) if (nr, nc) in band]
        adjacency[cell] = neigh
    assert all(len(neigh) == 2 for neigh in adjacency.values()), "inner band should be a single cycle"

    start_candidates = [pt for pt in _neighbors4(*gap, interior_mask.shape) if interior_mask[pt]]
    assert len(start_candidates) == 1, "gap should face exactly one interior start cell"
    start = start_candidates[0]

    inward = (start[0] - gap[0], start[1] - gap[1])
    preferred = (start[0] + _rotate_ccw(inward)[0], start[1] + _rotate_ccw(inward)[1])
    if preferred in adjacency[start]:
        curr = preferred
    else:
        curr = adjacency[start][0]

    cycle = [start]
    prev = start
    step_limit = len(band) + 1
    for _ in range(step_limit):
        if curr == start:
            break
        cycle.append(curr)
        nxts = [pt for pt in adjacency[curr] if pt != prev]
        assert len(nxts) == 1, "cycle traversal became ambiguous"
        prev, curr = curr, nxts[0]
    else:
        raise AssertionError("failed to close the inner cycle")

    assert curr == start, "inner cycle did not return to the start"
    assert len(cycle) == len(band), "cycle should visit every band cell exactly once"
    return cycle


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a clockwise inner boundary band whose length equals the number of interior markers in each gapped outline.

    Input structure:
    - Black background with one or more gray 4-connected outline components.
    - Each outline has exactly one 1-cell gap on one side.
    - Each enclosed region contains one or more same-color marker pixels on empty interior cells.

    Rule:
    - For each outline, close its gap conceptually, recover the unique enclosed region, and count its marker pixels N.
    - Starting from the cell one step inside the gap, move clockwise along the inner perimeter and paint the first N cells with the marker color.
    - Clear the original markers to background; keep the gray outline and the gap unchanged.

    Key insight:
    - The marker color chooses the band color, the marker count chooses the band length, and the gap fixes both the start cell and the clockwise travel direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    wall_components = _components(g == WALL)
    assert wall_components, "expected at least one gray outline"

    for component in wall_components:
        gap = _find_gap(component, g.shape)
        interior_mask, closed_wall = _closed_interior_mask(component, gap, g.shape)

        marker_values = g[interior_mask]
        marker_colors = {int(v) for v in np.unique(marker_values) if int(v) not in (BG, WALL)}
        assert len(marker_colors) == 1, "each region should contain markers of exactly one color"
        marker_color = next(iter(marker_colors))
        marker_count = int(np.count_nonzero(marker_values == marker_color))
        assert marker_count >= 1

        cycle = _inner_cycle(interior_mask, closed_wall, gap)
        assert marker_count <= len(cycle), "marker count should fit on the inner perimeter"

        marker_mask = interior_mask & (g != BG) & (g != WALL)
        out[marker_mask] = BG
        for r, c in cycle[:marker_count]:
            out[r, c] = marker_color

    return out


def _make_patch(rng: random.Random, marker_color: int) -> np.ndarray:
    inner_h = rng.randint(4, 8)
    inner_w = rng.randint(4, 8)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)

    patch[0, :] = WALL
    patch[-1, :] = WALL
    patch[:, 0] = WALL
    patch[:, -1] = WALL

    side = rng.choice(SIDES)
    if side == "top":
        gap = (0, rng.randint(1, inner_w))
    elif side == "bottom":
        gap = (inner_h + 1, rng.randint(1, inner_w))
    elif side == "left":
        gap = (rng.randint(1, inner_h), 0)
    else:
        gap = (rng.randint(1, inner_h), inner_w + 1)
    patch[gap] = BG

    area = inner_h * inner_w
    band_len = 2 * (inner_h + inner_w) - 4
    marker_cap = min(area, band_len, max(1, area - 1))
    marker_count = rng.randint(1, min(marker_cap, max(3, area // 2)))

    interior_cells = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    for r, c in rng.sample(interior_cells, marker_count):
        patch[r, c] = marker_color
    return patch


def _row_partitions(rng: random.Random, n: int) -> list[int]:
    if n == 1:
        return [1]
    if n == 2:
        return rng.choice(([2], [1, 1]))
    if n == 3:
        return rng.choice(([1, 2], [2, 1], [3]))
    return rng.choice(([2, 2], [1, 3], [3, 1]))


def _layout_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray | None:
    rng.shuffle(patches)
    row_counts = _row_partitions(rng, len(patches))
    rows: list[list[np.ndarray]] = []
    idx = 0
    for count in row_counts:
        rows.append(patches[idx : idx + count])
        idx += count

    row_specs = []
    for row in rows:
        gap_sizes = [rng.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        row_height = max(p.shape[0] for p in row)
        row_width = sum(p.shape[1] for p in row) + sum(gap_sizes)
        row_specs.append((row, gap_sizes, row_height, row_width))

    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    row_gap_sizes = [rng.randint(2, 4) for _ in range(max(0, len(rows) - 1))]

    total_h = top + bottom + sum(spec[2] for spec in row_specs) + sum(row_gap_sizes)
    total_w = left + right + max(spec[3] for spec in row_specs)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    y = top
    usable_w = total_w - left - right
    for row_idx, (row, gap_sizes, row_height, row_width) in enumerate(row_specs):
        x = left + rng.randint(0, usable_w - row_width)
        for patch_idx, patch in enumerate(row):
            ph, pw = patch.shape
            y_offset = y + rng.randint(0, row_height - ph)
            grid[y_offset : y_offset + ph, x : x + pw] = np.maximum(
                grid[y_offset : y_offset + ph, x : x + pw],
                patch,
            )
            x += pw
            if patch_idx < len(gap_sizes):
                x += gap_sizes[patch_idx]
        y += row_height
        if row_idx < len(row_gap_sizes):
            y += row_gap_sizes[row_idx]

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(1, 4)
    marker_colors = rng.sample(MARKER_COLORS, n_regions)
    patches = [_make_patch(rng, color) for color in marker_colors]
    grid = _layout_patches(rng, patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
