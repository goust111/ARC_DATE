import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GRID_MIN = 16
GRID_MAX = 30
REGION_COUNT_CHOICES = [2, 2, 3, 3, 4]
REGION_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _touches_edge(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _peel_layers(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    layers: list[set[tuple[int, int]]] = []
    while remaining:
        layer = {
            (r, c)
            for r, c in remaining
            if any((nr, nc) not in remaining for nr, nc in _neighbors4(r, c))
        }
        layers.append(layer)
        remaining -= layer
    return layers


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIR8:
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = mask.shape[0] - max(0, -dr)
        dst_c0 = max(0, dc)
        dst_c1 = mask.shape[1] - max(0, -dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _random_rectangle(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 8)
    w = rng.randint(2, 8)
    return np.ones((h, w), dtype=bool)


def _random_l_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 8)
    w = rng.randint(3, 8)
    thickness = rng.randint(1, max(1, min(h, w) // 2))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[h - thickness :, :] = True
    k = rng.randint(0, 3)
    return np.rot90(mask, k)


def _random_plus(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 7)
    w = rng.randint(3, 7)
    vr = rng.randint(1, max(1, h // 3))
    hc = rng.randint(1, max(1, w // 3))
    mask = np.zeros((h, w), dtype=bool)
    r0 = rng.randint(0, h - vr)
    c0 = rng.randint(0, w - hc)
    mask[r0 : r0 + vr, :] = True
    mask[:, c0 : c0 + hc] = True
    return _trim_mask(mask)


def _random_blob(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 8)
    w = rng.randint(3, 8)
    area = rng.randint(max(5, (h * w) // 3), min(h * w, max(6, (2 * h * w) // 3)))
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        choice = rng.choice(list(frontier))
        cells.add(choice)
        mask[choice] = True
    return _trim_mask(mask)


def _random_shape(rng: random.Random) -> np.ndarray:
    maker = rng.choice([_random_rectangle, _random_l_shape, _random_plus, _random_blob])
    return maker(rng)


def _pick_marker_cluster(adjacent_mask: np.ndarray, size: int, rng: random.Random) -> set[tuple[int, int]] | None:
    components = [comp for comp in _components(adjacent_mask) if len(comp) >= size]
    rng.shuffle(components)
    for comp in components:
        comp_set = set(comp)
        for _ in range(20):
            start = rng.choice(comp)
            cluster = {start}
            while len(cluster) < size:
                frontier = {
                    (nr, nc)
                    for r, c in cluster
                    for nr, nc in _neighbors4(r, c)
                    if (nr, nc) in comp_set and (nr, nc) not in cluster
                }
                if not frontier:
                    break
                cluster.add(rng.choice(list(frontier)))
            if len(cluster) == size:
                return cluster
    return None


def _make_region(branch: str, marker_size: int | None, rng: random.Random) -> dict | None:
    for _ in range(REGION_ATTEMPTS):
        base = _random_shape(rng)
        cells = {(int(r), int(c)) for r, c in np.argwhere(base)}
        layers = _peel_layers(cells)
        depth = len(layers)
        if depth == 0:
            continue

        if marker_size is None:
            if branch == "truncated":
                choices = [n for n in range(1, 5) if n > depth]
            else:
                choices = [n for n in range(1, 5) if n <= depth]
            if not choices:
                continue
            chosen_size = rng.choice(choices)
        else:
            chosen_size = marker_size
            if branch == "truncated" and chosen_size <= depth:
                continue
            if branch == "full" and chosen_size > depth:
                continue

        pad = 4
        canvas = np.zeros((base.shape[0] + 2 * pad, base.shape[1] + 2 * pad), dtype=bool)
        interior = canvas.copy()
        interior[pad : pad + base.shape[0], pad : pad + base.shape[1]] = base
        boundary = _dilate4(interior) & ~interior
        adjacent = _dilate4(boundary) & ~boundary & ~interior
        marker_cells = _pick_marker_cluster(adjacent, chosen_size, rng)
        if marker_cells is None:
            continue

        marker = np.zeros_like(interior, dtype=bool)
        for r, c in marker_cells:
            marker[r, c] = True

        occupied = interior | boundary | marker
        blocked = _dilate8(occupied)
        r0, c0, r1, c1 = _bbox(blocked)
        region = {
            "interior": interior[r0 : r1 + 1, c0 : c1 + 1],
            "boundary": boundary[r0 : r1 + 1, c0 : c1 + 1],
            "marker": marker[r0 : r1 + 1, c0 : c1 + 1],
            "blocked": blocked[r0 : r1 + 1, c0 : c1 + 1],
            "size": chosen_size,
            "depth": depth,
            "branch": branch,
        }
        return region
    return None


def _paint_region(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    region: dict,
    color: int,
    rng: random.Random,
    records: list[dict],
) -> bool:
    h, w = grid.shape
    ph, pw = region["blocked"].shape
    if ph > h or pw > w:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        window = blocked[top : top + ph, left : left + pw]
        if np.any(window & region["blocked"]):
            continue

        boundary_pts = [(top + int(r), left + int(c)) for r, c in np.argwhere(region["boundary"])]
        marker_pts = [(top + int(r), left + int(c)) for r, c in np.argwhere(region["marker"])]
        interior_pts = [(top + int(r), left + int(c)) for r, c in np.argwhere(region["interior"])]

        for r, c in boundary_pts:
            grid[r, c] = GREY
            expected[r, c] = GREY
        for r, c in marker_pts:
            grid[r, c] = color
            expected[r, c] = color

        for layer in _peel_layers(interior_pts)[: region["size"]]:
            for r, c in layer:
                expected[r, c] = color

        blocked[top : top + ph, left : left + pw] |= region["blocked"]
        records.append(
            {
                "interior": set(interior_pts),
                "marker": set(marker_pts),
                "color": color,
                "size": region["size"],
                "branch": region["branch"],
            }
        )
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint inward echo layers inside each grey-bounded cavity using the adjacent marker cluster's color and size.

    Input structure:
    - Black is the background and also the color of every enclosed cavity interior.
    - Several grey outlines form closed regions of different shapes, with one-pixel-wide borders.
    - Each region has exactly one 4-connected colored marker cluster touching the outside of its grey border.

    Rule:
    - Find every black connected component that does not touch the grid edge; each such component is a closed region interior.
    - For each interior, locate the unique colored marker cluster 4-adjacent to its grey border; the cluster's color is the echo color and its cell count is the number of inward layers to paint.
    - Repeatedly peel one 4-neighbor boundary layer from the interior and color that layer, stopping after the requested number of layers or when the interior is exhausted.
    - Leave the grey outlines and the outside marker clusters unchanged.

    Key insight:
    - The nearby marker is a two-part code: its position links it to one specific closed region, and its size controls how many inward echo layers of its own color appear inside that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg = [int(c) for c in np.unique(g) if int(c) != BG]
    assert non_bg, "expected grey boundaries and colored markers"
    grey = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    markers = []
    marker_mask = (g != BG) & (g != grey)
    for comp in _components(marker_mask):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "marker component must be monochrome"
        markers.append({"cells": set(comp), "color": colors.pop(), "size": len(comp)})

    out = g.copy()
    h, w = g.shape
    for comp in _components(g == BG):
        if _touches_edge(comp, h, w):
            continue
        interior = set(comp)
        boundary = {
            (nr, nc)
            for r, c in interior
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == grey
        }
        assert boundary, "enclosed region must have a grey boundary"

        hits = []
        for marker in markers:
            if any(
                0 <= nr < h and 0 <= nc < w and (nr, nc) in boundary
                for r, c in marker["cells"]
                for nr, nc in _neighbors4(r, c)
            ):
                hits.append(marker)
        assert len(hits) == 1, "expected one marker cluster per region"
        marker = hits[0]

        for layer in _peel_layers(interior)[: marker["size"]]:
            for r, c in layer:
                out[r, c] = marker["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    n_regions = rng.choice(REGION_COUNT_CHOICES)

    branches = ["full", "truncated"]
    while len(branches) < n_regions:
        branches.append(rng.choice(["full", "truncated"]))
    rng.shuffle(branches)

    colors = rng.sample(MARKER_COLORS, n_regions)

    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()
    blocked = np.zeros((h, w), dtype=bool)
    records: list[dict] = []

    for branch, color in zip(branches, colors):
        region = _make_region(branch, None, rng)
        if region is None:
            return None
        if not _paint_region(grid, expected, blocked, region, color, rng, records):
            return None

    if len({rec["branch"] for rec in records}) < 2:
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[GREY] <= max(counts[color] for color in colors):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
