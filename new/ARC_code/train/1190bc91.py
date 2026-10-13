from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import find_components, get_bg
import random


DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _component_coords(comp: dict) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in comp["coords"]]


def _ordered_straight_run(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "foreground run must be horizontal or vertical"

    if len(rows) == 1:
        row = next(iter(rows))
        cs = sorted(c for _r, c in coords)
        assert cs == list(range(cs[0], cs[-1] + 1)), "horizontal run must be contiguous"
        return [(row, c) for c in cs]

    col = next(iter(cols))
    rs = sorted(r for r, _c in coords)
    assert rs == list(range(rs[0], rs[-1] + 1)), "vertical run must be contiguous"
    return [(r, col) for r in rs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse grid by casting occluding diagonal rays from the longest run and filling the remaining diagonal-connected regions from the short bars.

    Input structure
    - Black-background grid with one unique longest 4-connected foreground object.
    - The longest object is a horizontal or vertical multicolor run; all other foreground objects are shorter monochrome bars.

    Rule
    - On a blank canvas, process the longest run from left to right or top to bottom; each pixel emits four 45-degree rays that stop at the first already-painted pixel.
    - Each remaining monochrome bar fills every still-blank pixel reachable from it by diagonal steps.

    Key insight
    - Earlier pixels in the long run occlude later diagonal rays, and those ray barriers partition the background into diagonal-connected regions for the short bars.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(comps) >= 2, "expected a main run plus at least one short bar"

    max_area = max(int(comp["area"]) for comp in comps)
    main_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(main_candidates) == 1, "main run must be the unique largest foreground object"

    main = main_candidates[0]
    main_coords = _component_coords(main)
    assert len({int(g[r, c]) for r, c in main_coords}) >= 2, "main run must be multicolor"
    ordered_main = _ordered_straight_run(main_coords)

    out = np.full_like(g, bg)
    for r, c in ordered_main:
        color = int(g[r, c])
        assert color != bg
        assert out[r, c] == bg, "main source pixel unexpectedly already painted"
        out[r, c] = color
        for dr, dc in DIAG_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr < H and 0 <= nc < W and out[nr, nc] == bg:
                out[nr, nc] = color
                nr += dr
                nc += dc

    main_set = set(main_coords)
    base = out.copy()
    fill_regions: list[tuple[int, set[tuple[int, int]]]] = []
    for comp in comps:
        coords = _component_coords(comp)
        if set(coords) == main_set:
            continue

        ordered_bar = _ordered_straight_run(coords)
        colors = {int(g[r, c]) for r, c in ordered_bar}
        assert len(colors) == 1, "short bar must be monochrome"
        color = next(iter(colors))
        assert color != bg
        assert all(base[r, c] == bg for r, c in ordered_bar), "short bar overlaps main rays"

        seen = set(ordered_bar)
        q = deque(ordered_bar)
        while q:
            r, c = q.popleft()
            for dr, dc in DIAG_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and base[nr, nc] == bg and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        fill_regions.append((color, seen))

    claimed: set[tuple[int, int]] = set()
    for color, region in fill_regions:
        assert claimed.isdisjoint(region), "short-bar diagonal regions must be disjoint"
        claimed.update(region)
        for r, c in region:
            out[r, c] = color

    for r, c in np.argwhere(g != bg):
        assert out[int(r), int(c)] == int(g[int(r), int(c)]), "input foreground must be preserved"

    return out

# --------------- generator (o5x) ---------------

# --- Configuration ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 4, 10
RUN_LENGTH_MIN, RUN_LENGTH_MAX = 3, 7    # main run pixel count
POS_MARGIN = 1                            # perpendicular position inset from grid edges
COLOR_MIN, COLOR_MAX = 1, 9              # non-background color range (inclusive)
MAX_RETRIES = 300                         # generate_input() retry budget

_DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ORTH_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Edge definitions: (name, pair_generator) -- each yields adjacent cell pairs along one edge.
_EDGE_DEFS = {
    "top":    lambda n: (((0, c), (0, c + 1)) for c in range(n - 1)),
    "bottom": lambda n: (((n - 1, c), (n - 1, c + 1)) for c in range(n - 1)),
    "left":   lambda n: (((r, 0), (r + 1, 0)) for r in range(n - 1)),
    "right":  lambda n: (((r, n - 1), (r + 1, n - 1)) for r in range(n - 1)),
}


def _diag_flood(ray, seeds):
    """BFS all zero-cells in *ray* diag-reachable from *seeds*. Returns the visited set (includes seeds)."""
    n = ray.shape[0]
    seen = set(seeds)
    q = deque(seeds)
    while q:
        r, c = q.popleft()
        for dr, dc in _DIAG_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and ray[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _cast_main_rays(n, orientation, length, pos, colors):
    """Place the main run and cast 4 diagonal rays from each pixel (with occlusion). Returns (ray_grid, coords) or (None, None)."""
    ray = np.zeros((n, n), dtype=int)
    if orientation == "H":
        coords = [(pos, c) for c in range(n - length, n)]
    else:
        coords = [(r, pos) for r in range(n - length, n)]

    for (r, c), color in zip(coords, colors):
        if ray[r, c] != 0:
            return None, None
        ray[r, c] = color
        for dr, dc in _DIAG_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr < n and 0 <= nc < n and ray[nr, nc] == 0:
                ray[nr, nc] = color
                nr += dr
                nc += dc
    return ray, coords


def _diag_labels(ray):
    """Label each diagonal-connected zero-region in *ray* with a unique integer. Non-zero cells get -1."""
    n = ray.shape[0]
    labels = np.full_like(ray, -1, dtype=int)
    next_label = 0
    for sr, sc in np.argwhere(ray == 0):
        sr, sc = int(sr), int(sc)
        if labels[sr, sc] >= 0:
            continue
        for cell in _diag_flood(ray, [(sr, sc)]):
            labels[cell] = next_label
        next_label += 1
    return labels


def _touches(cells, blocked):
    """True if any cell in *cells* is orthogonally adjacent to a cell in *blocked*."""
    for r, c in cells:
        for dr, dc in _ORTH_DIRS:
            if (r + dr, c + dc) in blocked:
                return True
    return False


def _edge_pairs(ray, labels, excluded_edge, blocked):
    """Find valid size-2 bar placements along grid edges (excluding *excluded_edge* and cells adjacent to *blocked*)."""
    n = ray.shape[0]
    pairs = []
    for edge_name, gen_fn in _EDGE_DEFS.items():
        if edge_name == excluded_edge:
            continue
        for pair in gen_fn(n):
            if all(ray[p] == 0 for p in pair) and not _touches(pair, blocked):
                pairs.append((pair, frozenset(int(labels[p]) for p in pair)))
    return pairs


def _choose_bars(ray, main_coords, orientation):
    """Select 1 or 2 compatible bar placements from grid edges."""
    excluded_edge = "right" if orientation == "H" else "bottom"
    labels = _diag_labels(ray)
    candidates = _edge_pairs(ray, labels, excluded_edge, set(main_coords))
    if not candidates:
        return None

    compatible_twos = []
    for i, (pair_a, labels_a) in enumerate(candidates):
        for pair_b, labels_b in candidates[i + 1:]:
            if labels_a & labels_b or _touches(pair_b, set(pair_a)):
                continue
            compatible_twos.append((pair_a, pair_b))

    if compatible_twos and random.choice((1, 2)) == 2:
        return list(random.choice(compatible_twos))
    return [random.choice(candidates)[0]]


def _bar_regions_disjoint(ray, bars):
    """Check that the diagonal flood regions of all bars are mutually disjoint."""
    claimed = set()
    for pair in bars:
        region = _diag_flood(ray, list(pair))
        if claimed & region:
            return False
        claimed.update(region)
    return True


def _build_output(ray, bars, bar_colors):
    """Overlay each bar's diagonal flood region onto the ray grid."""
    out = ray.copy()
    for pair, color in zip(bars, bar_colors):
        for r, c in _diag_flood(ray, list(pair)):
            out[r, c] = color
    return out


def _gen_attempt():
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    orientation = random.choice(("H", "V"))
    length = random.randint(RUN_LENGTH_MIN, min(n - 1, RUN_LENGTH_MAX))
    pos = random.randint(POS_MARGIN, n - 1 - POS_MARGIN)

    colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), length)
    ray, main_coords = _cast_main_rays(n, orientation, length, pos, colors)
    if ray is None:
        return None

    bars = _choose_bars(ray, main_coords, orientation)
    if not bars or not _bar_regions_disjoint(ray, bars):
        return None

    remaining = [c for c in range(COLOR_MIN, COLOR_MAX + 1) if c not in colors]
    bar_colors = random.sample(remaining, len(bars))

    inp = np.zeros((n, n), dtype=int)
    for (r, c), color in zip(main_coords, colors):
        inp[r, c] = color
    for pair, color in zip(bars, bar_colors):
        for r, c in pair:
            inp[r, c] = color

    expected = _build_output(ray, bars, bar_colors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 1190bc91 input")
