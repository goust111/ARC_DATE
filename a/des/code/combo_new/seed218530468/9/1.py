import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7)
GEN_RETRIES = 400
MASK_RETRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _border_and_interior(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    border = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                border[rr, cc] = True
                break
    interior = mask & ~border
    return border, interior


def _line_pair(coords: list[tuple[int, int]], axis: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    buckets: dict[int, list[int]] = defaultdict(list)
    if axis == "row":
        for r, c in coords:
            buckets[r].append(c)
        lines = [r for r, cols in buckets.items() if len(cols) >= 2]
        if not lines:
            return None
        row = random.choice(lines)
        cols = sorted(buckets[row])
        return (row, int(cols[0])), (row, int(cols[-1]))

    for r, c in coords:
        buckets[c].append(r)
    lines = [c for c, rows in buckets.items() if len(rows) >= 2]
    if not lines:
        return None
    col = random.choice(lines)
    rows = sorted(buckets[col])
    return (int(rows[0]), col), (int(rows[-1]), col)


def _carve_notch(mask: np.ndarray):
    h, w = mask.shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        if h < 6 or w < 5:
            return
        depth = random.randint(1, h - 4)
        c0 = random.randint(1, w - 3)
        c1 = random.randint(c0 + 1, w - 2)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, c0:c1 + 1] = False
        return

    if w < 6 or h < 5:
        return
    depth = random.randint(1, w - 4)
    r0 = random.randint(1, h - 3)
    r1 = random.randint(r0 + 1, h - 2)
    cols = slice(0, depth) if side == "left" else slice(w - depth, w)
    mask[r0:r1 + 1, cols] = False


def _region_spec(axis: str) -> dict | None:
    for _ in range(MASK_RETRIES):
        h = random.randint(6, 11)
        w = random.randint(6, 11)
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(0, 3)):
            _carve_notch(mask)

        comps = _components(mask)
        if len(comps) != 1:
            continue

        border, interior = _border_and_interior(mask)
        if int(interior.sum()) < 6:
            continue
        if len(_components(interior)) != 1:
            continue
        if len(_components(border)) != 1:
            continue

        interior_coords = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        row_pair = _line_pair(interior_coords, "row")
        col_pair = _line_pair(interior_coords, "col")
        if row_pair is None or col_pair is None:
            continue

        seed_a, seed_b = row_pair if axis == "row" else col_pair
        color_a, color_b = random.sample(SEED_COLORS, 2)
        seeds = [
            (int(seed_a[0]), int(seed_a[1]), int(color_a)),
            (int(seed_b[0]), int(seed_b[1]), int(color_b)),
        ]
        return {
            "axis": axis,
            "height": h,
            "width": w,
            "border": [tuple(map(int, rc)) for rc in np.argwhere(border)],
            "interior": interior_coords,
            "seeds": seeds,
        }
    return None


def _fill_region(out: np.ndarray, coords: list[tuple[int, int]], seeds: list[tuple[int, int, int]]):
    (r1, c1, col1), (r2, c2, col2) = seeds
    if r1 == r2:
        if c1 > c2:
            c1, c2 = c2, c1
            col1, col2 = col2, col1
        c_mid = (c1 + c2) // 2
        for r, c in coords:
            out[r, c] = col1 if c <= c_mid else col2
        return

    assert c1 == c2
    if r1 > r2:
        r1, r2 = r2, r1
        col1, col2 = col2, col1
    r_mid = (r1 + r2) // 2
    for r, c in coords:
        out[r, c] = col1 if r <= r_mid else col2


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey-bounded region by splitting its interior according to the two seed pixels.

    Input structure:
    - The background is black.
    - One or more 1-pixel-wide grey closed outlines enclose irregular regions.
    - Each enclosed region contains exactly two single-pixel seeds of different non-grey colors, aligned either on one row or on one column.

    Rule:
    - Treat grey as the boundary and find every non-grey/non-boundary component that does not touch the outer grid border.
    - In each enclosed component, read the two seed pixels and decide whether they indicate a horizontal split (same row) or a vertical split (same column).
    - If the seeds share a row, fill all cells at columns up to the midpoint with the left seed's color and all cells to the right with the right seed's color; if they share a column, do the analogous top/bottom fill by row midpoint.
    - Keep the grey outline unchanged and leave the black exterior background unchanged.

    Key insight:
    - The orientation of the two interior seeds is the hidden instruction for how the enclosed black area must be bisected and colored.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for comp in _components(g != GREY):
        if _touches_border(comp, h, w):
            continue
        seeds = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seeds) == 2
        assert seeds[0][2] != seeds[1][2]
        same_row = seeds[0][0] == seeds[1][0]
        same_col = seeds[0][1] == seeds[1][1]
        assert same_row != same_col
        _fill_region(out, comp, seeds)

    return out


def _orientations(n_regions: int) -> list[str]:
    if n_regions == 1:
        return [random.choice(("row", "col"))]
    extra_row = random.randint(0, 1)
    n_row = n_regions // 2 + extra_row
    n_col = n_regions - n_row
    axes = ["row"] * n_row + ["col"] * n_col
    random.shuffle(axes)
    return axes


def _place_specs(specs: list[dict], h: int, w: int) -> list[tuple[dict, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[tuple[dict, int, int]] = []
    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        sh, sw = int(spec["height"]), int(spec["width"])
        if sh + 2 > h or sw + 2 > w:
            return None
        positions = [(r0, c0) for r0 in range(1, h - sh) for c0 in range(1, w - sw)]
        random.shuffle(positions)
        spot = None
        for r0, c0 in positions:
            if occupied[r0 - 1:r0 + sh + 1, c0 - 1:c0 + sw + 1].any():
                continue
            spot = (r0, c0)
            break
        if spot is None:
            return None
        r0, c0 = spot
        occupied[r0 - 1:r0 + sh + 1, c0 - 1:c0 + sw + 1] = True
        placed.append((spec, r0, c0))
    return placed


def _render_input(h: int, w: int, placed: list[tuple[dict, int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec, r0, c0 in placed:
        for rr, cc in spec["border"]:
            grid[r0 + rr, c0 + cc] = GREY
        for rr, cc, color in spec["seeds"]:
            grid[r0 + rr, c0 + cc] = color
    return grid


def _expected_output(inp: np.ndarray, placed: list[tuple[dict, int, int]]) -> np.ndarray:
    out = inp.copy()
    for spec, r0, c0 in placed:
        coords = [(r0 + rr, c0 + cc) for rr, cc in spec["interior"]]
        seeds = [(r0 + rr, c0 + cc, color) for rr, cc, color in spec["seeds"]]
        _fill_region(out, coords, seeds)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    max_regions = min(4, max(1, (h * w) // 140))
    n_regions = random.randint(1, max_regions)

    specs: list[dict] = []
    for axis in _orientations(n_regions):
        spec = _region_spec(axis)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs(specs, h, w)
    if placed is None:
        return None

    inp = _render_input(h, w, placed)
    expected = _expected_output(inp, placed)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
