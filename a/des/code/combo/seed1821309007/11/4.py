import random
from collections import Counter, deque

import numpy as np


BG = 8
OBJ = 5
CORNER_COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]
SIZE_RANGE = (11, 18)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append({"coords": coords, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _region_from_point(r: float, c: float, h: int, w: int) -> str:
    if h == 1 or w == 1:
        raise AssertionError("grid too small")
    slope = (h - 1) / (w - 1)
    d1 = r - slope * c
    d2 = r - ((h - 1) - slope * c)
    eps = 1e-9
    if d1 < -eps and d2 < -eps:
        return "top"
    if d1 > eps and d2 > eps:
        return "bottom"
    if d1 > eps and d2 < -eps:
        return "left"
    if d1 < -eps and d2 > eps:
        return "right"
    raise AssertionError("point lies on an implicit diagonal")


def _corner_colors(grid: np.ndarray):
    h, w = grid.shape
    blocks = {
        "top": grid[:2, :2],
        "right": grid[:2, w - 2 : w],
        "left": grid[h - 2 : h, :2],
        "bottom": grid[h - 2 : h, w - 2 : w],
    }
    colors = {}
    for name, block in blocks.items():
        vals = np.unique(block)
        assert len(vals) == 1, "each corner marker must be monochrome"
        colors[name] = int(vals[0])
    assert len(set(colors.values())) == 4, "corner colors must be distinct"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object according to the triangular region containing its bbox center.

    The four regions are defined by the two invisible corner-to-corner diagonals.
    Their colors come from the 2x2 monochrome corner blocks: top-left -> top region,
    top-right -> right region, bottom-left -> left region, bottom-right -> bottom region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 5

    corner = _corner_colors(g)
    out = g.copy()

    obj_mask = g == OBJ
    obj_mask[:2, :2] = False
    obj_mask[:2, w - 2 : w] = False
    obj_mask[h - 2 : h, :2] = False
    obj_mask[h - 2 : h, w - 2 : w] = False

    components = _find_components(obj_mask)
    assert components, "expected at least one grey object"

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        region = _region_from_point(center_r, center_c, h, w)
        color = corner[region]
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _shape_variants():
    return [
        {(0, 0)},
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (0, 1), (1, 1)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (1, 0), (2, 0)},
        {(0, 0), (0, 1), (1, 1), (1, 2)},
        {(0, 1), (1, 0), (1, 1), (2, 1)},
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (1, 0), (1, 1), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (2, 0)},
    ]


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells, flip_r: bool, flip_c: bool, swap: bool):
    pts = set(cells)
    if flip_r:
        max_r = max(r for r, _ in pts)
        pts = {(max_r - r, c) for r, c in pts}
    if flip_c:
        max_c = max(c for _, c in pts)
        pts = {(r, max_c - c) for r, c in pts}
    if swap:
        pts = {(c, r) for r, c in pts}
    return _normalize(pts)


def _random_shape(rng: random.Random):
    base = rng.choice(_shape_variants())
    return _transform_shape(
        base,
        flip_r=bool(rng.randrange(2)),
        flip_c=bool(rng.randrange(2)),
        swap=bool(rng.randrange(2)),
    )


def _can_place(grid: np.ndarray, cells):
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and grid[nr, nc] != BG:
                    return False
    return True


def _sample_object_for_region(grid: np.ndarray, region: str, rng: random.Random):
    h, w = grid.shape
    for _ in range(200):
        shape = _random_shape(rng)
        sr = max(r for r, _ in shape) + 1
        sc = max(c for _, c in shape) + 1
        r0 = rng.randrange(2, h - sr - 1)
        c0 = rng.randrange(2, w - sc - 1)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        center_r = (min(rr) + max(rr)) / 2.0
        center_c = (min(cc) + max(cc)) / 2.0
        try:
            found = _region_from_point(center_r, center_c, h, w)
        except AssertionError:
            continue
        if found != region:
            continue
        if not _can_place(grid, cells):
            continue
        return cells
    return None


def _count_regions(grid: np.ndarray):
    counts = Counter()
    for comp in _find_components(grid == OBJ):
        r0, c0, r1, c1 = comp["bbox"]
        region = _region_from_point((r0 + r1) / 2.0, (c0 + c1) / 2.0, *grid.shape)
        counts[region] += 1
    return counts


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        h = rng.randint(*SIZE_RANGE)
        w = rng.randint(*SIZE_RANGE)
        grid = np.full((h, w), BG, dtype=int)

        region_to_corner = {
            "top": (slice(0, 2), slice(0, 2)),
            "right": (slice(0, 2), slice(w - 2, w)),
            "left": (slice(h - 2, h), slice(0, 2)),
            "bottom": (slice(h - 2, h), slice(w - 2, w)),
        }
        for region, color in zip(
            ["top", "right", "left", "bottom"],
            rng.sample(CORNER_COLOR_POOL, 4),
        ):
            rs, cs = region_to_corner[region]
            grid[rs, cs] = color

        active_regions = rng.sample(["top", "right", "left", "bottom"], rng.randint(2, 4))
        success = True
        for region in active_regions:
            for _ in range(1 + rng.randrange(2)):
                cells = _sample_object_for_region(grid, region, rng)
                if cells is None:
                    success = False
                    break
                for r, c in cells:
                    grid[r, c] = OBJ
            if not success:
                break
        if not success:
            continue

        counts = _count_regions(grid)
        if sum(counts.values()) < 2:
            continue
        if set(active_regions) - set(counts):
            continue
        return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
