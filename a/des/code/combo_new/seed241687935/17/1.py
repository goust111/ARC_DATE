import random
from collections import deque

import numpy as np


BACKGROUND = 0
FENCE_COLOR = 5
BRANCHES = ("h_left", "h_right", "v_top", "v_bottom")
GEN_RETRIES = 400
PALETTE = tuple(range(1, 10))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _side_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    top = _mode(grid[0, 1 : w - 1] if w > 2 else grid[0, :])
    bottom = _mode(grid[h - 1, 1 : w - 1] if w > 2 else grid[h - 1, :])
    left = _mode(grid[1 : h - 1, 0] if h > 2 else grid[:, 0])
    right = _mode(grid[1 : h - 1, w - 1] if h > 2 else grid[:, w - 1])
    return top, bottom, left, right


def _detect_fence_color(grid: np.ndarray) -> int | None:
    inner = np.asarray(grid[1:-1, 1:-1], dtype=int)
    non_bg = inner[inner != BACKGROUND]
    if non_bg.size == 0:
        return None
    return _mode(non_bg)


def _find_marker(grid: np.ndarray, region: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    region_set = set(region)
    seen: set[tuple[int, int]] = set()
    marker: list[tuple[int, int]] | None = None

    for start in region:
        if start in seen or int(grid[start]) == BACKGROUND:
            continue
        color = int(grid[start])
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in region_set and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if len(comp) != 2:
            continue
        (r1, c1), (r2, c2) = comp
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            continue
        if marker is not None:
            return None
        marker = [(int(r1), int(c1)), (int(r2), int(c2))]

    return marker


def _fill_color_for_marker(
    marker: list[tuple[int, int]],
    shape: tuple[int, int],
    side_colors: tuple[int, int, int, int],
) -> int:
    h, w = shape
    top, bottom, left, right = side_colors
    (r1, c1), (r2, c2) = marker

    if r1 == r2:
        c_left, c_right = sorted((c1, c2))
        dist_left = c_left
        dist_right = (w - 1) - c_right
        return left if dist_left <= dist_right else right

    r_top, r_bottom = sorted((r1, r2))
    dist_top = r_top
    dist_bottom = (h - 1) - r_bottom
    return top if dist_top <= dist_bottom else bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced region with the border color chosen by its marker direction and side proximity.

    Input structure:
    - The background is black.
    - The outer border is one cell thick and uses four distinct side colors.
    - Gray fence pixels form closed loops that enclose interior regions.
    - Each enclosed region contains exactly one two-pixel monochrome marker, placed horizontally or vertically.

    Rule:
    - Detect the fence color and treat the fence plus the outer border as walls.
    - Split the remaining cells into 4-connected regions and locate the unique two-pixel marker inside each marked region.
    - A horizontal marker compares its left cell's distance to the left edge with its right cell's distance to the right edge; a vertical marker analogously compares top vs bottom.
    - Use the nearer border side's color to fill every black cell in that region, leaving the fence and the marker unchanged.

    Key insight:
    The marker first chooses which axis matters, and only then does the nearer outer side on that axis choose the fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or min(g.shape) < 3:
        return out

    fence = _detect_fence_color(g)
    if fence is None:
        return out

    h, w = g.shape
    side_colors = _side_colors(g)

    mask = np.ones((h, w), dtype=bool)
    mask[0, :] = False
    mask[h - 1, :] = False
    mask[:, 0] = False
    mask[:, w - 1] = False
    mask &= g != fence

    for region in _components_from_mask(mask):
        marker = _find_marker(g, region)
        if marker is None:
            continue
        fill = _fill_color_for_marker(marker, g.shape, side_colors)
        for r, c in region:
            if int(g[r, c]) == BACKGROUND:
                out[r, c] = fill

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    out = set(cells)
    if rng.random() < 0.5:
        out = {(c, r) for r, c in out}
    if rng.random() < 0.5:
        max_r = max(r for r, _ in out)
        out = {(max_r - r, c) for r, c in out}
    if rng.random() < 0.5:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    return _normalize(out)


def _rect_shape(rng: random.Random) -> set[tuple[int, int]]:
    height = rng.randint(3, 7)
    width = rng.randint(3, 8)
    return {(r, c) for r in range(height) for c in range(width)}


def _elbow_shape(rng: random.Random) -> set[tuple[int, int]]:
    height = rng.randint(4, 8)
    width = rng.randint(4, 8)
    arm_h = rng.randint(2, height - 1)
    arm_w = rng.randint(2, width - 1)
    cells = {(r, c) for r in range(height) for c in range(width) if r < arm_h or c < arm_w}
    return _transform_shape(cells, rng)


def _make_region_shape(rng: random.Random) -> set[tuple[int, int]]:
    if rng.random() < 0.6:
        return _rect_shape(rng)
    return _elbow_shape(rng)


def _ring_cells(interior: set[tuple[int, int]]) -> set[tuple[int, int]]:
    ring: set[tuple[int, int]] = set()
    for r, c in interior:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in interior:
                ring.add((nr, nc))
    return ring


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _dilate8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            halo.add((r + dr, c + dc))
        halo.add((r, c))
    return halo


def _branch_candidates(
    interior: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = shape
    candidates = {branch: [] for branch in BRANCHES}
    for r, c in interior:
        if (r, c + 1) in interior:
            dist_left = c
            dist_right = (w - 1) - (c + 1)
            if dist_left < dist_right:
                candidates["h_left"].append(((r, c), (r, c + 1)))
            elif dist_right < dist_left:
                candidates["h_right"].append(((r, c), (r, c + 1)))
        if (r + 1, c) in interior:
            dist_top = r
            dist_bottom = (h - 1) - (r + 1)
            if dist_top < dist_bottom:
                candidates["v_top"].append(((r, c), (r + 1, c)))
            elif dist_bottom < dist_top:
                candidates["v_bottom"].append(((r, c), (r + 1, c)))
    return candidates


def _biased_values(lo: int, hi: int, toward_low: bool) -> list[int]:
    values = list(range(lo, hi + 1))
    if not values:
        return values
    mid = (lo + hi) // 2
    biased = [v for v in values if (v <= mid if toward_low else v >= mid)]
    return biased or values


def _place_region(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    branch: str,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = shape
    for _ in range(200):
        interior = _make_region_shape(rng)
        ring = _ring_cells(interior)
        full = interior | ring
        min_r, min_c, max_r, max_c = _bounds(full)

        top_lo = 2 - min_r
        top_hi = h - 3 - max_r
        left_lo = 2 - min_c
        left_hi = w - 3 - max_c
        if top_lo > top_hi or left_lo > left_hi:
            continue

        if branch == "v_top":
            top_values = _biased_values(top_lo, top_hi, toward_low=True)
            left_values = list(range(left_lo, left_hi + 1))
        elif branch == "v_bottom":
            top_values = _biased_values(top_lo, top_hi, toward_low=False)
            left_values = list(range(left_lo, left_hi + 1))
        elif branch == "h_left":
            top_values = list(range(top_lo, top_hi + 1))
            left_values = _biased_values(left_lo, left_hi, toward_low=True)
        else:
            top_values = list(range(top_lo, top_hi + 1))
            left_values = _biased_values(left_lo, left_hi, toward_low=False)

        top = rng.choice(top_values)
        left = rng.choice(left_values)
        abs_interior = _translate(interior, top, left)
        abs_ring = _translate(ring, top, left)
        halo = _dilate8(abs_interior | abs_ring)
        if any(cell in reserved for cell in halo):
            continue

        candidates = _branch_candidates(abs_interior, shape)[branch]
        if not candidates:
            continue

        marker = rng.choice(candidates)
        return abs_interior, abs_ring, marker
    return None


def _paint_border(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> None:
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right
    grid[0, 0] = top
    grid[0, -1] = top
    grid[-1, 0] = bottom
    grid[-1, -1] = bottom


def _validate_generated_grid(grid: np.ndarray, expected_regions: int) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    top, bottom, left, right = _side_colors(grid)
    assert len({top, bottom, left, right}) == 4

    fence = _detect_fence_color(grid)
    assert fence == FENCE_COLOR

    h, w = grid.shape
    mask = np.ones((h, w), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask &= grid != fence

    out = main(grid)
    region_count = 0
    for region in _components_from_mask(mask):
        marker = _find_marker(grid, region)
        if marker is None:
            continue
        region_count += 1
        fill = _fill_color_for_marker(marker, grid.shape, (top, bottom, left, right))
        saw_black = False
        for r, c in region:
            if int(grid[r, c]) == BACKGROUND:
                saw_black = True
                assert int(out[r, c]) == fill
            else:
                assert int(out[r, c]) == int(grid[r, c])
        assert saw_black
    assert region_count == expected_regions


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    border_choices = [c for c in PALETTE if c != FENCE_COLOR]
    top, bottom, left, right = rng.sample(border_choices, 4)
    _paint_border(grid, top, bottom, left, right)

    max_regions = max(2, min(5, (h * w) // 150 + 1))
    region_count = rng.randint(2, max_regions)

    branches = rng.sample(list(BRANCHES), k=min(region_count, len(BRANCHES)))
    while len(branches) < region_count:
        branches.append(rng.choice(BRANCHES))
    rng.shuffle(branches)

    marker_palette = [c for c in PALETTE if c not in {FENCE_COLOR, top, bottom, left, right}]
    reserved: set[tuple[int, int]] = set()

    for branch in branches:
        placement = _place_region((h, w), reserved, branch, rng)
        if placement is None:
            return None
        interior, ring, marker = placement
        reserved |= _dilate8(interior | ring)

        for r, c in ring:
            grid[r, c] = FENCE_COLOR

        marker_color = rng.choice(marker_palette)
        (r1, c1), (r2, c2) = marker
        grid[r1, c1] = marker_color
        grid[r2, c2] = marker_color

    _validate_generated_grid(grid, region_count)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
