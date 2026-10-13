import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
INNER_COLORS = (1, 3, 4, 6, 7, 8, 9)

EDGE_MARGIN = 1
GROUP_PADDING = 1
GRID_MIN = 14
GRID_MAX = 30
MAX_ATTEMPTS = 400
MAX_GROUPS = 3
MAX_DEPTH = 5


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_size = 2 * (h + w) - 4
    coord_set = set(coords)
    if len(coord_set) != expected_size:
        return None

    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    return r0, c0, r1, c1


def _extract_frames(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    frames: list[tuple[int, int, int, int]] = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, RED):
            continue
        for comp in _components(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue
            frames.append(bbox)
            for r, c in comp:
                frame_mask[r, c] = True
    return frames, frame_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each red seed into a diamond whose radius is encoded by its surrounding nested frames.

    Input structure:
    - The background is Black.
    - One or more groups of nested one-pixel-thick rectangular frames appear on the grid.
    - The outer frame of each group is Grey, inner frames use other colors, and Red seed pixels lie inside the innermost frame.

    Rule:
    - For each Red seed, count how many rectangular frames strictly contain that seed; this nested depth is its growth radius.
    - Paint every in-bounds cell whose Manhattan distance from the seed is at most that radius.
    - Do not overwrite any frame pixel: frame colors stay unchanged even when the diamond reaches them.
    - Merge all grown diamonds by simple union, keeping overlapping pixels Red.

    Key insight:
    - The nested-frame depth is the hidden numeric signal: it directly becomes the Manhattan radius of the Red diamond.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, frame_mask = _extract_frames(g)
    if not frames:
        return out

    for r, c in np.argwhere(g == RED):
        r = int(r)
        c = int(c)
        depth = sum(r0 < r < r1 and c0 < c < c1 for r0, c0, r1, c1 in frames)
        if depth <= 0:
            continue
        for dr in range(-depth, depth + 1):
            rr = r + dr
            if not (0 <= rr < g.shape[0]):
                continue
            reach = depth - abs(dr)
            c_start = max(0, c - reach)
            c_end = min(g.shape[1] - 1, c + reach)
            for cc in range(c_start, c_end + 1):
                if not frame_mask[rr, cc]:
                    out[rr, cc] = RED
    return out


def _stack_shape(depth: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    return inner_h + 4 * (depth - 1) + 2, inner_w + 4 * (depth - 1) + 2


def _sample_seed_positions(
    rng: random.Random, inner_h: int, inner_w: int, depth: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    preferred = [
        (r, c)
        for r, c in cells
        if min(r, c, inner_h - 1 - r, inner_w - 1 - c) < depth
    ]
    source = preferred if preferred else cells
    seed_count = rng.randint(1, min(3, len(cells)))

    def spaced_enough(pts: list[tuple[int, int]]) -> bool:
        for i in range(len(pts)):
            r1, c1 = pts[i]
            for j in range(i + 1, len(pts)):
                r2, c2 = pts[j]
                if abs(r1 - r2) + abs(c1 - c2) < 2:
                    return False
        return True

    for _ in range(40):
        pool = source if len(source) >= seed_count else cells
        pts = rng.sample(pool, seed_count)
        if seed_count == 1 or spaced_enough(pts):
            return pts
    return rng.sample(cells, seed_count)


def _make_stack(rng: random.Random, depth: int, inner_h: int, inner_w: int) -> np.ndarray:
    h, w = _stack_shape(depth, inner_h, inner_w)
    patch = np.full((h, w), BLACK, dtype=int)
    inner_colors = rng.sample(INNER_COLORS, depth - 1)
    for layer in range(depth):
        inset = 2 * layer
        r0, c0 = inset, inset
        r1, c1 = h - 1 - inset, w - 1 - inset
        color = GREY if layer == 0 else inner_colors[layer - 1]
        patch[r0 : r1 + 1, c0] = color
        patch[r0 : r1 + 1, c1] = color
        patch[r0, c0 : c1 + 1] = color
        patch[r1, c0 : c1 + 1] = color

    seed_offset = 2 * (depth - 1) + 1
    for sr, sc in _sample_seed_positions(rng, inner_h, inner_w, depth):
        patch[seed_offset + sr, seed_offset + sc] = RED
    return patch


def _region_is_free(occ: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - GROUP_PADDING)
    c0 = max(0, left - GROUP_PADDING)
    r1 = min(occ.shape[0], top + h + GROUP_PADDING)
    c1 = min(occ.shape[1], left + w + GROUP_PADDING)
    return not np.any(occ[r0:r1, c0:c1])


def _group_limit(height: int, width: int) -> int:
    area = height * width
    if area < 260:
        return 1
    if area < 500:
        return 2
    return MAX_GROUPS


def _sample_stack_spec(
    rng: random.Random, height: int, width: int
) -> tuple[int, int, int] | None:
    usable_h = height - 2 * EDGE_MARGIN
    usable_w = width - 2 * EDGE_MARGIN
    max_depth = min(MAX_DEPTH, usable_h // 4, usable_w // 4)
    if max_depth < 1:
        return None

    depth = rng.randint(1, max_depth)
    min_inner_h = 1 if depth >= 2 and rng.random() < 0.25 else 2
    min_inner_w = 1 if depth >= 2 and rng.random() < 0.25 else 2
    max_inner_h = min(7, usable_h - (4 * depth - 2))
    max_inner_w = min(7, usable_w - (4 * depth - 2))
    if max_inner_h < min_inner_h or max_inner_w < min_inner_w:
        return None

    inner_h = rng.randint(min_inner_h, max_inner_h)
    inner_w = rng.randint(min_inner_w, max_inner_w)
    if depth == 1 and inner_h == 1 and inner_w == 1:
        return None
    return depth, inner_h, inner_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BLACK, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    group_count = rng.randint(1, _group_limit(height, width))
    patches: list[np.ndarray] = []
    for _ in range(group_count):
        spec = _sample_stack_spec(rng, height, width)
        if spec is None:
            return None
        depth, inner_h, inner_w = spec
        patches.append(_make_stack(rng, depth, inner_h, inner_w))

    patches.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    for patch in patches:
        ph, pw = patch.shape
        if height - ph - EDGE_MARGIN < EDGE_MARGIN or width - pw - EDGE_MARGIN < EDGE_MARGIN:
            return None
        placed = False
        for _ in range(120):
            top = rng.randint(EDGE_MARGIN, height - ph - EDGE_MARGIN)
            left = rng.randint(EDGE_MARGIN, width - pw - EDGE_MARGIN)
            if not _region_is_free(occ, top, left, ph, pw):
                continue
            grid[top : top + ph, left : left + pw] = patch
            occ[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == RED) <= np.count_nonzero(grid == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
