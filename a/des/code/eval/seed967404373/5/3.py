from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SEED_COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _touches_boundary(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _seed_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[tuple[int, int], int, int, int]:
    seeds = [(r, c) for r, c in coords if int(grid[r, c]) not in (BLACK, GREY)]
    assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
    sr, sc = seeds[0]
    color = int(grid[sr, sc])

    walls = []
    for dr, dc in DIRS4:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == GREY:
            walls.append((dr, dc))

    assert len(walls) == 2, "seed must touch exactly two wall cells"
    vertical = [d for d in walls if d[0] != 0]
    horizontal = [d for d in walls if d[1] != 0]
    assert len(vertical) == 1 and len(horizontal) == 1, "wall cells must form a corner"

    row_sign = -vertical[0][0]
    col_sign = -horizontal[0][1]
    return (sr, sc), color, row_sign, col_sign


def _contiguous_shell_fill(
    coords: list[tuple[int, int]], seed: tuple[int, int], row_sign: int, col_sign: int
) -> list[tuple[int, int]]:
    sr, sc = seed
    shells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in coords:
        dr = row_sign * (r - sr)
        dc = col_sign * (c - sc)
        if dr < 0 or dc < 0:
            continue
        shells[max(dr, dc)].append((r, c))

    filled: list[tuple[int, int]] = []
    k = 0
    while k in shells:
        filled.extend(shells[k])
        k += 1
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed diagonally across its grey-walled room in staircase shells.

    Each enclosed non-grey component contains one colored seed and black room cells.
    The seed's two orthogonal grey neighbors reveal the corner orientation, which fixes
    the growth quadrant. Cells in that quadrant are grouped by Chebyshev distance from
    the seed; filling proceeds shell-by-shell and stops at the first missing shell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _components(g != GREY):
        if _touches_boundary(coords, g.shape):
            continue
        seed, color, row_sign, col_sign = _seed_info(g, coords)
        for r, c in _contiguous_shell_fill(coords, seed, row_sign, col_sign):
            out[r, c] = color
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
REGION_COUNT_CHOICES = (2, 3, 4)
REGION_GAP = 1

GROWTH_HEIGHT_RANGE = (5, 9)
GROWTH_WIDTH_RANGE = (5, 9)
AREA_RANGE = (10, 24)
SHAPE_RETRIES = 120
GEN_RETRIES = 300

MIN_FILLED_CELLS = 4
MIN_SHELL_DEPTH = 2
MAX_FILL_RATIO = 0.82


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _has_hole(mask: np.ndarray) -> bool:
    padded = np.pad(mask, 1, constant_values=False)
    outside = ~padded
    seen = np.zeros_like(outside, dtype=bool)
    queue = deque([(0, 0)])
    seen[0, 0] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < outside.shape[0] and 0 <= nc < outside.shape[1]:
                if outside[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

    return bool(np.any(outside & ~seen))


def _random_polyomino() -> np.ndarray | None:
    work_h = random.randint(*GROWTH_HEIGHT_RANGE)
    work_w = random.randint(*GROWTH_WIDTH_RANGE)
    max_area = min(AREA_RANGE[1], work_h * work_w - 1)
    if max_area < AREA_RANGE[0]:
        return None
    target_area = random.randint(AREA_RANGE[0], max_area)

    start = (random.randrange(work_h), random.randrange(work_w))
    cells: set[tuple[int, int]] = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < work_h and 0 <= nc < work_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target_area and frontier:
        chosen = random.choice(tuple(frontier))
        frontier.remove(chosen)
        cells.add(chosen)
        add_frontier(*chosen)

    if len(cells) != target_area:
        return None

    mask = _cells_to_mask(_normalize_cells(cells))
    if mask.shape[0] < 4 or mask.shape[1] < 4:
        return None
    if _has_hole(mask):
        return None
    return mask


def _corner_growth(mask: np.ndarray, seed: tuple[int, int], signs: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    filled = _contiguous_shell_fill(_mask_to_coords(mask), seed, signs[0], signs[1])
    if not filled:
        return [], -1
    sr, sc = seed
    shells = {
        max(signs[0] * (r - sr), signs[1] * (c - sc))
        for r, c in filled
        if signs[0] * (r - sr) >= 0 and signs[1] * (c - sc) >= 0
    }
    depth = max(shells) if shells else -1
    return filled, depth


def _corner_candidates(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    total = int(np.sum(mask))
    coords = set(_mask_to_coords(mask))
    candidates: list[dict] = []

    for r, c in coords:
        outside = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                outside.append((dr, dc))
        vertical = [d for d in outside if d[0] != 0]
        horizontal = [d for d in outside if d[1] != 0]
        if len(outside) != 2 or len(vertical) != 1 or len(horizontal) != 1:
            continue

        signs = (-vertical[0][0], -horizontal[0][1])
        filled, depth = _corner_growth(mask, (r, c), signs)
        fill_count = len(filled)
        if fill_count < MIN_FILLED_CELLS:
            continue
        if depth < MIN_SHELL_DEPTH:
            continue
        if fill_count >= total:
            continue
        if fill_count / total > MAX_FILL_RATIO:
            continue

        candidates.append(
            {
                "seed": (r, c),
                "signs": signs,
                "filled": filled,
                "depth": depth,
                "fill_count": fill_count,
            }
        )
    return candidates


def _build_region(target_signs: tuple[int, int], color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_RETRIES):
        mask = _random_polyomino()
        if mask is None:
            continue
        matches = [cand for cand in _corner_candidates(mask) if cand["signs"] == target_signs]
        if not matches:
            continue

        candidate = random.choice(matches)
        shape_coords = _mask_to_coords(mask)
        shifted_shape = {(r + 1, c + 1) for r, c in shape_coords}
        patch = np.full((mask.shape[0] + 2, mask.shape[1] + 2), BLACK, dtype=int)

        boundary: set[tuple[int, int]] = set()
        for r, c in shifted_shape:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in shifted_shape:
                    boundary.add((nr, nc))

        for r, c in boundary:
            patch[r, c] = GREY

        seed_r, seed_c = candidate["seed"]
        patch[seed_r + 1, seed_c + 1] = color

        out_patch = patch.copy()
        for r, c in candidate["filled"]:
            out_patch[r + 1, c + 1] = color
        return patch, out_patch
    return None


def _can_place(
    top: int,
    left: int,
    height: int,
    width: int,
    placed: list[tuple[int, int, int, int]],
    gap: int,
) -> bool:
    for pt, pl, ph, pw in placed:
        if not (
            top + height + gap <= pt
            or pt + ph + gap <= top
            or left + width + gap <= pl
            or pl + pw + gap <= left
        ):
            return False
    return True


def _place_rect(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _region_count_in_input(grid: np.ndarray) -> int:
    return sum(1 for comp in _components(grid != GREY) if not _touches_boundary(comp, grid.shape))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    region_count = random.choice(REGION_COUNT_CHOICES)
    colors = random.sample(SEED_COLORS, region_count)
    signs = [random.choice(SIGNS) for _ in range(region_count)]

    regions = []
    for color, sign in zip(colors, signs):
        region = _build_region(sign, color)
        if region is None:
            return None
        regions.append(region)

    order = sorted(range(region_count), key=lambda i: regions[i][0].shape[0] * regions[i][0].shape[1], reverse=True)
    inp = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    for idx in order:
        in_patch, out_patch = regions[idx]
        ph, pw = in_patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)

        chosen = None
        for top, left in positions:
            if _can_place(top, left, ph, pw, placed, REGION_GAP):
                chosen = (top, left)
                break
        if chosen is None:
            return None

        top, left = chosen
        _place_rect(inp, in_patch, top, left)
        _place_rect(expected, out_patch, top, left)
        placed.append((top, left, ph, pw))

    if _region_count_in_input(inp) != region_count:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
