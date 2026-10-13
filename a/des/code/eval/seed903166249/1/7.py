from __future__ import annotations

import itertools
import random

import numpy as np


AZURE = 8
ORANGE = 7
STAMP_COLORS = [1, 2, 3, 4, 5, 6, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_HEIGHT_RANGE = (11, 16)
GRID_WIDTH_RANGE = (11, 16)
STAMP_COUNT_OPTIONS = (3, 4, 5)
TIE_AREAS = (2, 4, 6)
IMBALANCED_AREAS = (3, 5, 7)
MAX_SHAPE_SIDE = 4
EDGE_MARGIN = 1
SHAPE_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 60
GENERATION_ATTEMPTS = 300

BRANCH_AZURE = "azure_majority"
BRANCH_ORANGE = "orange_majority"
BRANCH_TIE = "tie"
BRANCHES = (BRANCH_AZURE, BRANCH_ORANGE, BRANCH_TIE)


def main(grid: np.ndarray) -> np.ndarray:
    """
    On an azure/orange checkerboard, recolor each monochrome stamp to the
    opposite checkerboard color of the majority hidden beneath it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    even_color, odd_color = _infer_checkerboard_colors(g)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color in {even_color, odd_color}:
            continue
        for component in _color_components(g, color):
            even_count = sum((r + c) % 2 == 0 for r, c in component)
            odd_count = len(component) - even_count

            if even_count > odd_count:
                new_color = odd_color
            elif odd_count > even_count:
                new_color = even_color
            else:
                continue

            for r, c in component:
                out[r, c] = new_color
    return out


def _infer_checkerboard_colors(grid: np.ndarray) -> tuple[int, int]:
    parity = (np.indices(grid.shape).sum(axis=0) % 2) == 0
    colors = [int(c) for c in np.unique(grid)]
    best: tuple[int, int] | None = None
    best_support = -1

    for a, b in itertools.combinations(colors, 2):
        support = int(np.count_nonzero((grid == a) | (grid == b)))
        if support <= best_support:
            continue
        if _matches_checkerboard(grid, parity, a, b):
            best = (a, b)
            best_support = support
        if _matches_checkerboard(grid, parity, b, a):
            best = (b, a)
            best_support = support

    assert best is not None, "failed to infer checkerboard colors"
    return best


def _matches_checkerboard(
    grid: np.ndarray,
    parity: np.ndarray,
    even_color: int,
    odd_color: int,
) -> bool:
    even_mask = grid == even_color
    odd_mask = grid == odd_color
    if not even_mask.any() or not odd_mask.any():
        return False
    return bool(np.all(parity[even_mask]) and np.all(~parity[odd_mask]))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    components: list[list[tuple[int, int]]] = []

    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        seen[r0, c0] = True
        stack = [(r0, c0)]
        component: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(component)

    return components


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-stamp puzzle")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    azure_on_even = bool(random.getrandbits(1))
    grid = _build_checkerboard(h, w, azure_on_even)

    stamp_count = random.choice(STAMP_COUNT_OPTIONS)
    branches = [BRANCH_AZURE, BRANCH_ORANGE, BRANCH_TIE]
    while len(branches) < stamp_count:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)

    colors = random.sample(STAMP_COLORS, stamp_count)
    blocked = np.zeros((h, w), dtype=bool)

    for branch, color in zip(branches, colors):
        placement = _place_stamp(grid, blocked, azure_on_even, branch, color)
        if placement is None:
            return None

    if not _valid_generated_grid(grid, branches):
        return None

    return grid


def _build_checkerboard(h: int, w: int, azure_on_even: bool) -> np.ndarray:
    parity = (np.indices((h, w)).sum(axis=0) % 2) == 0
    even_color = AZURE if azure_on_even else ORANGE
    odd_color = ORANGE if azure_on_even else AZURE
    return np.where(parity, even_color, odd_color).astype(int)


def _place_stamp(
    grid: np.ndarray,
    blocked: np.ndarray,
    azure_on_even: bool,
    branch: str,
    color: int,
) -> tuple[list[tuple[int, int]], str] | None:
    h, w = grid.shape

    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _sample_shape(branch)
        if shape is None:
            continue

        sh = 1 + max(r for r, _ in shape)
        sw = 1 + max(c for _, c in shape)
        row_start = EDGE_MARGIN
        col_start = EDGE_MARGIN
        row_stop = h - sh - EDGE_MARGIN + 1
        col_stop = w - sw - EDGE_MARGIN + 1
        if row_stop <= row_start or col_stop <= col_start:
            continue

        anchors = [
            (r0, c0)
            for r0 in range(row_start, row_stop)
            for c0 in range(col_start, col_stop)
        ]
        random.shuffle(anchors)

        candidates: list[list[tuple[int, int]]] = []
        for r0, c0 in anchors:
            coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(blocked[r, c] for r, c in coords):
                continue
            if _branch_for_coords(coords, azure_on_even) != branch:
                continue
            candidates.append(coords)

        if not candidates:
            continue

        coords = random.choice(candidates)
        for r, c in coords:
            grid[r, c] = color
        _mark_blocked(blocked, coords)
        return coords, branch

    return None


def _sample_shape(branch: str) -> tuple[tuple[int, int], ...] | None:
    areas = TIE_AREAS if branch == BRANCH_TIE else IMBALANCED_AREAS

    for _ in range(SHAPE_ATTEMPTS):
        area = random.choice(areas)
        shape = _grow_connected_shape(area)
        shape = _random_transform(shape)
        if _shape_side(shape) > MAX_SHAPE_SIDE:
            continue

        even_count = sum((r + c) % 2 == 0 for r, c in shape)
        odd_count = len(shape) - even_count
        if branch == BRANCH_TIE and even_count == odd_count:
            return shape
        if branch != BRANCH_TIE and even_count != odd_count:
            return shape

    return None


def _grow_connected_shape(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        seen = set()

        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in seen:
                    continue
                seen.add(cand)
                degree = sum((cand[0] + ddr, cand[1] + ddc) in cells for ddr, ddc in DIR4)
                frontier.append(cand)
                weights.append((degree + 1) ** 2)

        cells.add(random.choices(frontier, weights=weights, k=1)[0])

    return _normalize_shape(cells)


def _random_transform(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    for _ in range(random.randrange(4)):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    if random.random() < 0.5:
        pts[:, 0] *= -1
    if random.random() < 0.5:
        pts[:, 1] *= -1
    return _normalize_shape(map(tuple, pts.tolist()))


def _normalize_shape(cells) -> tuple[tuple[int, int], ...]:
    cells = [(int(r), int(c)) for r, c in cells]
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_side(shape: tuple[tuple[int, int], ...]) -> int:
    height = 1 + max(r for r, _ in shape)
    width = 1 + max(c for _, c in shape)
    return max(height, width)


def _branch_for_coords(coords: list[tuple[int, int]], azure_on_even: bool) -> str:
    azure_count = 0
    for r, c in coords:
        if (((r + c) % 2) == 0) == azure_on_even:
            azure_count += 1
    orange_count = len(coords) - azure_count
    if azure_count > orange_count:
        return BRANCH_AZURE
    if orange_count > azure_count:
        return BRANCH_ORANGE
    return BRANCH_TIE


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _valid_generated_grid(grid: np.ndarray, expected_branches: list[str]) -> bool:
    try:
        even_color, odd_color = _infer_checkerboard_colors(grid)
    except AssertionError:
        return False

    if {even_color, odd_color} != {AZURE, ORANGE}:
        return False

    seen_branches: list[str] = []
    component_count = 0
    for color in map(int, np.unique(grid)):
        if color in {even_color, odd_color}:
            continue
        for component in _color_components(grid, color):
            component_count += 1
            seen_branches.append(_branch_for_coords(component, even_color == AZURE))

    if component_count != len(expected_branches):
        return False
    if not all(branch in seen_branches for branch in BRANCHES):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return out.shape == grid.shape
