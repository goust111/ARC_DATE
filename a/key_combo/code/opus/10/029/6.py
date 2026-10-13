from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PANEL = 6
HEIGHT = 6
TOTAL_WIDTH = 20
LEFT_SLICE = slice(0, 6)
MID_SLICE = slice(7, 13)
RIGHT_SLICE = slice(14, 20)
SEP_COLS = (6, 13)

GEN_MAX_ATTEMPTS = 500
GEN_PANEL_ATTEMPTS = 200
GEN_RIGHT_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < HEIGHT and 0 <= c < PANEL


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (HEIGHT, TOTAL_WIDTH), "expected a 6x20 grid"
    for col in SEP_COLS:
        assert np.all(g[:, col] == GREY), "expected grey separator columns"
    return g[:, LEFT_SLICE], g[:, MID_SLICE], g[:, RIGHT_SLICE]


def main(grid: np.ndarray) -> np.ndarray:
    """Use the occupancy complement of the left and middle panels as seed coordinates for a flood fill in the right panel.

    Input structure:
    - The grid is 6x20 and split into three 6x6 panels by two grey separator columns.
    - The left panel contains blue foreground pixels on black, the middle panel contains red foreground pixels on black.
    - The right panel contains black walkable cells and yellow blocker cells.

    Rule:
    - Align the left and middle panels cell-by-cell and mark a coordinate as occupied if the left cell is blue or the middle cell is red.
    - Every coordinate that is black in both source panels is in the complement set; project those coordinates onto the right panel.
    - If a projected complement coordinate lands on a black right-panel cell, make it a green seed, then flood-fill through 4-neighbor black cells while yellow cells block the spread.
    - Output only the transformed right 6x6 panel.

    Key insight:
    - The first two panels do not directly produce the answer; they only determine where seeds may appear, and the right panel's own obstacle topology decides the final filled region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    left, middle, right = _extract_panels(g)

    occupied = (left == BLUE) | (middle == RED)
    out = right.copy()
    q: deque[tuple[int, int]] = deque()
    seen = np.zeros((HEIGHT, PANEL), dtype=bool)

    for r, c in np.argwhere((~occupied) & (right == BLACK)):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = GREEN
        seen[rr, cc] = True
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(nr, nc) or seen[nr, nc] or out[nr, nc] != BLACK:
                continue
            seen[nr, nc] = True
            out[nr, nc] = GREEN
            q.append((nr, nc))

    return out


def _random_connected_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    cells = {(int(rng.integers(HEIGHT)), int(rng.integers(PANEL)))}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        frontier_seen = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(nr, nc) and (nr, nc) not in cells and (nr, nc) not in frontier_seen:
                    frontier.append((nr, nc))
                    frontier_seen.add((nr, nc))
        assert frontier, "frontier should exist while the component is smaller than the board"
        if rng.random() < 0.5:
            scores = [sum((rr, cc) in cells for rr, cc in _neighbors4(r, c)) for r, c in frontier]
            keep = max(scores)
            frontier = [cell for cell, score in zip(frontier, scores) if score == keep]
        cells.add(frontier[int(rng.integers(len(frontier)))])

    mask = np.zeros((HEIGHT, PANEL), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_source_masks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    want_overlap = bool(rng.integers(0, 2))
    for _ in range(GEN_PANEL_ATTEMPTS):
        left = _random_connected_mask(rng, int(rng.integers(3, 9)))
        middle = _random_connected_mask(rng, int(rng.integers(3, 9)))
        if np.array_equal(left, middle):
            continue
        overlap = bool(np.any(left & middle))
        if overlap != want_overlap:
            continue
        if not np.any(left & ~middle):
            continue
        if not np.any(middle & ~left):
            continue
        occupied = left | middle
        if int((~occupied).sum()) < 4:
            continue
        return left, middle
    return None


def _random_manhattan_path(
    rng: np.random.Generator, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    path = [(r, c)]
    while (r, c) != (gr, gc):
        options = []
        if r < gr:
            options.append((r + 1, c))
        elif r > gr:
            options.append((r - 1, c))
        if c < gc:
            options.append((r, c + 1))
        elif c > gc:
            options.append((r, c - 1))
        r, c = options[int(rng.integers(len(options)))]
        path.append((r, c))
    return path


def _grow_connected_region(
    rng: np.random.Generator,
    initial: set[tuple[int, int]],
    target_size: int,
    allowed: np.ndarray,
    prefer: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    region = set(initial)
    while len(region) < target_size:
        frontier: list[tuple[int, int]] = []
        frontier_seen = set()
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(nr, nc) and allowed[nr, nc] and (nr, nc) not in region and (nr, nc) not in frontier_seen:
                    frontier.append((nr, nc))
                    frontier_seen.add((nr, nc))
        if not frontier:
            break
        if prefer is not None and rng.random() < 0.75:
            preferred = [cell for cell in frontier if prefer[cell]]
            if preferred:
                frontier = preferred
        region.add(frontier[int(rng.integers(len(frontier)))])
    return region


def _safe_occ_mask(occupied: np.ndarray, main_region: set[tuple[int, int]]) -> np.ndarray:
    safe = occupied.copy()
    for r, c in main_region:
        safe[r, c] = False
    for r, c in main_region:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(nr, nc):
                safe[nr, nc] = False
    return safe


def _build_right_panel(rng: np.random.Generator, occupied: np.ndarray) -> np.ndarray | None:
    complement = ~occupied
    comp_coords = [tuple(map(int, rc)) for rc in np.argwhere(complement)]
    occ_coords = [tuple(map(int, rc)) for rc in np.argwhere(occupied)]
    want_pocket = rng.random() < 0.65

    if not comp_coords or not occ_coords:
        return None

    for _ in range(GEN_RIGHT_ATTEMPTS):
        seed_anchor = comp_coords[int(rng.integers(len(comp_coords)))]
        fill_anchor = occ_coords[int(rng.integers(len(occ_coords)))]
        main_region = set(_random_manhattan_path(rng, seed_anchor, fill_anchor))
        target_main = int(rng.integers(max(len(main_region) + 1, 5), 13))
        main_region = _grow_connected_region(
            rng,
            main_region,
            target_main,
            allowed=np.ones((HEIGHT, PANEL), dtype=bool),
            prefer=occupied,
        )

        seed_cells = [cell for cell in main_region if complement[cell]]
        fill_cells = [cell for cell in main_region if occupied[cell]]
        skipped_complements = [cell for cell in comp_coords if cell not in main_region]
        if not seed_cells or not fill_cells or not skipped_complements:
            continue

        pocket: set[tuple[int, int]] = set()
        if want_pocket:
            safe = _safe_occ_mask(occupied, main_region)
            safe_components = _components(safe)
            if not safe_components:
                continue
            comp = safe_components[int(rng.integers(len(safe_components)))]
            allowed = np.zeros((HEIGHT, PANEL), dtype=bool)
            for r, c in comp:
                allowed[r, c] = True
            pocket_start = comp[int(rng.integers(len(comp)))]
            pocket_target = int(rng.integers(1, min(4, len(comp)) + 1))
            pocket = _grow_connected_region(rng, {pocket_start}, pocket_target, allowed=allowed)

        right = np.full((HEIGHT, PANEL), YELLOW, dtype=int)
        for r, c in main_region | pocket:
            right[r, c] = BLACK

        if int(np.sum(right == YELLOW)) < 18:
            continue
        return right

    return None


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    left, middle, right = _extract_panels(grid)
    occupied = (left == BLUE) | (middle == RED)
    out = right.copy()
    for comp in _components(right == BLACK):
        if any(not occupied[r, c] for r, c in comp):
            for r, c in comp:
                out[r, c] = GREEN
    return out


def _is_connected_shape(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and 3 <= len(comps[0]) <= 8


def _validate_example(grid: np.ndarray) -> bool:
    try:
        left, middle, right = _extract_panels(grid)
    except AssertionError:
        return False

    if not np.all(np.isin(grid, [BLACK, BLUE, RED, YELLOW, GREY])):
        return False
    if not np.all(np.isin(left, [BLACK, BLUE])):
        return False
    if not np.all(np.isin(middle, [BLACK, RED])):
        return False
    if not np.all(np.isin(right, [BLACK, YELLOW])):
        return False

    left_mask = left == BLUE
    middle_mask = middle == RED
    if not _is_connected_shape(left_mask) or not _is_connected_shape(middle_mask):
        return False
    if np.array_equal(left_mask, middle_mask):
        return False

    occupied = left_mask | middle_mask
    complement = ~occupied
    if int(complement.sum()) < 4:
        return False

    seed_mask = complement & (right == BLACK)
    if int(seed_mask.sum()) < 1:
        return False
    if int(np.sum(complement & (right == YELLOW))) < 1:
        return False

    expected = _oracle_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    green_count = int(np.sum(actual == GREEN))
    seed_count = int(np.sum(seed_mask))
    if green_count <= seed_count:
        return False

    black_components = _components(right == BLACK)
    if not black_components:
        return False
    seeded_components = 0
    for comp in black_components:
        has_seed = any(complement[r, c] for r, c in comp)
        if has_seed:
            seeded_components += 1
            if not any(occupied[r, c] for r, c in comp):
                return False
    if seeded_components < 1:
        return False

    return True


def _assemble_grid(left_mask: np.ndarray, middle_mask: np.ndarray, right: np.ndarray) -> np.ndarray:
    grid = np.zeros((HEIGHT, TOTAL_WIDTH), dtype=int)
    grid[:, SEP_COLS[0]] = GREY
    grid[:, SEP_COLS[1]] = GREY
    left_panel = np.zeros((HEIGHT, PANEL), dtype=int)
    middle_panel = np.zeros((HEIGHT, PANEL), dtype=int)
    left_panel[left_mask] = BLUE
    middle_panel[middle_mask] = RED
    grid[:, LEFT_SLICE] = left_panel
    grid[:, MID_SLICE] = middle_panel
    grid[:, RIGHT_SLICE] = right
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = np.random.default_rng()
        masks = _generate_source_masks(rng)
        if masks is None:
            continue
        left_mask, middle_mask = masks
        right = _build_right_panel(rng, left_mask | middle_mask)
        if right is None:
            continue
        grid = _assemble_grid(left_mask, middle_mask, right)
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
