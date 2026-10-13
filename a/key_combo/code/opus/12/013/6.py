from __future__ import annotations

import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

PANEL_SIZE = 9
GRID_SHAPE = (9, 19)

RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(r: int, c: int, shape: tuple[int, int]) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
    assert np.all(g[:, PANEL_SIZE] == GREY), "expected a grey separator column"
    left = g[:, :PANEL_SIZE].copy()
    right = g[:, PANEL_SIZE + 1 :].copy()
    assert left.shape == (PANEL_SIZE, PANEL_SIZE)
    assert right.shape == (PANEL_SIZE, PANEL_SIZE)
    return left, right


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(nr, nc, mask.shape) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _domino_pairs(left: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.isin(left, (RED, GREEN))
    pairs: list[list[tuple[int, int]]] = []
    for comp in _mask_components(mask):
        if len(comp) != 2:
            continue
        colors = {int(left[r, c]) for r, c in comp}
        if colors == {RED, GREEN}:
            pairs.append(comp)
    return pairs


def _orange_objects(right: np.ndarray) -> list[list[tuple[int, int]]]:
    return [comp for comp in _mask_components(right == ORANGE) if comp]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the right-panel orange object whose area matches the count of left-panel red-green dominoes, then erase the matched dominoes.

    Input structure:
    - The input is a 9x19 grid split into two 9x9 yellow panels by one grey separator column.
    - The left panel contains blue decoration plus red/green pixels; complete dominoes are exactly 4-connected two-cell red-green components, while isolated red or green pixels are distractors.
    - The right panel contains 3 or 4 disjoint 4-connected orange objects on the same yellow background, and their areas are all different.

    Rule:
    - Count the complete red-green domino pairs in the left panel to obtain K.
    - In the right panel, find the unique orange object whose area is exactly K.
    - Overlay that object's panel-corresponding cells onto the left panel, but only where the left panel is yellow; blue and red/green cells are protected and are not overwritten.
    - Then recolor every complete domino pair: each red cell becomes yellow and each green cell becomes teal; isolated red/green distractors stay unchanged.

    Key insight:
    - The left-panel domino count is not the output itself; it selects which right-panel object should be transferred back to the left before the matched dominoes are erased.
    """
    left, right = _split_panels(grid)

    pairs = _domino_pairs(left)
    k = len(pairs)
    assert k > 0, "expected at least one complete domino pair"

    orange_objects = _orange_objects(right)
    areas = [len(obj) for obj in orange_objects]
    selected = [obj for obj in orange_objects if len(obj) == k]
    assert len(selected) == 1, f"expected one area-{k} orange object, got {areas}"
    chosen = selected[0]

    out = left.copy()
    for r, c in chosen:
        if int(out[r, c]) == YELLOW:
            out[r, c] = ORANGE

    for comp in pairs:
        for r, c in comp:
            out[r, c] = TEAL if int(left[r, c]) == GREEN else YELLOW

    return out


def _random_connected_shape(area: int) -> np.ndarray:
    dims = []
    for h in range(1, 5):
        for w in range(1, 5):
            if area <= h * w and h * w <= area + 3:
                dims.append((h, w))
    for _ in range(200):
        h, w = random.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = random.choice(frontier)
            cells.add(pick)
            mask[pick] = True
        if len(cells) != area:
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        mask = mask[rows][:, cols]
        if int(mask.sum()) == area:
            return mask
    raise RuntimeError(f"failed to build a connected shape of area {area}")


def _place_shape(
    mask: np.ndarray,
    forbidden: set[tuple[int, int]] | None = None,
    occupied: set[tuple[int, int]] | None = None,
    require_gap8: bool = False,
) -> set[tuple[int, int]] | None:
    forbidden = set() if forbidden is None else set(forbidden)
    occupied = set() if occupied is None else set(occupied)
    h, w = mask.shape
    placements = []
    for r0 in range(PANEL_SIZE - h + 1):
        for c0 in range(PANEL_SIZE - w + 1):
            coords = {
                (r0 + r, c0 + c)
                for r, c in zip(*np.where(mask))
            }
            if coords & forbidden:
                continue
            if coords & occupied:
                continue
            if require_gap8 and any(
                (nr, nc) in occupied
                for r, c in coords
                for nr, nc in _neighbors8(r, c)
            ):
                continue
            placements.append(coords)
    if not placements:
        return None
    return set(random.choice(placements))


def _single_rg_ok(cell: tuple[int, int], rg_cells: set[tuple[int, int]]) -> bool:
    if cell in rg_cells:
        return False
    for nr, nc in _neighbors4(*cell):
        if (nr, nc) in rg_cells:
            return False
    return True


def _pair_cells_ok(a: tuple[int, int], b: tuple[int, int], rg_cells: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
    if a in forbidden or b in forbidden or a in rg_cells or b in rg_cells:
        return False
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
        return False
    for cell in (a, b):
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) in rg_cells and (nr, nc) not in {a, b}:
                return False
    return True


def _grow_connected_subset(allowed: set[tuple[int, int]], min_area: int, max_area: int) -> set[tuple[int, int]] | None:
    allowed = set(allowed)
    if len(allowed) < min_area:
        return None
    target = random.randint(min_area, min(max_area, len(allowed)))
    starts = list(allowed)
    random.shuffle(starts)
    for start in starts[:40]:
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == target:
            return cells
    return None


def _render_left_panel(
    selected: set[tuple[int, int]],
    pair_specs: list[tuple[tuple[int, int], tuple[int, int], bool]],
    blue_cells: set[tuple[int, int]],
    noise_specs: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    left = np.full((PANEL_SIZE, PANEL_SIZE), YELLOW, dtype=int)
    for r, c in blue_cells:
        left[r, c] = BLUE
    for a, b, red_first in pair_specs:
        if red_first:
            left[a] = RED
            left[b] = GREEN
        else:
            left[a] = GREEN
            left[b] = RED
    for cell, color in noise_specs:
        left[cell] = color
    return left


def _render_right_panel(objects: list[set[tuple[int, int]]]) -> np.ndarray:
    right = np.full((PANEL_SIZE, PANEL_SIZE), YELLOW, dtype=int)
    for obj in objects:
        for r, c in obj:
            right[r, c] = ORANGE
    return right


def _assemble_grid(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    grid = np.full(GRID_SHAPE, YELLOW, dtype=int)
    grid[:, PANEL_SIZE] = GREY
    grid[:, :PANEL_SIZE] = left
    grid[:, PANEL_SIZE + 1 :] = right
    return grid


def _validate_sample(grid: np.ndarray, expect_noise_overlap: bool) -> None:
    left, right = _split_panels(grid)
    pairs = _domino_pairs(left)
    k = len(pairs)
    assert 2 <= k <= 5

    orange_objects = _orange_objects(right)
    assert len(orange_objects) in {3, 4}
    areas = [len(obj) for obj in orange_objects]
    assert len(set(areas)) == len(areas)
    assert areas.count(k) == 1

    selected = set(next(obj for obj in orange_objects if len(obj) == k))
    pair_cells = {cell for comp in pairs for cell in comp}
    blue_cells = set(map(tuple, np.argwhere(left == BLUE)))
    rg_cells = set(map(tuple, np.argwhere(np.isin(left, (RED, GREEN)))))
    noise_overlap = bool(selected & (rg_cells - pair_cells))

    assert not (selected & pair_cells), "selected object must not cover matched dominoes"
    assert not (selected & blue_cells), "selected object must not cover blue pixels"
    assert any(left[r, c] == YELLOW for r, c in selected), "selected object must paint at least one yellow cell"
    assert noise_overlap == expect_noise_overlap

    out = main(grid)
    assert out.shape == (PANEL_SIZE, PANEL_SIZE)
    for comp in pairs:
        colors = {int(left[r, c]) for r, c in comp}
        assert colors == {RED, GREEN}
        for r, c in comp:
            assert out[r, c] in {YELLOW, TEAL}
    for r, c in selected:
        if int(left[r, c]) == YELLOW:
            assert int(out[r, c]) == ORANGE
        else:
            assert int(out[r, c]) != ORANGE


def _attempt_generate() -> tuple[np.ndarray, bool] | None:
    num_objects = random.choice([3, 4])
    k = random.randint(2, 5)
    other_areas = random.sample([area for area in range(2, 7) if area != k], num_objects - 1)
    selected_mask = _random_connected_shape(k)
    selected = _place_shape(selected_mask)
    if selected is None:
        return None

    expect_noise_overlap = random.random() < (0.55 if k >= 3 else 0.35)
    total_noise = random.randint(1, 4)
    if expect_noise_overlap:
        inside_count = 1 if k <= 2 else random.randint(1, min(2, total_noise, k - 1))
    else:
        inside_count = 0
    total_noise = max(total_noise, inside_count)

    rg_cells: set[tuple[int, int]] = set()
    noise_specs: list[tuple[tuple[int, int], int]] = []

    if inside_count:
        inside_candidates = [cell for cell in selected if _single_rg_ok(cell, rg_cells)]
        random.shuffle(inside_candidates)
        if len(inside_candidates) < inside_count:
            return None
        for cell in inside_candidates[:inside_count]:
            color = random.choice([RED, GREEN])
            noise_specs.append((cell, color))
            rg_cells.add(cell)

    outside_candidates = [
        (r, c)
        for r in range(PANEL_SIZE)
        for c in range(PANEL_SIZE)
        if (r, c) not in selected
    ]
    random.shuffle(outside_candidates)
    while len(noise_specs) < total_noise:
        picked = None
        for cell in outside_candidates:
            if _single_rg_ok(cell, rg_cells):
                picked = cell
                break
        if picked is None:
            return None
        outside_candidates.remove(picked)
        noise_specs.append((picked, random.choice([RED, GREEN])))
        rg_cells.add(picked)

    forbidden_pairs = set(selected)
    pair_specs: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
    pair_options = []
    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if _in_bounds(nr, nc, (PANEL_SIZE, PANEL_SIZE)):
                    pair_options.append(((r, c), (nr, nc)))
    random.shuffle(pair_options)
    while len(pair_specs) < k:
        placed = False
        for a, b in pair_options:
            if _pair_cells_ok(a, b, rg_cells, forbidden_pairs):
                pair_specs.append((a, b, bool(random.getrandbits(1))))
                rg_cells.add(a)
                rg_cells.add(b)
                placed = True
                break
        if not placed:
            return None
        pair_options = [option for option in pair_options if option[0] not in rg_cells and option[1] not in rg_cells]

    blue_allowed = {
        (r, c)
        for r in range(PANEL_SIZE)
        for c in range(PANEL_SIZE)
        if (r, c) not in selected and (r, c) not in rg_cells
    }
    blue_cells = _grow_connected_subset(blue_allowed, 4, 8)
    if blue_cells is None:
        return None

    occupied_right = set(selected)
    right_objects = [set(selected)]
    for area in other_areas:
        placed = None
        for _ in range(60):
            mask = _random_connected_shape(area)
            coords = _place_shape(mask, occupied=occupied_right, require_gap8=True)
            if coords is None:
                continue
            placed = coords
            break
        if placed is None:
            return None
        right_objects.append(placed)
        occupied_right |= placed

    left = _render_left_panel(selected, pair_specs, blue_cells, noise_specs)
    right = _render_right_panel(right_objects)
    grid = _assemble_grid(left, right)
    return grid, expect_noise_overlap


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _attempt_generate()
        if built is None:
            continue
        grid, expect_noise_overlap = built
        try:
            _validate_sample(grid, expect_noise_overlap)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
