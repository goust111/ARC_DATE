from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
DIRS8: tuple[tuple[int, int], ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_seed_template(g: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "grid must contain the 3x3 seed template"
    seed_color = int(g[1, 1])
    assert seed_color != BACKGROUND, "template center must be the seed color"

    guide_colors = {
        int(g[1 + dr, 1 + dc])
        for dr, dc in DIRS8
        if int(g[1 + dr, 1 + dc]) != BACKGROUND
    }
    assert len(guide_colors) == 1, "template must use one growth-direction color"
    guide_color = next(iter(guide_colors))
    assert guide_color != seed_color, "template guide color must differ from the seed color"

    active = [
        (dr, dc)
        for dr, dc in DIRS8
        if int(g[1 + dr, 1 + dc]) == guide_color
    ]
    assert active, "template must encode at least one active growth direction"
    return seed_color, active


def _parse_palette(g: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = g.shape
    palette: list[int] = []
    for r in range(h):
        color = int(g[r, w - 1])
        if color == BACKGROUND:
            break
        palette.append(color)

    assert 2 <= len(palette) <= 5, "top-right palette must contain 2 to 5 colors"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette, {(r, w - 1) for r in range(len(palette))}


def _protected_cells(shape: tuple[int, int], palette_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    protected = {
        (r, c)
        for r in range(min(3, h))
        for c in range(min(3, w))
    }
    protected.update(palette_cells)
    return protected


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        comp: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow isolated seed pixels by the corner template and recolor each grown cluster by its size.

    Input structure:
    - The background is black.
    - The upper-left 3x3 block is a seed template: its center has color A, and its non-black neighbor cells have one guide color B.
    - A vertical palette in the upper-right corner lists colors for final cluster sizes 2, 3, 4, and onward.
    - Away from these corner markers, isolated pixels of color A may be near the grid edge or near non-black obstacle pixels.

    Rule:
    - Read the guide-color offsets around the upper-left center as the active one-step growth directions.
    - For every non-template pixel of color A, add pixels in those directions only where the target is in bounds, black, and not part of the corner markers.
    - Find the 8-connected components of all original central seeds plus their successful grown neighbors.
    - Recolor a component of size N with the palette color at position N - 1 from the top, while leaving the corner template, palette, and obstacles unchanged.

    Key insight:
    - The small 3x3 corner pattern is a direction stencil, boundary or obstacle clipping changes cluster size, and the right-corner palette maps that size to the final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seed_color, active_dirs = _parse_seed_template(g)
    palette, palette_cells = _parse_palette(g)
    protected = _protected_cells(g.shape, palette_cells)

    central_seeds = [
        (int(r), int(c))
        for r, c in np.argwhere(g == seed_color)
        if (int(r), int(c)) not in protected
    ]
    assert central_seeds, "expected at least one central seed pixel"

    grown = np.zeros(g.shape, dtype=bool)
    for r, c in central_seeds:
        grown[r, c] = True

    for r, c in central_seeds:
        for dr, dc in active_dirs:
            nr, nc = r + dr, c + dc
            if (
                _in_bounds(g.shape, nr, nc)
                and (nr, nc) not in protected
                and int(g[nr, nc]) == BACKGROUND
            ):
                grown[nr, nc] = True

    out = g.copy()
    for comp in _components8(grown):
        size = len(comp)
        palette_index = size - 2
        assert 0 <= palette_index < len(palette), "cluster size must be represented by the palette"
        color = int(palette[palette_index])
        for r, c in comp:
            out[r, c] = color

    for r, c in protected:
        out[r, c] = g[r, c]
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 26)
GRID_W_RANGE = (13, 26)
ACTIVE_COUNT_RANGE = (2, 5)
GENERATION_ATTEMPTS = 500
EXTRA_SEED_RANGE = (0, 2)
COLOR_POOL = tuple(range(1, 10))


def _gen_protected(shape: tuple[int, int], palette_len: int) -> set[tuple[int, int]]:
    h, w = shape
    protected = {
        (r, c)
        for r in range(min(3, h))
        for c in range(min(3, w))
    }
    protected.update((r, w - 1) for r in range(palette_len))
    if palette_len < h:
        protected.add((palette_len, w - 1))
    return protected


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                expanded.add((rr, cc))
    return expanded


def _shuffled_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    return cells


def _cells_available(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if (r, c) in protected or (r, c) in reserved or (r, c) in forbidden:
            return False
        if int(grid[r, c]) != BACKGROUND:
            return False
    return True


def _target_cells(center: tuple[int, int], active: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in active]


def _edge_options(
    grid: np.ndarray,
    active: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], int, list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = grid.shape
    options = []
    for center in _shuffled_cells((h, w)):
        r, c = center
        if not (r == 0 or c == 0 or r == h - 1 or c == w - 1):
            continue
        targets = _target_cells(center, active)
        in_bounds_targets = [cell for cell in targets if _in_bounds((h, w), cell[0], cell[1])]
        clipped = len(in_bounds_targets) < len(active)
        success_count = len(in_bounds_targets)
        if not clipped or not (1 <= success_count < len(active)):
            continue
        cells = [center] + in_bounds_targets
        if _cells_available(grid, cells, protected, reserved, forbidden):
            options.append((center, success_count, in_bounds_targets, in_bounds_targets))
    random.shuffle(options)
    return options


def _find_full_location(
    grid: np.ndarray,
    active: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    for center in _shuffled_cells(grid.shape):
        targets = _target_cells(center, active)
        if not all(_in_bounds(grid.shape, r, c) for r, c in targets):
            continue
        if _cells_available(grid, [center] + targets, protected, reserved, forbidden):
            return center, targets, targets
    return None


def _find_blocked_location(
    grid: np.ndarray,
    active: list[tuple[int, int]],
    success_count: int,
    protected: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    assert 1 <= success_count <= len(active)
    for center in _shuffled_cells(grid.shape):
        targets = _target_cells(center, active)
        if not all(_in_bounds(grid.shape, r, c) for r, c in targets):
            continue
        if not _cells_available(grid, [center] + targets, protected, reserved, forbidden):
            continue
        shuffled = targets[:]
        random.shuffle(shuffled)
        successes = shuffled[:success_count]
        blockers = shuffled[success_count:]
        return center, targets, successes, blockers
    return None


def _place_seed(
    grid: np.ndarray,
    seed_color: int,
    obstacle_color: int,
    center: tuple[int, int],
    all_targets: list[tuple[int, int]],
    successes: list[tuple[int, int]],
    blockers: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> None:
    r, c = center
    grid[r, c] = seed_color
    for rr, cc in blockers:
        grid[rr, cc] = obstacle_color

    reserved.add(center)
    reserved.update(all_targets)

    cluster_cells = {center, *successes}
    forbidden.update(_expanded(cluster_cells, grid.shape, radius=1))


def _draw_markers(
    grid: np.ndarray,
    seed_color: int,
    guide_color: int,
    active: list[tuple[int, int]],
    palette: list[int],
) -> None:
    grid[1, 1] = seed_color
    for dr, dc in active:
        grid[1 + dr, 1 + dc] = guide_color
    for r, color in enumerate(palette):
        grid[r, grid.shape[1] - 1] = int(color)


def _try_place_extra(
    grid: np.ndarray,
    seed_color: int,
    obstacle_color: int,
    active: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    success_count = random.randint(1, len(active))
    use_edge = success_count < len(active) and random.random() < 0.35
    if use_edge:
        options = [
            option
            for option in _edge_options(grid, active, protected, reserved, forbidden)
            if option[1] == success_count
        ]
        if options:
            center, _count, all_targets, successes = random.choice(options)
            _place_seed(
                grid,
                seed_color,
                obstacle_color,
                center,
                all_targets,
                successes,
                [],
                reserved,
                forbidden,
            )
            return True

    if success_count == len(active) and random.random() < 0.5:
        full = _find_full_location(grid, active, protected, reserved, forbidden)
        if full is None:
            return False
        center, all_targets, successes = full
        blockers: list[tuple[int, int]] = []
    else:
        blocked = _find_blocked_location(
            grid, active, success_count, protected, reserved, forbidden
        )
        if blocked is None:
            return False
        center, all_targets, successes, blockers = blocked

    _place_seed(
        grid,
        seed_color,
        obstacle_color,
        center,
        all_targets,
        successes,
        blockers,
        reserved,
        forbidden,
    )
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    active_count = random.randint(*ACTIVE_COUNT_RANGE)
    active = random.sample(list(DIRS8), active_count)

    colors = random.sample(COLOR_POOL, active_count + 3)
    seed_color, guide_color, obstacle_color = colors[:3]
    palette = colors[3:]

    grid = np.zeros((h, w), dtype=int)
    _draw_markers(grid, seed_color, guide_color, active, palette)

    protected = _gen_protected(grid.shape, len(palette))
    reserved = set(protected)
    forbidden = _expanded(protected, grid.shape, radius=1)
    covered_counts: set[int] = set()

    edge_candidates = _edge_options(grid, active, protected, reserved, forbidden)
    if not edge_candidates:
        return None
    edge_center, edge_count, edge_targets, edge_successes = random.choice(edge_candidates)
    _place_seed(
        grid,
        seed_color,
        obstacle_color,
        edge_center,
        edge_targets,
        edge_successes,
        [],
        reserved,
        forbidden,
    )
    covered_counts.add(edge_count)

    if active_count not in covered_counts:
        full = _find_full_location(grid, active, protected, reserved, forbidden)
        if full is None:
            return None
        center, all_targets, successes = full
        _place_seed(
            grid,
            seed_color,
            obstacle_color,
            center,
            all_targets,
            successes,
            [],
            reserved,
            forbidden,
        )
        covered_counts.add(active_count)

    remaining = [count for count in range(1, active_count + 1) if count not in covered_counts]
    random.shuffle(remaining)
    for count in remaining:
        blocked = _find_blocked_location(grid, active, count, protected, reserved, forbidden)
        if blocked is None:
            return None
        center, all_targets, successes, blockers = blocked
        _place_seed(
            grid,
            seed_color,
            obstacle_color,
            center,
            all_targets,
            successes,
            blockers,
            reserved,
            forbidden,
        )
        covered_counts.add(count)

    if not np.any(grid == obstacle_color):
        blocked = _find_blocked_location(
            grid,
            active,
            random.randint(1, active_count - 1),
            protected,
            reserved,
            forbidden,
        )
        if blocked is None:
            return None
        center, all_targets, successes, blockers = blocked
        _place_seed(
            grid,
            seed_color,
            obstacle_color,
            center,
            all_targets,
            successes,
            blockers,
            reserved,
            forbidden,
        )

    for _ in range(random.randint(*EXTRA_SEED_RANGE)):
        if not _try_place_extra(
            grid,
            seed_color,
            obstacle_color,
            active,
            protected,
            reserved,
            forbidden,
        ):
            break

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    for color in palette:
        if int(np.sum(out == int(color))) <= int(np.sum(grid == int(color))):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
