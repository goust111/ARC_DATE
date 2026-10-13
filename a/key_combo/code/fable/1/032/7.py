from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, in_bounds, neighbors4


_STEP2 = ((-2, 0), (2, 0), (0, -2), (0, 2))
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MARKER_COLORS = tuple(c for c in Color.NOT_BLACK if c != Color.GREY)

_LATTICE_SIZE_MIN = 4
_LATTICE_SIZE_MAX = 9
_SPLIT_PROB = 0.82
_MAX_ATTEMPTS = 400


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    marker_colors = colors_present(grid, exclude={Color.BLACK, Color.GREY})
    assert len(marker_colors) == 1, "expected exactly one non-black non-grey marker color"
    marker_color = int(marker_colors[0])
    marker_coords = np.argwhere(grid == marker_color)
    assert marker_coords.shape[0] == 1, "expected exactly one marker pixel"
    marker = tuple(map(int, marker_coords[0]))
    return marker, marker_color


def _reachable_black_mask(grid: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    visited = np.zeros_like(grid, dtype=bool)
    queue: deque[tuple[int, int]] = deque([marker])
    visited[marker] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEP2:
            nr, nc = r + dr, c + dc
            if not in_bounds(grid.shape, nr, nc):
                continue
            if visited[nr, nc]:
                continue
            if int(grid[r + dr // 2, c + dc // 2]) == Color.GREY:
                continue
            if int(grid[nr, nc]) != Color.BLACK:
                continue
            visited[nr, nc] = True
            queue.append((nr, nc))

    visited[marker] = False
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the marker-parity lattice cells that are reachable through unblocked two-step hops.

    Input structure:
    - A black background contains grey fixed pixels and exactly one non-grey non-black marker pixel.
    - The marker's row parity and column parity define one sparse parity lattice of cells two steps apart.
    - Grey pixels may sit between same-lattice cells and act as wall midpoints for those two-step hops.

    Rule:
    - Start from the unique marker pixel.
    - Move only to cells with the same row parity and column parity, using 4-neighbor steps on the half-resolution lattice.
    - A hop from one lattice cell to the next is blocked iff the midpoint cell between them is grey.
    - Recolor every reachable black lattice cell to the marker color, and leave grey pixels, the marker, and unreachable cells unchanged.

    Key insight:
    - This is not a full parity-lattice copy: the painted cells are exactly the marker's connected component in the half-resolution lattice after midpoint-grey walls delete specific hops.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker, marker_color = _find_marker(g)
    reachable = _reachable_black_mask(g, marker)

    out = g.copy()
    out[reachable] = marker_color
    return out


def _grid_shape_from_lattice(lh: int, lw: int, pr: int, pc: int) -> tuple[int, int]:
    extra_r = random.randint(0, 1)
    extra_c = random.randint(0, 1)
    h = 2 * int(lh) - 1 + int(pr) + extra_r
    w = 2 * int(lw) - 1 + int(pc) + extra_c
    assert h <= 30 and w <= 30
    return h, w


def _lattice_to_grid(pr: int, pc: int, lr: int, lc: int) -> tuple[int, int]:
    return int(pr + 2 * lr), int(pc + 2 * lc)


def _grow_component(lh: int, lw: int, area: int) -> np.ndarray:
    mask = np.zeros((lh, lw), dtype=bool)
    start = (random.randrange(lh), random.randrange(lw))
    mask[start] = True
    frontier = {nb for nb in neighbors4(*start) if 0 <= nb[0] < lh and 0 <= nb[1] < lw}
    center_r = (lh - 1) / 2.0
    center_c = (lw - 1) / 2.0

    while int(mask.sum()) < int(area):
        assert frontier, "connected growth frontier unexpectedly empty"
        choices = sorted(frontier)
        weights = []
        for r, c in choices:
            same_neighbors = sum(
                1
                for nr, nc in neighbors4(r, c)
                if 0 <= nr < lh and 0 <= nc < lw and mask[nr, nc]
            )
            edge_penalty = int(r in (0, lh - 1) or c in (0, lw - 1))
            center_bias = 1.0 / (1.0 + abs(r - center_r) + abs(c - center_c))
            weights.append(1.0 + 2.0 * same_neighbors + 0.6 * center_bias - 0.25 * edge_penalty)
        pick_r, pick_c = random.choices(choices, weights=weights, k=1)[0]
        mask[pick_r, pick_c] = True
        frontier.discard((pick_r, pick_c))
        for nr, nc in neighbors4(pick_r, pick_c):
            if 0 <= nr < lh and 0 <= nc < lw and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _choose_marker_cell(reach_mask: np.ndarray) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(reach_mask)]
    scored: list[tuple[int, tuple[int, int]]] = []
    for r, c in coords:
        degree = sum(
            1
            for nr, nc in neighbors4(r, c)
            if 0 <= nr < reach_mask.shape[0] and 0 <= nc < reach_mask.shape[1] and reach_mask[nr, nc]
        )
        scored.append((degree, (r, c)))
    best = max(score for score, _ in scored)
    pool = [coord for score, coord in scored if score >= best - 1]
    return random.choice(pool)


def _boundary_walls(reach_mask: np.ndarray, pr: int, pc: int) -> set[tuple[int, int]]:
    lh, lw = reach_mask.shape
    walls: set[tuple[int, int]] = set()
    for r, c in np.argwhere(reach_mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in _DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < lh and 0 <= nc < lw and not reach_mask[nr, nc]:
                gr, gc = _lattice_to_grid(pr, pc, rr, cc)
                walls.add((gr + dr, gc + dc))
    return walls


def _lattice_nodes_and_midpoints(
    h: int, w: int, lh: int, lw: int, pr: int, pc: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    nodes = {_lattice_to_grid(pr, pc, r, c) for r in range(lh) for c in range(lw)}
    midpoints: set[tuple[int, int]] = set()
    for r in range(lh):
        for c in range(lw):
            gr, gc = _lattice_to_grid(pr, pc, r, c)
            if c + 1 < lw:
                midpoints.add((gr, gc + 1))
            if r + 1 < lh:
                midpoints.add((gr + 1, gc))
    # Keep only in-bounds entries even when the canvas has extra trailing row/col.
    nodes = {(r, c) for r, c in nodes if 0 <= r < h and 0 <= c < w}
    midpoints = {(r, c) for r, c in midpoints if 0 <= r < h and 0 <= c < w}
    return nodes, midpoints


def _grow_irrelevant_blob(
    candidate_mask: np.ndarray,
    occupied: np.ndarray,
    max_size: int,
) -> list[tuple[int, int]]:
    free_coords = np.argwhere(candidate_mask & ~occupied)
    if free_coords.size == 0:
        return []
    start = tuple(map(int, free_coords[random.randrange(len(free_coords))]))
    blob = [start]
    occupied[start] = True
    frontier = {
        nb
        for nb in neighbors4(*start)
        if in_bounds(candidate_mask.shape, nb[0], nb[1]) and candidate_mask[nb] and not occupied[nb]
    }
    target = random.randint(1, max_size)

    while len(blob) < target and frontier:
        choices = sorted(frontier)
        frontier_weights = []
        for r, c in choices:
            support = sum(
                1
                for nr, nc in neighbors4(r, c)
                if in_bounds(candidate_mask.shape, nr, nc) and occupied[nr, nc]
            )
            frontier_weights.append(1.0 + support)
        pick = random.choices(choices, weights=frontier_weights, k=1)[0]
        frontier.discard(pick)
        if occupied[pick]:
            continue
        occupied[pick] = True
        blob.append(pick)
        for nb in neighbors4(*pick):
            if in_bounds(candidate_mask.shape, nb[0], nb[1]) and candidate_mask[nb] and not occupied[nb]:
                frontier.add(nb)
    return blob


def _paint_irrelevant_grey(grid: np.ndarray, candidate_mask: np.ndarray, target_pixels: int) -> None:
    occupied = grid == Color.GREY
    remaining = int(target_pixels)
    attempts = 0
    while remaining > 0 and attempts < 40:
        blob = _grow_irrelevant_blob(candidate_mask, occupied, max_size=min(remaining, 6))
        if not blob:
            break
        for r, c in blob:
            grid[r, c] = Color.GREY
        remaining -= len(blob)
        attempts += 1


def _same_lattice_mask(shape: tuple[int, int], pr: int, pc: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[pr::2, pc::2] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    pr = random.randint(0, 1)
    pc = random.randint(0, 1)
    lh = random.randint(_LATTICE_SIZE_MIN, _LATTICE_SIZE_MAX)
    lw = random.randint(_LATTICE_SIZE_MIN, _LATTICE_SIZE_MAX)
    h, w = _grid_shape_from_lattice(lh, lw, pr, pc)

    split_case = random.random() < _SPLIT_PROB
    total_nodes = lh * lw
    if split_case:
        min_area = max(4, total_nodes // 4)
        max_area = total_nodes - 1
        if min_area > max_area:
            return None
        reach_mask = _grow_component(lh, lw, random.randint(min_area, max_area))
    else:
        reach_mask = np.ones((lh, lw), dtype=bool)

    marker_lr, marker_lc = _choose_marker_cell(reach_mask)
    marker_r, marker_c = _lattice_to_grid(pr, pc, marker_lr, marker_lc)

    grid = np.full((h, w), Color.BLACK, dtype=int)
    marker_color = int(random.choice(_MARKER_COLORS))
    grid[marker_r, marker_c] = marker_color

    if split_case:
        for r, c in _boundary_walls(reach_mask, pr, pc):
            grid[r, c] = Color.GREY

    nodes, active_midpoints = _lattice_nodes_and_midpoints(h, w, lh, lw, pr, pc)
    node_mask = np.zeros((h, w), dtype=bool)
    for r, c in nodes:
        node_mask[r, c] = True
    candidate_mask = np.ones((h, w), dtype=bool)
    for r, c in nodes | active_midpoints:
        candidate_mask[r, c] = False

    extra_grey = random.randint(0, max(0, (h * w) // 20)) if split_case else random.randint(4, max(4, (h * w) // 14))
    _paint_irrelevant_grey(grid, candidate_mask, extra_grey)

    try:
        out = main(grid)
    except Exception:
        return None

    same_lattice = _same_lattice_mask(grid.shape, marker_r % 2, marker_c % 2)
    reachable = _reachable_black_mask(grid, (marker_r, marker_c))
    if not np.any(reachable):
        return None
    if not np.any(grid == Color.GREY):
        return None
    if not np.array_equal(out == marker_color, (grid == marker_color) | reachable):
        return None

    if split_case:
        if not np.any((grid == Color.BLACK) & same_lattice & ~reachable):
            return None
    else:
        if np.any((grid == Color.GREY) & node_mask):
            return None
        if not np.array_equal(reachable, (grid == Color.BLACK) & same_lattice):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-resolution flood-fill input")
