from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg


BG = Color.BLACK
TEMPLATE_COLOR = Color.BLUE
WALL_COLOR = Color.GREEN
FRAGMENT_COLORS = (Color.RED, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

GRID_MIN = 18
GRID_MAX = 28
ROOM_COUNT_CHOICES = (1, 2, 3)
ROOM_GAP_CHOICES = (2, 3, 4)
GEN_ATTEMPTS = 400
SIDE_CHOICES = ("top", "bottom", "left", "right")


def _boundary_reachable(passable: np.ndarray) -> np.ndarray:
    reachable = np.zeros_like(passable, dtype=bool)
    H, W = passable.shape
    q: deque[tuple[int, int]] = deque()

    for r in range(H):
        for c in (0, W - 1):
            if passable[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if passable[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _template_offsets(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(grid == TEMPLATE_COLOR)
    assert coords.size > 0, "template must exist"
    return _normalize(coords)


def _placement_cells(top: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    tr, tc = top
    return frozenset((tr + dr, tc + dc) for dr, dc in offsets)


def _candidate_placements(
    grid: np.ndarray,
    color: int,
    offsets: tuple[tuple[int, int], ...],
) -> list[frozenset[tuple[int, int]]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    assert len(coords) >= 2, "each fragment must have at least two pixels"
    H, W = grid.shape
    seen: set[tuple[int, int]] = set()
    candidates: list[frozenset[tuple[int, int]]] = []

    for r, c in coords:
        for dr, dc in offsets:
            top = (r - dr, c - dc)
            if top in seen:
                continue
            seen.add(top)
            full = _placement_cells(top, offsets)
            valid = True
            for rr, cc in full:
                if not (0 <= rr < H and 0 <= cc < W):
                    valid = False
                    break
                value = int(grid[rr, cc])
                if value not in (BG, color):
                    valid = False
                    break
            if valid and coords.issubset(full):
                candidates.append(full)
    return candidates


def _unique_template_placement(
    grid: np.ndarray,
    color: int,
    offsets: tuple[tuple[int, int], ...],
) -> frozenset[tuple[int, int]]:
    candidates = _candidate_placements(grid, color, offsets)
    uniq = {cand for cand in candidates}
    assert len(uniq) == 1, "fragment must determine a unique translated template placement"
    return next(iter(uniq))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete only the fragments that lie inside green-walled enclosed regions and erase the rest.

    Input structure:
    - Black is the background, blue is a complete template mask, and green is one 4-connected orthogonal wall object.
    - The green wall partitions the non-wall cells into boundary-connected exterior space and one or more enclosed interior regions.
    - Every other foreground color is a fragment whose pixels form a subset of one translated copy of the blue template.

    Rule:
    - Flood fill from the grid boundary through every non-green cell to mark the exterior side of the wall.
    - For each non-background, non-blue, non-green color, test whether all of its pixels lie in enclosed interior cells.
    - If they do, find the unique translated copy of the blue template that contains those pixels and fill the whole copy with that fragment color; otherwise erase that fragment to black.

    Key insight:
    - The green line is both the flood-fill barrier and the gate that decides whether a fragment should be completed or deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template = _template_offsets(g)
    exterior = _boundary_reachable(g != WALL_COLOR)
    interior = (g != WALL_COLOR) & ~exterior

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, TEMPLATE_COLOR, WALL_COLOR)]
    assert colors, "expected at least one fragment color"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2
        if all(interior[r, c] for r, c in coords):
            full = _unique_template_placement(g, color, template)
            for r, c in full:
                out[r, c] = color
        else:
            out[coords[:, 0], coords[:, 1]] = BG

    return out


def _random_template() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        target = random.randint(4, min(7, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 == r0 or c1 == c0:
            continue
        norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        if len(norm) >= 4:
            return norm
    raise RuntimeError("failed to sample template")


def _room_dims(template: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _ in template) + 1
    max_c = max(c for _, c in template) + 1
    inner_h = random.randint(max_r + 1, max_r + 3)
    inner_w = random.randint(max_c + 1, max_c + 3)
    return inner_h + 2, inner_w + 2


def _draw_box(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR


def _sample_rooms(H: int, W: int, template: tuple[tuple[int, int], ...]) -> tuple[list[tuple[int, int, int, int]], str] | None:
    side = random.choice(SIDE_CHOICES)
    room_count = random.choice(ROOM_COUNT_CHOICES)
    dims = [_room_dims(template) for _ in range(room_count)]

    if side in ("top", "bottom"):
        widths = [w for _, w in dims]
        gaps = [random.choice(ROOM_GAP_CHOICES) for _ in range(max(0, room_count - 1))]
        total_w = sum(widths) + sum(gaps)
        if total_w > W - 2:
            return None
        start_c = random.randint(1, W - total_w - 1)
        max_h = max(h for h, _ in dims)
        if side == "top":
            if H - max_h - 4 < 4:
                return None
            base_r = random.randint(4, H - max_h - 4)
        else:
            if H - max_h - 5 < 1:
                return None
            base_r = random.randint(1, H - max_h - 5)

        rooms = []
        c = start_c
        for h, w in dims:
            delta = random.randint(-1, 1)
            r0 = min(max(base_r + delta, 1), H - h - 2)
            if side == "top":
                r0 = max(r0, 3)
            r1 = r0 + h - 1
            rooms.append((r0, c, r1, c + w - 1))
            c += w
            if gaps:
                c += gaps[len(rooms) - 1] if len(rooms) - 1 < len(gaps) else 0
        return rooms, side

    heights = [h for h, _ in dims]
    gaps = [random.choice(ROOM_GAP_CHOICES) for _ in range(max(0, room_count - 1))]
    total_h = sum(heights) + sum(gaps)
    if total_h > H - 2:
        return None
    start_r = random.randint(1, H - total_h - 1)
    max_w = max(w for _, w in dims)
    if side == "left":
        if W - max_w - 4 < 4:
            return None
        base_c = random.randint(4, W - max_w - 4)
    else:
        if W - max_w - 5 < 1:
            return None
        base_c = random.randint(1, W - max_w - 5)

    rooms = []
    r = start_r
    for h, w in dims:
        delta = random.randint(-1, 1)
        c0 = min(max(base_c + delta, 1), W - w - 2)
        if side == "left":
            c0 = max(c0, 3)
        c1 = c0 + w - 1
        rooms.append((r, c0, r + h - 1, c1))
        r += h
        if gaps:
            r += gaps[len(rooms) - 1] if len(rooms) - 1 < len(gaps) else 0
    return rooms, side


def _build_wall_grid(H: int, W: int, template: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, list[np.ndarray]] | None:
    sampled = _sample_rooms(H, W, template)
    if sampled is None:
        return None
    rooms, side = sampled
    grid = np.full((H, W), BG, dtype=int)
    for room in rooms:
        _draw_box(grid, room)

    if side == "top":
        bus = min(r0 for r0, _, _, _ in rooms) - 2
        anchors = []
        for r0, c0, _, c1 in rooms:
            ac = random.randint(c0 + 1, c1 - 1)
            anchors.append(ac)
            grid[bus : r0 + 1, ac] = WALL_COLOR
        grid[bus, min(anchors) : max(anchors) + 1] = WALL_COLOR
    elif side == "bottom":
        bus = max(r1 for _, _, r1, _ in rooms) + 2
        anchors = []
        for _, c0, r1, c1 in rooms:
            ac = random.randint(c0 + 1, c1 - 1)
            anchors.append(ac)
            grid[r1 : bus + 1, ac] = WALL_COLOR
        grid[bus, min(anchors) : max(anchors) + 1] = WALL_COLOR
    elif side == "left":
        bus = min(c0 for _, c0, _, _ in rooms) - 2
        anchors = []
        for r0, c0, r1, _ in rooms:
            ar = random.randint(r0 + 1, r1 - 1)
            anchors.append(ar)
            grid[ar, bus : c0 + 1] = WALL_COLOR
        grid[min(anchors) : max(anchors) + 1, bus] = WALL_COLOR
    else:
        bus = max(c1 for _, _, _, c1 in rooms) + 2
        anchors = []
        for r0, _, r1, c1 in rooms:
            ar = random.randint(r0 + 1, r1 - 1)
            anchors.append(ar)
            grid[ar, c1 : bus + 1] = WALL_COLOR
        grid[min(anchors) : max(anchors) + 1, bus] = WALL_COLOR

    exterior = _boundary_reachable(grid != WALL_COLOR)
    inside_bg = (grid == BG) & ~exterior
    components = components_mask(inside_bg, connectivity=4)
    if len(components) != len(rooms):
        return None
    return grid, components


def _candidate_tops(
    free_mask: np.ndarray,
    reserved: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    H, W = free_mask.shape
    max_r = max(r for r, _ in offsets)
    max_c = max(c for _, c in offsets)
    out: list[tuple[int, int]] = []
    for top_r in range(H - max_r):
        for top_c in range(W - max_c):
            cells = [(top_r + dr, top_c + dc) for dr, dc in offsets]
            if all(free_mask[r, c] and not reserved[r, c] for r, c in cells):
                out.append((top_r, top_c))
    return out


def _valid_anchor_subsets(
    base_grid: np.ndarray,
    full_cells: frozenset[tuple[int, int]],
    color: int,
    offsets: tuple[tuple[int, int], ...],
) -> list[frozenset[tuple[int, int]]]:
    cells = sorted(full_cells)
    valids: list[frozenset[tuple[int, int]]] = []
    for size in range(2, len(cells)):
        for combo in itertools.combinations(cells, size):
            subset = frozenset(combo)
            trial = base_grid.copy()
            for r, c in subset:
                trial[r, c] = color
            try:
                if _unique_template_placement(trial, color, offsets) == full_cells:
                    valids.append(subset)
            except AssertionError:
                continue
    return valids


def _paint_cells(grid: np.ndarray, cells: frozenset[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    template = _random_template()
    built = _build_wall_grid(H, W, template)
    if built is None:
        return None
    grid, interior_components = built

    exterior = _boundary_reachable(grid != WALL_COLOR)
    reserved = grid != BG

    ext_tops = _candidate_tops((grid == BG) & exterior, reserved, template)
    if not ext_tops:
        return None
    template_top = random.choice(ext_tops)
    template_cells = _placement_cells(template_top, template)
    _paint_cells(grid, template_cells, TEMPLATE_COLOR)
    for r, c in template_cells:
        reserved[r, c] = True

    colors = list(FRAGMENT_COLORS)
    random.shuffle(colors)
    internal_count = random.randint(1, len(interior_components))
    chosen_components = random.sample(interior_components, internal_count)
    external_count = random.randint(1, min(3, len(colors) - internal_count))

    fragment_specs: list[tuple[int, frozenset[tuple[int, int]], bool]] = []

    for comp in chosen_components:
        mask = np.zeros((H, W), dtype=bool)
        mask[comp[:, 0], comp[:, 1]] = True
        tops = _candidate_tops(mask, reserved, template)
        if not tops:
            return None
        top = random.choice(tops)
        full = _placement_cells(top, template)
        color = colors[len(fragment_specs)]
        subsets = _valid_anchor_subsets(grid, full, color, template)
        if not subsets:
            return None
        subset = random.choice(subsets)
        fragment_specs.append((color, subset, True))
        for r, c in full:
            reserved[r, c] = True

    for _ in range(external_count):
        tops = _candidate_tops((grid == BG) & exterior, reserved, template)
        if not tops:
            return None
        top = random.choice(tops)
        full = _placement_cells(top, template)
        color = colors[len(fragment_specs)]
        subsets = _valid_anchor_subsets(grid, full, color, template)
        if not subsets:
            return None
        subset = random.choice(subsets)
        fragment_specs.append((color, subset, False))
        for r, c in full:
            reserved[r, c] = True

    for color, subset, _ in fragment_specs:
        _paint_cells(grid, subset, color)

    try:
        _verify_example(grid)
    except AssertionError:
        return None
    return grid


def _verify_example(grid: np.ndarray) -> None:
    out = main(grid)
    assert grid.shape == out.shape
    assert np.any(grid == WALL_COLOR) and np.any(grid == TEMPLATE_COLOR)

    template = _template_offsets(grid)
    exterior = _boundary_reachable(grid != WALL_COLOR)
    interior = (grid != WALL_COLOR) & ~exterior
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, TEMPLATE_COLOR, WALL_COLOR)]
    assert colors

    saw_completion = False
    saw_erasure = False
    for color in colors:
        coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        full = _unique_template_placement(grid, color, template)
        if all(interior[r, c] for r, c in coords):
            for r, c in full:
                assert out[r, c] == color
            if full != frozenset(coords):
                saw_completion = True
        else:
            assert not np.any(out == color)
            saw_erasure = True

    assert saw_completion, "need at least one fragment that gets completed"
    assert saw_erasure, "need at least one fragment that gets erased"
    assert np.array_equal(out[grid == WALL_COLOR], grid[grid == WALL_COLOR])
    assert np.array_equal(out[grid == TEMPLATE_COLOR], grid[grid == TEMPLATE_COLOR])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-completion ARC input")
