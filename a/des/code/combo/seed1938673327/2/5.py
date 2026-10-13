from __future__ import annotations

from collections import deque
import random

import numpy as np


ORANGE = 7
GREY = 5
RED = 2
GREEN = 3
YELLOW = 4

LEGEND_COLORS = (RED, GREEN, YELLOW)
OBJECT_COLOR_POOL = (0, 1, 6, 8, 9)
GENERATION_ATTEMPTS = 300


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[list[tuple[int, int]]] = []

    if h >= 3:
        vertical_starts = {(0, 0), (0, w - 1), (h - 3, 0), (h - 3, w - 1)}
        for r0, c0 in vertical_starts:
            if 0 <= r0 <= h - 3 and 0 <= c0 < w:
                colors = tuple(int(g[r0 + i, c0]) for i in range(3))
                if colors == LEGEND_COLORS:
                    candidates.append([(r0 + i, c0) for i in range(3)])

    if w >= 3:
        horizontal_starts = {(0, 0), (0, w - 3), (h - 1, 0), (h - 1, w - 3)}
        for r0, c0 in horizontal_starts:
            if 0 <= r0 < h and 0 <= c0 <= w - 3:
                colors = tuple(int(g[r0, c0 + i]) for i in range(3))
                if colors == LEGEND_COLORS:
                    candidates.append([(r0, c0 + i) for i in range(3)])

    assert len(candidates) == 1, "expected exactly one corner legend"
    positions = candidates[0]
    colors = tuple(int(g[r, c]) for r, c in positions)
    return positions, colors


def _object_components(
    grid: np.ndarray,
    background: int,
    wall: int,
    legend_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            if color == background or color == wall or (r, c) in legend_cells:
                continue

            visited[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc]:
                        continue
                    if (nr, nc) in legend_cells:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            components.append(coords)

    return components


def _contact_edges(grid: np.ndarray, coords: list[tuple[int, int]], wall: int) -> int:
    h, w = grid.shape
    total = 0
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == wall:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by how many shared edges it has with the grey walls.

    Objects touching the walls on 1-2 edges become the first legend color, 3-4 edges
    become the second legend color, and 5+ edges become the third legend color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    background = _majority_color(g)
    legend_positions, legend_colors = _find_legend(g)
    legend_cells = set(legend_positions)

    out = g.copy()
    for coords in _object_components(g, background, GREY, legend_cells):
        contacts = _contact_edges(g, coords, GREY)
        assert contacts >= 1, "every generated object must touch a wall"
        if contacts <= 2:
            new_color = legend_colors[0]
        elif contacts <= 4:
            new_color = legend_colors[1]
        else:
            new_color = legend_colors[2]
        rows, cols = zip(*coords)
        out[rows, cols] = int(new_color)

    return out


def _bucket_target(contact_edges: int) -> int:
    if contact_edges <= 2:
        return RED
    if contact_edges <= 4:
        return GREEN
    return YELLOW


def _legend_array(orientation: str) -> np.ndarray:
    if orientation == "vertical":
        return np.array([[RED], [GREEN], [YELLOW]], dtype=int)
    assert orientation == "horizontal"
    return np.array([[RED, GREEN, YELLOW]], dtype=int)


def _place_legend(grid: np.ndarray, corner: str, orientation: str) -> set[tuple[int, int]]:
    patch = _legend_array(orientation)
    ph, pw = patch.shape
    h, w = grid.shape

    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        assert corner == "br"
        r0, c0 = h - ph, w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return {(r0 + r, c0 + c) for r in range(ph) for c in range(pw)}


def _add_frontier(
    frontier: set[tuple[int, int]],
    used: set[tuple[int, int]],
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    cell: tuple[int, int],
) -> None:
    r, c = cell
    for nr, nc in _neighbors4(r, c):
        if row_min <= nr <= row_max and col_min <= nc <= col_max and (nr, nc) not in used:
            frontier.add((nr, nc))


def _make_motif(contact_edges: int, color: int) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    assert 1 <= contact_edges <= 6

    left_slack = random.randint(0, 1)
    right_slack = random.randint(0, 1)
    wall_len = contact_edges + left_slack + right_slack
    arm_len = random.choice((0, 0, 1, 2))
    extra_rows = random.randint(1, 3)
    extra_cells = random.randint(0, contact_edges + 2)

    wall_row = arm_len
    contact_row = wall_row + 1
    height = contact_row + 1 + extra_rows
    width = wall_len
    contact_start = left_slack

    patch = np.full((height, width), ORANGE, dtype=int)
    patch[wall_row, :] = GREY
    if arm_len:
        endpoint_col = 0 if random.random() < 0.5 else width - 1
        patch[: wall_row + 1, endpoint_col] = GREY

    object_cells: set[tuple[int, int]] = {
        (contact_row, contact_start + offset) for offset in range(contact_edges)
    }
    row_min, row_max = contact_row + 1, height - 1
    col_min, col_max = 0, width - 1
    frontier: set[tuple[int, int]] = set()
    for cell in object_cells:
        _add_frontier(frontier, object_cells, row_min, row_max, col_min, col_max, cell)

    target_size = len(object_cells) + extra_cells
    while frontier and len(object_cells) < target_size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        object_cells.add(cell)
        _add_frontier(frontier, object_cells, row_min, row_max, col_min, col_max, cell)

    for r, c in object_cells:
        patch[r, c] = color

    rotation = random.randrange(4)
    rotated = np.rot90(patch, k=rotation)
    coords = list(zip(*np.where(rotated == color)))
    return rotated, [(int(r), int(c)) for r, c in coords], contact_edges


def _can_place_rect(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    r0 = top - 1
    c0 = left - 1
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    color: int,
    target_color: int,
    objects: list[tuple[list[tuple[int, int]], int]],
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [
        (r, c)
        for r in range(1, h - ph)
        for c in range(1, w - pw)
        if _can_place_rect(occupied, r, c, ph, pw)
    ]
    if not candidates:
        return False

    top, left = random.choice(candidates)
    grid[top : top + ph, left : left + pw] = patch
    occupied[top : top + ph, left : left + pw] = True

    coords = list(zip(*np.where(patch == color)))
    global_coords = [(top + int(r), left + int(c)) for r, c in coords]
    objects.append((global_coords, target_color))
    return True


def _grid_shape_for(patches: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(area * 1.6))) + 4)
    height = random.randint(side, side + 5)
    width = random.randint(side, side + 7)
    return height, width


def _sample_contacts() -> list[int]:
    contacts = [
        random.choice((1, 2)),
        random.choice((3, 4)),
        random.choice((5, 6)),
    ]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        bucket = random.randrange(3)
        if bucket == 0:
            contacts.append(random.choice((1, 2)))
        elif bucket == 1:
            contacts.append(random.choice((3, 4)))
        else:
            contacts.append(random.choice((5, 6)))
    random.shuffle(contacts)
    return contacts


def _mark_legend_padding(
    occupied: np.ndarray,
    legend_cells: set[tuple[int, int]],
) -> None:
    h, w = occupied.shape
    for r, c in legend_cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                occupied[rr, cc] = True


def _expected_output(
    grid: np.ndarray,
    objects: list[tuple[list[tuple[int, int]], int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for coords, target_color in objects:
        rows, cols = zip(*coords)
        out[rows, cols] = int(target_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    contacts = _sample_contacts()
    colors = random.sample(list(OBJECT_COLOR_POOL), len(contacts))
    motifs = [_make_motif(contact_edges, color) for contact_edges, color in zip(contacts, colors)]
    patches = [patch for patch, _coords, _contacts in motifs]

    height, width = _grid_shape_for(patches)
    grid = np.full((height, width), ORANGE, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    orientation = random.choice(("vertical", "horizontal"))
    legend_cells = _place_legend(grid, corner, orientation)
    _mark_legend_padding(occupied, legend_cells)

    placed_objects: list[tuple[list[tuple[int, int]], int]] = []
    order = list(range(len(motifs)))
    random.shuffle(order)
    for idx in order:
        patch, _coords, contact_edges = motifs[idx]
        if not _place_patch(
            grid,
            occupied,
            patch,
            colors[idx],
            _bucket_target(contact_edges),
            placed_objects,
        ):
            return None

    expected = _expected_output(grid, placed_objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
