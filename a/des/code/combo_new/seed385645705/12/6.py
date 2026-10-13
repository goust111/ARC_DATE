import random
from collections import deque

import numpy as np


BG = 0
WALL = 9
GEN_ATTEMPTS = 300
TEMPLATE_GAP = 2
TEMPLATE_PATTERNS = (
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _top_wall_row(grid: np.ndarray) -> int:
    wall_rows = np.flatnonzero(np.any(grid == WALL, axis=1))
    if wall_rows.size == 0:
        raise ValueError("expected at least one wall row")
    return int(wall_rows[0])


def _extract_templates(grid: np.ndarray, top_wall: int) -> dict[int, np.ndarray]:
    top = grid[:top_wall]
    templates: dict[int, np.ndarray] = {}
    colors = [int(color) for color in np.unique(top) if int(color) not in {BG, WALL}]
    for color in colors:
        pts = np.argwhere(top == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        patch = top[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        templates[color] = patch == color
    if not templates:
        raise ValueError("expected at least one template")
    return templates


def _room_specs(grid: np.ndarray, top_wall: int) -> list[dict[str, object]]:
    zone = grid[top_wall:]
    rooms: list[dict[str, object]] = []
    for comp in _components(zone != WALL):
        pts = [(int(r + top_wall), int(c)) for r, c in comp]
        colors = {int(grid[r, c]) for r, c in pts if int(grid[r, c]) not in {BG, WALL}}
        if not colors:
            continue
        if len(colors) != 1:
            raise ValueError("each room must contain exactly one marker color")
        color = next(iter(colors))
        count = sum(int(grid[r, c]) == color for r, c in pts)
        if not (1 <= count <= 4):
            raise ValueError("marker count must be between 1 and 4")
        rooms.append(
            {
                "points": pts,
                "color": int(color),
                "count": int(count),
                "bbox": _bbox(pts),
            }
        )
    if not rooms:
        raise ValueError("expected at least one marked room")
    return rooms


def _scale_pattern(pattern: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(pattern, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the matching top-strip template, scaled by the room's marker count.

    Input structure:
    - The background is black, and maroon wall lines divide the lower part of the grid into enclosed rooms.
    - Above the walls sits a strip of small template rectangles; each template uses one non-black color to draw a repeated texture inside its own tight bounding box.
    - Every room contains 1 to 4 marker pixels of one template color, and all markers inside a room share that color.

    Rule:
    - Use the first wall row to separate the template strip from the room area.
    - For each room, the marker color selects the template of the same color, and the number of markers gives the scale factor from 1x to 4x.
    - Enlarge the template by replacing every template cell with a factor-by-factor block, then tile that enlarged pattern across the room starting from the room bbox's top-left corner, clipping at the room boundary.
    - Marker pixels are overwritten by the tiled pattern, the top template strip is erased to black, and the maroon walls stay unchanged.

    Key insight:
    - The room markers encode two independent attributes at once: color chooses which template to use, while count chooses how much to scale that template before tiling it.
    """
    g = np.asarray(grid, dtype=int)
    top_wall = _top_wall_row(g)
    templates = _extract_templates(g, top_wall)

    out = np.full_like(g, BG)
    out[g == WALL] = WALL

    for room in _room_specs(g, top_wall):
        color = int(room["color"])
        scale = int(room["count"])
        if color not in templates:
            raise ValueError("room marker color missing from template strip")
        tiled = _scale_pattern(templates[color], scale)
        th, tw = tiled.shape
        r0, c0, _, _ = room["bbox"]
        for r, c in room["points"]:
            out[r, c] = color if tiled[(r - r0) % th, (c - c0) % tw] else BG
    return out


def _room_grid_size(room_rows: int, room_cols: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return room_rows * cell_h + room_rows + 1, room_cols * cell_w + room_cols + 1


def _cell_start(index: int, cell_size: int) -> int:
    return 1 + index * (cell_size + 1)


def _adjacent_cells(cell: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < room_rows and 0 <= nc < room_cols:
            out.append((int(nr), int(nc)))
    return out


def _partition_cells(
    rng: random.Random, room_rows: int, room_cols: int, room_count: int
) -> dict[tuple[int, int], int] | None:
    cells = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    seeds = rng.sample(cells, room_count)
    owner = {cell: idx for idx, cell in enumerate(seeds)}
    unassigned = {cell for cell in cells if cell not in owner}

    while unassigned:
        frontier: list[tuple[tuple[int, int], list[int]]] = []
        for cell in unassigned:
            neighbor_ids = {owner[nbr] for nbr in _adjacent_cells(cell, room_rows, room_cols) if nbr in owner}
            if neighbor_ids:
                frontier.append((cell, sorted(neighbor_ids)))
        if not frontier:
            return None
        cell, candidates = rng.choice(frontier)
        owner[cell] = rng.choice(candidates)
        unassigned.remove(cell)
    return owner


def _build_structure(
    owner: dict[tuple[int, int], int], room_rows: int, room_cols: int, cell_h: int, cell_w: int
) -> np.ndarray:
    h, w = _room_grid_size(room_rows, room_cols, cell_h, cell_w)
    structure = np.full((h, w), WALL, dtype=int)

    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = _cell_start(rr, cell_h)
            c0 = _cell_start(cc, cell_w)
            structure[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG

    for rr in range(room_rows):
        for cc in range(room_cols):
            this_id = owner[(rr, cc)]
            r0 = _cell_start(rr, cell_h)
            c0 = _cell_start(cc, cell_w)
            if rr + 1 < room_rows and owner[(rr + 1, cc)] == this_id:
                structure[r0 + cell_h, c0 : c0 + cell_w] = BG
            if cc + 1 < room_cols and owner[(rr, cc + 1)] == this_id:
                structure[r0 : r0 + cell_h, c0 + cell_w] = BG
    return structure


def _sample_marker_positions(
    rng: random.Random, points: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    best = points[:count]
    best_score = -1
    for _ in range(20):
        sample = rng.sample(points, count)
        score = 0
        for i, (r1, c1) in enumerate(sample):
            for r2, c2 in sample[i + 1 :]:
                score += abs(r1 - r2) + abs(c1 - c2)
        if score > best_score:
            best = sample
            best_score = score
    return best


def _template_strip_width(n_templates: int) -> int:
    return n_templates * 3 + (n_templates - 1) * TEMPLATE_GAP


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        top_wall = _top_wall_row(grid)
        templates = _extract_templates(grid, top_wall)
        rooms = _room_specs(grid, top_wall)
    except Exception:
        return False

    if len(templates) < 2 or len(rooms) < 4:
        return False
    if {int(room["count"]) for room in rooms} != {1, 2, 3, 4}:
        return False
    if any(int(room["color"]) not in templates for room in rooms):
        return False

    out = main(grid)
    return out.shape == grid.shape and np.array_equal(out[grid == WALL], np.full(np.count_nonzero(grid == WALL), WALL))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    cell_h = rng.randint(3, 4)
    cell_w = rng.randint(3, 4)
    coarse_cells = room_rows * room_cols
    room_count = rng.randint(4, min(6, coarse_cells))
    n_templates = rng.randint(2, min(4, room_count))

    owner = _partition_cells(rng, room_rows, room_cols, room_count)
    if owner is None:
        return None
    structure = _build_structure(owner, room_rows, room_cols, cell_h, cell_w)
    room_components = _components(structure != WALL)
    if len(room_components) != room_count:
        return None

    top_wall = 5
    structure_h, structure_w = structure.shape
    template_w = _template_strip_width(n_templates)
    grid_h = top_wall + structure_h + rng.randint(0, 2)
    grid_w_min = max(structure_w, template_w)
    if grid_h > 30 or grid_w_min > 30:
        return None
    grid_w = grid_w_min + rng.randint(0, 30 - grid_w_min)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    structure_left = rng.randint(0, grid_w - structure_w)
    grid[top_wall : top_wall + structure_h, structure_left : structure_left + structure_w] = structure

    available_colors = [color for color in range(1, 10) if color != WALL]
    template_colors = rng.sample(available_colors, n_templates)
    pattern_ids = rng.sample(range(len(TEMPLATE_PATTERNS)), n_templates)
    templates = [(template_colors[i], TEMPLATE_PATTERNS[pattern_ids[i]]) for i in range(n_templates)]

    template_left = rng.randint(0, grid_w - template_w)
    cursor = template_left
    for color, pattern in templates:
        row0 = rng.randint(0, top_wall - 1 - pattern.shape[0])
        patch = grid[row0 : row0 + pattern.shape[0], cursor : cursor + pattern.shape[1]]
        patch[pattern] = color
        cursor += pattern.shape[1] + TEMPLATE_GAP

    counts = [1, 2, 3, 4]
    while len(counts) < room_count:
        counts.append(rng.randint(1, 4))
    rng.shuffle(counts)

    room_colors = template_colors[:]
    while len(room_colors) < room_count:
        room_colors.append(rng.choice(template_colors))
    rng.shuffle(room_colors)

    order = list(range(room_count))
    rng.shuffle(order)
    for idx, room_idx in enumerate(order):
        comp = room_components[room_idx]
        count = counts[idx]
        color = room_colors[idx]
        global_points = [(r + top_wall, c + structure_left) for r, c in comp]
        markers = _sample_marker_positions(rng, global_points, count)
        for r, c in markers:
            grid[r, c] = color

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
