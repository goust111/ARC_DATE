from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
MAROON = 9
MOVABLE_COLORS = (2, 3, 4, 5, 6, 7, 8)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "down",
    "bottom": "up",
    "left": "right",
    "right": "left",
}
PUSH_TO_CONTACT_SIDE = {
    "up": "bottom",
    "down": "top",
    "left": "right",
    "right": "left",
}


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "area": int(arr.shape[0])})
    return comps


def detect_push_direction(grid: np.ndarray, coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    coord_set = {tuple(cell) for cell in coords.tolist()}
    sides = set()
    for r, c in coord_set:
        if in_bounds(grid.shape, r - 1, c) and int(grid[r - 1, c]) == MAROON and (r - 1, c) not in coord_set:
            sides.add("top")
        if in_bounds(grid.shape, r + 1, c) and int(grid[r + 1, c]) == MAROON and (r + 1, c) not in coord_set:
            sides.add("bottom")
        if in_bounds(grid.shape, r, c - 1) and int(grid[r, c - 1]) == MAROON and (r, c - 1) not in coord_set:
            sides.add("left")
        if in_bounds(grid.shape, r, c + 1) and int(grid[r, c + 1]) == MAROON and (r, c + 1) not in coord_set:
            sides.add("right")
    assert len(sides) == 1, "each movable object must touch maroon on exactly one side"
    side = next(iter(sides))
    push_name = OPPOSITE_SIDE[side]
    return push_name, DIRS[push_name]


def steps_to_boundary(shape: tuple[int, int], coords: np.ndarray, direction: tuple[int, int]) -> int:
    dr, dc = direction
    rows = coords[:, 0]
    cols = coords[:, 1]
    h, w = shape
    if dr == -1:
        return int(rows.min())
    if dr == 1:
        return int(h - 1 - rows.max())
    if dc == -1:
        return int(cols.min())
    return int(w - 1 - cols.max())


def push_objects(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    movable_colors = set(int(v) for v in np.unique(g) if int(v) not in {bg, BLUE, MAROON})
    comps = find_color_components(g, movable_colors)

    out = g.copy()
    placements = []
    barrier_mask = np.zeros(g.shape, dtype=bool)
    for comp in comps:
        coords = comp["coords"]
        color = int(comp["color"])
        _, direction = detect_push_direction(g, coords)
        steps = steps_to_boundary(g.shape, coords, direction)
        assert steps >= 1, "generated instances keep every push non-trivial"
        shifted = coords + np.array(direction, dtype=int) * steps
        out[coords[:, 0], coords[:, 1]] = bg
        placements.append({"color": color, "coords": shifted})

    for item in placements:
        coords = item["coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] == bg), "pushed barriers must not collide"
        out[coords[:, 0], coords[:, 1]] = int(item["color"])
        barrier_mask[coords[:, 0], coords[:, 1]] = True
    return out, barrier_mask, placements


def ordered_blue_seeds(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
    seeds = sorted(map(tuple, np.argwhere(np.asarray(grid, dtype=int) == BLUE).tolist()))
    assert len(seeds) == 3, "expected exactly three blue seed pixels"
    d1 = (seeds[1][0] - seeds[0][0], seeds[1][1] - seeds[0][1])
    d2 = (seeds[2][0] - seeds[1][0], seeds[2][1] - seeds[1][1])
    assert d1 == d2, "seed spacing must be equal"
    assert abs(d1[0]) == abs(d1[1]) and abs(d1[0]) >= 1, "seeds must lie on a 45-degree diagonal"
    return seeds, abs(int(d1[0])), seeds[1]


def square_border_coords(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    top = cr - radius
    bottom = cr + radius
    left = cc - radius
    right = cc + radius
    coords = []
    for c in range(left, right + 1):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return coords


def iter_ring_layers(shape: tuple[int, int], center: tuple[int, int], spacing: int):
    radius = spacing
    while True:
        coords = square_border_coords(center, radius)
        inb = [(r, c) for r, c in coords if in_bounds(shape, r, c)]
        if not inb:
            break
        yield radius, coords, inb
        radius += spacing


def draw_rings(
    base: np.ndarray,
    center: tuple[int, int],
    spacing: int,
    blocker_mask: np.ndarray,
    barrier_mask: np.ndarray | None = None,
):
    out = np.asarray(base, dtype=int).copy()
    layer_info = []
    for radius, _coords, inb in iter_ring_layers(out.shape, center, spacing):
        drawn = 0
        blocked = 0
        blocked_barrier = 0
        for r, c in inb:
            if blocker_mask[r, c]:
                blocked += 1
                if barrier_mask is not None and barrier_mask[r, c]:
                    blocked_barrier += 1
                continue
            out[r, c] = BLUE
            drawn += 1
        layer_info.append(
            {
                "radius": radius,
                "drawn": drawn,
                "blocked": blocked,
                "blocked_barrier": blocked_barrier,
            }
        )
    return out, layer_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push marker-touched objects to the edge, then grow blue concentric square frames that are cut by the pushed barriers.

    Input structure:
    - The background is black, with 2-3 small non-blue, non-maroon 4-connected objects in distinct colors.
    - Each movable object touches maroon marker pixels on exactly one cardinal side, which determines its outward push direction.
    - Exactly three blue seed pixels lie on one 45-degree diagonal with equal spacing.

    Rule:
    - Move every non-blue, non-maroon object straight away from its maroon contact side until that object reaches the grid boundary, erasing its old position and keeping maroon markers fixed.
    - Read the blue seeds in row-major order, measure their common spacing S, and use the middle seed as the center for concentric square borders at radii S, 2S, 3S, ...
    - For each border cell that is in bounds, draw blue unless that cell is occupied by a pushed barrier or by a maroon marker; blocked cells stay in their existing color.
    - Stop once the next full square border has no in-bounds cells at all; the original three seed pixels remain blue.

    Key insight:
    - The pushed objects are not the final target: once they become edge barriers, they asymmetrically punch holes into otherwise regular concentric square growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pushed, barrier_mask, _ = push_objects(g)
    _seeds, spacing, center = ordered_blue_seeds(g)
    blocker_mask = barrier_mask | (pushed == MAROON)
    out, _ = draw_rings(pushed, center, spacing, blocker_mask, barrier_mask)
    return out


def random_polyomino(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(rng.choice(sorted(frontier)))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def boundary_face_count(shape: list[tuple[int, int]], side: str) -> int:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    if side == "up":
        face = min(rows)
        return sum(r == face for r, _ in shape)
    if side == "down":
        face = max(rows)
        return sum(r == face for r, _ in shape)
    if side == "left":
        face = min(cols)
        return sum(c == face for _, c in shape)
    face = max(cols)
    return sum(c == face for _, c in shape)


def place_shape_on_boundary(
    shape: list[tuple[int, int]],
    side: str,
    offset: int,
    shape_hw: tuple[int, int],
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape_hw
    gh, gw = grid_shape
    if side == "up":
        anchor = (0, offset)
    elif side == "down":
        anchor = (gh - h, offset)
    elif side == "left":
        anchor = (offset, 0)
    else:
        anchor = (offset, gw - w)
    ar, ac = anchor
    return [(ar + r, ac + c) for r, c in shape]


def boundary_face_coords(final_coords: list[tuple[int, int]], side: str, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    if side == "up":
        return [(r, c) for r, c in final_coords if r == 0]
    if side == "down":
        return [(r, c) for r, c in final_coords if r == gh - 1]
    if side == "left":
        return [(r, c) for r, c in final_coords if c == 0]
    return [(r, c) for r, c in final_coords if c == gw - 1]


def side_contact_cells(coords: list[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    if side == "up":
        edge = min(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if side == "down":
        edge = max(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if side == "left":
        edge = min(c for _, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    edge = max(c for _, c in coords)
    return [(r, c) for r, c in coords if c == edge]


def expand8(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def build_seed_layout(
    rng: random.Random,
) -> tuple[int, int, int, int, str, int, tuple[int, int], list[tuple[int, int]]]:
    spacing = rng.choice([3, 4])
    active_layer = rng.choice([2, 3])
    active_side = rng.choice(["up", "down", "left", "right"])
    orient = rng.choice(["main", "anti"])

    min_dim = max(18, (active_layer + 2) * spacing + 3, 6 * spacing + 1)
    h = rng.randint(min_dim, min(30, min_dim + 4))
    w = rng.randint(min_dim, min(30, min_dim + 4))

    if active_side == "up":
        cr = active_layer * spacing
    elif active_side == "down":
        cr = h - 1 - active_layer * spacing
    else:
        cr = rng.randint(2 * spacing + 1, h - 2 * spacing - 2)

    if active_side == "left":
        cc = active_layer * spacing
    elif active_side == "right":
        cc = w - 1 - active_layer * spacing
    else:
        cc = rng.randint(2 * spacing + 1, w - 2 * spacing - 2)

    if orient == "main":
        seeds = [(cr - spacing, cc - spacing), (cr, cc), (cr + spacing, cc + spacing)]
    else:
        seeds = [(cr - spacing, cc + spacing), (cr, cc), (cr + spacing, cc - spacing)]
    for r, c in seeds:
        assert in_bounds((h, w), r, c)
    return h, w, spacing, active_layer, active_side, cr, (cr, cc), seeds


def active_boundary_segment(
    center: tuple[int, int], spacing: int, layer: int, side: str, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    radius = spacing * layer
    segment = set()
    for r, c in square_border_coords(center, radius):
        if not in_bounds(shape, r, c):
            continue
        if side == "up" and r == 0:
            segment.add((r, c))
        elif side == "down" and r == shape[0] - 1:
            segment.add((r, c))
        elif side == "left" and c == 0:
            segment.add((r, c))
        elif side == "right" and c == shape[1] - 1:
            segment.add((r, c))
    return segment


def try_place_object(
    rng: random.Random,
    side: str,
    color: int,
    grid_shape: tuple[int, int],
    all_ring_cells: set[tuple[int, int]],
    boundary_ring_cells: set[tuple[int, int]],
    required_hits: set[tuple[int, int]],
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    require_hit: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    gh, gw = grid_shape
    for _ in range(120):
        size = rng.randint(2, 4)
        shape = random_polyomino(rng, size)
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        shape_hw = (max(rows) + 1, max(cols) + 1)
        if require_hit and boundary_face_count(shape, side) < 2:
            continue

        if side in {"up", "down"}:
            offsets = list(range(0, gw - shape_hw[1] + 1))
        else:
            offsets = list(range(0, gh - shape_hw[0] + 1))
        rng.shuffle(offsets)

        for offset in offsets:
            final_coords = place_shape_on_boundary(shape, side, offset, shape_hw, grid_shape)
            face = boundary_face_coords(final_coords, side, grid_shape)
            if not face:
                continue
            orth = [c for _r, c in face] if side in {"up", "down"} else [r for r, _c in face]
            if min(orth) < 1 or max(orth) > (gw - 2 if side in {"up", "down"} else gh - 2):
                continue
            face_set = set(face)
            hits_required = bool(face_set & required_hits)
            hits_any_ring = bool(face_set & boundary_ring_cells)
            if require_hit and not hits_required:
                continue
            if not require_hit and hits_any_ring:
                continue

            final_set = set(final_coords)
            if final_set & output_occ:
                continue
            if final_set & protected:
                continue

            max_step = steps_to_boundary(
                grid_shape,
                np.array(final_coords, dtype=int),
                tuple(-v for v in DIRS[side]),
            )
            steps = list(range(2, max_step + 1))
            rng.shuffle(steps)
            for step in steps:
                dr, dc = DIRS[side]
                original_coords = [(r - dr * step, c - dc * step) for r, c in final_coords]
                original_set = set(original_coords)
                if original_set & input_occ:
                    continue
                if original_set & protected:
                    continue
                if original_set & all_ring_cells:
                    continue

                path_ok = True
                for t in range(step + 1):
                    shifted = {(r + dr * t, c + dc * t) for r, c in original_coords}
                    if t == 0:
                        probe = shifted
                    else:
                        probe = shifted - original_set
                    if probe & protected:
                        path_ok = False
                        break
                    if probe & (input_occ - original_set):
                        path_ok = False
                        break
                if not path_ok:
                    continue

                contact_side = PUSH_TO_CONTACT_SIDE[side]
                contact_cells = side_contact_cells(original_coords, contact_side)
                rng.shuffle(contact_cells)
                for contact in contact_cells:
                    mr = contact[0] - dr
                    mc = contact[1] - dc
                    marker_cells = [(mr, mc)]
                    if not in_bounds(grid_shape, mr, mc):
                        continue
                    if (mr, mc) in input_occ or (mr, mc) in output_occ or (mr, mc) in protected:
                        continue
                    if (mr, mc) in all_ring_cells:
                        continue

                    if rng.random() < 0.5:
                        tangents = [(0, 1), (0, -1)] if side in {"up", "down"} else [(1, 0), (-1, 0)]
                        rng.shuffle(tangents)
                        for tr, tc in tangents:
                            nr, nc = mr + tr, mc + tc
                            if not in_bounds(grid_shape, nr, nc):
                                continue
                            if (nr, nc) in input_occ or (nr, nc) in output_occ or (nr, nc) in protected:
                                continue
                            if (nr, nc) in original_set or (nr, nc) in all_ring_cells:
                                continue
                            marker_cells.append((nr, nc))
                            break

                    marker_set = set(marker_cells)
                    unrelated = expand8(input_occ | protected) - expand8(original_set)
                    if marker_set & unrelated:
                        continue
                    return original_coords, final_coords, marker_cells
    return None


def build_marker_only_grid(shape: tuple[int, int], seeds: list[tuple[int, int]], markers: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in seeds:
        grid[r, c] = BLUE
    for r, c in markers:
        grid[r, c] = MAROON
    return grid


def validate_instance(grid: np.ndarray) -> bool:
    try:
        seeds, spacing, center = ordered_blue_seeds(grid)
        movable = find_color_components(grid, set(MOVABLE_COLORS))
        markers = find_color_components(grid, {MAROON})
        assert 2 <= len(movable) <= 3
        assert all(2 <= comp["area"] <= 4 for comp in movable)
        assert all(1 <= comp["area"] <= 2 for comp in markers)
        assert len({int(comp["color"]) for comp in movable}) == len(movable)

        all_ring_cells = set()
        for _radius, _coords, inb in iter_ring_layers(grid.shape, center, spacing):
            all_ring_cells.update(inb)

        pushed, barrier_mask, _ = push_objects(grid)
        blocker_mask = barrier_mask | (pushed == MAROON)
        out, info = draw_rings(pushed, center, spacing, blocker_mask, barrier_mask)
        assert any(layer["blocked_barrier"] > 0 and layer["drawn"] > 0 for layer in info)
        assert np.any(out != pushed)

        marker_cells = {tuple(cell) for cell in np.argwhere(grid == MAROON).tolist()}
        marker_only = build_marker_only_grid(grid.shape, seeds, marker_cells)
        no_push, _ = draw_rings(marker_only, center, spacing, marker_only == MAROON)
        assert np.any(out != no_push)
        assert np.array_equal(out, main(grid))

        for cell in marker_cells:
            assert cell not in all_ring_cells
        for comp in movable:
            for cell in map(tuple, comp["coords"].tolist()):
                assert cell not in all_ring_cells
            detect_push_direction(grid, comp["coords"])
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, spacing, active_layer, active_side, _cr, center, seeds = build_seed_layout(rng)
    grid_shape = (h, w)

    all_ring_cells: set[tuple[int, int]] = set()
    boundary_ring_cells: set[tuple[int, int]] = set()
    for _radius, _coords, inb in iter_ring_layers(grid_shape, center, spacing):
        for cell in inb:
            all_ring_cells.add(cell)
            if cell[0] in {0, h - 1} or cell[1] in {0, w - 1}:
                boundary_ring_cells.add(cell)

    required_hits = active_boundary_segment(center, spacing, active_layer, active_side, grid_shape)
    if len(required_hits) < 4:
        return None

    grid = np.full(grid_shape, BG, dtype=int)
    for r, c in seeds:
        grid[r, c] = BLUE

    protected = set(seeds)
    input_occ = set(seeds)
    output_occ = set(seeds)

    num_objects = rng.choice([2, 3])
    other_sides = [s for s in ["up", "down", "left", "right"] if s != active_side]
    sides = [active_side] + rng.sample(other_sides, num_objects - 1)
    colors = rng.sample(list(MOVABLE_COLORS), num_objects)

    all_markers: set[tuple[int, int]] = set()
    for index, (side, color) in enumerate(zip(sides, colors)):
        placed = try_place_object(
            rng,
            side,
            color,
            grid_shape,
            all_ring_cells,
            boundary_ring_cells,
            required_hits,
            input_occ,
            output_occ,
            protected,
            require_hit=(index == 0),
        )
        if placed is None:
            return None
        original_coords, final_coords, marker_cells = placed
        for r, c in original_coords:
            grid[r, c] = color
        for r, c in marker_cells:
            grid[r, c] = MAROON
        input_occ.update(original_coords)
        input_occ.update(marker_cells)
        output_occ.update(final_coords)
        output_occ.update(marker_cells)
        all_markers.update(marker_cells)

    if len(all_markers) < num_objects:
        return None
    if not validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
