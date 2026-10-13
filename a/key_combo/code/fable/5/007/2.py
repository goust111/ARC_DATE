from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background_components(walls: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    """Return 4-connected non-wall components and whether each touches the patch edge."""
    h, w = walls.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool]] = []

    for sr in range(h):
        for sc in range(w):
            if bool(walls[sr, sc]) or bool(seen[sr, sc]):
                continue
            seen[sr, sc] = True
            queue = deque([(sr, sc)])
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                touches_edge |= r == 0 or r == h - 1 or c == 0 or c == w - 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and not bool(walls[rr, cc])
                        and not bool(seen[rr, cc])
                    ):
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            components.append((cells, bool(touches_edge)))
    return components


def _door_direction(walls: np.ndarray) -> tuple[int, int]:
    """Read the unique non-corner gap in the rectangular outer wall."""
    h, w = walls.shape
    missing: set[tuple[int, int]] = set()
    for c in range(w):
        if not bool(walls[0, c]):
            missing.add((0, c))
        if not bool(walls[h - 1, c]):
            missing.add((h - 1, c))
    for r in range(h):
        if not bool(walls[r, 0]):
            missing.add((r, 0))
        if not bool(walls[r, w - 1]):
            missing.add((r, w - 1))

    assert len(missing) == 1, "each frame must have exactly one outer-wall gap"
    r, c = next(iter(missing))
    assert r not in (0, h - 1) or c not in (0, w - 1), "the door cannot be a corner"
    if r == 0:
        return DIRS["up"]
    if r == h - 1:
        return DIRS["down"]
    if c == 0:
        return DIRS["left"]
    assert c == w - 1
    return DIRS["right"]


def _analyze_frame(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size, "frame color is absent"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all((crop == BACKGROUND) | (crop == int(color)))), (
        "different frames must not enter one another's bounding boxes"
    )
    walls = crop == int(color)
    direction = _door_direction(walls)
    components = _background_components(walls)
    enclosed = sum(not touches_edge for _cells, touches_edge in components)
    leaking = sum(touches_edge for _cells, touches_edge in components)
    assert leaking == 1, "the door must expose exactly one room to the exterior"
    assert enclosed >= 1, "a frame must contain at least one still-enclosed room"
    return {
        "color": int(color),
        "coords": coords.astype(int),
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "distance": int(enclosed),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every partitioned frame toward its door by its number of sealed rooms.

    Input structure:
    - A black grid contains several mutually separated, differently colored rectangular
      frames; same-colored inner walls partition each frame into rooms.
    - Every outer wall has exactly one one-cell, non-corner door gap, and exactly one
      interior background region reaches that gap. Valid translated frames do not overlap.

    Rule:
    - For each color, use its full-grid bounding box as that frame's coordinate system.
      The side containing the sole outer-wall gap gives the outward motion direction.
    - Count the 4-connected background components inside the box that do not touch its
      boundary. Clear all old walls, then translate every colored wall pixel by that count;
      discard translated pixels that fall beyond the grid edge.

    Key insight:
    - The door independently encodes direction while topology encodes distance: the room
      connected through the door is already leaking and therefore is not counted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert bool(np.all((0 <= g) & (g <= 9))), "expected ARC palette values"

    colors = [int(v) for v in np.unique(g) if int(v) != BACKGROUND]
    assert colors, "expected at least one colored frame"
    frames = [_analyze_frame(g, color) for color in colors]

    h, w = g.shape
    destinations: list[tuple[dict, np.ndarray]] = []
    occupied = np.zeros((h, w), dtype=bool)
    for frame in frames:
        dr, dc = frame["direction"]
        distance = int(frame["distance"])
        shifted = frame["coords"] + np.array([dr * distance, dc * distance], dtype=int)
        in_bounds = (
            (0 <= shifted[:, 0])
            & (shifted[:, 0] < h)
            & (0 <= shifted[:, 1])
            & (shifted[:, 1] < w)
        )
        visible = shifted[in_bounds]
        if visible.size:
            assert not bool(occupied[visible[:, 0], visible[:, 1]].any()), (
                "translated frames must not overlap"
            )
            occupied[visible[:, 0], visible[:, 1]] = True
        destinations.append((frame, visible))

    out = np.full_like(g, BACKGROUND)
    for frame, visible in destinations:
        if visible.size:
            out[visible[:, 0], visible[:, 1]] = int(frame["color"])
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_frame(color: int, direction: tuple[int, int], sealed: int, require_merge: bool) -> np.ndarray:
    """Build a rectilinear room plan whose door leaves exactly `sealed` rooms closed."""
    total_rooms = int(sealed) + 1
    choices = []
    for nr in range(1, 4):
        for nc in range(1, 4):
            cells = nr * nc
            if total_rooms <= cells <= 6 and (not require_merge or cells > total_rooms):
                choices.append((nr, nc))
    assert choices
    nr, nc = random.choice(choices)

    # Two-cell rooms make both the partitions and one-cell openings easy to see.
    room_heights = [random.choice((2, 2, 3)) for _ in range(nr)]
    room_widths = [random.choice((2, 2, 3)) for _ in range(nc)]
    row_lines = [0]
    col_lines = [0]
    for size in room_heights:
        row_lines.append(row_lines[-1] + size + 1)
    for size in room_widths:
        col_lines.append(col_lines[-1] + size + 1)

    patch = np.zeros((row_lines[-1] + 1, col_lines[-1] + 1), dtype=int)
    for r in row_lines:
        patch[r, :] = int(color)
    for c in col_lines:
        patch[:, c] = int(color)

    # Start with a rectangular grid of cells, then open selected inner wall segments.
    # Union-find ensures every opening reduces the actual topological room count by one.
    parent = list(range(nr * nc))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    adjacencies: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(nr):
        for c in range(nc - 1):
            gap_r = random.randint(row_lines[r] + 1, row_lines[r + 1] - 1)
            adjacencies.append((r * nc + c, r * nc + c + 1, (gap_r, col_lines[c + 1])))
    for r in range(nr - 1):
        for c in range(nc):
            gap_c = random.randint(col_lines[c] + 1, col_lines[c + 1] - 1)
            adjacencies.append((r * nc + c, (r + 1) * nc + c, (row_lines[r + 1], gap_c)))
    random.shuffle(adjacencies)

    openings_needed = nr * nc - total_rooms
    opened = 0
    for a, b, (gr, gc) in adjacencies:
        if opened == openings_needed:
            break
        if union(a, b):
            patch[gr, gc] = BACKGROUND
            opened += 1
    assert opened == openings_needed

    dr, dc = direction
    if dr == -1:
        door_choices = [(0, c) for c in range(1, patch.shape[1] - 1) if c not in col_lines]
    elif dr == 1:
        door_choices = [
            (patch.shape[0] - 1, c)
            for c in range(1, patch.shape[1] - 1)
            if c not in col_lines
        ]
    elif dc == -1:
        door_choices = [(r, 0) for r in range(1, patch.shape[0] - 1) if r not in row_lines]
    else:
        door_choices = [
            (r, patch.shape[1] - 1)
            for r in range(1, patch.shape[0] - 1)
            if r not in row_lines
        ]
    patch[random.choice(door_choices)] = BACKGROUND

    walls = patch == int(color)
    assert _door_direction(walls) == direction
    comps = _background_components(walls)
    assert sum(not edge for _cells, edge in comps) == sealed
    assert sum(edge for _cells, edge in comps) == 1
    assert len(comps) == total_rooms
    return patch


def _motion_distance_to_edge(
    shape: tuple[int, int], patch_shape: tuple[int, int], r: int, c: int, direction: tuple[int, int]
) -> int:
    h, w = shape
    ph, pw = patch_shape
    dr, dc = direction
    if dr == -1:
        return int(r)
    if dr == 1:
        return int(h - (r + ph))
    if dc == -1:
        return int(c)
    return int(w - (c + pw))


def _expanded_box_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    return not bool(mask[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any())


def _try_generate() -> np.ndarray | None:
    h = random.randint(25, 30)
    w = random.randint(25, 30)
    object_count = random.choice((3, 4))
    colors = random.sample(range(1, 10), object_count)

    first_dir_name = random.choice(tuple(DIRS))
    second_dir_name = random.choice([name for name in DIRS if name != first_dir_name])
    d1, d2 = DIRS[first_dir_name], DIRS[second_dir_name]
    a, b = random.sample(range(1, 5), 2)

    # Same direction/different distance and same distance/different direction are both
    # visible in every generated puzzle. The first object deliberately clips by one line.
    specs = [
        (colors[0], d1, a, True, True),
        (colors[1], d1, b, False, False),
        (colors[2], d2, a, False, False),
    ]
    if object_count == 4:
        extra_direction = random.choice(tuple(DIRS.values()))
        extra_distance = random.randint(1, 4)
        specs.append((colors[3], extra_direction, extra_distance, False, random.random() < 0.5))
    patches = []
    for color, direction, sealed, clipped, require_merge in specs:
        patch = _make_frame(color, direction, sealed, require_merge)
        if patch.shape[0] > 12 or patch.shape[1] > 12:
            return None
        patches.append((color, direction, sealed, clipped, patch))

    grid = np.zeros((h, w), dtype=int)
    input_occupied = np.zeros((h, w), dtype=bool)
    output_occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for color, direction, sealed, should_clip, patch in patches:
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            edge_distance = _motion_distance_to_edge(grid.shape, patch.shape, r, c, direction)
            if should_clip:
                if edge_distance != sealed - 1:
                    continue
            elif edge_distance < sealed + 1:
                continue
            if not _expanded_box_clear(input_occupied, r, c, r + ph - 1, c + pw - 1):
                continue

            local = np.argwhere(patch != BACKGROUND)
            source = local + np.array([r, c], dtype=int)
            dr, dc = direction
            shifted = source + np.array([dr * sealed, dc * sealed], dtype=int)
            in_bounds = (
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < h)
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < w)
            )
            visible = shifted[in_bounds]
            clipped_count = int((~in_bounds).sum())
            if should_clip:
                if clipped_count == 0 or len(visible) < 0.6 * len(source):
                    continue
            elif clipped_count != 0:
                continue
            vr0, vc0 = visible.min(axis=0)
            vr1, vc1 = visible.max(axis=0)
            vr0, vc0, vr1, vc1 = map(int, (vr0, vc0, vr1, vc1))
            if not _expanded_box_clear(output_occupied, vr0, vc0, vr1, vc1):
                continue

            # Keep different objects' old and new footprints visually separate. An
            # object's own old/new overlap is allowed and is the natural meaning of slide.
            if not _expanded_box_clear(output_occupied, r, c, r + ph - 1, c + pw - 1):
                continue
            if not _expanded_box_clear(input_occupied, vr0, vc0, vr1, vc1):
                continue
            chosen = (r, c, source, visible, clipped_count)
            break

        if chosen is None:
            return None

        r, c, source, visible, clipped_count = chosen
        region = grid[r : r + ph, c : c + pw]
        region[patch != BACKGROUND] = int(color)
        input_occupied[r : r + ph, c : c + pw] = True
        output_occupied[visible[:, 0], visible[:, 1]] = True
        placed.append(
            {
                "color": int(color),
                "direction": direction,
                "distance": int(sealed),
                "patch": patch,
                "source": source,
                "visible": visible,
                "clipped": int(clipped_count),
            }
        )

    expected = np.zeros_like(grid)
    for obj in placed:
        vis = obj["visible"]
        expected[vis[:, 0], vis[:, 1]] = obj["color"]

    # Semantic validation, not merely main(generate_input()) closure.
    if len({obj["direction"] for obj in placed}) < 2:
        return None
    if len({obj["distance"] for obj in placed}) < 2:
        return None
    if not (placed[0]["direction"] == placed[1]["direction"] and placed[0]["distance"] != placed[1]["distance"]):
        return None
    if not (placed[0]["distance"] == placed[2]["distance"] and placed[0]["direction"] != placed[2]["direction"]):
        return None
    if placed[0]["clipped"] <= 0 or any(obj["clipped"] for obj in placed[1:]):
        return None

    for obj in placed:
        analysis = _analyze_frame(grid, obj["color"])
        if analysis["direction"] != obj["direction"] or analysis["distance"] != obj["distance"]:
            return None
        walls = obj["patch"] == obj["color"]
        comps = _background_components(walls)
        # There is exactly one extra room, the door-connected one. Thus shifting by
        # all visible rooms would necessarily be one cell farther than the rule.
        if len(comps) != obj["distance"] + 1:
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _try_generate()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid door-and-room sliding puzzle")
