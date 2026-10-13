import random
from collections import deque

import numpy as np


Color = type(
    "Color",
    (),
    {
        "BLACK": 0,
        "BLUE": 1,
        "RED": 2,
        "GREEN": 3,
        "YELLOW": 4,
        "GREY": 5,
        "PINK": 6,
        "ORANGE": 7,
        "TEAL": 8,
        "MAROON": 9,
    },
)


FOREGROUND_COLORS = [Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL]
GEN_RETRIES = 500
GRID_MIN = 13
GRID_MAX = 18
SPRITE_COUNT_RANGE = (4, 6)


SHAPES = (
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)], dtype=int),
    np.array([(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], dtype=int),
    np.array([(0, 2), (0, 1), (1, 1), (1, 0), (2, 0)], dtype=int),
)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return _round_half_up((r0 + r1) / 2.0), _round_half_up((c0 + c1) / 2.0)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(pts, dtype=int))
    return components


def _extract_sprites(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    maroon = np.argwhere(grid == Color.MAROON)
    assert maroon.shape == (1, 2), "expected exactly one maroon attractor pixel"
    attractor = int(maroon[0, 0]), int(maroon[0, 1])

    sprites = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (Color.BLACK, Color.MAROON)):
        for coords in _find_components_of_color(grid, color):
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return attractor, sprites


def _direction_from_center(center: tuple[int, int], attractor: tuple[int, int]) -> tuple[str, bool]:
    dr = int(attractor[0] - center[0])
    dc = int(attractor[1] - center[1])
    assert not (dr == 0 and dc == 0), "sprite center cannot coincide with the attractor"
    if abs(dr) >= abs(dc):
        assert dr != 0, "vertical tie requires non-zero row offset"
        return ("down" if dr > 0 else "up"), abs(dr) == abs(dc)
    return ("right" if dc > 0 else "left"), False


def _shift_to_edge(bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, _r1, _c1 = bbox
    if direction == "up":
        return r0
    if direction == "down":
        return h - 1 - bbox[2]
    if direction == "left":
        return c0
    assert direction == "right"
    return w - 1 - bbox[3]


def _moved_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    shift = _shift_to_edge(bbox, direction, shape)
    if direction == "up":
        delta = (-shift, 0)
    elif direction == "down":
        delta = (shift, 0)
    elif direction == "left":
        delta = (0, -shift)
    else:
        delta = (0, shift)
    moved = coords + np.array(delta, dtype=int)
    return moved, _bbox_from_coords(moved), shift


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _edge_distance(bbox: tuple[int, int, int, int], direction: str, grid_shape: tuple[int, int]) -> int:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    return {
        "up": int(r0),
        "down": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }[direction]


def _min_edge_distance(bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> int:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    return min(int(r0), int(c0), int(h - 1 - r1), int(w - 1 - c1))


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, pt)) for pt in coords}


def _touches_existing(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return True
    return False


def _mark(coords: np.ndarray, occupied: np.ndarray) -> None:
    occupied[coords[:, 0], coords[:, 1]] = True


def _sweep_set(coords: np.ndarray, direction: str, shift: int) -> set[tuple[int, int]]:
    if direction == "up":
        delta = (-1, 0)
    elif direction == "down":
        delta = (1, 0)
    elif direction == "left":
        delta = (0, -1)
    else:
        delta = (0, 1)
    cells = set()
    for t in range(shift + 1):
        dr = delta[0] * t
        dc = delta[1] * t
        for rr, cc in coords:
            cells.add((int(rr + dr), int(cc + dc)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite toward the attractor-chosen cardinal edge, then keep only the nearest and farthest landed sprites.

    Input structure:
    - A black background contains exactly one maroon single-pixel attractor.
    - It also contains 4-connected non-black, non-maroon sprites; each sprite is handled independently.
    - Each sprite's reference point is the rounded center of its bounding box.

    Rule:
    - For each sprite, compute the row/column offset from its bounding-box center to the attractor.
    - Quantize that offset to one of the four cardinal directions by the larger-magnitude axis; ties choose the vertical axis.
    - Translate the whole sprite in that direction until its bounding-box front reaches the grid edge.
    - After all sprites land, compute each landed bounding-box center's Manhattan distance to the attractor.
    - Recolor the unique nearest landed sprite blue and the unique farthest landed sprite red, erase every other sprite, and erase the attractor.

    Key insight:
    The input positions encode per-sprite gravity directions, but the output depends on the post-gravity distance ranking of the landed sprites.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    attractor, sprites = _extract_sprites(g)
    assert len(sprites) >= 2, "need at least two sprites to define nearest and farthest"

    landed = []
    occupied = np.zeros((h, w), dtype=bool)
    for sprite in sprites:
        center = _bbox_center(sprite["bbox"])
        direction, _tie = _direction_from_center(center, attractor)
        moved, moved_bbox, shift = _moved_coords(sprite["coords"], sprite["bbox"], direction, g.shape)
        assert shift > 0, "sprites must have a non-zero fall distance"
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h) & (0 <= moved[:, 1]) & (moved[:, 1] < w))
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "landed sprites must not overlap"
        occupied[moved[:, 0], moved[:, 1]] = True
        landed.append(
            {
                "coords": moved,
                "bbox": moved_bbox,
                "distance": _manhattan(_bbox_center(moved_bbox), attractor),
            }
        )

    distances = [item["distance"] for item in landed]
    assert len(distances) == len(set(distances)), "landed sprite distances must be unique"
    near_i = min(range(len(landed)), key=lambda i: landed[i]["distance"])
    far_i = max(range(len(landed)), key=lambda i: landed[i]["distance"])
    assert near_i != far_i

    out = np.zeros_like(g)
    out[landed[near_i]["coords"][:, 0], landed[near_i]["coords"][:, 1]] = Color.BLUE
    out[landed[far_i]["coords"][:, 0], landed[far_i]["coords"][:, 1]] = Color.RED
    return out


def _generate_attractor(h: int, w: int) -> tuple[int, int]:
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    random.shuffle(rows)
    random.shuffle(cols)
    center = (h // 2, w // 2)
    for r in rows:
        for c in cols:
            if (r, c) != center:
                return r, c
    raise RuntimeError("failed to place attractor")


def _tag_matches(tag: str, direction: str, is_tie: bool) -> bool:
    if tag.endswith("_tie"):
        return direction == tag[:-4] and is_tie
    return direction == tag and not is_tie


def _enumerate_candidates(
    shape: np.ndarray,
    tag: str,
    attractor: tuple[int, int],
    grid_shape: tuple[int, int],
    input_occ: np.ndarray,
    final_occ: np.ndarray,
    final_distances: set[int],
    existing_finals: list[set[tuple[int, int]]],
    existing_sweeps: list[set[tuple[int, int]]],
):
    h, w = grid_shape
    shp_h = int(shape[:, 0].max()) + 1
    shp_w = int(shape[:, 1].max()) + 1
    candidates = []

    for r0 in range(h - shp_h + 1):
        for c0 in range(w - shp_w + 1):
            coords = shape + np.array([r0, c0], dtype=int)
            if _touches_existing(coords, input_occ):
                continue
            if any(_manhattan((int(rr), int(cc)), attractor) <= 1 for rr, cc in coords):
                continue

            bbox = (r0, c0, r0 + shp_h - 1, c0 + shp_w - 1)
            center = _bbox_center(bbox)
            direction, is_tie = _direction_from_center(center, attractor)
            if not _tag_matches(tag, direction, is_tie):
                continue

            moved, moved_bbox, shift = _moved_coords(coords, bbox, direction, grid_shape)
            if shift <= 0:
                continue
            if _touches_existing(moved, final_occ):
                continue
            if tuple(attractor) in _coord_set(moved):
                continue

            final_distance = _manhattan(_bbox_center(moved_bbox), attractor)
            if final_distance in final_distances:
                continue

            sweep = _sweep_set(coords, direction, shift)
            moved_set = _coord_set(moved)
            conflict = False
            for other_final, other_sweep in zip(existing_finals, existing_sweeps):
                if sweep & other_final or other_sweep & moved_set:
                    conflict = True
                    break
            if conflict:
                continue

            candidates.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "direction": direction,
                    "is_tie": is_tie,
                    "final_coords": moved,
                    "final_bbox": moved_bbox,
                    "shift": shift,
                    "final_distance": final_distance,
                    "misleading": _edge_distance(bbox, direction, grid_shape) > _min_edge_distance(bbox, grid_shape),
                    "sweep": sweep,
                    "final_set": moved_set,
                }
            )
    return candidates


def _build_expected(grid_shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    near = min(placed, key=lambda item: item["final_distance"])
    far = max(placed, key=lambda item: item["final_distance"])
    out[near["final_coords"][:, 0], near["final_coords"][:, 1]] = Color.BLUE
    out[far["final_coords"][:, 0], far["final_coords"][:, 1]] = Color.RED
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        attractor, sprites = _extract_sprites(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    if not (4 <= len(sprites) <= 6):
        return False
    if attractor[0] in (0, h - 1) or attractor[1] in (0, w - 1):
        return False
    if attractor == (h // 2, w // 2):
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False

    directions = []
    landed = []
    occupied = np.zeros((h, w), dtype=bool)
    initial_distances = []
    nearest_edge_shortcut = True
    tie_seen = False
    sweeps = []
    finals = []

    for sprite in sprites:
        coords = sprite["coords"]
        if not (3 <= len(coords) <= 5):
            return False
        if any(_manhattan((int(rr), int(cc)), attractor) <= 1 for rr, cc in coords):
            return False

        center = _bbox_center(sprite["bbox"])
        direction, is_tie = _direction_from_center(center, attractor)
        directions.append(direction)
        tie_seen |= is_tie
        initial_distances.append(_manhattan(center, attractor))
        if _edge_distance(sprite["bbox"], direction, grid.shape) > _min_edge_distance(sprite["bbox"], grid.shape):
            nearest_edge_shortcut = False

        moved, moved_bbox, shift = _moved_coords(coords, sprite["bbox"], direction, grid.shape)
        if shift <= 0 or np.any(occupied[moved[:, 0], moved[:, 1]]):
            return False
        if tuple(attractor) in _coord_set(moved):
            return False
        occupied[moved[:, 0], moved[:, 1]] = True

        sweep = _sweep_set(coords, direction, shift)
        moved_set = _coord_set(moved)
        for other_final, other_sweep in zip(finals, sweeps):
            if sweep & other_final or other_sweep & moved_set:
                return False

        sweeps.append(sweep)
        finals.append(moved_set)
        landed.append(
            {
                "color": sprite["color"],
                "distance": _manhattan(_bbox_center(moved_bbox), attractor),
                "coords": moved,
            }
        )

    final_distances = [item["distance"] for item in landed]
    if len(final_distances) != len(set(final_distances)):
        return False
    if len(set(directions)) < 3:
        return False
    if nearest_edge_shortcut:
        return False

    initial_near = min(range(len(sprites)), key=lambda i: initial_distances[i])
    initial_far = max(range(len(sprites)), key=lambda i: initial_distances[i])
    final_near = min(range(len(landed)), key=lambda i: landed[i]["distance"])
    final_far = max(range(len(landed)), key=lambda i: landed[i]["distance"])
    if initial_near == final_near and initial_far == final_far:
        return False
    if tie_seen and not any(abs(attractor[0] - _bbox_center(sprite["bbox"])[0]) == abs(attractor[1] - _bbox_center(sprite["bbox"])[1]) for sprite in sprites):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    attractor = _generate_attractor(h, w)
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    colors = random.sample(FOREGROUND_COLORS, sprite_count)

    direction_tags = ["up", "down", "left", "right"]
    while len(direction_tags) < sprite_count:
        direction_tags.append(random.choice(["up", "down", "left", "right"]))
    random.shuffle(direction_tags)

    vertical_indices = [i for i, tag in enumerate(direction_tags) if tag in ("up", "down")]
    if vertical_indices and random.random() < 0.65:
        idx = random.choice(vertical_indices)
        direction_tags[idx] = f"{direction_tags[idx]}_tie"

    must_mislead_idx = random.randrange(sprite_count)

    grid = np.zeros((h, w), dtype=int)
    grid[attractor] = Color.MAROON
    input_occ = np.zeros((h, w), dtype=bool)
    final_occ = np.zeros((h, w), dtype=bool)
    final_distances: set[int] = set()
    existing_finals: list[set[tuple[int, int]]] = []
    existing_sweeps: list[set[tuple[int, int]]] = []
    placed = []

    for idx, (color, tag) in enumerate(zip(colors, direction_tags)):
        all_candidates = []
        shapes = list(SHAPES)
        random.shuffle(shapes)
        for shape in shapes:
            all_candidates.extend(
                _enumerate_candidates(
                    shape,
                    tag,
                    attractor,
                    grid.shape,
                    input_occ,
                    final_occ,
                    final_distances,
                    existing_finals,
                    existing_sweeps,
                )
            )

        if idx == must_mislead_idx:
            misleaders = [cand for cand in all_candidates if cand["misleading"]]
            if not misleaders:
                return None
            all_candidates = misleaders

        if not all_candidates:
            return None

        choice = random.choice(all_candidates)
        grid[choice["coords"][:, 0], choice["coords"][:, 1]] = color
        _mark(choice["coords"], input_occ)
        _mark(choice["final_coords"], final_occ)
        final_distances.add(choice["final_distance"])
        existing_finals.append(choice["final_set"])
        existing_sweeps.append(choice["sweep"])
        placed.append({**choice, "color": color})

    expected = _build_expected(grid.shape, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-distance ARC input")
