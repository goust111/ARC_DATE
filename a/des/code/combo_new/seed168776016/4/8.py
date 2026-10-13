import random
from collections import Counter

import numpy as np


GRID_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 400
ROOM_COUNT_OPTIONS = (1, 2, 3, 4)
ROOM_SIZE_RANGE = (5, 9)
ROOM_PADDING = 1
DIRECTIONS = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _door_side(door: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = door
    r0, c0, r1, c1 = bbox
    sides: list[str] = []
    if r == r0:
        sides.append("top")
    if r == r1:
        sides.append("bottom")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    if len(sides) != 1:
        raise AssertionError("door must lie on exactly one wall side")
    return sides[0]


def _extract_rooms(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    counts = Counter(map(int, g.flat))
    non_bg = [color for color in counts if color != bg]
    if not non_bg:
        raise AssertionError("expected at least one wall color")
    wall = max(non_bg, key=lambda color: counts[color])

    rooms: list[dict[str, object]] = []
    for comp in _components(g == wall):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            raise AssertionError("room bbox must be at least 5x5")

        border = _border_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set.issubset(border):
            raise AssertionError("wall component must stay on its bounding-box border")
        if len(comp_set) != len(border) - 1:
            raise AssertionError("room wall must be a rectangle with exactly one missing border pixel")

        door_cells = [cell for cell in border if int(g[cell]) != wall]
        if len(door_cells) != 1:
            raise AssertionError("expected exactly one door cell on the border")
        door = door_cells[0]
        if int(g[door]) != bg:
            raise AssertionError("door must stay background-colored")

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((interior != bg) & (interior != wall))
        if seed_coords.shape[0] != 1:
            raise AssertionError("expected exactly one non-wall seed inside each room")
        sr, sc = map(int, seed_coords[0])
        sr += r0 + 1
        sc += c0 + 1

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door": door,
                "door_side": _door_side(door, (r0, c0, r1, c1)),
                "seed": (sr, sc),
                "seed_color": int(g[sr, sc]),
            }
        )

    if not rooms:
        raise AssertionError("expected at least one room")

    rooms.sort(key=lambda room: tuple(room["bbox"]))
    return bg, wall, rooms


def _apply_growth(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, _wall, rooms = _extract_rooms(g)
    out = g.copy()

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        sr, sc = room["seed"]
        color = int(room["seed_color"])
        door_side = room["door_side"]

        if door_side == "top":
            out[sr:r1, c0 + 1 : c1] = color
        elif door_side == "bottom":
            out[r0 + 1 : sr + 1, c0 + 1 : c1] = color
        elif door_side == "left":
            out[r0 + 1 : r1, sc:c1] = color
        elif door_side == "right":
            out[r0 + 1 : r1, c0 + 1 : sc + 1] = color
        else:
            raise AssertionError("unknown door side")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room's seed into a full-width or full-height block pointing away from the room's door.

    Input structure:
    - The background is the most common color.
    - Several one-cell-thick rectangular wall frames share one wall color, and
      each frame has exactly one missing wall pixel that acts as a door.
    - Inside every room, all pixels are background except for one colored seed
      pixel touching one inner wall.

    Rule:
    - Treat each wall connected component as one room and recover its bounding
      box, unique door position, and unique interior seed.
    - Keep the walls, the door gap, and everything outside the rooms unchanged.
    - If the door is on the top or bottom wall, fill the room across its full
      interior width from the seed's row away from the door until the opposite
      wall.
    - If the door is on the left or right wall, fill the room across its full
      interior height from the seed's column away from the door until the
      opposite wall.

    Key insight:
    - The seed does not grow toward the opening; the opening only chooses the
      opposite growth direction, and the growth produces a rectangular slab
      rather than a one-pixel ray.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    return _apply_growth(g)


def _boxes_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(
    box: tuple[int, int, int, int], pad: int
) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _noncorner_inner_index(lo: int, hi: int, rng: random.Random) -> int:
    inner_lo = lo + 2
    inner_hi = hi - 2
    if inner_lo <= inner_hi:
        return rng.randint(inner_lo, inner_hi)
    return rng.randint(lo + 1, hi - 1)


def _draw_room_border(
    grid: np.ndarray, box: tuple[int, int, int, int], wall: int
) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall


def _door_cell(
    box: tuple[int, int, int, int], side: str, rng: random.Random
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    if side == "right":
        return rng.randint(r0 + 1, r1 - 1), c1
    raise AssertionError("unknown side")


def _seed_cell(
    box: tuple[int, int, int, int], side: str, rng: random.Random
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0 + 1, _noncorner_inner_index(c0, c1, rng)
    if side == "bottom":
        return r1 - 1, _noncorner_inner_index(c0, c1, rng)
    if side == "left":
        return _noncorner_inner_index(r0, r1, rng), c0 + 1
    if side == "right":
        return _noncorner_inner_index(r0, r1, rng), c1 - 1
    raise AssertionError("unknown side")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rooms = rng.choice(ROOM_COUNT_OPTIONS)
    min_side = {1: 10, 2: 14, 3: 18, 4: 22}[n_rooms]
    height = rng.randint(min_side, GRID_LIMIT)
    width = rng.randint(min_side, GRID_LIMIT)

    palette = rng.sample(range(10), 2 + n_rooms)
    bg = palette[0]
    wall = palette[1]
    seed_colors = palette[2:]

    grid = np.full((height, width), bg, dtype=int)
    directions = rng.sample(DIRECTIONS, n_rooms)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for i in range(n_rooms):
        box = None
        for _ in range(150):
            room_h = rng.randint(ROOM_SIZE_RANGE[0], min(ROOM_SIZE_RANGE[1], height - 2))
            room_w = rng.randint(ROOM_SIZE_RANGE[0], min(ROOM_SIZE_RANGE[1], width - 2))
            top = rng.randint(1, height - room_h - 1)
            left = rng.randint(1, width - room_w - 1)
            candidate = (top, left, top + room_h - 1, left + room_w - 1)
            expanded = _expanded_box(candidate, ROOM_PADDING)
            if any(_boxes_intersect(expanded, _expanded_box(other, ROOM_PADDING)) for other in placed_boxes):
                continue
            box = candidate
            break
        if box is None:
            return None

        _draw_room_border(grid, box, wall)
        door = _door_cell(box, directions[i], rng)
        grid[door] = bg
        seed_side = rng.choice(DIRECTIONS)
        seed = _seed_cell(box, seed_side, rng)
        grid[seed] = seed_colors[i]
        placed_boxes.append(box)

    counts = Counter(map(int, grid.flat))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(out.shape) > GRID_LIMIT or max(grid.shape) > GRID_LIMIT:
        return None
    if int(np.count_nonzero(out != grid)) < n_rooms:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth input")
