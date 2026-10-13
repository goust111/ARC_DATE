from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE_COLORS = tuple(c for c in range(10) if c not in {BLACK, RED, GREY})
DIRECTIONS = ("up", "down", "left", "right")
DIR_TO_SIDE = {"up": "top", "down": "bottom", "left": "left", "right": "right"}

GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
ROOM_COUNT_RANGE = (2, 5)
ROOM_HEIGHT_RANGE = (4, 9)
ROOM_WIDTH_RANGE = (4, 10)
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 150


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
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _resize_nearest(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = tile.shape
    row_idx = (np.arange(out_h) * src_h) // out_h
    col_idx = (np.arange(out_w) * src_w) // out_w
    return tile[row_idx[:, None], col_idx[None, :]]


def _extract_templates(grid: np.ndarray) -> dict[str, np.ndarray]:
    template_mask = ~np.isin(grid, (BLACK, RED, GREY))
    templates: dict[str, np.ndarray] = {}
    for comp in _components(template_mask):
        r0, c0, r1, c1 = _bbox(comp)
        edges = []
        if r0 == 0:
            edges.append("top")
        if r1 == grid.shape[0] - 1:
            edges.append("bottom")
        if c0 == 0:
            edges.append("left")
        if c1 == grid.shape[1] - 1:
            edges.append("right")
        if len(edges) != 1:
            continue
        side = edges[0]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(template_mask[r0 : r1 + 1, c0 : c1 + 1]), "template must fill its bbox"
        templates[side] = patch.copy()
    assert set(templates) == {"top", "bottom", "left", "right"}, "expected one template per edge"
    return templates


def _room_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    passable = grid != GREY
    rooms = []
    for comp in _components(passable):
        if _touches_boundary(comp, grid.shape):
            continue
        vals = {int(grid[r, c]) for r, c in comp}
        if not vals <= {BLACK, RED}:
            continue
        red_count = sum(int(grid[r, c]) == RED for r, c in comp)
        if red_count == 0:
            continue
        rooms.append(comp)
    assert rooms, "expected at least one enclosed room"
    return rooms


def _arrow_direction(grid: np.ndarray, room: list[tuple[int, int]]) -> str:
    reds = [(r, c) for r, c in room if int(grid[r, c]) == RED]
    assert 3 <= len(reds) <= 4, "arrow should contain 3 or 4 red cells"
    rows = [r for r, _c in reds]
    cols = [c for _r, c in reds]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    candidates: list[str] = []
    if max_c > min_c and cols.count(max_c) == 1 and cols.count(min_c) >= 2:
        candidates.append("right")
    if max_c > min_c and cols.count(min_c) == 1 and cols.count(max_c) >= 2:
        candidates.append("left")
    if max_r > min_r and rows.count(min_r) == 1 and rows.count(max_r) >= 2:
        candidates.append("up")
    if max_r > min_r and rows.count(max_r) == 1 and rows.count(min_r) >= 2:
        candidates.append("down")

    assert len(candidates) == 1, "arrow direction must be unambiguous"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the edge template selected by the red arrow direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates = _extract_templates(g)
    out = g.copy()

    for room in _room_components(g):
        r0, c0, r1, c1 = _bbox(room)
        direction = _arrow_direction(g, room)
        template = templates[DIR_TO_SIDE[direction]]
        scaled = _resize_nearest(template, r1 - r0 + 1, c1 - c0 + 1)
        for r, c in room:
            out[r, c] = int(scaled[r - r0, c - c0])

    return out


def _template_pattern(size: int, style: str, colors: list[int]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=int)
    if style == "checker":
        for r in range(size):
            for c in range(size):
                arr[r, c] = colors[(r + c) % 2]
    elif style == "stripes":
        for c in range(size):
            arr[:, c] = colors[c % len(colors)]
    elif style == "bands":
        for r in range(size):
            arr[r, :] = colors[r % len(colors)]
    elif style == "diag":
        for r in range(size):
            for c in range(size):
                arr[r, c] = colors[(r + 2 * c) % len(colors)]
    elif style == "frame":
        arr[:, :] = colors[0]
        arr[1:-1, 1:-1] = colors[1]
        if size == 4:
            arr[1:3, 1:3] = colors[1]
            arr[2 - 1, 2 - 1] = colors[2]
            arr[1, 2] = colors[2]
            arr[2, 1] = colors[2]
        else:
            arr[1, 1] = colors[2]
    else:  # quadrants
        rmid = size // 2
        cmid = size // 2
        arr[:rmid, :cmid] = colors[0]
        arr[:rmid, cmid:] = colors[1]
        arr[rmid:, :cmid] = colors[1]
        arr[rmid:, cmid:] = colors[2]
        if size % 2 == 1:
            arr[rmid, :] = colors[0]
            arr[:, cmid] = colors[2]
    return arr


def _make_templates() -> dict[str, np.ndarray]:
    sides = ["top", "bottom", "left", "right"]
    styles = ["checker", "stripes", "bands", "diag", "frame", "quadrants"]
    used: set[bytes] = set()
    templates: dict[str, np.ndarray] = {}
    for side in sides:
        for _ in range(100):
            size = random.choice((3, 4))
            style = random.choice(styles)
            cols = random.sample(PALETTE_COLORS, 3)
            tile = _template_pattern(size, style, cols)
            sig = tile.tobytes()
            if sig in used:
                continue
            used.add(sig)
            templates[side] = tile
            break
        else:
            raise AssertionError("failed to build unique templates")
    return templates


def _place_templates(grid: np.ndarray, templates: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    h, w = grid.shape
    occupied: set[tuple[int, int]] = set()

    top = templates["top"]
    c0 = (w - top.shape[1]) // 2
    grid[0 : top.shape[0], c0 : c0 + top.shape[1]] = top
    for r in range(top.shape[0]):
        for c in range(c0, c0 + top.shape[1]):
            occupied.add((r, c))

    bottom = templates["bottom"]
    c0 = (w - bottom.shape[1]) // 2
    r0 = h - bottom.shape[0]
    grid[r0:h, c0 : c0 + bottom.shape[1]] = bottom
    for r in range(r0, h):
        for c in range(c0, c0 + bottom.shape[1]):
            occupied.add((r, c))

    left = templates["left"]
    r0 = (h - left.shape[0]) // 2
    grid[r0 : r0 + left.shape[0], 0 : left.shape[1]] = left
    for r in range(r0, r0 + left.shape[0]):
        for c in range(left.shape[1]):
            occupied.add((r, c))

    right = templates["right"]
    r0 = (h - right.shape[0]) // 2
    c0 = w - right.shape[1]
    grid[r0 : r0 + right.shape[0], c0:w] = right
    for r in range(r0, r0 + right.shape[0]):
        for c in range(c0, w):
            occupied.add((r, c))

    return occupied


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _make_room_mask(height: int, width: int) -> np.ndarray:
    for _ in range(100):
        mask = np.ones((height, width), dtype=bool)
        notch_count = random.choice((0, 1, 1, 2, 2))
        corners = ["tl", "tr", "bl", "br"]
        random.shuffle(corners)
        for corner in corners[:notch_count]:
            max_h = max(1, height - 2)
            max_w = max(1, width - 2)
            notch_h = random.randint(1, max_h)
            notch_w = random.randint(1, max_w)
            if corner == "tl":
                mask[:notch_h, :notch_w] = False
            elif corner == "tr":
                mask[:notch_h, width - notch_w :] = False
            elif corner == "bl":
                mask[height - notch_h :, :notch_w] = False
            else:
                mask[height - notch_h :, width - notch_w :] = False

        if not _connected(mask):
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        area = int(mask.sum())
        if area < (height * width * 3) // 5:
            continue
        return mask
    raise AssertionError("failed to make room mask")


ARROW_VARIANTS: dict[str, list[list[tuple[int, int]]]] = {
    "right": [
        [(0, 0), (1, 1), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
    ],
    "left": [
        [(0, 1), (1, 0), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "up": [
        [(0, 1), (1, 0), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
    ],
    "down": [
        [(0, 0), (0, 2), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
    ],
}


def _variant_shape(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _place_arrow(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    best: list[tuple[int, int, list[tuple[int, int]]]] = []
    fallback: list[tuple[int, int, list[tuple[int, int]]]] = []
    for cells in random.sample(ARROW_VARIANTS[direction], k=len(ARROW_VARIANTS[direction])):
        ah, aw = _variant_shape(cells)
        for r0 in range(mask.shape[0] - ah + 1):
            for c0 in range(mask.shape[1] - aw + 1):
                placed = [(r0 + r, c0 + c) for r, c in cells]
                if not all(mask[r, c] for r, c in placed):
                    continue
                fallback.append((r0, c0, placed))
                if np.all(mask[r0 : r0 + ah, c0 : c0 + aw]):
                    best.append((r0, c0, placed))
    choices = best if best else fallback
    assert choices, "room must be able to contain its arrow"
    return random.choice(choices)[2]


def _room_geometry(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    interior = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    walls: set[tuple[int, int]] = set()
    for r, c in interior:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in interior:
                walls.add((nr, nc))
    return interior, walls


def _expand_buffer(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_room(
    grid: np.ndarray,
    occupancy: set[tuple[int, int]],
    direction: str,
) -> bool:
    height = random.randint(*ROOM_HEIGHT_RANGE)
    width = random.randint(*ROOM_WIDTH_RANGE)
    mask = _make_room_mask(height, width)
    arrow_cells = _place_arrow(mask, direction)
    interior_rel, walls_rel = _room_geometry(mask)

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(2, grid.shape[0] - height - 3)
        c0 = random.randint(2, grid.shape[1] - width - 3)

        interior_abs = {(r0 + r, c0 + c) for r, c in interior_rel}
        walls_abs = {(r0 + r, c0 + c) for r, c in walls_rel}
        footprint = interior_abs | walls_abs
        if any(
            r < 1 or c < 1 or r >= grid.shape[0] - 1 or c >= grid.shape[1] - 1
            for r, c in footprint
        ):
            continue

        buffer_cells = _expand_buffer(footprint)
        if any(cell in occupancy for cell in buffer_cells):
            continue

        for r, c in walls_abs:
            grid[r, c] = GREY
        for r, c in {(r0 + r, c0 + c) for r, c in arrow_cells}:
            grid[r, c] = RED

        occupancy |= buffer_cells
        return True

    return False


def _balanced_directions(room_count: int) -> list[str]:
    base = list(DIRECTIONS)
    random.shuffle(base)
    dirs = [base[i % 4] for i in range(room_count)]
    random.shuffle(dirs)
    return dirs


def _valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = np.full((h, w), BLACK, dtype=int)

        templates = _make_templates()
        occupancy = _expand_buffer(_place_templates(grid, templates))

        room_count = random.randint(*ROOM_COUNT_RANGE)
        directions = _balanced_directions(room_count)
        random.shuffle(directions)

        ok = True
        for direction in directions:
            if not _place_room(grid, occupancy, direction):
                ok = False
                break
        if not ok:
            continue
        if not _valid_input(grid):
            continue
        return grid

    raise AssertionError("generate_input failed after retries")
