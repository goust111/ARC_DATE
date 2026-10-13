import random

import numpy as np


BG = 0
YELLOW = 4
GEN_ATTEMPTS = 200
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touching_edges(coords: np.ndarray, h: int, w: int) -> set[str]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    edges: set[str] = set()
    if np.any(rows == 0):
        edges.add("top")
    if np.any(rows == h - 1):
        edges.add("bottom")
    if np.any(cols == 0):
        edges.add("left")
    if np.any(cols == w - 1):
        edges.add("right")
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each foreground object according to how many grid edges it touches.

    0 touched edges: put one yellow pixel at the component bbox center.
    1 touched edge: paint a 1-pixel yellow line on the touched side of the object.
    2 touched edges: paint the four corners of the component bbox yellow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for coords in _components(g):
        rows = coords[:, 0]
        cols = coords[:, 1]
        min_r = int(rows.min())
        max_r = int(rows.max())
        min_c = int(cols.min())
        max_c = int(cols.max())
        touched = _touching_edges(coords, h, w)
        touch_count = len(touched)
        assert touch_count in (0, 1, 2), "objects must touch 0, 1, or 2 grid edges"

        if touch_count == 0:
            out[(min_r + max_r) // 2, (min_c + max_c) // 2] = YELLOW
            continue

        if touch_count == 1:
            edge = next(iter(touched))
            if edge == "top":
                mask = rows == min_r
            elif edge == "bottom":
                mask = rows == max_r
            elif edge == "left":
                mask = cols == min_c
            else:
                mask = cols == max_c
            line = coords[mask]
            out[line[:, 0], line[:, 1]] = YELLOW
            continue

        for r, c in ((min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)):
            out[r, c] = YELLOW

    return out


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    reserved[r0:r1, c0:c1] = True


def _randint(rng: random.Random, low: int, high: int) -> int | None:
    if high < low:
        return None
    return rng.randint(low, high)


def _place_corner(
    rng: random.Random,
    reserved: np.ndarray,
    corner: str,
) -> tuple[int, int, int, int] | None:
    h, w = reserved.shape
    for _ in range(40):
        height = rng.randint(2, min(5, h - 2))
        width = rng.randint(2, min(5, w - 2))
        if corner == "tl":
            top, left = 0, 0
        elif corner == "tr":
            top, left = 0, w - width
        elif corner == "bl":
            top, left = h - height, 0
        else:
            top, left = h - height, w - width
        if _can_place(reserved, top, left, height, width):
            _reserve(reserved, top, left, height, width)
            return top, left, height, width
    return None


def _place_edge(
    rng: random.Random,
    reserved: np.ndarray,
    edge: str,
) -> tuple[int, int, int, int] | None:
    h, w = reserved.shape
    for _ in range(60):
        if edge in ("top", "bottom"):
            height = rng.randint(2, min(4, h - 2))
            width = rng.randint(3, min(6, w - 2))
            left = _randint(rng, 1, w - width - 1)
            if left is None:
                continue
            top = 0 if edge == "top" else h - height
        else:
            height = rng.randint(3, min(6, h - 2))
            width = rng.randint(2, min(4, w - 2))
            top = _randint(rng, 1, h - height - 1)
            if top is None:
                continue
            left = 0 if edge == "left" else w - width
        if _can_place(reserved, top, left, height, width):
            _reserve(reserved, top, left, height, width)
            return top, left, height, width
    return None


def _place_inner(
    rng: random.Random,
    reserved: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = reserved.shape
    for _ in range(80):
        height = rng.randint(3, min(6, h - 2))
        width = rng.randint(3, min(6, w - 2))
        top = _randint(rng, 1, h - height - 1)
        left = _randint(rng, 1, w - width - 1)
        if top is None or left is None:
            continue
        if _can_place(reserved, top, left, height, width):
            _reserve(reserved, top, left, height, width)
            return top, left, height, width
    return None


def _validate_input(grid: np.ndarray) -> bool:
    components = _components(grid)
    if len(components) < 3:
        return False
    counts = {0: 0, 1: 0, 2: 0}
    h, w = grid.shape
    colors: set[int] = set()
    for coords in components:
        color = int(grid[coords[0, 0], coords[0, 1]])
        if color in colors or color == YELLOW:
            return False
        colors.add(color)
        touch_count = len(_touching_edges(coords, h, w))
        if touch_count not in counts:
            return False
        counts[touch_count] += 1
    return all(counts[k] >= 1 for k in counts) and int(np.count_nonzero(grid == BG)) > grid.size // 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 22)
    w = rng.randint(14, 22)
    reserved = np.zeros((h, w), dtype=bool)

    counts = {0: rng.randint(1, 2), 1: rng.randint(1, 2), 2: rng.randint(1, 2)}
    total = sum(counts.values())
    colors = rng.sample(PALETTE, total)
    placements: list[tuple[int, int, int, int]] = []

    corners = ["tl", "tr", "bl", "br"]
    rng.shuffle(corners)
    for corner in corners[: counts[2]]:
        rect = _place_corner(rng, reserved, corner)
        if rect is None:
            return None
        placements.append(rect)

    edges = ["top", "bottom", "left", "right"]
    for _ in range(counts[1]):
        rect = _place_edge(rng, reserved, rng.choice(edges))
        if rect is None:
            return None
        placements.append(rect)

    for _ in range(counts[0]):
        rect = _place_inner(rng, reserved)
        if rect is None:
            return None
        placements.append(rect)

    rng.shuffle(placements)
    grid = np.zeros((h, w), dtype=int)
    for color, (top, left, height, width) in zip(colors, placements):
        grid[top : top + height, left : left + width] = color

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
