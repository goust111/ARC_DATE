import random

import numpy as np


BACKGROUND = 8
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
EDGES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 500
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(g: np.ndarray) -> list[dict]:
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    out = []

    for sr in range(height):
        for sc in range(width):
            color = int(g[sr, sc])
            if color == BACKGROUND or seen[sr, sc]:
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))

            out.append({"color": color, "coords": np.array(coords, dtype=int)})

    return out


def _tail_cells(coords: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    area = int(len(coords))

    touched = []
    if np.any(rows == 0):
        touched.append("top")
    if np.any(rows == height - 1):
        touched.append("bottom")
    if np.any(cols == 0):
        touched.append("left")
    if np.any(cols == width - 1):
        touched.append("right")

    if not touched:
        return []
    assert len(touched) == 1, "each edge object must touch exactly one grid edge"

    edge = touched[0]
    cells = []
    if edge == "top":
        col = (c0 + c1) // 2
        for r in range(r1 + 1, min(height, r1 + 1 + area)):
            cells.append((r, col))
    elif edge == "bottom":
        col = (c0 + c1) // 2
        for r in range(r0 - 1, max(-1, r0 - area - 1), -1):
            cells.append((r, col))
    elif edge == "left":
        row = (r0 + r1) // 2
        for c in range(c1 + 1, min(width, c1 + 1 + area)):
            cells.append((row, c))
    else:
        assert edge == "right"
        row = (r0 + r1) // 2
        for c in range(c0 - 1, max(-1, c0 - area - 1), -1):
            cells.append((row, c))

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Add an area-length same-color inward tail to every object touching one grid edge.

    Input structure:
    - The background is Azure.
    - Several monochrome 4-connected foreground objects are separated on the grid.
    - Some objects touch exactly one of the four grid edges; other objects are fully interior.

    Rule:
    - For each edge-touching foreground object, count its pixels to get area N.
    - From the object's side opposite the touched edge, draw a one-pixel-wide straight tail inward for N cells.
    - Align vertical tails to the object's bbox column midpoint and horizontal tails to its bbox row midpoint.
    - Truncate any tail that reaches the far grid boundary; leave non-edge objects unchanged.

    Key insight:
    - Edge contact supplies the tail direction, while object area independently supplies the tail length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)

    out = g.copy()
    for comp in _components(g):
        color = int(comp["color"])
        for r, c in _tail_cells(comp["coords"], height, width):
            out[r, c] = color
    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


VERTICAL_EDGE_MASKS = [
    _mask(["1", "1"]),
    _mask(["111"]),
    _mask(["111", "010"]),
    _mask(["010", "111", "010"]),
    _mask(["111", "111"]),
    _mask(["111", "111", "010"]),
    _mask(["11111", "00100"]),
    _mask(["11111", "01110", "00100"]),
    _mask(["11111", "11111"]),
    _mask(["11111", "11111", "00100"]),
    _mask(["11111", "11111", "01110"]),
]

INTERIOR_MASKS = [
    _mask(["11", "10"]),
    _mask(["010", "111"]),
    _mask(["11", "11"]),
    _mask(["010", "111", "010"]),
    _mask(["110", "011"]),
    _mask(["111", "010", "010"]),
]


def _oriented_mask(edge: str) -> np.ndarray:
    base = random.choice(VERTICAL_EDGE_MASKS)
    if edge == "top":
        return base.copy()
    if edge == "bottom":
        return np.flipud(base).copy()

    left = base.T.copy()
    if edge == "left":
        return left
    assert edge == "right"
    return np.fliplr(left).copy()


def _object_and_tail(mask: np.ndarray, row: int, col: int, edge: str, height: int, width: int):
    local = np.argwhere(mask)
    obj = [(int(row + r), int(col + c)) for r, c in local]
    coords = np.array(obj, dtype=int)
    tail = _tail_cells(coords, height, width)
    wanted = int(mask.sum())
    return obj, tail, len(tail) < wanted


def _dilate(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _edge_position_options(edge: str, mask: np.ndarray, height: int, width: int):
    mh, mw = map(int, mask.shape)
    if edge == "top":
        row = 0
        return [(row, col) for col in range(1, width - mw)]
    if edge == "bottom":
        row = height - mh
        return [(row, col) for col in range(1, width - mw)]
    if edge == "left":
        col = 0
        return [(row, col) for row in range(1, height - mh)]

    assert edge == "right"
    col = width - mw
    return [(row, col) for row in range(1, height - mh)]


def _place_edge_object(
    grid: np.ndarray,
    edge: str,
    color: int,
    blocked: set[tuple[int, int]],
    claimed: set[tuple[int, int]],
) -> bool | None:
    height, width = map(int, grid.shape)
    for _ in range(PLACE_ATTEMPTS):
        mask = _oriented_mask(edge)
        options = _edge_position_options(edge, mask, height, width)
        if not options:
            continue
        row, col = random.choice(options)
        obj, tail, truncated = _object_and_tail(mask, row, col, edge, height, width)
        cells = set(obj) | set(tail)
        if cells & blocked or cells & claimed:
            continue
        if any(grid[r, c] != BACKGROUND for r, c in obj):
            continue
        if any(grid[r, c] != BACKGROUND for r, c in tail):
            continue

        for r, c in obj:
            grid[r, c] = color
        claimed.update(cells)
        blocked.update(_dilate(cells, height, width))
        return truncated

    return None


def _place_interior_object(
    grid: np.ndarray,
    color: int,
    blocked: set[tuple[int, int]],
    claimed: set[tuple[int, int]],
) -> bool:
    height, width = map(int, grid.shape)
    for _ in range(PLACE_ATTEMPTS):
        mask = random.choice(INTERIOR_MASKS)
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        if random.choice([False, True]):
            mask = np.flipud(mask)
        if random.choice([False, True]):
            mask = mask.T

        mh, mw = map(int, mask.shape)
        if height - mh <= 2 or width - mw <= 2:
            continue
        row = random.randint(1, height - mh - 1)
        col = random.randint(1, width - mw - 1)
        obj = {
            (int(row + r), int(col + c))
            for r, c in np.argwhere(mask)
        }
        if obj & blocked or obj & claimed:
            continue
        if any(grid[r, c] != BACKGROUND for r, c in obj):
            continue

        for r, c in obj:
            grid[r, c] = color
        claimed.update(obj)
        blocked.update(_dilate(obj, height, width))
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    colors = random.sample(FOREGROUND_COLORS, random.randint(5, 7))
    random.shuffle(colors)

    blocked: set[tuple[int, int]] = set()
    claimed: set[tuple[int, int]] = set()
    truncation_seen = False
    full_tail_seen = False

    edge_order = list(EDGES)
    random.shuffle(edge_order)
    for i, edge in enumerate(edge_order):
        truncated = _place_edge_object(grid, edge, colors[i], blocked, claimed)
        if truncated is None:
            return None
        truncation_seen |= bool(truncated)
        full_tail_seen |= not bool(truncated)

    for color in colors[4:]:
        if not _place_interior_object(grid, color, blocked, claimed):
            return None

    if not (truncation_seen and full_tail_seen):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
