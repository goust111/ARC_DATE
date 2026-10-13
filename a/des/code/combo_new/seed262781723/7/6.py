from collections import deque

import numpy as np


DIR_VECTORS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}

PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
DIR_ORDER = ("right", "left", "up", "down")


def _components8(grid: np.ndarray, bg: int = 0) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append((color, np.array(coords, dtype=int)))
    return components


def _parse_arrow(coords: np.ndarray) -> tuple[str, tuple[int, int], int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[rs - r0, cs - c0] = True

    if h == 3 and w >= 4 and len(coords) == w + 1:
        top = np.flatnonzero(mask[0])
        middle = np.flatnonzero(mask[1])
        bottom = np.flatnonzero(mask[2])
        if (
            len(top) == 1
            and len(bottom) == 1
            and int(top[0]) == int(bottom[0])
            and len(middle) == w - 1
        ):
            root_c = int(top[0])
            if not np.array_equal(np.flatnonzero(~mask[1]), np.array([root_c])):
                raise AssertionError("horizontal arrow must have one root gap")
            if root_c == w - 2:
                return "right", (r0 + 1, c1), w - 2
            if root_c == 1:
                return "left", (r0 + 1, c0), w - 2

    if w == 3 and h >= 4 and len(coords) == h + 1:
        left = np.flatnonzero(mask[:, 0])
        middle = np.flatnonzero(mask[:, 1])
        right = np.flatnonzero(mask[:, 2])
        if (
            len(left) == 1
            and len(right) == 1
            and int(left[0]) == int(right[0])
            and len(middle) == h - 1
        ):
            root_r = int(left[0])
            if not np.array_equal(np.flatnonzero(~mask[:, 1]), np.array([root_r])):
                raise AssertionError("vertical arrow must have one root gap")
            if root_r == h - 2:
                return "down", (r1, c0 + 1), h - 2
            if root_r == 1:
                return "up", (r0, c0 + 1), h - 2

    raise AssertionError("foreground component is not a valid arrow")


def _square_bounds(tip: tuple[int, int], direction: str, length: int) -> tuple[int, int, int, int]:
    dr, dc = DIR_VECTORS[direction]
    landing_r = int(tip[0] + dr * length)
    landing_c = int(tip[1] + dc * length)
    if direction in ("right", "down"):
        return landing_r, landing_c, landing_r + length, landing_c + length
    return landing_r - length + 1, landing_c - length + 1, landing_r + 1, landing_c + 1


def _paint_square(out: np.ndarray, color: int, tip: tuple[int, int], direction: str, length: int) -> None:
    r0, c0, r1, c1 = _square_bounds(tip, direction, length)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(out.shape[0], r1)
    cc1 = min(out.shape[1], c1)
    if rr0 < rr1 and cc0 < cc1:
        out[rr0:rr1, cc0:cc1] = int(color)


def _arrow_cells(tip: tuple[int, int], direction: str, length: int) -> set[tuple[int, int]]:
    r, c = tip
    cells = {(int(r), int(c))}
    if direction == "right":
        root_c = c - 1
        cells.add((r - 1, root_c))
        cells.add((r + 1, root_c))
        for k in range(length):
            cells.add((r, root_c - 1 - k))
    elif direction == "left":
        root_c = c + 1
        cells.add((r - 1, root_c))
        cells.add((r + 1, root_c))
        for k in range(length):
            cells.add((r, root_c + 1 + k))
    elif direction == "up":
        root_r = r + 1
        cells.add((root_r, c - 1))
        cells.add((root_r, c + 1))
        for k in range(length):
            cells.add((root_r + 1 + k, c))
    else:
        assert direction == "down"
        root_r = r - 1
        cells.add((root_r, c - 1))
        cells.add((root_r, c + 1))
        for k in range(length):
            cells.add((root_r - 1 - k, c))
    return {(int(rr), int(cc)) for rr, cc in cells}


def _visible_square_cells(shape: tuple[int, int], tip: tuple[int, int], direction: str, length: int) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = _square_bounds(tip, direction, length)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h, r1)
    cc1 = min(w, c1)
    return {(r, c) for r in range(rr0, rr1) for c in range(cc0, cc1)}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _tip_ranges(h: int, w: int, direction: str, length: int, clipped: bool) -> tuple[range, range] | None:
    if direction == "right":
        r_lo, r_hi = 2, min(h - 3, h - length)
        if clipped:
            c_lo, c_hi = max(length + 2, w - 2 * length + 1), w - length - 1
        else:
            c_lo, c_hi = length + 2, w - 2 * length
    elif direction == "left":
        r_lo, r_hi = max(2, length - 1), h - 3
        if clipped:
            c_lo, c_hi = length, min(w - length - 3, 2 * length - 2)
        else:
            c_lo, c_hi = 2 * length - 1, w - length - 3
    elif direction == "up":
        c_lo, c_hi = max(2, length - 1), w - 3
        if clipped:
            r_lo, r_hi = length, min(h - length - 3, 2 * length - 2)
        else:
            r_lo, r_hi = 2 * length - 1, h - length - 3
    else:
        assert direction == "down"
        c_lo, c_hi = 2, min(w - 3, w - length)
        if clipped:
            r_lo, r_hi = max(length + 2, h - 2 * length + 1), h - length - 1
        else:
            r_lo, r_hi = length + 2, h - 2 * length

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _feasible_lengths(h: int, w: int, direction: str, clipped: bool) -> list[int]:
    lengths: list[int] = []
    for length in range(2, min(8, h, w)):
        if _tip_ranges(h, w, direction, length, clipped) is not None:
            lengths.append(length)
    return lengths


def _render_specs(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        _paint_square(
            out,
            int(spec["color"]),
            tuple(spec["tip"]),
            str(spec["direction"]),
            int(spec["length"]),
        )
    return out


def _permuted(items: list[object], rng: np.random.Generator) -> list[object]:
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def _choose_directions(count: int, rng: np.random.Generator) -> list[str]:
    base = list(rng.permutation(np.array(DIR_ORDER, dtype=object)))
    if count <= len(base):
        result = base[:count]
    else:
        extras = list(rng.choice(np.array(DIR_ORDER, dtype=object), size=count - len(base), replace=True))
        result = base + extras
    return [str(v) for v in _permuted(result, rng)]


def _choose_clip_flags(count: int, rng: np.random.Generator) -> list[bool]:
    base = [False] * (count // 2) + [True] * (count - count // 2)
    return [bool(v) for v in _permuted(base, rng)]


def _grid_size(count: int, rng: np.random.Generator) -> tuple[int, int]:
    low = 10 + 2 * count
    h = int(rng.integers(low, 29))
    w = int(rng.integers(low, 29))
    return h, w


def _place_one(
    shape: tuple[int, int],
    direction: str,
    clipped: bool,
    color: int,
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    h, w = shape
    lengths = _feasible_lengths(h, w, direction, clipped)
    if not lengths:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        length = int(lengths[int(rng.integers(len(lengths)))])
        ranges = _tip_ranges(h, w, direction, length, clipped)
        if ranges is None:
            continue
        rows, cols = ranges
        positions = [(r, c) for r in rows for c in cols]
        positions = [(int(r), int(c)) for r, c in _permuted(positions, rng)]
        for tip in positions:
            arrow = _arrow_cells(tip, direction, length)
            square = _visible_square_cells(shape, tip, direction, length)
            if not square:
                continue
            full_area = length * length
            if clipped and len(square) == full_area:
                continue
            if not clipped and len(square) != full_area:
                continue
            if any(cell in blocked_input for cell in arrow):
                continue
            if any(cell in blocked_output for cell in square):
                continue
            return {
                "color": color,
                "direction": direction,
                "length": length,
                "tip": tip,
                "arrow": arrow,
                "square": square,
                "clipped": clipped,
            }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 6))
    h, w = _grid_size(count, rng)
    colors = [int(v) for v in rng.choice(np.array(PALETTE), size=count, replace=False)]
    directions = _choose_directions(count, rng)
    clip_flags = _choose_clip_flags(count, rng)

    specs: list[dict[str, object]] = []
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()

    order = list(range(count))
    order.sort(key=lambda i: (not clip_flags[i], directions[i]))
    order = [int(i) for i in _permuted(order, rng)]

    for idx in order:
        spec = _place_one(
            (h, w),
            directions[idx],
            clip_flags[idx],
            colors[idx],
            blocked_input,
            blocked_output,
            rng,
        )
        if spec is None:
            return None
        specs.append(spec)
        blocked_input |= _halo(set(spec["arrow"]), (h, w))
        blocked_output |= _halo(set(spec["square"]), (h, w))

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        for r, c in set(spec["arrow"]):
            grid[r, c] = int(spec["color"])

    expected = _render_specs((h, w), specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each arrow in its pointing direction and replace it with a square whose size matches the tail length.

    Input structure:
    - Background is black.
    - Foreground consists of isolated single-color arrow shapes.
    - Each arrow has a 3-pixel triangular head pointing up, down, left, or right, plus a 1-pixel-wide tail of length L extending backward from the head root.
    - Different arrows do not touch each other.

    Rule:
    - Erase all arrow pixels.
    - For each arrow, recover its pointing direction from the 3-pixel head and recover tail length L from the arrow's bounding geometry.
    - Starting from the tip pixel, move L cells forward along the arrow direction to get a landing point.
    - Draw a solid L x L square in the arrow color, using that landing point as the top-left corner for right/down arrows or the bottom-right corner for left/up arrows, clipping at the grid boundary if needed.

    Key insight:
    - The asymmetric 3-pixel head encodes direction, while the tail length simultaneously determines both the launch distance and the size of the output square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for color, coords in _components8(g, bg=0):
        direction, tip, length = _parse_arrow(coords)
        _paint_square(out, int(color), tip, direction, int(length))
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
