import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

SEED_COLORS = (4, 6, 7, 8, 9)
ALL_SIZES = [(h, w) for h in range(4, 9) for w in range(4, 9)]
FULL_SIZE_OPTIONS = [size for size in ALL_SIZES if (size[0] - 2) * (size[1] - 2) <= 20]

FRAME_COUNT_OPTIONS = (1, 2, 3)
FRAME_COUNT_WEIGHTS = (2, 3, 2)
GRID_MIN = 14
GRID_MAX = 30
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _right_of(direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return dc, -dr


def _gap_index(side: np.ndarray) -> int | None:
    assert int(side[0]) == GRAY and int(side[-1]) == GRAY
    internal = side[1:-1]
    missing = np.flatnonzero(internal != GRAY)
    if missing.size == 0:
        return None
    assert missing.size == 1
    idx = int(missing[0]) + 1
    assert int(side[idx]) == BLACK
    other = np.delete(internal, missing[0])
    assert np.all(other == GRAY)
    return idx


def _parse_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 4 and w >= 4

    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    assert int(np.count_nonzero(patch == GRAY)) == 2 * h + 2 * w - 5
    assert np.all(patch[~perimeter] != GRAY)

    gap_candidates = {
        "up": _gap_index(patch[0, :]),
        "right": _gap_index(patch[:, -1]),
        "down": _gap_index(patch[-1, :]),
        "left": _gap_index(patch[:, 0]),
    }
    present = [(name, idx) for name, idx in gap_candidates.items() if idx is not None]
    assert len(present) == 1
    gap_dir, idx = present[0]

    if gap_dir == "up":
        gap_local = (0, idx)
    elif gap_dir == "down":
        gap_local = (h - 1, idx)
    elif gap_dir == "left":
        gap_local = (idx, 0)
    else:
        gap_local = (idx, w - 1)

    expected_gray = perimeter.copy()
    expected_gray[gap_local] = False
    assert np.array_equal(patch == GRAY, expected_gray)

    interior = patch[1:-1, 1:-1]
    seed_mask = (interior != BLACK) & (interior != GRAY)
    assert int(np.count_nonzero(seed_mask)) == 1
    assert np.all((interior == BLACK) | seed_mask)
    seed_local = tuple(map(int, np.argwhere(seed_mask)[0]))
    seed_color = int(interior[seed_local])

    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "gap_dir": gap_dir,
        "gap": (r0 + gap_local[0], c0 + gap_local[1]),
        "seed": (r0 + 1 + seed_local[0], c0 + 1 + seed_local[1]),
        "seed_color": seed_color,
    }


def _quadrant(seed: tuple[int, int], cell: tuple[int, int], gap_dir: str) -> int:
    sr, sc = seed
    r, c = cell
    dr, dc = r - sr, c - sc
    fdr, fdc = DIRS[gap_dir]
    rdr, rdc = _right_of(gap_dir)
    front = dr * fdr + dc * fdc
    right = dr * rdr + dc * rdc

    if front > 0 and right >= 0:
        return 1
    if right > 0 and front <= 0:
        return 2
    if front < 0 and right <= 0:
        return 3
    if right < 0 and front >= 0:
        return 4
    raise AssertionError("only the seed cell lies on both dividers")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    frames = [_parse_frame(g, comp) for comp in _components_of_color(g, GRAY)]
    assert frames, "expected at least one gray frame"

    h_total, w_total = g.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        fh, fw = frame["shape"]
        area = (fh - 2) * (fw - 2)
        gap_r, gap_c = frame["gap"]
        seed_r, seed_c = frame["seed"]
        seed_color = frame["seed_color"]
        dr, dc = DIRS[frame["gap_dir"]]

        for step in range(area):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h_total and 0 <= cc < w_total):
                break
            out[rr, cc] = seed_color

        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if (rr, cc) == (seed_r, seed_c):
                    continue
                quadrant = _quadrant((seed_r, seed_c), (rr, cc), frame["gap_dir"])
                if quadrant == 1:
                    out[rr, cc] = seed_color
                elif quadrant == 2:
                    out[rr, cc] = RED
                elif quadrant == 3:
                    out[rr, cc] = BLUE
                else:
                    out[rr, cc] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored ray through each frame gap and recolor the frame interior by oriented quadrants.

    Input structure:
    - The background is black.
    - Several one-pixel-thick gray rectangular frames appear on the grid, and each frame has exactly one single-cell gap on one side.
    - Every frame encloses exactly one non-black, non-gray seed pixel somewhere inside the frame interior.

    Rule:
    - For each gray frame, detect which side contains the gap; that side defines the outward exit direction.
    - Starting at the gap cell itself, draw a straight ray outward in the seed color, with length equal to the interior area `(H-2) * (W-2)`, truncated by the grid boundary.
    - Recolor the interior of the frame using the seed row and seed column as the two dividers.
    - Treat the gap direction as "forward" and label the four quadrants clockwise as 1 to 4; fill quadrant 1 with the seed color, quadrant 2 with red, quadrant 3 with blue, and quadrant 4 with green.
    - Keep the seed pixel and the existing gray frame pixels unchanged.

    Key insight:
    - The missing frame cell does two jobs at once: it determines the ray direction outside the box and also fixes the rotated coordinate system that decides how the four interior quadrants are colored.
    """
    return _solve(grid)


def _placement_ranges(
    grid_h: int,
    grid_w: int,
    frame_h: int,
    frame_w: int,
    direction: str,
    area: int,
    want_truncated: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_low, row_high = 1, grid_h - frame_h - 1
    col_low, col_high = 1, grid_w - frame_w - 1
    if row_low > row_high or col_low > col_high:
        return None

    if direction == "up":
        if want_truncated:
            row_high = min(row_high, area - 2)
        else:
            row_low = max(row_low, area - 1)
    elif direction == "down":
        limit = grid_h - frame_h - area + 1
        if want_truncated:
            row_low = max(row_low, limit + 1)
        else:
            row_high = min(row_high, limit)
    elif direction == "left":
        if want_truncated:
            col_high = min(col_high, area - 2)
        else:
            col_low = max(col_low, area - 1)
    else:
        limit = grid_w - frame_w - area + 1
        if want_truncated:
            col_low = max(col_low, limit + 1)
        else:
            col_high = min(col_high, limit)

    if row_low > row_high or col_low > col_high:
        return None
    return (row_low, row_high), (col_low, col_high)


def _gap_abs(spec: dict) -> tuple[int, int]:
    r0, c0 = spec["top_left"]
    gr, gc = spec["gap_local"]
    return r0 + gr, c0 + gc


def _seed_abs(spec: dict) -> tuple[int, int]:
    r0, c0 = spec["top_left"]
    sr, sc = spec["seed_local"]
    return r0 + sr, c0 + sc


def _ray_cells(spec: dict, grid_h: int, grid_w: int) -> set[tuple[int, int]]:
    gap_r, gap_c = _gap_abs(spec)
    dr, dc = DIRS[spec["gap_dir"]]
    cells: set[tuple[int, int]] = set()
    for step in range(spec["area"]):
        rr = gap_r + dr * step
        cc = gap_c + dc * step
        if not (0 <= rr < grid_h and 0 <= cc < grid_w):
            break
        cells.add((rr, cc))
    return cells


def _bbox_cells(spec: dict) -> set[tuple[int, int]]:
    r0, c0 = spec["top_left"]
    h, w = spec["shape"]
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expand_cells(cells: set[tuple[int, int]], grid_h: int, grid_w: int, margin: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - margin), min(grid_h, r + margin + 1)):
            for cc in range(max(0, c - margin), min(grid_w, c + margin + 1)):
                expanded.add((rr, cc))
    return expanded


def _paint_input(grid: np.ndarray, spec: dict) -> None:
    r0, c0 = spec["top_left"]
    h, w = spec["shape"]
    grid[r0, c0 : c0 + w] = GRAY
    grid[r0 + h - 1, c0 : c0 + w] = GRAY
    grid[r0 : r0 + h, c0] = GRAY
    grid[r0 : r0 + h, c0 + w - 1] = GRAY
    grid[_gap_abs(spec)] = BLACK
    grid[_seed_abs(spec)] = spec["seed_color"]


def _frame_flags(rng: random.Random, count: int) -> list[bool]:
    flags = [bool(rng.getrandbits(1)) for _ in range(count)]
    if count >= 2:
        flags[0] = False
        flags[1] = True
        rng.shuffle(flags)
    return flags


def _sample_spec(rng: random.Random, grid_h: int, grid_w: int, want_truncated: bool) -> dict | None:
    size_options = ALL_SIZES if want_truncated else FULL_SIZE_OPTIONS
    for _ in range(PLACE_RETRIES):
        frame_h, frame_w = rng.choice(size_options)
        area = (frame_h - 2) * (frame_w - 2)
        direction = rng.choice(tuple(DIRS))
        ranges = _placement_ranges(grid_h, grid_w, frame_h, frame_w, direction, area, want_truncated)
        if ranges is None:
            continue
        (row_low, row_high), (col_low, col_high) = ranges
        r0 = rng.randint(row_low, row_high)
        c0 = rng.randint(col_low, col_high)

        if direction == "up":
            gap_local = (0, rng.randint(1, frame_w - 2))
        elif direction == "down":
            gap_local = (frame_h - 1, rng.randint(1, frame_w - 2))
        elif direction == "left":
            gap_local = (rng.randint(1, frame_h - 2), 0)
        else:
            gap_local = (rng.randint(1, frame_h - 2), frame_w - 1)

        spec = {
            "top_left": (r0, c0),
            "shape": (frame_h, frame_w),
            "gap_dir": direction,
            "gap_local": gap_local,
            "seed_local": (rng.randint(1, frame_h - 2), rng.randint(1, frame_w - 2)),
            "seed_color": rng.choice(SEED_COLORS),
            "area": area,
            "want_truncated": want_truncated,
        }
        return spec
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    min_side = min(GRID_MAX, GRID_MIN + 4 * (frame_count - 1))
    grid_h = rng.randint(min_side, GRID_MAX)
    grid_w = rng.randint(min_side, GRID_MAX)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    reserved: set[tuple[int, int]] = set()
    specs: list[dict] = []
    for want_truncated in _frame_flags(rng, frame_count):
        placed = False
        for _ in range(PLACE_RETRIES):
            spec = _sample_spec(rng, grid_h, grid_w, want_truncated)
            if spec is None:
                continue
            footprint = _bbox_cells(spec) | _ray_cells(spec, grid_h, grid_w)
            expanded = _expand_cells(footprint, grid_h, grid_w, margin=1)
            if expanded & reserved:
                continue
            reserved |= expanded
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        _paint_input(grid, spec)

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
