import numpy as np


BG_COLOR = 0
CROSS_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
ODD_SIZES = np.arange(13, 30, 2, dtype=int)
MAX_SEEDS_PER_QUADRANT = 4
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_cross(grid: np.ndarray) -> tuple[int, int, int, int]:
    height, width = map(int, grid.shape)
    bg = _most_common_color(grid)
    candidates: list[tuple[int, int, int]] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((color, int(rows[0]), int(cols[0])))

    assert len(candidates) == 1, "expected one full-row/full-column cross"
    cross_color, mid_r, mid_c = candidates[0]
    assert 0 < mid_r < height - 1 and 0 < mid_c < width - 1, "cross must split the grid"
    return int(cross_color), int(mid_r), int(mid_c), int(bg)


def _quadrant_specs(height: int, width: int, mid_r: int, mid_c: int):
    return [
        ("TL", 0, mid_r, 0, mid_c, -1, -1),
        ("TR", 0, mid_r, mid_c + 1, width, -1, 1),
        ("BL", mid_r + 1, height, 0, mid_c, 1, -1),
        ("BR", mid_r + 1, height, mid_c + 1, width, 1, 1),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each quadrant's colored seeds outward along the diagonal away from the central cross.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column form a central cross that splits the grid into four quadrants.
    - Each quadrant contains one or more isolated single-pixel seeds of one color, and the seeds do not touch the cross.

    Rule:
    - Count how many seed pixels lie in each quadrant.
    - For every seed, draw a same-colored diagonal ray pointing away from the center: up-left, up-right, down-left, or down-right according to the quadrant.
    - The ray length equals that quadrant's seed count, includes the seed itself, and is truncated at the grid edge.

    Key insight:
    - The central cross reveals both the quadrant boundaries and the outward diagonal direction, while the number of seeds inside a quadrant sets the shared ray length for that quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    cross_color, mid_r, mid_c, bg = _detect_cross(g)
    out = g.copy()

    for _name, r0, r1, c0, c1, dr, dc in _quadrant_specs(height, width, mid_r, mid_c):
        window = g[r0:r1, c0:c1]
        seed_mask = (window != bg) & (window != cross_color)
        seed_colors = np.unique(window[seed_mask])
        assert seed_colors.size == 1, "expected exactly one seed color per quadrant"

        seed_color = int(seed_colors[0])
        local_coords = np.argwhere(window == seed_color)
        ray_length = int(local_coords.shape[0])
        assert ray_length >= 1, "expected at least one seed in each quadrant"

        for local_r, local_c in local_coords:
            start_r = int(r0 + local_r)
            start_c = int(c0 + local_c)
            for step in range(ray_length):
                rr = int(start_r + step * dr)
                cc = int(start_c + step * dc)
                if not (0 <= rr < height and 0 <= cc < width):
                    break
                if rr == mid_r or cc == mid_c:
                    break
                out[rr, cc] = seed_color

    return out


def _ray_capacity(r: int, c: int, dr: int, dc: int, height: int, width: int) -> int:
    row_room = r if dr < 0 else height - 1 - r
    col_room = c if dc < 0 else width - 1 - c
    return 1 + min(int(row_room), int(col_room))


def _diag_key(r: int, c: int, dr: int, dc: int) -> tuple[str, int]:
    if dr == dc:
        return ("diff", int(r - c))
    return ("sum", int(r + c))


def _candidate_positions(name: str, mid_r: int, mid_c: int, height: int, width: int) -> list[tuple[int, int]]:
    if name == "TL":
        rows = range(0, mid_r - 1)
        cols = range(0, mid_c - 1)
    elif name == "TR":
        rows = range(0, mid_r - 1)
        cols = range(mid_c + 2, width)
    elif name == "BL":
        rows = range(mid_r + 2, height)
        cols = range(0, mid_c - 1)
    else:
        rows = range(mid_r + 2, height)
        cols = range(mid_c + 2, width)
    return [(int(r), int(c)) for r in rows for c in cols]


def _is_compatible(
    pos: tuple[int, int],
    seeds: list[tuple[int, int]],
    used_diagonals: set[tuple[str, int]],
    dr: int,
    dc: int,
) -> bool:
    r, c = pos
    if _diag_key(r, c, dr, dc) in used_diagonals:
        return False
    for other_r, other_c in seeds:
        if max(abs(int(r - other_r)), abs(int(c - other_c))) <= 1:
            return False
    return True


def _place_quadrant_seeds(
    candidates: list[tuple[int, int]],
    count: int,
    dr: int,
    dc: int,
    height: int,
    width: int,
    want_truncated: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    truncated = [
        pos for pos in candidates if _ray_capacity(pos[0], pos[1], dr, dc, height, width) < count
    ]
    full = [
        pos for pos in candidates if _ray_capacity(pos[0], pos[1], dr, dc, height, width) >= count
    ]

    for _ in range(80):
        seeds: list[tuple[int, int]] = []
        used_diagonals: set[tuple[str, int]] = set()

        first_pool = list(candidates)
        if count > 1:
            if want_truncated and truncated:
                first_pool = list(truncated)
            elif (not want_truncated) and full:
                first_pool = list(full)
        rng.shuffle(first_pool)

        chosen_first = None
        for pos in first_pool:
            if _is_compatible(pos, seeds, used_diagonals, dr, dc):
                chosen_first = pos
                break
        if chosen_first is None:
            continue

        seeds.append(chosen_first)
        used_diagonals.add(_diag_key(chosen_first[0], chosen_first[1], dr, dc))

        rest = list(candidates)
        rng.shuffle(rest)
        for pos in rest:
            if len(seeds) >= count:
                break
            if _is_compatible(pos, seeds, used_diagonals, dr, dc):
                seeds.append(pos)
                used_diagonals.add(_diag_key(pos[0], pos[1], dr, dc))

        if len(seeds) == count:
            return seeds

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(ODD_SIZES))
    width = int(rng.choice(ODD_SIZES))
    mid_r = height // 2
    mid_c = width // 2

    max_count = min(MAX_SEEDS_PER_QUADRANT, mid_r - 1, mid_c - 1)
    if max_count < 2:
        return None

    counts = [int(rng.integers(1, max_count + 1)) for _ in range(4)]
    if len(set(counts)) == 1:
        return None
    if sum(count > 1 for count in counts) < 2:
        return None

    colors = [int(color) for color in rng.choice(PALETTE, size=4, replace=False)]
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[mid_r, :] = CROSS_COLOR
    grid[:, mid_c] = CROSS_COLOR

    full_any = False
    truncated_any = False

    for idx, (name, _r0, _r1, _c0, _c1, dr, dc) in enumerate(_quadrant_specs(height, width, mid_r, mid_c)):
        candidates = _candidate_positions(name, mid_r, mid_c, height, width)
        if len(candidates) < counts[idx]:
            return None

        want_truncated = bool(rng.integers(0, 2))
        seeds = _place_quadrant_seeds(
            candidates,
            counts[idx],
            dr,
            dc,
            height,
            width,
            want_truncated,
            rng,
        )
        if seeds is None:
            return None

        for r, c in seeds:
            grid[r, c] = colors[idx]
            capacity = _ray_capacity(r, c, dr, dc, height, width)
            if counts[idx] > 1 and capacity < counts[idx]:
                truncated_any = True
            if counts[idx] > 1 and capacity >= counts[idx]:
                full_any = True

    if not full_any or not truncated_any:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
