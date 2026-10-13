import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (8, 19)
GRID_W_RANGE = (11, 24)
PALETTE_LEN_RANGE = (3, 7)
PANEL_COL_OPTIONS = (0, 1, 2)
SEED_COUNT_RANGE = (2, 7)
GEN_RETRIES = 400
POSITION_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _find_palette(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for c in range(w - 1):
        if np.any(grid[:, c + 1] != bg):
            continue
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size < 3:
            continue
        if int(rows[-1]) - int(rows[0]) + 1 != int(rows.size):
            continue
        colors = [int(grid[int(r), c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((int(c), int(rows[0]), colors))
    assert candidates, "expected a vertical palette column followed by a spacer"
    return min(candidates, key=lambda item: (item[0], -len(item[2]), item[1]))


def _simulate_growth(
    grid: np.ndarray,
    panel_c: int,
    palette: list[int],
    bg: int,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    main_c0 = panel_c + 2
    radius_of = {int(color): idx + 1 for idx, color in enumerate(palette)}

    seeds: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(g != bg):
        r = int(r)
        c = int(c)
        if c == panel_c:
            continue
        if c < main_c0:
            continue
        color = int(g[r, c])
        assert color in radius_of, "seed color must appear in the palette"
        seeds.append((r, c, color, int(radius_of[color])))

    occupied = np.zeros((h, w), dtype=bool)
    seed_mask = np.zeros((h, w), dtype=bool)
    for r, c, _color, _radius in seeds:
        occupied[r, c] = True
        seed_mask[r, c] = True

    stats = {
        "painted": 0,
        "ray_blocks": 0,
        "seed_blocks": 0,
        "boundary_blocks": 0,
    }

    # Draw stronger seeds first so weaker rays stop when they run into them.
    for r, c, color, radius in sorted(seeds, key=lambda item: (-item[3], item[0], item[1], item[2])):
        for dr, dc in DIRS4:
            for step in range(1, radius + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < h and main_c0 <= cc < w):
                    stats["boundary_blocks"] += 1
                    break
                if occupied[rr, cc]:
                    if seed_mask[rr, cc]:
                        stats["seed_blocks"] += 1
                    else:
                        stats["ray_blocks"] += 1
                    break
                out[rr, cc] = color
                occupied[rr, cc] = True
                stats["painted"] += 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a cross whose arm length is read from the left palette.

    The palette is a vertical list of distinct colors. A seed's radius equals the
    1-based row index of its color inside that palette. Rays only grow inside the
    main area to the right of the spacer column, and lower-radius rays stop when
    they encounter rays already drawn by higher-radius seeds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else _most_common_color(g)
    panel_c, _panel_r0, palette = _find_palette(g, bg)
    out, _stats = _simulate_growth(g, panel_c, palette, bg)
    return out


def _render_input(
    h: int,
    w: int,
    panel_c: int,
    panel_r0: int,
    palette: list[int],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for idx, color in enumerate(palette):
        grid[panel_r0 + idx, panel_c] = int(color)
    for r, c, color in seeds:
        grid[r, c] = int(color)
    return grid


def _chebyshev_far_enough(
    pos: tuple[int, int],
    seeds: list[tuple[int, int, int]],
    allow_same_row_col: bool = False,
) -> bool:
    r, c = pos
    for rr, cc, _color in seeds:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
        if not allow_same_row_col and (r == rr or c == cc) and abs(r - rr) + abs(c - cc) <= 3:
            return False
    return True


def _random_palette(rng: np.random.Generator, length: int) -> list[int]:
    colors = rng.choice(np.arange(1, 10, dtype=int), size=length, replace=False)
    return [int(c) for c in colors]


def _random_seed_color(rng: np.random.Generator, palette: list[int]) -> int:
    return int(palette[int(rng.integers(0, len(palette)))])


def _try_place_cross_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    main_c0: int,
    radius_of: dict[int, int],
) -> list[tuple[int, int, int]] | None:
    colors = list(radius_of)
    if len(colors) < 2:
        return None
    for _ in range(POSITION_RETRIES):
        color_a, color_b = [int(x) for x in rng.choice(np.array(colors, dtype=int), size=2, replace=False)]
        ra = int(radius_of[color_a])
        rb = int(radius_of[color_b])
        if ra == rb:
            continue
        ir = int(rng.integers(1, h - 1))
        ic = int(rng.integers(main_c0 + 1, w - 1))

        a_cols = [c for c in range(max(main_c0, ic - ra), min(w - 1, ic + ra) + 1) if c != ic]
        b_rows = [r for r in range(max(0, ir - rb), min(h - 1, ir + rb) + 1) if r != ir]
        if not a_cols or not b_rows:
            continue

        rng.shuffle(a_cols)
        rng.shuffle(b_rows)
        for a_c in a_cols:
            for b_r in b_rows:
                seeds = [(ir, int(a_c), color_a), (int(b_r), ic, color_b)]
                if max(abs(ir - b_r), abs(a_c - ic)) <= 1:
                    continue
                return seeds
    return None


def _try_place_line_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    main_c0: int,
    radius_of: dict[int, int],
) -> list[tuple[int, int, int]] | None:
    colors = list(radius_of)
    if len(colors) < 2:
        return None
    for _ in range(POSITION_RETRIES):
        color_a, color_b = [int(x) for x in rng.choice(np.array(colors, dtype=int), size=2, replace=False)]
        ra = int(radius_of[color_a])
        rb = int(radius_of[color_b])
        if ra == rb:
            continue
        reach = max(ra, rb)
        same_row = bool(rng.integers(0, 2))
        if same_row:
            r = int(rng.integers(0, h))
            left = int(rng.integers(main_c0, w - 2))
            max_gap = min(reach, w - left - 1)
            if max_gap < 2:
                continue
            gap = int(rng.integers(2, max_gap + 1))
            right = left + gap
            if right >= w:
                continue
            return [(r, left, color_a), (r, right, color_b)]
        c = int(rng.integers(main_c0, w))
        top = int(rng.integers(0, h - 1))
        max_gap = min(reach, h - top - 1)
        if max_gap < 2:
            continue
        gap = int(rng.integers(2, max_gap + 1))
        bottom = top + gap
        if bottom >= h:
            continue
        return [(top, c, color_a), (bottom, c, color_b)]
    return None


def _try_place_random_seed(
    rng: np.random.Generator,
    h: int,
    w: int,
    main_c0: int,
    seeds: list[tuple[int, int, int]],
    color: int,
    edge_bias: bool,
) -> tuple[int, int, int] | None:
    for _ in range(POSITION_RETRIES):
        if edge_bias and bool(rng.integers(0, 2)):
            side = int(rng.integers(0, 4))
            if side == 0:
                r = 0
                c = int(rng.integers(main_c0, w))
            elif side == 1:
                r = h - 1
                c = int(rng.integers(main_c0, w))
            elif side == 2:
                r = int(rng.integers(0, h))
                c = main_c0
            else:
                r = int(rng.integers(0, h))
                c = w - 1
        else:
            r = int(rng.integers(0, h))
            c = int(rng.integers(main_c0, w))
        if _chebyshev_far_enough((r, c), seeds):
            return (r, c, int(color))
    return None


def _scene_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    stats: dict[str, int],
    special_mode: str,
    intended_panel_c: int,
) -> bool:
    if stats["painted"] < 4:
        return False
    if np.count_nonzero(expected) > int(inp.size * 0.42):
        return False
    if np.array_equal(inp, expected):
        return False

    non_bg = np.argwhere(inp != BG)
    if non_bg.size == 0:
        return False

    try:
        panel_c, _panel_r0, _palette = _find_palette(inp, BG)
        actual = main(inp)
    except Exception:
        return False

    if panel_c != intended_panel_c:
        return False
    if not np.array_equal(actual, expected):
        return False
    if special_mode == "cross" and (stats["ray_blocks"] + stats["seed_blocks"] == 0):
        return False
    if special_mode == "line" and stats["seed_blocks"] == 0:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))

    palette_len_hi = min(PALETTE_LEN_RANGE[1], h + 1, 10)
    if palette_len_hi <= PALETTE_LEN_RANGE[0]:
        return None
    palette_len = int(rng.integers(PALETTE_LEN_RANGE[0], palette_len_hi))
    panel_c_choices = [c for c in PANEL_COL_OPTIONS if c + 3 < w]
    if not panel_c_choices:
        return None
    panel_c = int(rng.choice(np.array(panel_c_choices, dtype=int)))
    panel_r0 = int(rng.integers(0, h - palette_len + 1))
    main_c0 = panel_c + 2
    if w - main_c0 < 5:
        return None

    palette = _random_palette(rng, palette_len)
    radius_of = {int(color): idx + 1 for idx, color in enumerate(palette)}
    seed_target = int(rng.integers(SEED_COUNT_RANGE[0], SEED_COUNT_RANGE[1]))
    seed_target = min(seed_target, max(2, (h * (w - main_c0)) // 8))
    if seed_target < 2:
        return None

    special_mode = ("none", "cross", "line")[int(rng.integers(0, 3))]
    seeds: list[tuple[int, int, int]] = []

    if special_mode == "cross":
        pair = _try_place_cross_pair(rng, h, w, main_c0, radius_of)
        if pair is None:
            return None
        seeds.extend(pair)
    elif special_mode == "line":
        pair = _try_place_line_pair(rng, h, w, main_c0, radius_of)
        if pair is None:
            return None
        seeds.extend(pair)

    while len(seeds) < seed_target:
        color = _random_seed_color(rng, palette)
        edge_bias = bool(rng.integers(0, 3) == 0)
        seed = _try_place_random_seed(rng, h, w, main_c0, seeds, color, edge_bias=edge_bias)
        if seed is None:
            return None
        seeds.append(seed)

    if len({color for _r, _c, color in seeds}) < 2:
        return None

    inp = _render_input(h, w, panel_c, panel_r0, palette, seeds)
    expected, stats = _simulate_growth(inp, panel_c, palette, BG)
    if not _scene_ok(inp, expected, stats, special_mode, panel_c):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
