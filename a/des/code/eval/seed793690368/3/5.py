import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _palette_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _palette_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_palette_bbox(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    if r1 - r0 != c1 - c0:
        return False
    if not np.all(g[r0, c0 : c1 + 1] == GREY):
        return False
    if not np.all(g[r1, c0 : c1 + 1] == GREY):
        return False
    if not np.all(g[r0 : r1 + 1, c0] == GREY):
        return False
    if not np.all(g[r0 : r1 + 1, c1] == GREY):
        return False

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    if inner.shape[0] < 3 or inner.shape[1] < 3:
        return False
    if inner[0, 0] != BLACK:
        return False

    row_headers = [int(x) for x in inner[1:, 0]]
    col_headers = [int(x) for x in inner[0, 1:]]
    if len(set(row_headers)) != len(row_headers):
        return False
    if len(set(col_headers)) != len(col_headers):
        return False
    if set(row_headers) != set(col_headers):
        return False
    if any(x in (BLACK, GREY) for x in row_headers + col_headers):
        return False

    row_index = {color: i for i, color in enumerate(row_headers)}
    col_index = {color: i for i, color in enumerate(col_headers)}
    matrix = inner[1:, 1:]
    if np.any(matrix == GREY):
        return False

    for a in row_headers:
        for b in row_headers:
            if int(matrix[row_index[a], col_index[b]]) != int(matrix[row_index[b], col_index[a]]):
                return False
    return True


def _find_palette_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    r1, c1 = h - 1, w - 1
    candidates = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if _is_palette_bbox(g, r0, c0, r1, c1):
                candidates.append((r0, c0, r1, c1))
    if not candidates:
        raise AssertionError("palette not found")
    return max(candidates, key=lambda box: (box[0], box[1]))


def _parse_palette(g: np.ndarray) -> tuple[tuple[int, int, int, int], set[int], dict[tuple[int, int], int]]:
    bbox = _find_palette_bbox(g)
    r0, c0, r1, c1 = bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    row_headers = [int(x) for x in inner[1:, 0]]
    col_headers = [int(x) for x in inner[0, 1:]]
    matrix = inner[1:, 1:]

    row_index = {color: i for i, color in enumerate(row_headers)}
    col_index = {color: i for i, color in enumerate(col_headers)}
    colors = set(row_headers)

    mix: dict[tuple[int, int], int] = {}
    for a in colors:
        for b in colors:
            mix[(a, b)] = int(matrix[row_index[a], col_index[b]])
    return bbox, colors, mix


def _collect_seeds(g: np.ndarray, bbox: tuple[int, int, int, int], palette_colors: set[int]) -> list[tuple[int, int, int]]:
    mask = ~_palette_mask(g.shape, bbox)
    seeds = []
    for r, c in np.argwhere(mask & (g != BLACK) & (g != GREY)):
        color = int(g[r, c])
        if color in palette_colors:
            seeds.append((int(r), int(c), color))
    if len({color for _, _, color in seeds}) < 2:
        raise AssertionError("expected at least two seed colors")
    return seeds


def _simulate(
    g: np.ndarray,
    seeds: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    mix: dict[tuple[int, int], int],
) -> tuple[np.ndarray, list[list[set[int]]]]:
    h, w = g.shape
    palette = _palette_mask(g.shape, bbox)
    seed_positions = {(r, c) for r, c, _ in seeds}
    covered = [[set() for _ in range(w)] for _ in range(h)]

    for r, c, color in seeds:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            while _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != GREY:
                covered[nr][nc].add(color)
                nr += dr
                nc += dc

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if palette[r, c] or (r, c) in seed_positions or int(g[r, c]) == GREY:
                continue
            colors = covered[r][c]
            if not colors:
                continue
            if len(colors) == 1:
                out[r, c] = next(iter(colors))
                continue
            results = {mix[(a, b)] for a, b in combinations(sorted(colors), 2)}
            if len(results) != 1:
                raise AssertionError("ambiguous multi-seed overlap")
            out[r, c] = next(iter(results))
    return out, covered


def _stop_reason(
    g: np.ndarray,
    border: np.ndarray,
    seed: tuple[int, int, int],
    dr: int,
    dc: int,
) -> str:
    r, c, _ = seed
    nr, nc = r + dr, c + dc
    while _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != GREY:
        nr += dr
        nc += dc
    if not _in_bounds(g.shape, nr, nc):
        return "boundary"
    if border[nr, nc]:
        return "palette"
    return "wall"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bbox, palette_colors, mix = _parse_palette(g)
    seeds = _collect_seeds(g, bbox, palette_colors)
    out, _ = _simulate(g, seeds, bbox, mix)
    return out


def _make_mix_map(
    base_colors: list[int],
    mode: str,
    rng: random.Random,
) -> dict[tuple[int, int], int]:
    remaining = [color for color in COLOR_POOL if color not in base_colors]
    mix: dict[tuple[int, int], int] = {}
    for color in base_colors:
        mix[(color, color)] = color

    if mode == "triple":
        mix_color = rng.choice(remaining)
        for a, b in combinations(base_colors, 2):
            mix[(a, b)] = mix_color
            mix[(b, a)] = mix_color
        return mix

    palette = rng.sample(remaining, rng.randint(1, min(3, len(remaining))))
    for a, b in combinations(base_colors, 2):
        color = rng.choice(palette)
        mix[(a, b)] = color
        mix[(b, a)] = color
    return mix


def _paint_palette(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    order: list[int],
    mix: dict[tuple[int, int], int],
) -> None:
    r0, c0, r1, c1 = bbox
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner[:, :] = BLACK
    inner[0, 0] = BLACK
    inner[0, 1:] = order
    inner[1:, 0] = order
    for i, a in enumerate(order, start=1):
        for j, b in enumerate(order, start=1):
            inner[i, j] = mix[(a, b)]


def _outside_palette(h: int, w: int, bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return not (r0 <= r <= r1 and c0 <= c <= c1)


def _can_place_wall(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    cells: list[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(g.shape, r, c):
            return False
        if not _outside_palette(g.shape[0], g.shape[1], bbox, r, c):
            return False
        if int(g[r, c]) != BLACK:
            return False
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1):
            return False
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc) and (nr, nc) not in cell_set and int(g[nr, nc]) == GREY:
                return False
    return True


def _add_wall(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    cells: list[tuple[int, int]],
) -> bool:
    if not _can_place_wall(g, bbox, cells):
        return False
    for r, c in cells:
        g[r, c] = GREY
    return True


def _sample_pair_topstrip(
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    r0, c0, _, _ = bbox
    if r0 < 4 or w - c0 < 3:
        return None
    target_r = rng.randint(1, r0 - 3)
    target_c = rng.randint(c0, w - 2)
    left_max = min(c0 - 2, target_c - 1)
    if left_max < 1:
        return None
    seed_row_c = rng.randint(1, left_max)
    row_options = [rr for rr in range(r0) if rr != target_r]
    if not row_options:
        return None
    seed_col_r = rng.choice(row_options)
    seeds = [
        (target_r, seed_row_c, colors[0]),
        (seed_col_r, target_c, colors[1]),
    ]
    return seeds, (target_r, target_c)


def _sample_pair_leftstrip(
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    r0, c0, _, _ = bbox
    if c0 < 4 or h - r0 < 3:
        return None
    target_r = rng.randint(r0, h - 2)
    target_c = rng.randint(1, c0 - 3)
    upper_options = list(range(1, target_r))
    if not upper_options:
        return None
    seed_col_r = rng.choice(upper_options)
    seed_row_c = rng.randint(0, target_c - 1)
    seeds = [
        (target_r, seed_row_c, colors[0]),
        (seed_col_r, target_c, colors[1]),
    ]
    return seeds, (target_r, target_c)


def _sample_triple_topstrip(
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    r0, c0, _, _ = bbox
    if r0 < 4 or w < 6:
        return None
    target_r = rng.randint(1, r0 - 3)
    target_c = rng.randint(max(c0, 2), w - 3)
    left_options = list(range(1, target_c))
    right_options = list(range(target_c + 1, w - 1))
    row_options = [rr for rr in range(r0) if rr != target_r]
    if not left_options or not right_options or not row_options:
        return None
    left_c = rng.choice(left_options)
    right_c = rng.choice(right_options)
    seed_col_r = rng.choice(row_options)
    seeds = [
        (target_r, left_c, colors[0]),
        (target_r, right_c, colors[1]),
        (seed_col_r, target_c, colors[2]),
    ]
    return seeds, (target_r, target_c)


def _sample_triple_leftstrip(
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    r0, c0, _, _ = bbox
    if c0 < 4 or h < 6:
        return None
    target_r = rng.randint(max(r0, 2), h - 3)
    target_c = rng.randint(1, c0 - 3)
    upper_options = list(range(1, target_r))
    lower_options = list(range(target_r + 1, h - 1))
    left_options = list(range(0, target_c))
    if not upper_options or not lower_options or not left_options:
        return None
    upper_r = rng.choice(upper_options)
    lower_r = rng.choice(lower_options)
    seed_row_c = rng.choice(left_options)
    seeds = [
        (upper_r, target_c, colors[0]),
        (lower_r, target_c, colors[1]),
        (target_r, seed_row_c, colors[2]),
    ]
    return seeds, (target_r, target_c)


def _place_extra_seed(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    seeds: list[tuple[int, int, int]],
    color: int,
    rng: random.Random,
) -> bool:
    occupied = {(r, c) for r, c, _ in seeds}
    candidates = []
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if not _outside_palette(g.shape[0], g.shape[1], bbox, r, c):
                continue
            if int(g[r, c]) != BLACK:
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc, _ in seeds):
                continue
            if (r, c) in occupied:
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r, c = rng.choice(candidates)
    g[r, c] = color
    seeds.append((r, c, color))
    return True


def _place_blocker_topstrip(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    target: tuple[int, int],
    rng: random.Random,
) -> bool:
    r0, c0, _, _ = bbox
    target_r, target_c = target
    candidates = []
    for wall_c in range(target_c + 1, g.shape[1] - 1):
        for length in range(2, min(5, r0)):
            start_min = max(1, target_r - length + 1)
            start_max = min(target_r, r0 - 1 - length)
            for start_r in range(start_min, start_max + 1):
                cells = [(start_r + k, wall_c) for k in range(length)]
                if _can_place_wall(g, bbox, cells):
                    candidates.append(cells)
    if not candidates:
        return False
    return _add_wall(g, bbox, rng.choice(candidates))


def _place_blocker_leftstrip(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    target: tuple[int, int],
    rng: random.Random,
) -> bool:
    _, c0, _, _ = bbox
    target_r, target_c = target
    candidates = []
    for wall_r in range(target_r + 1, g.shape[0] - 1):
        for length in range(2, min(5, c0)):
            start_min = max(1, target_c - length + 1)
            start_max = min(target_c, c0 - 1 - length)
            for start_c in range(start_min, start_max + 1):
                cells = [(wall_r, start_c + k) for k in range(length)]
                if _can_place_wall(g, bbox, cells):
                    candidates.append(cells)
    if not candidates:
        return False
    return _add_wall(g, bbox, rng.choice(candidates))


def _place_random_walls(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    count: int,
    rng: random.Random,
) -> None:
    for _ in range(count):
        placed = False
        for _ in range(80):
            orientation = rng.choice(("h", "v"))
            length = rng.randint(2, 5)
            if orientation == "h":
                r = rng.randint(1, g.shape[0] - 2)
                c = rng.randint(1, g.shape[1] - 1 - length)
                cells = [(r, c + k) for k in range(length)]
            else:
                r = rng.randint(1, g.shape[0] - 1 - length)
                c = rng.randint(1, g.shape[1] - 2)
                cells = [(r + k, c) for k in range(length)]
            if _add_wall(g, bbox, cells):
                placed = True
                break
        if not placed:
            return


def _build_template(
    mode: str,
    rng: random.Random,
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    base_colors: list[int],
) -> tuple[str, list[tuple[int, int, int]], tuple[int, int]] | None:
    if mode == "pair":
        name = rng.choice(("pair_topstrip", "pair_leftstrip"))
        if name == "pair_topstrip":
            result = _sample_pair_topstrip(rng, h, w, bbox, base_colors[:2])
        else:
            result = _sample_pair_leftstrip(rng, h, w, bbox, base_colors[:2])
    else:
        name = rng.choice(("triple_topstrip", "triple_leftstrip"))
        if name == "triple_topstrip":
            result = _sample_triple_topstrip(rng, h, w, bbox, base_colors[:3])
        else:
            result = _sample_triple_leftstrip(rng, h, w, bbox, base_colors[:3])
    if result is None:
        return None
    seeds, target = result
    return name, seeds, target


def _seed_grid(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    order: list[int],
    mix: dict[tuple[int, int], int],
) -> np.ndarray:
    g = np.full((h, w), BLACK, dtype=int)
    _paint_palette(g, bbox, order, mix)
    return g


def _validate_instance(
    g: np.ndarray,
    seeds: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    mix: dict[tuple[int, int], int],
    mode: str,
) -> bool:
    try:
        expected, covered = _simulate(g, seeds, bbox, mix)
        observed = main(g)
    except AssertionError:
        return False
    if not np.array_equal(expected, observed):
        return False

    palette = _palette_mask(g.shape, bbox)
    seed_positions = {(r, c) for r, c, _ in seeds}
    max_overlap = 0
    mixed_cells = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if palette[r, c] or (r, c) in seed_positions or int(g[r, c]) == GREY:
                continue
            overlap = len(covered[r][c])
            max_overlap = max(max_overlap, overlap)
            if overlap >= 2:
                mixed_cells += 1
    if mixed_cells == 0:
        return False
    if mode == "pair" and max_overlap != 2:
        return False
    if mode == "triple" and max_overlap < 3:
        return False

    border = _palette_border_mask(g.shape, bbox)
    reasons = {_stop_reason(g, border, seed, dr, dc) for seed in seeds for dr, dc in DIR4}
    return reasons == {"boundary", "palette", "wall"}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(("pair", "triple"))
    k = 3 if mode == "triple" else rng.choice((2, 3))
    palette_side = k + 3
    h = rng.randint(palette_side + 6, 22)
    w = rng.randint(palette_side + 6, 22)
    bbox = (h - palette_side, w - palette_side, h - 1, w - 1)

    base_colors = rng.sample(list(COLOR_POOL), k)
    order = base_colors[:]
    rng.shuffle(order)
    mix = _make_mix_map(base_colors, mode, rng)

    built = _build_template(mode, rng, h, w, bbox, base_colors)
    if built is None:
        return None
    template_name, seeds, target = built

    g = _seed_grid(h, w, bbox, order, mix)
    for r, c, color in seeds:
        if int(g[r, c]) != BLACK:
            return None
        g[r, c] = color

    if mode == "pair" and k == 3 and rng.random() < 0.6:
        extra_color = next(color for color in base_colors if color not in {seed[2] for seed in seeds})
        _place_extra_seed(g, bbox, seeds, extra_color, rng)

    if "topstrip" in template_name:
        if not _place_blocker_topstrip(g, bbox, target, rng):
            return None
    else:
        if not _place_blocker_leftstrip(g, bbox, target, rng):
            return None

    _place_random_walls(g, bbox, rng.randint(0, 2), rng)
    if not _validate_instance(g, seeds, bbox, mix, mode):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
