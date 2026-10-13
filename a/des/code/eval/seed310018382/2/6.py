import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RADIUS_RANGE = (1, 4)
COLOR_COUNT_RANGE = (2, 5)
HEIGHT_RANGE = (10, 20)
LEFT_PAD_MAX = 4
MAX_GENERATION_ATTEMPTS = 400
MODE_ORDER = ("clean", "overlap", "clip", "both")
_MODE_CURSOR = 0


def _parse_palette(row: np.ndarray) -> list[tuple[int, int, int]]:
    entries: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    c = 0
    width = int(row.shape[0])
    while c < width:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        assert color not in seen, "palette colors must be distinct"
        seen.add(color)
        gap = 0
        cc = c + 1
        while cc < width and int(row[cc]) == BG:
            gap += 1
            cc += 1
        entries.append((color, c, gap))
        c = cc
    assert entries, "palette row must contain at least one color"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2

    palette = _parse_palette(g[0])
    colors = [color for color, _, _ in palette]
    radii = {color: gap for color, _, gap in palette}
    seeds: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}

    for r in range(1, height):
        for c in range(width):
            color = int(g[r, c])
            if color != BG and color in radii:
                seeds[color].append((r, c))

    out = np.full_like(g, BG)
    for color in reversed(colors):
        radius = int(radii[color])
        for r, c in seeds[color]:
            r0 = max(0, int(r - radius))
            r1 = min(height, int(r + radius + 1))
            c0 = max(0, int(c - radius))
            c1 = min(width, int(c + radius + 1))
            out[r0:r1, c0:c1] = int(color)

    out[0, :] = g[0, :]
    return out


def _body_box(seed: tuple[int, int], radius: int, height: int, width: int) -> tuple[int, int, int, int]:
    r, c = seed
    return (
        max(1, int(r - radius)),
        min(height - 1, int(r + radius)),
        max(0, int(c - radius)),
        min(width - 1, int(c + radius)),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(a, b):
        return 0
    r0 = max(a[0], b[0])
    r1 = min(a[1], b[1])
    c0 = max(a[2], b[2])
    c1 = min(a[3], b[3])
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _box_area(box: tuple[int, int, int, int]) -> int:
    return int((box[1] - box[0] + 1) * (box[3] - box[2] + 1))


def _seed_clips(seed: tuple[int, int], radius: int, height: int, width: int) -> bool:
    r, c = seed
    return bool(r - radius < 0 or r + radius >= height or c - radius < 0 or c + radius >= width)


def _sample_interior_seed(radius: int, height: int, width: int) -> tuple[int, int] | None:
    r0 = radius + 1
    r1 = height - 1 - radius
    c0 = radius
    c1 = width - 1 - radius
    if r0 > r1 or c0 > c1:
        return None
    return random.randint(r0, r1), random.randint(c0, c1)


def _sample_clipped_seed(radius: int, height: int, width: int) -> tuple[int, int] | None:
    sides: list[str] = []
    if radius + 1 <= height - 1 - radius:
        if radius > 0:
            sides.extend(["left", "right"])
    if radius <= width - 1 - radius:
        sides.append("bottom")
    random.shuffle(sides)
    for side in sides:
        if side == "left":
            c_hi = min(width - 1, radius - 1)
            if c_hi < 0:
                continue
            seed = (random.randint(radius + 1, height - 1 - radius), random.randint(0, c_hi))
        elif side == "right":
            c_lo = max(0, width - radius)
            if c_lo > width - 1:
                continue
            seed = (random.randint(radius + 1, height - 1 - radius), random.randint(c_lo, width - 1))
        else:
            r_lo = max(1, height - radius)
            if r_lo > height - 1:
                continue
            seed = (random.randint(r_lo, height - 1), random.randint(radius, width - 1 - radius))
        if seed[0] - radius >= 1 and _seed_clips(seed, radius, height, width):
            return seed
    return None


def _can_place_seed(
    idx: int,
    seed: tuple[int, int],
    placed: list[tuple[int, int] | None],
    radii: list[int],
    height: int,
    width: int,
    overlap_pair: tuple[int, int] | None,
) -> bool:
    r, c = seed
    if not (1 <= r < height and 0 <= c < width):
        return False
    box = _body_box(seed, radii[idx], height, width)
    for other_idx, other_seed in enumerate(placed):
        if other_seed is None:
            continue
        if max(abs(r - other_seed[0]), abs(c - other_seed[1])) <= 1:
            return False
        other_box = _body_box(other_seed, radii[other_idx], height, width)
        allowed = overlap_pair is not None and tuple(sorted((idx, other_idx))) == overlap_pair
        if not allowed and _boxes_overlap(box, other_box):
            return False
    return True


def _layout_features(
    placed: list[tuple[int, int]],
    radii: list[int],
    height: int,
    width: int,
) -> tuple[bool, bool]:
    has_overlap = False
    for i in range(len(placed)):
        box_i = _body_box(placed[i], radii[i], height, width)
        for j in range(i + 1, len(placed)):
            if _boxes_overlap(box_i, _body_box(placed[j], radii[j], height, width)):
                has_overlap = True
                break
        if has_overlap:
            break
    has_clip = any(_seed_clips(seed, radius, height, width) for seed, radius in zip(placed, radii))
    return has_overlap, has_clip


def _build_grid(
    colors: list[int],
    radii: list[int],
    left_pad: int,
    height: int,
    width: int,
    seeds: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    c = left_pad
    for color, radius in zip(colors, radii):
        grid[0, c] = int(color)
        c += 1 + int(radius)
    for color, (r, c) in zip(colors, seeds):
        grid[r, c] = int(color)
    return grid


def _candidate_ok(
    grid: np.ndarray,
    colors: list[int],
    seeds: list[tuple[int, int]],
    radii: list[int],
    mode: str,
    overlap_pair: tuple[int, int] | None,
) -> bool:
    height, width = map(int, grid.shape)
    if int(np.count_nonzero(grid[1:, :] != BG)) != len(seeds):
        return False
    if any(np.any(grid[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] == color) < 1 for color, (r, c) in zip(colors, seeds)):
        return False

    has_overlap, has_clip = _layout_features(seeds, radii, height, width)
    if mode == "clean" and (has_overlap or has_clip):
        return False
    if mode == "overlap" and (not has_overlap or has_clip):
        return False
    if mode == "clip" and (has_overlap or not has_clip):
        return False
    if mode == "both" and (not has_overlap or not has_clip):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    body = out[1:, :]
    for color in colors:
        if not np.any(body == int(color)):
            return False

    if overlap_pair is not None:
        hi_idx, lo_idx = overlap_pair
        hi_box = _body_box(seeds[hi_idx], radii[hi_idx], height, width)
        lo_box = _body_box(seeds[lo_idx], radii[lo_idx], height, width)
        area = _intersection_area(hi_box, lo_box)
        if area <= 0:
            return False
        r0 = max(hi_box[0], lo_box[0])
        r1 = min(hi_box[1], lo_box[1])
        c0 = max(hi_box[2], lo_box[2])
        c1 = min(hi_box[3], lo_box[3])
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == int(colors[hi_idx])):
            return False

    return bool(np.array_equal(out[0, :], grid[0, :]))


def _place_seeds(
    radii: list[int],
    height: int,
    width: int,
    mode: str,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None] | None:
    n = len(radii)
    overlap_pair = None
    if mode in ("overlap", "both"):
        overlap_pair = tuple(sorted(random.sample(range(n), 2)))

    clip_idx = None
    if mode == "clip":
        clip_idx = random.randrange(n)
    elif mode == "both":
        choices = [i for i in range(n) if overlap_pair is None or i not in overlap_pair]
        if not choices:
            return None
        clip_idx = random.choice(choices)

    placed: list[tuple[int, int] | None] = [None] * n

    def place_one(idx: int, clipped: bool, must_overlap_with: int | None = None) -> bool:
        for _ in range(160):
            seed = _sample_clipped_seed(radii[idx], height, width) if clipped else _sample_interior_seed(radii[idx], height, width)
            if seed is None:
                return False
            if not _can_place_seed(idx, seed, placed, radii, height, width, overlap_pair):
                continue
            if must_overlap_with is not None:
                other_seed = placed[must_overlap_with]
                assert other_seed is not None
                box = _body_box(seed, radii[idx], height, width)
                other_box = _body_box(other_seed, radii[must_overlap_with], height, width)
                inter = _intersection_area(box, other_box)
                if inter < 2:
                    continue
                if inter >= _box_area(box) or inter >= _box_area(other_box):
                    continue
            placed[idx] = seed
            return True
        return False

    ordered_special: list[tuple[int, bool, int | None]] = []
    if overlap_pair is not None:
        ordered_special.append((overlap_pair[0], False, None))
        ordered_special.append((overlap_pair[1], False, overlap_pair[0]))
    if clip_idx is not None:
        ordered_special.append((clip_idx, True, None))

    used_special = {idx for idx, _, _ in ordered_special}
    for idx, clipped, must_overlap_with in ordered_special:
        if placed[idx] is not None:
            continue
        if not place_one(idx, clipped, must_overlap_with):
            return None

    remaining = [idx for idx in range(n) if idx not in used_special]
    random.shuffle(remaining)
    for idx in remaining:
        if not place_one(idx, False, None):
            return None

    if any(seed is None for seed in placed):
        return None
    return [seed for seed in placed if seed is not None], overlap_pair


def _generate_candidate(mode: str) -> np.ndarray | None:
    min_colors = 3 if mode == "both" else COLOR_COUNT_RANGE[0]
    n_colors = random.randint(min_colors, COLOR_COUNT_RANGE[1])
    colors = random.sample(PALETTE, n_colors)
    radii = [random.randint(*RADIUS_RANGE) for _ in range(n_colors)]
    if len(set(radii)) == 1:
        tweak_idx = random.randrange(n_colors)
        options = [r for r in range(RADIUS_RANGE[0], RADIUS_RANGE[1] + 1) if r != radii[tweak_idx]]
        radii[tweak_idx] = random.choice(options)

    base_width = sum(1 + radius for radius in radii)
    max_radius = max(radii)
    min_left_pad = max(0, 2 * max_radius + 1 - base_width)
    if min_left_pad > LEFT_PAD_MAX:
        return None
    left_pad = random.randint(min_left_pad, LEFT_PAD_MAX)
    width = base_width + left_pad

    min_height = max(HEIGHT_RANGE[0], 2 * max_radius + 2)
    if min_height > HEIGHT_RANGE[1]:
        return None
    height = random.randint(min_height, HEIGHT_RANGE[1])

    placed = _place_seeds(radii, height, width, mode)
    if placed is None:
        return None
    seeds, overlap_pair = placed

    grid = _build_grid(colors, radii, left_pad, height, width, seeds)
    if _candidate_ok(grid, colors, seeds, radii, mode, overlap_pair):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _MODE_CURSOR
    mode = MODE_ORDER[_MODE_CURSOR % len(MODE_ORDER)]
    _MODE_CURSOR += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for mode={mode}")
