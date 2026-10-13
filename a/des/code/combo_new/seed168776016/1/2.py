import itertools
import random

import numpy as np


BLACK = 0
GRAY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
GROUP_COUNT_RANGE = (2, 4)
COUNT_VALUES = (2, 3, 4, 5, 6)
EVEN_COUNTS = (2, 4, 6)
ODD_COUNTS = (3, 5)
BOX_SIZE_RANGE = (3, 10)
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


def _extract_groups(grid: np.ndarray) -> list[dict]:
    groups = []
    for color in (int(c) for c in np.unique(grid) if int(c) != BLACK):
        coords_arr = np.argwhere(grid == color)
        coords = [tuple(map(int, rc)) for rc in coords_arr]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        groups.append(
            {
                "color": color,
                "coords": coords,
                "count": len(coords),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return groups


def _border_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _rectangles_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _separated_offsets(length: int) -> list[int]:
    offsets = list(range(0, length, 2))
    last = length - 1
    if offsets[-1] != last:
        if last - offsets[-1] == 1:
            offsets[-1] = last
        else:
            offsets.append(last)
    return offsets


def _slot_capacity(height: int, width: int) -> int:
    return len(_separated_offsets(height)) * len(_separated_offsets(width))


def _sample_rect_size(count: int, rng: random.Random) -> tuple[int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        height = rng.randint(*BOX_SIZE_RANGE)
        width = rng.randint(*BOX_SIZE_RANGE)
        if _slot_capacity(height, width) >= count:
            return height, width
    return None


def _random_rect(
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    r0 = rng.randint(0, grid_h - height)
    c0 = rng.randint(0, grid_w - width)
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _choose_counts(num_groups: int, overlap_mode: bool, rng: random.Random) -> list[int]:
    parities = [0, 1] + [rng.randint(0, 1) for _ in range(num_groups - 2)]
    rng.shuffle(parities)
    counts = [
        rng.choice(EVEN_COUNTS if parity == 0 else ODD_COUNTS)
        for parity in parities
    ]
    if overlap_mode and counts[0] == counts[1]:
        options = [value for value in COUNT_VALUES if value != counts[0]]
        counts[1] = rng.choice(options)
    return counts


def _place_rectangles(
    grid_h: int,
    grid_w: int,
    counts: list[int],
    overlap_mode: bool,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []

    if overlap_mode:
        size_a = _sample_rect_size(counts[0], rng)
        size_b = _sample_rect_size(counts[1], rng)
        if size_a is None or size_b is None:
            return None
        for _ in range(PLACEMENT_RETRIES):
            rect_a = _random_rect(size_a[0], size_a[1], grid_h, grid_w, rng)
            rect_b = _random_rect(size_b[0], size_b[1], grid_h, grid_w, rng)
            if rect_a == rect_b:
                continue
            if not _rectangles_intersect(rect_a, rect_b):
                continue
            if not (_border_cells(rect_a) & _border_cells(rect_b)):
                continue
            rects.extend([rect_a, rect_b])
            break
        if len(rects) != 2:
            return None

    start_index = len(rects)
    for group_index in range(start_index, len(counts)):
        size = _sample_rect_size(counts[group_index], rng)
        if size is None:
            return None
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            rect = _random_rect(size[0], size[1], grid_h, grid_w, rng)
            if any(_rectangles_intersect(rect, other, pad=1) for other in rects):
                continue
            placed = rect
            break
        if placed is None:
            return None
        rects.append(placed)

    return rects


def _point_combinations(
    rect: tuple[int, int, int, int],
    count: int,
) -> list[tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = rect
    row_offsets = _separated_offsets(r1 - r0 + 1)
    col_offsets = _separated_offsets(c1 - c0 + 1)
    candidates = [
        (r0 + dr, c0 + dc)
        for dr in row_offsets
        for dc in col_offsets
    ]
    combos = []
    for combo in itertools.combinations(candidates, count):
        rows = [r for r, _ in combo]
        cols = [c for _, c in combo]
        if min(rows) != r0 or max(rows) != r1:
            continue
        if min(cols) != c0 or max(cols) != c1:
            continue
        combos.append(combo)
    return combos


def _combo_far_from_used(
    combo: tuple[tuple[int, int], ...],
    used_points: set[tuple[int, int]],
) -> bool:
    for r, c in combo:
        for ur, uc in used_points:
            if abs(r - ur) <= 1 and abs(c - uc) <= 1:
                return False
    return True


def _assign_points(
    rects: list[tuple[int, int, int, int]],
    counts: list[int],
    rng: random.Random,
) -> list[tuple[tuple[int, int], ...]] | None:
    shared_border_by_group = [set() for _ in rects]
    for i, rect_i in enumerate(rects):
        border_i = _border_cells(rect_i)
        for j, rect_j in enumerate(rects):
            if i == j:
                continue
            shared_border_by_group[i] |= border_i & _border_cells(rect_j)

    options_by_group: list[list[tuple[tuple[int, int], ...]]] = []
    for i, (rect, count) in enumerate(zip(rects, counts)):
        combos = _point_combinations(rect, count)
        if not combos:
            return None
        rng.shuffle(combos)
        combos.sort(
            key=lambda combo: sum(point in shared_border_by_group[i] for point in combo)
        )
        options_by_group.append(combos)

    order = sorted(range(len(rects)), key=lambda idx: len(options_by_group[idx]))
    chosen: list[tuple[tuple[int, int], ...] | None] = [None] * len(rects)

    def backtrack(step: int, used_points: set[tuple[int, int]]) -> bool:
        if step == len(order):
            return True
        group_index = order[step]
        for combo in options_by_group[group_index]:
            combo_set = set(combo)
            if combo_set & used_points:
                continue
            if not _combo_far_from_used(combo, used_points):
                continue
            chosen[group_index] = combo
            if backtrack(step + 1, used_points | combo_set):
                return True
            chosen[group_index] = None
        return False

    if not backtrack(0, set()):
        return None
    return [combo for combo in chosen if combo is not None]


def _build_grid(
    grid_h: int,
    grid_w: int,
    colors: list[int],
    points_by_group: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for color, points in zip(colors, points_by_group):
        for r, c in points:
            grid[r, c] = color
    return grid


def _validate_generated(
    grid: np.ndarray,
    colors: list[int],
    counts: list[int],
    rects: list[tuple[int, int, int, int]],
    overlap_mode: bool,
) -> bool:
    groups = {info["color"]: info for info in _extract_groups(grid)}
    if set(groups) != set(colors):
        return False

    for color, count, rect in zip(colors, counts, rects):
        info = groups[color]
        if info["count"] != count or info["bbox"] != rect:
            return False
        coords = info["coords"]
        for i, (r0, c0) in enumerate(coords):
            for r1, c1 in coords[i + 1 :]:
                if abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1:
                    return False

    if not any(count % 2 == 0 for count in counts):
        return False
    if not any(count % 2 == 1 for count in counts):
        return False

    out = main(grid)
    if np.array_equal(out, grid) or not np.any(out == GRAY):
        return False

    if overlap_mode:
        shared = _border_cells(rects[0]) & _border_cells(rects[1])
        if not shared or counts[0] == counts[1]:
            return False
        source_cells = {coord for info in groups.values() for coord in info["coords"]}
        visible_shared = [cell for cell in shared if cell not in source_cells]
        if not visible_shared:
            return False
        winner = 0 if counts[0] > counts[1] else 1
        if not all(out[r, c] == colors[winner] for r, c in visible_shared):
            return False
    else:
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _rectangles_intersect(rects[i], rects[j]):
                    return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered single pixels by color into bounding-box frames and gray-fill the even-sized ones.

    Input structure:
    - The background is black.
    - Several non-black colors each appear as a small set of isolated single pixels.
    - Pixels with the same color form one group, and each group's extent is given by its tight axis-aligned bounding box.

    Rule:
    - For each color group, compute the smallest axis-aligned bounding box that contains all pixels of that color.
    - Draw that box's 1-pixel-thick border in the group's color.
    - If the group has an even number of source pixels, fill the box interior with gray; if it has an odd number, leave the interior black.
    - When box borders overlap, groups with more source pixels are drawn later and therefore stay on top, while all original source pixels are restored last and never get covered.

    Key insight:
    - The scattered points are not moved individually; each color first becomes a single spatial group via its bounding box, and the group's parity decides whether the enclosed area is filled.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    groups = _extract_groups(g)

    for info in groups:
        if info["count"] % 2 == 1:
            continue
        r0, c0, r1, c1 = info["bbox"]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            out[r0 + 1 : r1, c0 + 1 : c1] = GRAY

    for info in sorted(
        groups,
        key=lambda item: (
            item["count"],
            item["bbox"][0],
            item["bbox"][1],
            item["bbox"][2],
            item["bbox"][3],
        ),
    ):
        color = info["color"]
        r0, c0, r1, c1 = info["bbox"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    for info in groups:
        for r, c in info["coords"]:
            out[r, c] = info["color"]

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    num_groups = rng.randint(*GROUP_COUNT_RANGE)
    overlap_mode = rng.random() < 0.5

    counts = _choose_counts(num_groups, overlap_mode, rng)
    rects = _place_rectangles(grid_h, grid_w, counts, overlap_mode, rng)
    if rects is None:
        return None

    points_by_group = _assign_points(rects, counts, rng)
    if points_by_group is None:
        return None

    colors = rng.sample(SOURCE_COLORS, num_groups)
    grid = _build_grid(grid_h, grid_w, colors, points_by_group)

    if _validate_generated(grid, colors, counts, rects, overlap_mode):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
