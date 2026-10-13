from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, FRAME})
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PATTERN_KINDS = ("h2", "v2", "h3", "v3", "quad", "cross")
GRID_SIZE_BOUNDS = (14, 31)
FRAME_COUNT_OPTIONS = (1, 2, 3, 4)
OUTER_HEIGHT_BOUNDS = (5, 12)
OUTER_WIDTH_BOUNDS = (5, 12)
PLACEMENT_ATTEMPTS = 80
PATTERN_ATTEMPTS = 60
GENERATION_RETRIES = 400


def _collect_component(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    stack = [start]
    seen[start] = True
    coords: list[tuple[int, int]] = []
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        coords.append((int(r), int(c)))
        for dr, dc in NEIGHBORS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return np.asarray(coords, dtype=int)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == FRAME
    seen = np.zeros_like(mask, dtype=bool)
    frames: list[tuple[int, int, int, int]] = []
    for r, c in zip(*np.where(mask), strict=False):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        coords = _collect_component(mask, (r, c), seen)
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(local, expected):
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _ordered_counts(values: np.ndarray) -> tuple[dict[int, int], list[int]]:
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values.ravel():
        color = int(value)
        if color not in counts:
            counts[color] = 0
            order.append(color)
        counts[color] += 1
    return counts, order


def _side_choice(values: np.ndarray, area_counts: dict[int, int]) -> tuple[int, int]:
    counts, order = _ordered_counts(values)
    best_color = int(order[0])
    best_count = int(counts[best_color])
    best_area = int(area_counts.get(best_color, 0))
    for color in order[1:]:
        count = int(counts[color])
        area = int(area_counts.get(color, 0))
        if (count, area) > (best_count, best_area):
            best_color = int(color)
            best_count = count
            best_area = area
    return best_color, best_count


def _corner_choice(
    first_color: int,
    first_strength: int,
    second_color: int,
    second_strength: int,
    area_counts: dict[int, int],
) -> int:
    if int(first_color) == int(second_color):
        return int(first_color)
    first_area = int(area_counts.get(int(first_color), 0))
    second_area = int(area_counts.get(int(second_color), 0))
    if first_area != second_area:
        return int(first_color if first_area > second_area else second_color)
    if first_strength != second_strength:
        return int(first_color if first_strength > second_strength else second_color)
    return int(first_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed interior's edge-adjacent color distribution onto its border.

    Input structure:
    - Background is black.
    - One or more grey one-pixel-thick rectangular frames sit on the canvas.
    - Each frame encloses an interior made of 2-3 non-grey color regions.

    Rule:
    - Detect every grey rectangular frame and inspect its enclosed interior.
    - For each side, read the interior row or column directly against that wall and
      find its majority color.
    - Recolor the corresponding grey border side to that majority color while
      leaving the interior unchanged.
    - Set each corner to the color of the adjacent side whose color occupies the
      larger total area inside the frame.

    Key insight:
    - The frame is a projection surface: each wall copies the dominant color from
      the interior strip directly opposite it, and the corners are resolved by the
      larger neighboring interior region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue

        area_counts, _order = _ordered_counts(interior)
        top_color, top_strength = _side_choice(interior[0, :], area_counts)
        bottom_color, bottom_strength = _side_choice(interior[-1, :], area_counts)
        left_color, left_strength = _side_choice(interior[:, 0], area_counts)
        right_color, right_strength = _side_choice(interior[:, -1], area_counts)

        out[r0, c0 + 1 : c1] = int(top_color)
        out[r1, c0 + 1 : c1] = int(bottom_color)
        out[r0 + 1 : r1, c0] = int(left_color)
        out[r0 + 1 : r1, c1] = int(right_color)

        out[r0, c0] = _corner_choice(top_color, top_strength, left_color, left_strength, area_counts)
        out[r0, c1] = _corner_choice(top_color, top_strength, right_color, right_strength, area_counts)
        out[r1, c0] = _corner_choice(bottom_color, bottom_strength, left_color, left_strength, area_counts)
        out[r1, c1] = _corner_choice(bottom_color, bottom_strength, right_color, right_strength, area_counts)

    return out


def _strict_majority(values: np.ndarray) -> tuple[bool, int]:
    counts, order = _ordered_counts(values)
    ranked = sorted(((int(count), int(color)) for color, count in counts.items()), reverse=True)
    if len(ranked) == 1:
        return True, int(order[0])
    return bool(ranked[0][0] > ranked[1][0]), int(ranked[0][1])


def _interior_is_valid(interior: np.ndarray) -> bool:
    colors = set(int(v) for v in np.unique(interior))
    if not (2 <= len(colors) <= 3):
        return False
    if BG in colors or FRAME in colors:
        return False

    area_counts, _order = _ordered_counts(interior)
    strips = (
        interior[0, :],
        interior[-1, :],
        interior[:, 0],
        interior[:, -1],
    )
    side_info: list[tuple[int, int]] = []
    for strip in strips:
        ok, _winner = _strict_majority(strip)
        if not ok:
            return False
        side_info.append(_side_choice(strip, area_counts))

    corner_pairs = (
        (side_info[0][0], side_info[2][0]),
        (side_info[0][0], side_info[3][0]),
        (side_info[1][0], side_info[2][0]),
        (side_info[1][0], side_info[3][0]),
    )
    for first, second in corner_pairs:
        if int(first) != int(second) and int(area_counts[first]) == int(area_counts[second]):
            return False
    return True


def _rand_color_sample(k: int) -> list[int]:
    return random.sample(PALETTE, k)


def _build_h_bands(h: int, w: int, n_colors: int) -> np.ndarray | None:
    colors = _rand_color_sample(n_colors)
    if n_colors == 2:
        cuts = [cut for cut in range(1, h) if cut != h - cut]
        if not cuts:
            return None
        cut = random.choice(cuts)
        arr = np.empty((h, w), dtype=int)
        arr[:cut, :] = colors[0]
        arr[cut:, :] = colors[1]
        return arr

    options = []
    for a in range(1, h - 1):
        for b in range(1, h - a):
            c = h - a - b
            if len({a, b, c}) == 3:
                options.append((a, b, c))
    if not options:
        return None
    a, b, c = random.choice(options)
    arr = np.empty((h, w), dtype=int)
    arr[:a, :] = colors[0]
    arr[a : a + b, :] = colors[1]
    arr[a + b :, :] = colors[2]
    return arr


def _build_v_bands(h: int, w: int, n_colors: int) -> np.ndarray | None:
    colors = _rand_color_sample(n_colors)
    if n_colors == 2:
        cuts = [cut for cut in range(1, w) if cut != w - cut]
        if not cuts:
            return None
        cut = random.choice(cuts)
        arr = np.empty((h, w), dtype=int)
        arr[:, :cut] = colors[0]
        arr[:, cut:] = colors[1]
        return arr

    options = []
    for a in range(1, w - 1):
        for b in range(1, w - a):
            c = w - a - b
            if len({a, b, c}) == 3:
                options.append((a, b, c))
    if not options:
        return None
    a, b, c = random.choice(options)
    arr = np.empty((h, w), dtype=int)
    arr[:, :a] = colors[0]
    arr[:, a : a + b] = colors[1]
    arr[:, a + b :] = colors[2]
    return arr


def _build_quadrants(h: int, w: int) -> np.ndarray | None:
    if h < 3 or w < 3:
        return None
    cut_rs = [cut for cut in range(1, h) if cut != h - cut]
    cut_cs = [cut for cut in range(1, w) if cut != w - cut]
    if not cut_rs or not cut_cs:
        return None
    colors = _rand_color_sample(3)
    cut_r = random.choice(cut_rs)
    cut_c = random.choice(cut_cs)
    arr = np.empty((h, w), dtype=int)
    arr[:cut_r, :cut_c] = colors[0]
    arr[:cut_r, cut_c:] = colors[1]
    arr[cut_r:, :cut_c] = colors[2]
    arr[cut_r:, cut_c:] = random.choice((colors[0], colors[1], colors[2]))
    return arr


def _build_cross(h: int, w: int) -> np.ndarray | None:
    if h < 4 or w < 4:
        return None
    colors = _rand_color_sample(3)
    arr = np.full((h, w), colors[0], dtype=int)
    v_width = random.randint(1, max(1, w // 3))
    h_height = random.randint(1, max(1, h // 3))
    v_start = random.randint(0, w - v_width)
    h_start = random.randint(0, h - h_height)
    arr[:, v_start : v_start + v_width] = colors[1]
    arr[h_start : h_start + h_height, :] = colors[2]
    return arr


def _make_interior(h: int, w: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        kind = random.choice(PATTERN_KINDS)
        if kind == "h2":
            interior = _build_h_bands(h, w, 2)
        elif kind == "v2":
            interior = _build_v_bands(h, w, 2)
        elif kind == "h3":
            interior = _build_h_bands(h, w, 3)
        elif kind == "v3":
            interior = _build_v_bands(h, w, 3)
        elif kind == "quad":
            interior = _build_quadrants(h, w)
        else:
            interior = _build_cross(h, w)
        if interior is not None and _interior_is_valid(interior):
            return interior
    return None


def _place_frame(grid: np.ndarray, occupied: np.ndarray, r0: int, c0: int, interior: np.ndarray) -> None:
    h, w = interior.shape
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior

    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r1 + 2)
    cc1 = min(grid.shape[1], c1 + 2)
    occupied[rr0:rr1, cc0:cc1] = True


def _attempt_layout() -> np.ndarray | None:
    h = random.randrange(*GRID_SIZE_BOUNDS)
    w = random.randrange(*GRID_SIZE_BOUNDS)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    placed = 0
    for _ in range(frame_count):
        success = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            outer_h = random.randrange(OUTER_HEIGHT_BOUNDS[0], min(OUTER_HEIGHT_BOUNDS[1], h - 1))
            outer_w = random.randrange(OUTER_WIDTH_BOUNDS[0], min(OUTER_WIDTH_BOUNDS[1], w - 1))
            if outer_h > h or outer_w > w:
                continue
            if h - outer_h < 0 or w - outer_w < 0:
                continue
            r0 = random.randint(0, h - outer_h)
            c0 = random.randint(0, w - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 2)
            cc1 = min(w, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            interior = _make_interior(outer_h - 2, outer_w - 2)
            if interior is None:
                continue
            _place_frame(grid, occupied, r0, c0, interior)
            success = True
            placed += 1
            break
        if not success:
            break

    if placed != frame_count:
        return None
    if len(_find_frames(grid)) != placed:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_layout()
        if grid is None:
            continue
        out = main(grid)
        if out.shape == grid.shape and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
