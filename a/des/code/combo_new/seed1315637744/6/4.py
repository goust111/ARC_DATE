import random

import numpy as np


BG = 0
DIVIDER_COLOR = 5
BRANCHES = ("horizontal", "vertical", "rotational", "none")
MAX_GENERATION_ATTEMPTS = 400
_BRANCH_CURSOR = 0


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    matches = []
    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color == col_color:
                matches.append((row_color, r, c))

    assert len(matches) == 1, "expected exactly one full-row/full-column divider"
    color, mid_r, mid_c = matches[0]
    assert 0 < mid_r < h - 1 and 0 < mid_c < w - 1
    assert h == 2 * mid_r + 1 and w == 2 * mid_c + 1
    return int(color), int(mid_r), int(mid_c)


def _trim_patch(patch: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(patch != bg)
    assert coords.size > 0, "seed patch cannot be empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _symmetry_flags(patch: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(patch, np.flipud(patch))),
        bool(np.array_equal(patch, np.fliplr(patch))),
        bool(np.array_equal(patch, np.rot90(patch, 2))),
    )


def _classify_patch(patch: np.ndarray) -> str:
    horizontal, vertical, rotational = _symmetry_flags(patch)
    if horizontal and not vertical and not rotational:
        return "horizontal"
    if vertical and not horizontal and not rotational:
        return "vertical"
    if rotational and not horizontal and not vertical:
        return "rotational"
    if not horizontal and not vertical and not rotational:
        return "none"
    raise AssertionError("seed must belong to exactly one supported symmetry class")


def _paste_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for r0, c0 in np.argwhere(patch != bg):
        r = top + int(r0)
        c = left + int(c0)
        assert 0 <= r < h and 0 <= c < w
        if (r, c) in corners:
            continue
        value = int(patch[r0, c0])
        existing = int(out[r, c])
        assert existing in (bg, value), "copied seed pixels must not conflict"
        out[r, c] = value


def _branch_transforms(seed: np.ndarray, branch: str) -> dict[str, np.ndarray]:
    if branch == "horizontal":
        return {
            "ru": np.fliplr(seed),
            "lb": np.flipud(seed),
            "rd": np.flipud(np.fliplr(seed)),
        }
    if branch == "vertical":
        return {
            "ru": np.rot90(seed, -1),
            "rd": np.rot90(seed, 2),
            "lb": np.rot90(seed, 1),
        }
    if branch == "rotational":
        return {"rd": seed.copy()}
    assert branch == "none"
    return {"ru": seed.copy(), "lb": seed.copy(), "rd": seed.copy()}


def _quadrant_origins(mid_r: int, mid_c: int) -> dict[str, tuple[int, int]]:
    return {
        "ru": (0, mid_c + 1),
        "lb": (mid_r + 1, 0),
        "rd": (mid_r + 1, mid_c + 1),
    }


def _cleared_quadrant(quadrant: np.ndarray, marker_pos: tuple[int, int], bg: int) -> np.ndarray:
    out = quadrant.copy()
    out[marker_pos] = bg
    return out


def _seed_from_grid(grid: np.ndarray, bg: int, mid_r: int, mid_c: int) -> tuple[np.ndarray, int, int]:
    tl = _cleared_quadrant(grid[:mid_r, :mid_c], (0, 0), bg)
    coords = np.argwhere(tl != bg)
    assert coords.size > 0, "top-left quadrant must contain a seed pattern"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    seed = tl[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    return seed, int(r0), int(c0)


def _expected_output(grid: np.ndarray, seed: np.ndarray, top: int, left: int, branch: str) -> np.ndarray:
    mid_r = grid.shape[0] // 2
    mid_c = grid.shape[1] // 2
    out = grid.copy()
    for name, transformed in _branch_transforms(seed, branch).items():
        origin_r, origin_c = _quadrant_origins(mid_r, mid_c)[name]
        _paste_patch(out, transformed, origin_r + top, origin_c + left, BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the top-left seed into the other quadrants by first detecting which symmetry class the seed itself has.

    Input structure
    - The grid is split into four equal quadrants by one full horizontal divider row and one full vertical divider column of the same non-background color.
    - The top-left quadrant contains a multicolor seed pattern on the background, while the other three quadrants are empty except for single-pixel corner markers.
    - The four corners hold two marker colors arranged as top-left = bottom-right and top-right = bottom-left.

    Rule
    - Ignore the corner markers when extracting the top-left seed and trim the seed to its non-background bounding box.
    - If the seed has horizontal reflection symmetry, stamp its left-right flip in the top-right quadrant, its top-bottom flip in the bottom-left quadrant, and the double flip in the bottom-right quadrant.
    - If the seed has vertical reflection symmetry, stamp 90-degree clockwise, 270-degree clockwise, and 180-degree rotated copies into the top-right, bottom-left, and bottom-right quadrants respectively.
    - If the seed has only 180-degree rotational symmetry, copy it only into the bottom-right quadrant; if it has no symmetry, copy it unchanged into all three empty quadrants. Keep the divider and corner markers unchanged.

    Key insight
    - The corner colors are distractors: the correct branch is chosen solely from the seed pattern's intrinsic symmetry type, then the matching quadrant-filling rule is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    divider_color, mid_r, mid_c = _find_divider(g, bg)

    tl_marker = int(g[0, 0])
    tr_marker = int(g[0, -1])
    assert tl_marker != tr_marker
    assert tl_marker == int(g[-1, -1])
    assert tr_marker == int(g[-1, 0])
    assert tl_marker not in (bg, divider_color)
    assert tr_marker not in (bg, divider_color)

    qh, qw = mid_r, mid_c
    ru = _cleared_quadrant(g[:qh, mid_c + 1 :], (0, qw - 1), bg)
    lb = _cleared_quadrant(g[mid_r + 1 :, :qw], (qh - 1, 0), bg)
    rd = _cleared_quadrant(g[mid_r + 1 :, mid_c + 1 :], (qh - 1, qw - 1), bg)
    assert np.all(ru == bg) and np.all(lb == bg) and np.all(rd == bg)

    seed, top, left = _seed_from_grid(g, bg, mid_r, mid_c)
    branch = _classify_patch(seed)

    out = g.copy()
    origins = _quadrant_origins(mid_r, mid_c)
    for name, transformed in _branch_transforms(seed, branch).items():
        origin_r, origin_c = origins[name]
        _paste_patch(out, transformed, origin_r + top, origin_c + left, bg)
    return out


def _horizontal_orbits(height: int, width: int) -> list[list[tuple[int, int]]]:
    orbits = []
    for r in range((height + 1) // 2):
        mirror_r = height - 1 - r
        for c in range(width):
            if r == mirror_r:
                orbits.append([(r, c)])
            else:
                orbits.append([(r, c), (mirror_r, c)])
    return orbits


def _vertical_orbits(height: int, width: int) -> list[list[tuple[int, int]]]:
    orbits = []
    for r in range(height):
        for c in range((width + 1) // 2):
            mirror_c = width - 1 - c
            if c == mirror_c:
                orbits.append([(r, c)])
            else:
                orbits.append([(r, c), (r, mirror_c)])
    return orbits


def _rotational_orbits(height: int, width: int) -> list[list[tuple[int, int]]]:
    orbits = []
    seen = set()
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            mate = (height - 1 - r, width - 1 - c)
            if mate == (r, c):
                orbit = [(r, c)]
            else:
                orbit = [(r, c), mate]
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _paint_from_orbits(height: int, width: int, orbits: list[list[tuple[int, int]]], colors: list[int]) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    orbit_order = orbits[:]
    random.shuffle(orbit_order)
    min_count = max(3, int(np.ceil(len(orbit_order) * 0.25)))
    max_count = max(min_count, int(np.ceil(len(orbit_order) * 0.55)))
    count = random.randint(min_count, min(max_count, len(orbit_order)))
    active_orbits = orbit_order[:count]

    color_count = min(random.randint(2, 4), len(colors), len(active_orbits))
    chosen = random.sample(colors, color_count)
    assignments = chosen + [random.choice(chosen) for _ in range(len(active_orbits) - color_count)]
    random.shuffle(assignments)

    for orbit, color in zip(active_orbits, assignments, strict=False):
        for r, c in orbit:
            patch[r, c] = color
    return patch


def _paint_random_patch(height: int, width: int, colors: list[int]) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    cells = [(r, c) for r in range(height) for c in range(width)]
    min_count = max(5, int(np.ceil(height * width * 0.25)))
    max_count = min(height * width - 1, max(min_count, int(np.floor(height * width * 0.55))))
    count = random.randint(min_count, max_count)
    positions = random.sample(cells, count)

    color_count = min(random.randint(2, 4), len(colors), count)
    chosen = random.sample(colors, color_count)
    assignments = chosen + [random.choice(chosen) for _ in range(count - color_count)]
    random.shuffle(assignments)

    for (r, c), color in zip(positions, assignments, strict=False):
        patch[r, c] = color
    return patch


def _valid_patch(patch: np.ndarray) -> bool:
    trimmed = _trim_patch(patch)
    if trimmed.shape[0] < 3 or trimmed.shape[1] < 3:
        return False
    colors = {int(color) for color in np.unique(trimmed) if int(color) != BG}
    if len(colors) < 2:
        return False
    area = int(np.count_nonzero(trimmed != BG))
    if area < 5:
        return False
    density = area / float(trimmed.size)
    return density <= 0.7


def _make_patch(branch: str, max_h: int, max_w: int, exclude: set[int]) -> np.ndarray | None:
    colors = [color for color in range(10) if color not in exclude]
    if len(colors) < 2:
        return None

    for _ in range(200):
        if branch == "vertical":
            limit = min(max_h, max_w)
            if limit < 3:
                return None
            height = random.randint(3, limit)
            width = random.randint(3, limit)
        else:
            if max_h < 3 or max_w < 3:
                return None
            height = random.randint(3, max_h)
            width = random.randint(3, max_w)

        if branch == "horizontal":
            candidate = _paint_from_orbits(height, width, _horizontal_orbits(height, width), colors)
        elif branch == "vertical":
            candidate = _paint_from_orbits(height, width, _vertical_orbits(height, width), colors)
        elif branch == "rotational":
            candidate = _paint_from_orbits(height, width, _rotational_orbits(height, width), colors)
        else:
            candidate = _paint_random_patch(height, width, colors)

        if not _valid_patch(candidate):
            continue
        trimmed = _trim_patch(candidate)
        try:
            classified = _classify_patch(trimmed)
        except AssertionError:
            continue
        if classified == branch:
            return trimmed
    return None


def _next_branch() -> str:
    global _BRANCH_CURSOR
    branch = BRANCHES[_BRANCH_CURSOR % len(BRANCHES)]
    _BRANCH_CURSOR += 1
    return branch


def _build_input(qh: int, qw: int, seed: np.ndarray, top: int, left: int, marker_a: int, marker_b: int) -> np.ndarray:
    h, w = 2 * qh + 1, 2 * qw + 1
    grid = np.full((h, w), BG, dtype=int)
    mid_r, mid_c = qh, qw
    grid[mid_r, :] = DIVIDER_COLOR
    grid[:, mid_c] = DIVIDER_COLOR

    grid[0, 0] = marker_a
    grid[0, -1] = marker_b
    grid[-1, 0] = marker_b
    grid[-1, -1] = marker_a

    _paste_patch(grid, seed, top, left, BG)
    return grid


def _gen_attempt(branch: str) -> np.ndarray | None:
    qh = random.randint(7, 14)
    qw = random.randint(7, 14)

    marker_pool = [color for color in range(10) if color not in (BG, DIVIDER_COLOR)]
    marker_a, marker_b = random.sample(marker_pool, 2)
    seed = _make_patch(branch, qh - 2, qw - 2, {BG, DIVIDER_COLOR, marker_a, marker_b})
    if seed is None:
        return None

    seed_h, seed_w = seed.shape
    if branch == "vertical":
        used_h = used_w = max(seed_h, seed_w)
    else:
        used_h, used_w = seed_h, seed_w

    if qh - used_h - 1 < 1 or qw - used_w - 1 < 1:
        return None

    top = random.randint(1, qh - used_h - 1)
    left = random.randint(1, qw - used_w - 1)
    grid = _build_input(qh, qw, seed, top, left, marker_a, marker_b)
    expected = _expected_output(grid, seed, top, left, branch)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    branch = _next_branch()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {branch} sample")
