from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 3
OBSTACLE = 5
OCCLUDER = 7
PATTERN_COLORS = (1, 2, 4, 6, 8, 9)
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _adjacent_to_color(grid: np.ndarray, pos: Coord, color: int) -> bool:
    return any(
        _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color
        for rr, cc in _neighbors4(*pos)
    )


def _marker_pos(grid: np.ndarray) -> Coord:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (1, 2), "expected exactly one marker pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _sightline_to_obstacle(grid: np.ndarray, start: Coord) -> tuple[Coord, Coord]:
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = start
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = sr + dr, sc + dc
        previous_bg: Coord | None = None
        while _in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value == OBSTACLE:
                assert previous_bg is not None, "marker cannot touch obstacle"
                hits.append(((dr, dc), previous_bg))
                break
            if value != BG:
                break
            previous_bg = (r, c)
            r += dr
            c += dc

    assert len(hits) == 1, "expected exactly one clear sightline to obstacle"
    return hits[0]


def _walk_contour(grid: np.ndarray, start: Coord, direction: Coord) -> Coord:
    dr, dc = direction
    turns = [(dr, -1), (dr, 1)] if dr else [(-1, dc), (1, dc)]
    paths: list[tuple[int, Coord]] = []
    for tr, tc in turns:
        r, c = start
        length = 0
        while True:
            nr, nc = r + tr, c + tc
            if not _in_bounds(grid.shape, nr, nc):
                break
            if int(grid[nr, nc]) != BG:
                break
            if not _adjacent_to_color(grid, (nr, nc), OBSTACLE):
                break
            r, c = nr, nc
            length += 1
        paths.append((length, (r, c)))

    best_len = max(length for length, _ in paths)
    best_endpoints = {endpoint for length, endpoint in paths if length == best_len}
    assert len(best_endpoints) == 1, "expected a unique longest contour walk"
    return best_endpoints.pop()


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BG) & (grid != MARKER) & (grid != OBSTACLE)
    return _bbox(mask)


def _fill_occlusion(grid: np.ndarray, axis: str) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    pr0, pc0, pr1, pc1 = _pattern_bbox(out)
    orange_cells = np.argwhere(out == OCCLUDER)
    assert orange_cells.size > 0, "expected an occlusion rectangle"
    or0, oc0, or1, oc1 = _bbox(out == OCCLUDER)
    assert np.all(out[or0 : or1 + 1, oc0 : oc1 + 1] == OCCLUDER), "occlusion must be rectangular"

    for r, c in orange_cells:
        rr = int(r)
        cc = int(c)
        if axis == "vertical":
            src = (rr, pc0 + pc1 - cc)
        else:
            src = (pr0 + pr1 - rr, cc)
        sr, sc = src
        assert pr0 <= sr <= pr1 and pc0 <= sc <= pc1, "reflection source must stay inside pattern bbox"
        assert int(out[sr, sc]) != OCCLUDER, "chosen axis must reflect from visible pattern cells"
        out[rr, cc] = int(out[sr, sc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the orange-occluded symmetric pattern using the obstacle contour walk to choose the reflection axis.

    Input structure:
    - A black background contains one large colored rectangular pattern and one solid orange rectangular occlusion inside it.
    - Outside that pattern sits one gray connected obstacle and one single green marker pixel with exactly one clear cardinal sightline to the obstacle.
    - The visible pattern is symmetric across both its horizontal and vertical midlines, while the occlusion is placed so only one of those axes exposes the hidden cells.

    Rule:
    - Slide from the green marker along its unique sightline to the last background cell before the gray obstacle.
    - From there, try the two diagonal contour-following turns and continue along the longer background path that stays 4-neighbor-adjacent to the obstacle.
    - Compare the marker's start-to-end displacement: a horizontal-major displacement selects the vertical reflection axis, and a vertical-major displacement selects the horizontal reflection axis.
    - Replace each orange pixel by the color at its mirror position across the selected axis; leave every other pixel unchanged.

    Key insight:
    - The contour walk does not place or move any visible object; its net direction is only a code that tells you which symmetry axis reveals the hidden side of the pattern.
    """
    g = np.asarray(grid, dtype=int)
    start = _marker_pos(g)
    direction, contour_start = _sightline_to_obstacle(g, start)
    end = _walk_contour(g, contour_start, direction)
    dr = int(end[0] - start[0])
    dc = int(end[1] - start[1])
    assert abs(dr) != abs(dc), "direction signal must have a unique principal axis"
    axis = "vertical" if abs(dc) > abs(dr) else "horizontal"
    return _fill_occlusion(g, axis)


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    mask = src != BG
    region = dst[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((region != BG) & mask), "objects must not overlap"
    region[mask] = src[mask]


def _mirror_pattern(seed: np.ndarray, h: int, w: int) -> np.ndarray:
    pattern = np.empty((h, w), dtype=int)
    for r in range(h):
        rr = min(r, h - 1 - r)
        for c in range(w):
            cc = min(c, w - 1 - c)
            pattern[r, c] = int(seed[rr, cc])
    return pattern


def _generate_pattern(h: int, w: int) -> np.ndarray:
    seed_h = (h + 1) // 2
    seed_w = (w + 1) // 2
    for _ in range(80):
        palette = random.sample(list(PATTERN_COLORS), k=random.randint(3, min(5, len(PATTERN_COLORS))))
        seed = np.array([[random.choice(palette) for _ in range(seed_w)] for _ in range(seed_h)], dtype=int)
        pattern = _mirror_pattern(seed, h, w)
        if len(np.unique(pattern)) >= 3:
            return pattern
    raise RuntimeError("failed to generate a diverse symmetric pattern")


def _occlude_pattern(pattern: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = pattern.shape
    inp = pattern.copy()

    if axis == "vertical":
        max_trim = max(1, h // 2 - 1)
        trim = random.randint(1, max_trim)
        top = trim
        bottom = h - trim
        max_band = max(2, w // 2 - 1)
        band_w = random.randint(2, max_band)
        if random.random() < 0.5:
            left, right = 0, band_w
        else:
            left, right = w - band_w, w
        inp[top:bottom, left:right] = OCCLUDER
    else:
        max_trim = max(1, w // 2 - 1)
        trim = random.randint(1, max_trim)
        left = trim
        right = w - trim
        max_band = max(2, h // 2 - 1)
        band_h = random.randint(2, max_band)
        if random.random() < 0.5:
            top, bottom = 0, band_h
        else:
            top, bottom = h - band_h, h
        inp[top:bottom, left:right] = OCCLUDER

    assert np.any(inp == OCCLUDER)
    return inp, pattern.copy()


def _gen_boundary(h: int, start_row: int, walk_len: int, start_left: int) -> list[int]:
    lefts = [0] * h
    lefts[start_row] = start_left

    for r in range(start_row - 1, -1, -1):
        if lefts[r + 1] == 0:
            lefts[r] = 0
        else:
            lefts[r] = lefts[r + 1] - random.randint(0, 1)

    for i in range(1, walk_len + 1):
        lefts[start_row + i] = start_left + i

    stop_row = start_row + walk_len + 1
    if stop_row < h:
        lefts[stop_row] = lefts[stop_row - 1]
        for r in range(stop_row + 1, h):
            if lefts[r - 1] < 4 and random.random() < 0.5:
                lefts[r] = lefts[r - 1] + 1
            else:
                lefts[r] = lefts[r - 1]
    return lefts


def _base_scene() -> np.ndarray | None:
    h = random.randint(4, 6)
    walk_len = random.randint(1, min(3, h - 3))
    start_row = random.randint(1, h - walk_len - 2)
    gap = random.randint(2, 4)
    max_start_left = min(gap, 4 - walk_len)
    if max_start_left < 1:
        return None
    start_left = random.randint(1, max_start_left)

    lefts = _gen_boundary(h, start_row, walk_len, start_left)
    min_left = min(lefts)
    max_left = max(lefts)
    sum_lefts = sum(lefts)

    w_min = max(3, max_left + 1, min_left + 3, (8 + sum_lefts + h - 1) // h)
    w_max = min(5, (15 + sum_lefts) // h)
    if w_min > w_max:
        return None
    width = random.randint(w_min, w_max)
    area = h * width - sum_lefts
    if not (8 <= area <= 15):
        return None

    c0 = gap - start_left + 1
    scene = np.full((h, c0 + width), BG, dtype=int)
    for rr, left in enumerate(lefts):
        scene[rr, c0 + left : c0 + width] = OBSTACLE
    scene[start_row, 0] = MARKER
    return scene


def _scene_signal_axis(scene: np.ndarray) -> str:
    start = _marker_pos(scene)
    direction, contour_start = _sightline_to_obstacle(scene, start)
    end = _walk_contour(scene, contour_start, direction)
    dr = int(end[0] - start[0])
    dc = int(end[1] - start[1])
    assert abs(dr) != abs(dc)
    return "vertical" if abs(dc) > abs(dr) else "horizontal"


def _generate_scene(target_axis: str) -> np.ndarray:
    displacement_major = "horizontal" if target_axis == "vertical" else "vertical"
    rotations = [0, 2] if displacement_major == "horizontal" else [1, 3]

    for _ in range(120):
        base = _base_scene()
        if base is None:
            continue
        scene = np.rot90(base, random.choice(rotations)).copy()
        if _scene_signal_axis(scene) == target_axis:
            return scene
    raise RuntimeError("failed to generate contour scene")


def _assemble_grid(pattern_in: np.ndarray, pattern_out: np.ndarray, scene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph, pw = pattern_in.shape
    sh, sw = scene.shape
    sep = 2
    extra = lambda: random.randint(1, 3)
    jitter = lambda: random.randint(0, 2)
    corner = random.choice(["tl", "tr", "bl", "br"])

    if corner == "tl":
        scene_r0, scene_c0 = 0, 0
        pattern_r0 = sh + sep + jitter()
        pattern_c0 = sw + sep + jitter()
        h = pattern_r0 + ph + extra()
        w = pattern_c0 + pw + extra()
    elif corner == "tr":
        pattern_r0 = sh + sep + jitter()
        pattern_c0 = extra()
        scene_r0 = 0
        scene_c0 = pattern_c0 + pw + sep + jitter()
        h = pattern_r0 + ph + extra()
        w = scene_c0 + sw + extra()
    elif corner == "bl":
        pattern_r0 = extra()
        pattern_c0 = sw + sep + jitter()
        scene_r0 = pattern_r0 + ph + sep + jitter()
        scene_c0 = 0
        h = scene_r0 + sh + extra()
        w = pattern_c0 + pw + extra()
    else:
        pattern_r0 = extra()
        pattern_c0 = extra()
        scene_r0 = pattern_r0 + ph + sep + jitter()
        scene_c0 = pattern_c0 + pw + sep + jitter()
        h = scene_r0 + sh + extra()
        w = scene_c0 + sw + extra()

    assert h <= 30 and w <= 30
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw] = pattern_in
    out[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw] = pattern_out
    _paste_nonzero(inp, scene, scene_r0, scene_c0)
    _paste_nonzero(out, scene, scene_r0, scene_c0)
    return inp, out


def _attempt_input() -> np.ndarray | None:
    axis = random.choice(["vertical", "horizontal"])
    scene = _generate_scene(axis)
    ph = random.randint(9, 13)
    pw = random.randint(9, 13)
    pattern = _generate_pattern(ph, pw)
    pattern_in, pattern_out = _occlude_pattern(pattern, axis)
    inp, expected = _assemble_grid(pattern_in, pattern_out, scene)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
