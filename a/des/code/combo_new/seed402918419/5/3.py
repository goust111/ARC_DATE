import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
RED = 2

BG_CHOICES = (1, 3, 4, 5, 6, 7, 8, 9)
SIDE_NAMES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    black = np.argwhere(grid == BLACK)
    if black.size == 0:
        raise ValueError("missing black frame")

    r0, c0, r1, c1 = _bbox(black)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("frame too small")

    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(frame[0, :] == BLACK):
        raise ValueError("top border broken")
    if not np.all(frame[-1, :] == BLACK):
        raise ValueError("bottom border broken")
    if not np.all(frame[:, 0] == BLACK):
        raise ValueError("left border broken")
    if not np.all(frame[:, -1] == BLACK):
        raise ValueError("right border broken")
    if np.any(frame[1:-1, 1:-1] == BLACK):
        raise ValueError("frame interior must stay empty")

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(np.count_nonzero(grid == BLACK)) != expected:
        raise ValueError("unexpected black pixels outside frame")
    return r0, c0, r1, c1


def _find_direction_marker(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    red = np.argwhere(grid == RED)
    if len(red) != 1:
        raise ValueError("expected exactly one red marker")

    r0, c0, r1, c1 = frame
    r, c = map(int, red[0])
    if r == r0 - 1 and c0 <= c <= c1:
        return "top", (r, c)
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom", (r, c)
    if c == c0 - 1 and r0 <= r <= r1:
        return "left", (r, c)
    if c == c1 + 1 and r0 <= r <= r1:
        return "right", (r, c)
    raise ValueError("red marker is not adjacent to a frame side")


def _collect_markers(
    grid: np.ndarray, bg: int, frame: tuple[int, int, int, int]
) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    markers: dict[int, tuple[int, int]] = {}
    for rr, cc in np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        r = int(r0 + 1 + rr)
        c = int(c0 + 1 + cc)
        color = int(grid[r, c])
        if color in (BLACK, RED):
            raise ValueError("interior marker uses reserved color")
        if color in markers:
            raise ValueError("marker colors must be unique")
        markers[color] = (r, c)
    if len(markers) not in (2, 3):
        raise ValueError("expected 2 or 3 interior markers")
    return markers


def _collect_stamp_pixels(
    grid: np.ndarray, bg: int, frame: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame
    mask = np.ones(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    pixels = []
    for r, c in np.argwhere(mask & (grid != bg) & (grid != BLACK) & (grid != RED)):
        pixels.append((int(r), int(c), int(grid[r, c])))
    if not pixels:
        raise ValueError("missing stamp")
    return pixels


def _find_translation(
    grid: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    markers: dict[int, tuple[int, int]],
    stamp_pixels: list[tuple[int, int, int]],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c, color in stamp_pixels:
        by_color[int(color)].append((int(r), int(c)))

    delta_sets: list[set[tuple[int, int]]] = []
    for color, marker_pos in markers.items():
        if color not in by_color:
            raise ValueError("stamp is missing a marker color")
        mr, mc = marker_pos
        delta_sets.append({(int(mr - sr), int(mc - sc)) for sr, sc in by_color[color]})

    candidates = set.intersection(*delta_sets)
    valid: list[tuple[int, int]] = []
    marker_occupants = {pos: color for color, pos in markers.items()}
    for dr, dc in sorted(candidates):
        covered = {color: False for color in markers}
        ok = True
        for sr, sc, color in stamp_pixels:
            tr = int(sr + dr)
            tc = int(sc + dc)
            if not (r0 + 1 <= tr <= r1 - 1 and c0 + 1 <= tc <= c1 - 1):
                ok = False
                break
            existing = int(grid[tr, tc])
            if existing != bg and existing != color:
                ok = False
                break
            if marker_occupants.get((tr, tc)) == color:
                covered[color] = True
        if ok and all(covered.values()):
            valid.append((int(dr), int(dc)))

    if len(valid) != 1:
        raise ValueError("expected a unique stamp translation")
    return valid[0]


def _project_from_side(
    grid: np.ndarray, frame: tuple[int, int, int, int], side: str
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = frame
    inner = out[r0 + 1 : r1, c0 + 1 : c1].copy()
    inner_h, inner_w = map(int, inner.shape)

    if side == "top":
        for step, row in enumerate(range(r0 - 1, -1, -1)):
            out[int(row), c0 + 1 : c1] = inner[int(step % inner_h), :]
    elif side == "bottom":
        for step, row in enumerate(range(r1 + 1, out.shape[0])):
            out[int(row), c0 + 1 : c1] = inner[int(inner_h - 1 - (step % inner_h)), :]
    elif side == "left":
        for step, col in enumerate(range(c0 - 1, -1, -1)):
            out[r0 + 1 : r1, int(col)] = inner[:, int(step % inner_w)]
    elif side == "right":
        for step, col in enumerate(range(c1 + 1, out.shape[1])):
            out[r0 + 1 : r1, int(col)] = inner[:, int(inner_w - 1 - (step % inner_w))]
    else:
        raise ValueError(f"unknown side {side}")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Place the external stamp inside the hollow frame by color anchors, then project the completed interior outward.

    Input structure:
    - The dominant color is the background.
    - A single black hollow rectangle encloses 2-3 differently colored singleton markers on background cells.
    - Outside the frame there is one small multicolor stamp whose color set matches the marker colors, plus one red direction pixel on the exterior midpoint of a frame side.

    Rule:
    - Find the unique translation that puts, for every marker color, a same-colored stamp pixel on that marker without rotating or flipping the stamp.
    - Draw the whole translated stamp inside the frame, keeping the frame itself unchanged.
    - Erase the original outside stamp and the red direction pixel, then repeat the interior rows or columns outward from the marked side in side-to-center order until the grid boundary.

    Key insight:
    - The colored interior markers determine only the stamp translation, while the red pixel independently selects which side emits the repeated outward projection.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame = _find_frame(g)
    side, red_pos = _find_direction_marker(g, frame)
    markers = _collect_markers(g, bg, frame)
    stamp_pixels = _collect_stamp_pixels(g, bg, frame)
    dr, dc = _find_translation(g, bg, frame, markers, stamp_pixels)

    out = g.copy()
    for r, c, _color in stamp_pixels:
        out[int(r), int(c)] = bg
    out[int(red_pos[0]), int(red_pos[1])] = bg

    for r, c, color in stamp_pixels:
        out[int(r + dr), int(c + dc)] = int(color)

    return _project_from_side(out, frame, side)


def _crop_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cells)


def _random_connected_shape(rng: random.Random, size: int) -> list[tuple[int, int]]:
    for _ in range(200):
        cells = {(rng.randrange(4), rng.randrange(4))}
        while len(cells) < size:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                cells.add((int(nr), int(nc)))
        shape = _crop_cells(cells)
        height = 1 + max(r for r, _ in shape)
        width = 1 + max(c for _, c in shape)
        if height <= 4 and width <= 4:
            return shape
    raise RuntimeError("failed to build a connected stamp")


def _choose_interior_dim(min_needed: int, rng: random.Random) -> int:
    choices = [dim for dim in (5, 7, 9, 11, 13) if dim >= min_needed]
    return int(rng.choice(choices))


def _projection_cells(
    frame: tuple[int, int, int, int], side: str, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    h, w = shape
    cells: set[tuple[int, int]] = set()
    if side == "top":
        for r in range(0, r0):
            for c in range(c0 + 1, c1):
                cells.add((int(r), int(c)))
    elif side == "bottom":
        for r in range(r1 + 1, h):
            for c in range(c0 + 1, c1):
                cells.add((int(r), int(c)))
    elif side == "left":
        for r in range(r0 + 1, r1):
            for c in range(0, c0):
                cells.add((int(r), int(c)))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c1 + 1, w):
                cells.add((int(r), int(c)))
    return cells


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_stamp_positions(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    side: str,
    stamp_shape: tuple[int, int],
    red_pos: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid_shape
    sh, sw = stamp_shape
    r0, c0, r1, c1 = frame
    padded_frame = (max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1))
    projection = _projection_cells(frame, side, grid_shape)
    choices: list[tuple[int, int]] = []

    for sr in range(0, h - sh + 1):
        for sc in range(0, w - sw + 1):
            stamp_box = (sr, sc, sr + sh - 1, sc + sw - 1)
            if _rectangles_overlap(stamp_box, padded_frame):
                continue
            if red_pos[0] >= sr and red_pos[0] < sr + sh and red_pos[1] >= sc and red_pos[1] < sc + sw:
                continue
            if any((sr + dr, sc + dc) in projection for dr in range(sh) for dc in range(sw)):
                continue
            choices.append((int(sr), int(sc)))

    if not choices:
        return None
    return rng.choice(choices)


def _build_stamp(
    rng: random.Random, marker_colors: list[int]
) -> tuple[list[tuple[int, int, int]], dict[int, tuple[int, int]], tuple[int, int]]:
    size = rng.randint(len(marker_colors) + 1, min(6, len(marker_colors) + 3))
    shape = _random_connected_shape(rng, size)
    color_cells: list[tuple[int, int, int]] = []
    anchor_by_color: dict[int, tuple[int, int]] = {}

    anchor_indices = rng.sample(range(len(shape)), len(marker_colors))
    index_to_color = {}
    for color, idx in zip(marker_colors, anchor_indices):
        index_to_color[int(idx)] = int(color)

    for idx, (r, c) in enumerate(shape):
        if idx in index_to_color:
            color = index_to_color[idx]
            anchor_by_color[color] = (int(r), int(c))
        else:
            color = int(rng.choice(marker_colors))
        color_cells.append((int(r), int(c), int(color)))

    height = 1 + max(r for r, _c, _color in color_cells)
    width = 1 + max(c for _r, c, _color in color_cells)
    return color_cells, anchor_by_color, (int(height), int(width))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.choice(BG_CHOICES))
    side = str(rng.choice(SIDE_NAMES))
    marker_count = int(rng.choice((2, 3)))
    marker_colors = rng.sample([c for c in BG_CHOICES if c != bg], marker_count)

    stamp_cells, anchors, (stamp_h, stamp_w) = _build_stamp(rng, marker_colors)
    inner_h = _choose_interior_dim(stamp_h, rng)
    inner_w = _choose_interior_dim(stamp_w, rng)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    grid_h = rng.randint(max(14, frame_h + 4), 30)
    grid_w = rng.randint(max(14, frame_w + 4), 30)

    top_min = 2 if side == "top" else 1
    bottom_min = 2 if side == "bottom" else 1
    left_min = 2 if side == "left" else 1
    right_min = 2 if side == "right" else 1

    if grid_h < frame_h + top_min + bottom_min or grid_w < frame_w + left_min + right_min:
        return None

    r0 = rng.randint(top_min, grid_h - frame_h - bottom_min)
    c0 = rng.randint(left_min, grid_w - frame_w - right_min)
    frame = (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1))

    if side == "top":
        red_pos = (int(r0 - 1), int(c0 + frame_w // 2))
    elif side == "bottom":
        red_pos = (int(r0 + frame_h), int(c0 + frame_w // 2))
    elif side == "left":
        red_pos = (int(r0 + frame_h // 2), int(c0 - 1))
    else:
        red_pos = (int(r0 + frame_h // 2), int(c0 + frame_w))

    stamp_origin = _find_stamp_positions(
        (grid_h, grid_w), frame, side, (stamp_h, stamp_w), red_pos, rng
    )
    if stamp_origin is None:
        return None
    stamp_r, stamp_c = stamp_origin

    place_r = rng.randint(0, inner_h - stamp_h)
    place_c = rng.randint(0, inner_w - stamp_w)

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    fr0, fc0, fr1, fc1 = frame
    grid[fr0, fc0 : fc1 + 1] = BLACK
    grid[fr1, fc0 : fc1 + 1] = BLACK
    grid[fr0 : fr1 + 1, fc0] = BLACK
    grid[fr0 : fr1 + 1, fc1] = BLACK
    grid[red_pos[0], red_pos[1]] = RED

    for color, (ar, ac) in anchors.items():
        grid[fr0 + 1 + place_r + ar, fc0 + 1 + place_c + ac] = int(color)

    for rr, cc, color in stamp_cells:
        grid[stamp_r + rr, stamp_c + cc] = int(color)

    try:
        markers = _collect_markers(grid, bg, frame)
        stamp_pixels = _collect_stamp_pixels(grid, bg, frame)
        _find_translation(grid, bg, frame, markers, stamp_pixels)
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != bg) <= np.count_nonzero(grid != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
