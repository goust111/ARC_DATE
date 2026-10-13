from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
DIRS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ROT_ORDER = ("up", "left", "down", "right")  # one np.rot90 step rotates up -> left
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def is_contiguous(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return bool(mask.any()) and len(components_mask(mask)) == 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("patch out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _midpoint(side: int, direction: str) -> tuple[int, int]:
    h = side // 2
    if direction == "up":
        return 0, h
    if direction == "down":
        return side - 1, h
    if direction == "left":
        return h, 0
    if direction == "right":
        return h, side - 1
    raise ValueError(direction)


def _rotation_steps(src: str, dst: str) -> int:
    return (ROT_ORDER.index(dst) - ROT_ORDER.index(src)) % 4


def _detect_nose_dir(patch: np.ndarray, bg: int) -> str:
    side = patch.shape[0]
    candidates: list[tuple[str, int, int]] = []
    for direction in DIRS:
        r, c = _midpoint(side, direction)
        color = int(patch[r, c])
        if color == bg:
            continue
        freq = int(np.count_nonzero(patch == color))
        same_neighbors = 0
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < side and 0 <= nc < side and int(patch[nr, nc]) == color:
                same_neighbors += 1
        candidates.append((direction, freq, same_neighbors))

    assert candidates, "expected a midpoint nose candidate"
    min_freq = min(freq for _direction, freq, _same_neighbors in candidates)
    narrowed = [item for item in candidates if item[1] == min_freq]
    if len(narrowed) > 1:
        min_same = min(same_neighbors for _direction, _freq, same_neighbors in narrowed)
        narrowed = [item for item in narrowed if item[2] == min_same]
    assert len(narrowed) == 1, "expected one uniquely identifiable nose direction"
    return narrowed[0][0]


def _nearest_marker(markers: np.ndarray, center: tuple[int, int], direction: str) -> tuple[int, int] | None:
    cr, cc = center
    if direction == "up":
        hits = [(int(r), int(c)) for r, c in markers if int(c) == cc and int(r) < cr]
        return max(hits, default=None, key=lambda rc: rc[0])
    if direction == "down":
        hits = [(int(r), int(c)) for r, c in markers if int(c) == cc and int(r) > cr]
        return min(hits, default=None, key=lambda rc: rc[0])
    if direction == "left":
        hits = [(int(r), int(c)) for r, c in markers if int(r) == cr and int(c) < cc]
        return max(hits, default=None, key=lambda rc: rc[1])
    hits = [(int(r), int(c)) for r, c in markers if int(r) == cr and int(c) > cc]
    return min(hits, default=None, key=lambda rc: rc[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the nearest gray marker on each sightline with a copy whose nose faces back to the source sprite.

    Input structure:
    - One colored sprite occupies an odd-by-odd square bounding box, and one edge midpoint cell is a differently colored nose pixel.
    - Several isolated gray marker pixels lie on the sprite bbox center row or center column, with zero, one, or more markers on each of the four rays.
    - Gray differs from the sprite colors and from the background.

    Rule:
    - Find the unique non-gray sprite, keep it unchanged, and use its bbox center as the sightline origin.
    - On each of the four directions, only the nearest gray marker is used: center-align a copy of the sprite on that marker and rotate it by 0/90/180/270 so the nose points toward the original sprite.
    - Remove every gray marker from the output, so farther same-ray markers disappear instead of being stamped.

    Key insight:
    - Marker position chooses where to stamp, but the sprite rotation must be inferred from the original nose side and the marker's direction relative to the source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg != GREY

    sprite_mask = (g != bg) & (g != GREY)
    components = components_mask(sprite_mask)
    assert len(components) == 1, "expected one non-gray sprite"

    sprite_bbox = bbox_coords(components[0])
    r0, c0, r1, c1 = sprite_bbox
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side % 2 == 1, "expected an odd square sprite bbox"

    patch = crop_bbox(g, sprite_bbox)
    center = (r0 + side // 2, c0 + side // 2)
    nose_dir = _detect_nose_dir(patch, bg)

    markers = np.argwhere(g == GREY)
    for mr, mc in markers:
        assert int(mr) == center[0] or int(mc) == center[1], "markers must stay on the sightlines"

    out = g.copy()
    out[out == GREY] = bg
    half = side // 2
    for direction in DIRS:
        marker = _nearest_marker(markers, center, direction)
        if marker is None:
            continue
        desired_nose = OPPOSITE[direction]
        rotated = np.rot90(patch, _rotation_steps(nose_dir, desired_nose))
        paste(out, rotated, marker[0] - half, marker[1] - half, transparent=bg)

    return out


def _gen_sprite_patch(side: int, bg: int, body_color: int, nose_color: int, nose_dir: str) -> np.ndarray:
    mask = np.ones((side, side), dtype=bool)
    interior = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    center = (side // 2, side // 2)
    random.shuffle(interior)
    if center in interior and random.random() < 0.55:
        interior.remove(center)
        interior = [center] + interior

    min_removals = 1 if interior and random.random() < 0.75 else 0
    target_removals = random.randint(min_removals, len(interior)) if interior else 0
    removed = 0
    for cell in interior:
        if removed >= target_removals:
            break
        trial = mask.copy()
        trial[cell] = False
        if is_contiguous(trial):
            mask = trial
            removed += 1

    patch = np.full((side, side), bg, dtype=int)
    patch[mask] = body_color
    nr, nc = _midpoint(side, nose_dir)
    patch[nr, nc] = nose_color
    return patch


def _gen_marker_counts(side: int) -> dict[str, int]:
    counts = {direction: 0 for direction in DIRS}
    if side == 7:
        active = random.sample(DIRS, random.choice((2, 3)))
        for direction in active:
            counts[direction] = 1
        if random.random() < 0.7:
            counts[random.choice(active)] = 2
        return counts

    active = random.sample(DIRS, random.choice((2, 3, 4)))
    for direction in active:
        counts[direction] = 1 if random.random() < 0.45 else 2
    if all(counts[direction] == 1 for direction in active):
        counts[random.choice(active)] = 2
    return counts


def _gen_distances(side: int, count: int) -> list[int]:
    if count == 0:
        return []
    half = side // 2
    distances = [random.randint(side, side + 2)]
    if count >= 2:
        distances.append(distances[0] + random.randint(half + 3, half + 4))
    return distances


def _side_need(side: int, distances: list[int]) -> int:
    half = side // 2
    if not distances:
        return half
    return max(distances[-1], distances[0] + half)


def _ray_coord(center: tuple[int, int], direction: str, distance: int) -> tuple[int, int]:
    r, c = center
    if direction == "up":
        return r - distance, c
    if direction == "down":
        return r + distance, c
    if direction == "left":
        return r, c - distance
    return r, c + distance


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    patch: np.ndarray,
    sprite_origin: tuple[int, int],
    nose_dir: str,
    markers_by_dir: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    side = patch.shape[0]
    half = side // 2
    out = np.full(shape, bg, dtype=int)
    paste(out, patch, sprite_origin[0], sprite_origin[1], transparent=bg)
    for direction, markers in markers_by_dir.items():
        if not markers:
            continue
        rotated = np.rot90(patch, _rotation_steps(nose_dir, OPPOSITE[direction]))
        mr, mc = markers[0]
        paste(out, rotated, mr - half, mc - half, transparent=bg)
    return out


def _disjoint_boxes(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for rr0, cc0, rr1, cc1 in boxes[i + 1 :]:
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False
    return True


def _validate_layout(
    inp: np.ndarray,
    expected: np.ndarray,
    patch: np.ndarray,
    sprite_origin: tuple[int, int],
    nose_dir: str,
    markers_by_dir: dict[str, list[tuple[int, int]]],
    nose_color: int,
    bg: int,
) -> bool:
    side = patch.shape[0]
    half = side // 2

    if np.any(expected == GREY):
        return False
    if np.count_nonzero(expected == nose_color) != 1 + sum(bool(markers) for markers in markers_by_dir.values()):
        return False
    sr, sc = sprite_origin
    if not np.array_equal(expected[sr : sr + side, sc : sc + side], patch):
        return False

    boxes = [(sr, sc, sr + side - 1, sc + side - 1)]
    for direction, markers in markers_by_dir.items():
        if not markers:
            continue
        mr, mc = markers[0]
        rr = mr - half
        cc = mc - half
        rotated = np.rot90(patch, _rotation_steps(nose_dir, OPPOSITE[direction]))
        if not np.array_equal(expected[rr : rr + side, cc : cc + side], rotated):
            return False
        boxes.append((rr, cc, rr + side - 1, cc + side - 1))
        for blocked in markers[1:]:
            if int(expected[blocked]) != bg:
                return False

    if not _disjoint_boxes(boxes):
        return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice((3, 5, 5, 7))
    counts = _gen_marker_counts(side)
    distances = {direction: _gen_distances(side, counts[direction]) for direction in DIRS}

    min_height = _side_need(side, distances["up"]) + 1 + _side_need(side, distances["down"])
    min_width = _side_need(side, distances["left"]) + 1 + _side_need(side, distances["right"])
    if min_height > 30 or min_width > 30:
        return None

    bg = random.choice(COLOR_POOL)
    sprite_colors = [color for color in COLOR_POOL if color not in {bg, GREY}]
    body_color, nose_color = random.sample(sprite_colors, 2)
    nose_dir = random.choice(DIRS)
    patch = _gen_sprite_patch(side, bg, body_color, nose_color, nose_dir)

    extra_rows = random.randint(0, 30 - min_height)
    top_extra = random.randint(0, extra_rows)
    extra_cols = random.randint(0, 30 - min_width)
    left_extra = random.randint(0, extra_cols)

    height = min_height + extra_rows
    width = min_width + extra_cols
    center = (_side_need(side, distances["up"]) + top_extra, _side_need(side, distances["left"]) + left_extra)
    sprite_origin = (center[0] - side // 2, center[1] - side // 2)

    inp = np.full((height, width), bg, dtype=int)
    paste(inp, patch, sprite_origin[0], sprite_origin[1], transparent=bg)

    markers_by_dir: dict[str, list[tuple[int, int]]] = {direction: [] for direction in DIRS}
    for direction in DIRS:
        for distance in distances[direction]:
            marker = _ray_coord(center, direction, distance)
            markers_by_dir[direction].append(marker)
            inp[marker] = GREY

    expected = _render_expected(inp.shape, bg, patch, sprite_origin, nose_dir, markers_by_dir)
    if not _validate_layout(inp, expected, patch, sprite_origin, nose_dir, markers_by_dir, nose_color, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid line-of-sight stamp input")
