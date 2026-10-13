from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
MAX_GENERATION_ATTEMPTS = 600

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) == bg:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[Coord] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            assert arr.size > 0
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            box = (int(r0), int(c0), int(r1), int(c1))
            comps.append(
                {
                    "coords": arr,
                    "bbox": box,
                    "patch": _crop_bbox(g, box),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _rotate_patch(patch: np.ndarray, delta_dir: int) -> np.ndarray:
    if delta_dir == 1:
        return np.rot90(patch, -1)
    if delta_dir == 3:
        return np.rot90(patch, 1)
    raise AssertionError("expected a quarter-turn direction delta")


def _center_from_bbox(box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 1 and w % 2 == 1
    return (r0 + r1) // 2, (c0 + c1) // 2


def _direction_and_distance(pivot: Coord, center: Coord) -> tuple[int, int]:
    dr = int(center[0] - pivot[0])
    dc = int(center[1] - pivot[1])
    assert (dr == 0) ^ (dc == 0), "copy center must lie on one cardinal ray"
    if dr < 0:
        return 0, -dr
    if dc > 0:
        return 1, dc
    if dr > 0:
        return 2, dr
    return 3, -dc


def _place_patch_by_center(dst: np.ndarray, patch: np.ndarray, center: Coord, bg: int) -> None:
    h, w = patch.shape
    assert h % 2 == 1 and w % 2 == 1
    r0 = int(center[0] - h // 2)
    c0 = int(center[1] - w // 2)
    r1 = r0 + h
    c1 = c0 + w
    assert 0 <= r0 and 0 <= c0 and r1 <= dst.shape[0] and c1 <= dst.shape[1]
    patch_mask = patch != bg
    dst_view = dst[r0:r1, c0:c1]
    assert not np.any((dst_view != bg) & patch_mask), "new copy overlaps existing content"
    dst_view[patch_mask] = patch[patch_mask]


def _singleton_pivot(grid: np.ndarray, bg: int) -> Coord:
    counts = Counter(map(int, grid.ravel()))
    pivot_colors = [color for color, count in counts.items() if color != bg and int(count) == 1]
    assert len(pivot_colors) == 1, "expected exactly one uniquely colored pivot pixel"
    pivot_color = pivot_colors[0]
    coords = np.argwhere(grid == pivot_color)
    assert coords.shape[0] == 1
    return int(coords[0, 0]), int(coords[0, 1])


def _parse_orbit(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pivot = _singleton_pivot(g, bg)

    work = g.copy()
    work[pivot] = bg
    components = _find_components(work, bg)
    assert len(components) == 3, "expected three sprite copies"

    items = []
    for comp in components:
        center = _center_from_bbox(comp["bbox"])
        dir_idx, dist = _direction_and_distance(pivot, center)
        items.append(
            {
                "bbox": comp["bbox"],
                "patch": comp["patch"],
                "center": center,
                "dir": dir_idx,
                "dist": int(dist),
                "area": comp["area"],
            }
        )

    items.sort(key=lambda item: item["dist"])
    dists = [item["dist"] for item in items]
    assert dists[0] < dists[1] < dists[2]
    step1 = dists[1] - dists[0]
    step2 = dists[2] - dists[1]
    assert step1 == step2 and step1 > 0

    diff1 = (items[1]["dir"] - items[0]["dir"]) % 4
    diff2 = (items[2]["dir"] - items[1]["dir"]) % 4
    assert diff1 == diff2 and diff1 in (1, 3), "expected a consistent clockwise or counterclockwise turn"
    delta_dir = diff1

    for prev, cur in zip(items, items[1:]):
        rotated = _rotate_patch(prev["patch"], delta_dir)
        assert np.array_equal(rotated, cur["patch"]), "copy orientations must follow the orbit turn"

    return {
        "bg": bg,
        "pivot": pivot,
        "items": items,
        "step": step1,
        "delta_dir": delta_dir,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate one more rotating sprite frame in a growing spiral around a unique pivot pixel.

    Input structure:
    - A background grid contains one uniquely colored pivot pixel and three connected copies of the
      same odd-sized sprite; the sprite itself has no 90-degree rotational symmetry.
    - Each copy has a unique bbox center, and that center lies on one cardinal ray from the pivot.
    - When the copies are ordered by pivot distance, their directions differ by 90 degrees and
      their distances form an arithmetic progression.
    - The sprite image is quarter-turned in the same sense between consecutive copies.

    Rule:
    - Find the pivot and sort the three copies by increasing pivot distance.
    - Infer whether the orbit turns clockwise or counterclockwise, and recover the common distance
      step from the ordered copy centers.
    - Rotate the farthest copy by one more matching quarter-turn and place it on the next cardinal
      ray at one more distance step from the pivot, using the sprite center as the orbit anchor.
    - Keep the pivot pixel and the original three copies unchanged.

    Key insight:
    - The visible copies are consecutive frames of a spiral motion, not a missing symmetric quartet:
      the next object is determined only by extrapolating direction, radius, and sprite rotation
      together.
    """
    parsed = _parse_orbit(np.asarray(grid, dtype=int))
    out = np.asarray(grid, dtype=int).copy()

    last = parsed["items"][-1]
    next_patch = _rotate_patch(last["patch"], parsed["delta_dir"])
    next_dir = (last["dir"] + parsed["delta_dir"]) % 4
    next_dist = last["dist"] + parsed["step"]
    dr, dc = DIRS[next_dir]
    next_center = (
        int(parsed["pivot"][0] + dr * next_dist),
        int(parsed["pivot"][1] + dc * next_dist),
    )
    _place_patch_by_center(out, next_patch, next_center, parsed["bg"])
    return out


def _rotation_cycle(sprite: np.ndarray, delta_dir: int) -> list[np.ndarray]:
    patches = [np.asarray(sprite, dtype=int).copy()]
    for _ in range(3):
        patches.append(_rotate_patch(patches[-1], delta_dir))
    return patches


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_mask(h: int, w: int, target_cells: int) -> np.ndarray | None:
    center = (h // 2, w // 2)
    cells = {center}
    frontier = set(_neighbors4(*center))
    while len(cells) < target_cells:
        choices = [(r, c) for r, c in frontier if 0 <= r < h and 0 <= c < w and (r, c) not in cells]
        if not choices:
            return None
        pick = random.choice(choices)
        cells.add(pick)
        frontier.discard(pick)
        for nbr in _neighbors4(*pick):
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in cells:
                frontier.add(nbr)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_cells = max(5, area // 3)
    for _ in range(200):
        target = random.randint(min_cells, area)
        mask = _grow_connected_mask(height, width, target)
        if mask is None or not mask[height // 2, width // 2] or not _touches_all_sides(mask):
            continue

        palette_size = random.randint(2, min(4, len(PALETTE)))
        palette = random.sample(PALETTE, palette_size)
        sprite = np.zeros((height, width), dtype=int)
        coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
        random.shuffle(coords)
        for color, (r, c) in zip(palette, coords):
            sprite[r, c] = int(color)
        for r, c in coords[palette_size:]:
            sprite[r, c] = int(random.choice(palette))

        rotations = [np.rot90(sprite, k) for k in range(4)]
        fingerprints = {(rot.shape, rot.tobytes()) for rot in rotations}
        if len(fingerprints) != 4:
            continue
        return sprite
    return None


def _copy_bbox_relative(center_rel: Coord, patch: np.ndarray) -> BBox:
    h, w = patch.shape
    return (
        int(center_rel[0] - h // 2),
        int(center_rel[1] - w // 2),
        int(center_rel[0] + h // 2),
        int(center_rel[1] + w // 2),
    )


def _occupied_cells(center: Coord, patch: np.ndarray, bg: int) -> set[Coord]:
    h, w = patch.shape
    top = center[0] - h // 2
    left = center[1] - w // 2
    coords = set()
    for r in range(h):
        for c in range(w):
            if int(patch[r, c]) != bg:
                coords.add((top + r, left + c))
    return coords


def _chebyshev_touches(a: set[Coord], b: set[Coord]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _assemble_input(
    sprite: np.ndarray,
    pivot_color: int,
    start_dir: int,
    delta_dir: int,
    first_dist: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    bg = BLACK
    patches = _rotation_cycle(sprite, delta_dir)
    dirs = [int((start_dir + i * delta_dir) % 4) for i in range(4)]
    dists = [int(first_dist + i * step) for i in range(4)]
    centers_rel = [(DIRS[d][0] * dist, DIRS[d][1] * dist) for d, dist in zip(dirs, dists)]

    boxes = [_copy_bbox_relative(center_rel, patch) for center_rel, patch in zip(centers_rel, patches)]
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    height = max_r - min_r + 1 + margin_top + margin_bottom
    width = max_c - min_c + 1 + margin_left + margin_right
    if height > 30 or width > 30:
        return None

    pivot = (int(-min_r + margin_top), int(-min_c + margin_left))
    input_grid = np.zeros((height, width), dtype=int)
    output_grid = np.zeros((height, width), dtype=int)
    input_grid[pivot] = int(pivot_color)
    output_grid[pivot] = int(pivot_color)

    all_objects: list[set[Coord]] = [{pivot}]
    for idx, (center_rel, patch) in enumerate(zip(centers_rel, patches)):
        center = (pivot[0] + center_rel[0], pivot[1] + center_rel[1])
        coords = _occupied_cells(center, patch, bg)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in coords):
            return None
        if any(coords & prev for prev in all_objects):
            return None
        if any(_chebyshev_touches(coords, prev) for prev in all_objects):
            return None
        all_objects.append(coords)
        target = input_grid if idx < 3 else output_grid
        _place_patch_by_center(target, patch, center, bg)
        if idx < 3:
            _place_patch_by_center(output_grid, patch, center, bg)

    return input_grid, output_grid


def _generation_parameters() -> tuple[int, int, int, int, int, int]:
    height = random.choice([3, 5])
    width = random.choice([3, 5])
    start_dir = random.randrange(4)
    delta_dir = random.choice([1, 3])
    while True:
        step = random.randint(1, 3)
        first_dist = random.randint(4, 7)
        if first_dist + 3 * step <= 13:
            break
    return height, width, start_dir, delta_dir, first_dist, step


def _generator_accepts(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        parsed = _parse_orbit(input_grid)
        output = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(output, expected_output):
        return False
    if np.array_equal(output, input_grid):
        return False

    last = parsed["items"][-1]
    next_patch = _rotate_patch(last["patch"], parsed["delta_dir"])
    next_dir = (last["dir"] + parsed["delta_dir"]) % 4
    next_dist = last["dist"] + parsed["step"]
    dr, dc = DIRS[next_dir]
    next_center = (parsed["pivot"][0] + dr * next_dist, parsed["pivot"][1] + dc * next_dist)
    predicted = input_grid.copy()
    _place_patch_by_center(predicted, next_patch, next_center, parsed["bg"])
    return np.array_equal(predicted, expected_output)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height, width, start_dir, delta_dir, first_dist, step = _generation_parameters()
        sprite = _make_sprite(height, width)
        if sprite is None:
            continue

        sprite_colors = sorted(set(map(int, sprite.ravel())) - {BLACK})
        pivot_choices = [color for color in PALETTE if color not in sprite_colors]
        if not pivot_choices:
            continue
        pivot_color = random.choice(pivot_choices)

        assembled = _assemble_input(sprite, pivot_color, start_dir, delta_dir, first_dist, step)
        if assembled is None:
            continue
        input_grid, expected_output = assembled

        if _generator_accepts(input_grid, expected_output):
            return input_grid

    raise RuntimeError("failed to generate a valid spiral-orbit ARC sample")
