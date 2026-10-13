import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (2, 5)
ROT90_SIZES = (5, 7, 9)
ROT180_SIZES = (5, 7, 9)
MAX_PLACEMENT_TRIES = 250
GENERATION_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    comp = _components(mask)
    return len(comp) == 1 and comp[0].shape[0] == coords.shape[0]


def _orbit_closure(mask: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "rot180":
        return mask | mask[::-1, ::-1]
    assert symmetry == "rot90"
    assert mask.shape[0] == mask.shape[1]
    result = np.zeros_like(mask, dtype=bool)
    cur = mask.copy()
    for _ in range(4):
        result |= cur
        cur = np.rot90(cur)
    return result


def _legend_blocks_on_side(grid: np.ndarray, bg: int, c0: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    assert c0 in (0, w - 2)
    inward = c0 + 2 if c0 == 0 else c0 - 1
    blocks = []
    for r in range(h - 1):
        sub = grid[r : r + 2, c0 : c0 + 2]
        if np.any(sub == bg):
            continue
        if int(sub[0, 0]) != int(sub[1, 0]):
            continue
        if int(sub[0, 1]) != int(sub[1, 1]):
            continue
        if int(sub[0, 0]) == int(sub[0, 1]):
            continue
        if r > 0 and np.any(grid[r - 1, c0 : c0 + 2] != bg):
            continue
        if r + 2 < h and np.any(grid[r + 2, c0 : c0 + 2] != bg):
            continue
        if 0 <= inward < w and np.any(grid[r : r + 2, inward] != bg):
            continue
        blocks.append((r, int(sub[0, 0]), int(sub[0, 1])))
    return blocks


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, str]], np.ndarray]:
    h, w = grid.shape
    side_blocks = []
    if w >= 2:
        left = _legend_blocks_on_side(grid, bg, 0)
        if len(left) == 2:
            side_blocks.append((0, left))
        right = _legend_blocks_on_side(grid, bg, w - 2)
        if len(right) == 2:
            side_blocks.append((w - 2, right))
    assert len(side_blocks) == 1, "expected exactly one edge legend with two 2x2 blocks"

    c0, blocks = side_blocks[0]
    blocks = sorted(blocks, key=lambda item: item[0])
    marker_to_info = {
        blocks[0][1]: (blocks[0][2], "rot90"),
        blocks[1][1]: (blocks[1][2], "rot180"),
    }
    assert len(marker_to_info) == 2, "legend marker colors must be distinct"

    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _marker, _repair in blocks:
        legend_mask[r : r + 2, c0 : c0 + 2] = True
    return marker_to_info, legend_mask


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    marker_to_info, legend_mask = _parse_legend(g, bg)

    object_mask = (g != bg) & ~legend_mask
    comps = _components(object_mask)
    assert comps, "expected at least one non-legend component"

    out = g.copy()
    for coords in comps:
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) in marker_to_info]
        assert len(marker_cells) == 1, "each object must contain exactly one legend marker color"
        marker_color = int(g[marker_cells[0]])
        repair_color, symmetry = marker_to_info[marker_color]

        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        occupied = sub != bg
        if symmetry == "rot90":
            assert occupied.shape[0] == occupied.shape[1], "90-degree objects must live in square boxes"
        expected = _orbit_closure(occupied, symmetry)
        fill_mask = expected & ~occupied
        out[r0 : r1 + 1, c0 : c1 + 1][fill_mask] = repair_color

    return out


def _rot90_orbits(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    orbits = []
    for r in range(size):
        for c in range(size):
            orbit = []
            rr, cc = r, c
            for _ in range(4):
                orbit.append((rr, cc))
                rr, cc = cc, size - 1 - rr
            key = tuple(sorted(set(orbit)))
            if key not in seen:
                seen.add(key)
                orbits.append(key)
    return orbits


def _rot180_orbits(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    orbits = []
    for r in range(height):
        for c in range(width):
            key = tuple(sorted({(r, c), (height - 1 - r, width - 1 - c)}))
            if key not in seen:
                seen.add(key)
                orbits.append(key)
    return orbits


def _orbit_adjacent(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    for r, c in orbit:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if mask[rr, cc]:
                    return True
    return False


def _build_rot90_mask(rng: random.Random) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    size = rng.choice(ROT90_SIZES)
    mid = size // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[mid, :] = True
    mask[:, mid] = True

    available = [orbit for orbit in _rot90_orbits(size) if not all(mask[r, c] for r, c in orbit)]
    target = rng.randint(2, min(8, len(available)))
    added = []
    while len(added) < target:
        adjacent = [orbit for orbit in available if _orbit_adjacent(mask, orbit)]
        if not adjacent:
            break
        orbit = rng.choice(adjacent)
        for r, c in orbit:
            mask[r, c] = True
        added.append(orbit)
        available.remove(orbit)

    assert added, "90-degree object needs at least one non-axis orbit"
    assert _is_connected(mask)
    return mask, added


def _build_rot180_mask(rng: random.Random) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    height = rng.choice(ROT180_SIZES)
    width = rng.choice(ROT180_SIZES)
    mid_r = height // 2
    mid_c = width // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[mid_r, :] = True
    mask[:, mid_c] = True

    available = [orbit for orbit in _rot180_orbits(height, width) if not all(mask[r, c] for r, c in orbit)]
    target = rng.randint(2, min(10, len(available)))
    added = []
    while len(added) < target:
        adjacent = [orbit for orbit in available if _orbit_adjacent(mask, orbit)]
        if not adjacent:
            break
        orbit = rng.choice(adjacent)
        for r, c in orbit:
            mask[r, c] = True
        added.append(orbit)
        available.remove(orbit)

    assert added, "180-degree object needs at least one off-axis orbit"
    assert _is_connected(mask)
    return mask, added


def _damage_mask(
    mask: np.ndarray,
    extra_orbits: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(80):
        damaged = mask.copy()
        chosen = rng.sample(extra_orbits, rng.randint(1, min(3, len(extra_orbits))))
        removed: list[tuple[int, int]] = []
        for orbit in chosen:
            orbit_cells = list(orbit)
            remove_count = 1 if len(orbit_cells) == 2 else rng.randint(1, len(orbit_cells) - 1)
            for r, c in rng.sample(orbit_cells, remove_count):
                damaged[r, c] = False
                removed.append((r, c))
        if removed and _is_connected(damaged):
            return damaged, removed
    raise AssertionError("failed to remove a connected set of defect pixels")


def _make_object(
    symmetry: str,
    marker_color: int,
    body_color: int,
    rng: random.Random,
) -> np.ndarray:
    if symmetry == "rot90":
        mask, extras = _build_rot90_mask(rng)
    else:
        mask, extras = _build_rot180_mask(rng)

    damaged, removed = _damage_mask(mask, extras, rng)
    assert removed
    cells = [tuple(map(int, coord)) for coord in np.argwhere(damaged)]
    marker_r, marker_c = rng.choice(cells)
    obj = np.full(mask.shape, BG, dtype=int)
    obj[damaged] = body_color
    obj[marker_r, marker_c] = marker_color
    return obj


def _bbox_free(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0], r0 + h + 1)
    cc1 = min(used.shape[1], c0 + w + 1)
    return not used[rr0:rr1, cc0:cc1].any()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    types = ["rot90", "rot180"]
    while len(types) < object_count:
        types.append(rng.choice(["rot90", "rot180"]))
    rng.shuffle(types)

    reserved = rng.sample(PALETTE, 4)
    marker90, repair90, marker180, repair180 = reserved
    body_choices = [color for color in PALETTE if color not in reserved]

    objects = []
    for symmetry in types:
        body_color = rng.choice(body_choices)
        marker_color = marker90 if symmetry == "rot90" else marker180
        obj = _make_object(symmetry, marker_color, body_color, rng)
        objects.append({"symmetry": symmetry, "grid": obj})

    max_obj_h = max(item["grid"].shape[0] for item in objects)
    max_obj_w = max(item["grid"].shape[1] for item in objects)
    min_h = max(GRID_MIN, max_obj_h + 2)
    min_w = max(GRID_MIN, max_obj_w + 6)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = rng.randint(min_h, GRID_MAX)
    width = rng.randint(min_w, GRID_MAX)
    side = rng.choice(["left", "right"])

    row_candidates = [(r0, r1) for r0 in range(height - 1) for r1 in range(r0 + 3, height - 1)]
    if not row_candidates:
        return None
    top_row, bottom_row = rng.choice(row_candidates)

    grid = np.full((height, width), BG, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    if side == "left":
        legend_col = 0
        used[:, :4] = True
    else:
        legend_col = width - 2
        used[:, width - 4 :] = True

    grid[top_row : top_row + 2, legend_col] = marker90
    grid[top_row : top_row + 2, legend_col + 1] = repair90
    grid[bottom_row : bottom_row + 2, legend_col] = marker180
    grid[bottom_row : bottom_row + 2, legend_col + 1] = repair180

    place_min_c = 4 if side == "left" else 0
    place_max_c = width - 1 if side == "left" else width - 5

    for item in objects:
        obj = item["grid"]
        oh, ow = obj.shape
        if place_max_c - place_min_c + 1 < ow:
            return None
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            r0 = rng.randint(0, height - oh)
            c0 = rng.randint(place_min_c, place_max_c - ow + 1)
            if not _bbox_free(used, r0, c0, oh, ow):
                continue
            patch = obj != BG
            if np.any(grid[r0 : r0 + oh, c0 : c0 + ow][patch] != BG):
                continue
            grid[r0 : r0 + oh, c0 : c0 + ow][patch] = obj[patch]
            used[r0 : r0 + oh, c0 : c0 + ow] |= patch
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != BG:
        return None

    try:
        out = _solve_grid(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each damaged pattern by reading a border legend that maps marker colors to a repair color and rotation symmetry.

    Input structure:
    - bg is the most common color.
    - Two small 2x2 legend blocks sit on the left or right edge; each block has a marker-color column and a repair-color column.
    - Away from the legend, each 8-connected object occupies a compact bbox and contains exactly one marker-colored pixel.
    - Objects are missing some pixels, so their occupied cells no longer satisfy the intended 90-degree or 180-degree rotational symmetry.

    Rule:
    - Read the upper legend block as the 90-degree entry and the lower legend block as the 180-degree entry.
    - For each object, find its unique marker pixel and use the legend to decide both the symmetry type and the repair color.
    - Inside that object's bbox, rotate the occupied-cell mask by the required symmetry and take the union of all rotated copies.
    - Any cell that should be occupied under this rotational closure but is currently bg is a defect; fill every such defect with the object's repair color.
    - Keep the legend and the marker pixels themselves unchanged.

    Key insight:
    - The legend encodes two layers at once: row order gives the symmetry type, while the colored columns give marker -> repair mapping.
    """
    return _solve_grid(grid)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
