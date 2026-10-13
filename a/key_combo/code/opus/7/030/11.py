import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
OBJECT_COLORS = (1, 2, 3, 4)
OBJECT_SIZE = 5
BBOX_SIDE = 3
ROTATION_K_BY_CORNER_COUNT = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0}
CORNER_COORDS = ((0, 0), (0, 2), (2, 0), (2, 2))
MAX_GENERATION_ATTEMPTS = 300
ROW_RANGE = (15, 21)
COL_RANGE = (15, 21)
SLOT_STEP = 5
COUNT_TEMPLATE_FAMILIES = [
    (0, 1, 2),
    (1, 2, 3),
    (0, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 3, 2, 1),
    (0, 2, 3, 1, 2),
    (3, 1, 2, 0),
]
_COUNT_TEMPLATE_ORDER = list(range(len(COUNT_TEMPLATE_FAMILIES)))
random.shuffle(_COUNT_TEMPLATE_ORDER)
_COUNT_TEMPLATE_CURSOR = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray) -> list[dict]:
    seen = np.zeros_like(grid, dtype=bool)
    components: list[dict] = []
    H, W = grid.shape

    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _corner_count(mask: np.ndarray) -> int:
    return sum(int(mask[r, c]) for r, c in CORNER_COORDS)


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) != OBJECT_SIZE:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < BBOX_SIDE and 0 <= nc < BBOX_SIDE and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == OBJECT_SIZE


def _enumerate_masks() -> dict[int, list[np.ndarray]]:
    groups = {corner_count: [] for corner_count in range(5)}
    for cells in itertools.combinations(range(BBOX_SIDE * BBOX_SIDE), OBJECT_SIZE):
        mask = np.zeros((BBOX_SIDE, BBOX_SIDE), dtype=bool)
        for cell in cells:
            mask[cell // BBOX_SIDE, cell % BBOX_SIDE] = True
        rows, cols = np.where(mask)
        if rows.ptp() != BBOX_SIDE - 1 or cols.ptp() != BBOX_SIDE - 1:
            continue
        if not _mask_is_connected(mask):
            continue
        groups[_corner_count(mask)].append(mask)
    return groups


MASKS_BY_CORNER_COUNT = _enumerate_masks()

# Under the task's 4-connected, 3x3, five-pixel constraints, the n=4 case is
# geometrically impossible. main() still supports it if such an input is given.
GENERATOR_MASKS_BY_CORNER_COUNT = {
    0: MASKS_BY_CORNER_COUNT[0],
    1: MASKS_BY_CORNER_COUNT[1],
    2: [
        mask
        for mask in MASKS_BY_CORNER_COUNT[2]
        if not np.array_equal(mask, np.rot90(mask, ROTATION_K_BY_CORNER_COUNT[2]))
    ],
    3: MASKS_BY_CORNER_COUNT[3],
}


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each 3x3 five-pixel object by the angle encoded by its occupied bbox corners.

    Input structure:
    - The grid uses black (0) as background and contains several non-overlapping monochrome foreground objects.
    - Each object is 4-connected, occupies exactly five cells, and spans a 3x3 bounding box.
    - Different objects may have different colors, but the transformation depends only on each object's local 3x3 mask.

    Rule:
    - Find every 4-connected foreground object and crop its 3x3 bounding box.
    - Count how many bbox corners belong to the object mask.
    - Map that corner count to a rotation: 0 -> 0 degrees, 1 -> 90 degrees clockwise, 2 -> 180 degrees, 3 -> 270 degrees clockwise, 4 -> 0 degrees.
    - Rotate the object's 3x3 mask around its center by that amount and stamp the same color back into the original bbox.

    Key insight:
    - The corner-occupancy count is not a label to output directly; it is a hidden parameter that selects how much each object should rotate in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _find_components(g):
        assert comp["area"] == OBJECT_SIZE, "each object must contain exactly five cells"
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 - r0 + 1 == BBOX_SIDE and c1 - c0 + 1 == BBOX_SIDE, "each object must span a 3x3 bbox"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        mask = patch == color
        corner_count = _corner_count(mask)
        assert corner_count in ROTATION_K_BY_CORNER_COUNT, "unexpected corner-count class"

        rotated_mask = np.rot90(mask, ROTATION_K_BY_CORNER_COUNT[corner_count])
        out[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
        out[r0 : r1 + 1, c0 : c1 + 1][rotated_mask] = color

    return out


def _choose_corner_counts() -> list[int]:
    global _COUNT_TEMPLATE_CURSOR

    family_index = _COUNT_TEMPLATE_ORDER[_COUNT_TEMPLATE_CURSOR % len(_COUNT_TEMPLATE_ORDER)]
    _COUNT_TEMPLATE_CURSOR += 1
    counts = list(COUNT_TEMPLATE_FAMILIES[family_index])
    random.shuffle(counts)
    return counts


def _sample_object_colors(object_count: int) -> list[int]:
    base = list(OBJECT_COLORS)
    random.shuffle(base)
    colors = base[: min(object_count, len(base))]
    while len(colors) < object_count:
        colors.append(random.choice(OBJECT_COLORS))
    random.shuffle(colors)
    return colors


def _sample_masks(corner_counts: list[int]) -> list[np.ndarray]:
    masks = []
    used_keys_by_count: dict[int, set[bytes]] = {}
    for corner_count in corner_counts:
        choices = GENERATOR_MASKS_BY_CORNER_COUNT[corner_count]
        random_order = choices.copy()
        random.shuffle(random_order)
        chosen = None
        used_keys = used_keys_by_count.setdefault(corner_count, set())
        for mask in random_order:
            key = mask.tobytes()
            if key not in used_keys:
                chosen = mask
                used_keys.add(key)
                break
        if chosen is None:
            chosen = random.choice(choices)
        masks.append(chosen.copy())
    return masks


def _slot_positions(limit: int) -> list[int]:
    return list(range(0, limit - BBOX_SIDE + 1, SLOT_STEP))


def _place_objects(shape: tuple[int, int], masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    H, W = shape
    row_slots = _slot_positions(H)
    col_slots = _slot_positions(W)
    slots = [(r0, c0) for r0 in row_slots for c0 in col_slots]
    if len(slots) < len(masks):
        return None

    grid = np.zeros((H, W), dtype=int)
    for (r0, c0), mask, color in zip(random.sample(slots, len(masks)), masks, colors):
        patch = np.zeros((BBOX_SIDE, BBOX_SIDE), dtype=int)
        patch[mask] = color
        grid[r0 : r0 + BBOX_SIDE, c0 : c0 + BBOX_SIDE] = patch
    return grid


def _valid_input(grid: np.ndarray, expected_corner_counts: list[int]) -> bool:
    comps = _find_components(grid)
    if len(comps) != len(expected_corner_counts):
        return False
    found_corner_counts = []
    for comp in comps:
        if comp["area"] != OBJECT_SIZE:
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 != BBOX_SIDE or c1 - c0 + 1 != BBOX_SIDE:
            return False
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        found_corner_counts.append(_corner_count(patch == int(comp["color"])))
    return len(set(found_corner_counts)) >= 2 and sorted(found_corner_counts) == sorted(expected_corner_counts)


def _gen_attempt() -> np.ndarray | None:
    corner_counts = _choose_corner_counts()
    H = random.randint(*ROW_RANGE)
    W = random.randint(*COL_RANGE)
    masks = _sample_masks(corner_counts)
    colors = _sample_object_colors(len(corner_counts))
    grid = _place_objects((H, W), masks, colors)
    if grid is None:
        return None
    if not _valid_input(grid, corner_counts):
        return None
    output = main(grid)
    if output.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
