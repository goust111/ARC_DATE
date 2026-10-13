from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, label_mask


BACKGROUND_COLOR = Color.BLACK
MARKER_COLOR = Color.GRAY
BODY_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

OBJECT_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_HEIGHT_RANGE = (5, 9)
OBJECT_WIDTH_RANGE = (5, 10)
PLACEMENT_GAP = 2
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 250


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(component["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _count_holes(occupancy: np.ndarray) -> int:
    void_labels, num_voids = label_mask(~occupancy, connectivity=4)
    if num_voids == 0:
        return 0

    border_ids = set(map(int, void_labels[0, :]))
    border_ids.update(map(int, void_labels[-1, :]))
    border_ids.update(map(int, void_labels[:, 0]))
    border_ids.update(map(int, void_labels[:, -1]))
    border_ids.discard(0)
    return sum(region_id not in border_ids for region_id in range(1, int(num_voids) + 1))


def _analyze_object(grid: np.ndarray, component: dict) -> dict:
    bbox = tuple(map(int, component["bbox"]))
    crop = crop_bbox(grid, bbox)
    occupancy = _component_mask(component)

    body_colors = sorted(set(map(int, np.unique(crop[occupancy]))) - {int(MARKER_COLOR)})
    assert len(body_colors) == 1, "each object must have exactly one non-gray body color"
    body_color = int(body_colors[0])

    marker_mask = occupancy & (crop == int(MARKER_COLOR))
    restored = crop.copy()
    restored[marker_mask] = body_color

    return {
        "bbox": bbox,
        "crop": crop,
        "occupancy": occupancy,
        "body_color": body_color,
        "marker_count": int(marker_mask.sum()),
        "hole_count": _count_holes(occupancy),
        "restored_crop": restored,
        "area": int(occupancy.sum()),
    }


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = find_components(g, bg=BACKGROUND_COLOR, connectivity=4, monochromatic=False)
    assert components, "expected at least one object"
    return [_analyze_object(g, component) for component in components]


def _matching_object(objects: list[dict]) -> dict:
    matches = [obj for obj in objects if int(obj["hole_count"]) == int(obj["marker_count"])]
    assert len(matches) == 1, "expected a unique self-matching object"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique holed object whose gray marker count matches its own hole count.

    Input structure
    - The grid has a black background with several separated non-black objects.
    - Each object is monochrome except for gray marker pixels painted on its body.
    - Each object contains one or more enclosed black holes inside its own shape.

    Rule
    - Treat every connected non-black component as one object, counting gray pixels as part of that object.
    - For each object, count its gray markers and the enclosed black void components inside its bbox.
    - Exactly one object has equal counts; crop that object's bbox and recolor its gray markers back to the object's body color while keeping holes black.

    Key insight
    - The gray pixels are a self-referential claim about the same object's hidden hole count, not a pointer to some other shape.
    """
    objects = _analyze_objects(np.asarray(grid, dtype=int))
    return _matching_object(objects)["restored_crop"].copy()


def _sample_counts(rng: random.Random, num_objects: int) -> tuple[list[int], list[int], int]:
    hole_counts = rng.sample(list(range(1, 6)), num_objects)
    target_index = rng.randrange(num_objects)

    marker_counts: list[int | None] = [None] * num_objects
    marker_counts[target_index] = hole_counts[target_index]

    other_indices = [idx for idx in range(num_objects) if idx != target_index]
    donor_indices = other_indices[:]
    while True:
        rng.shuffle(donor_indices)
        if all(src != dst for src, dst in zip(donor_indices, other_indices)):
            break
    for src, dst in zip(donor_indices, other_indices):
        marker_counts[dst] = hole_counts[src]

    return hole_counts, [int(count) for count in marker_counts], int(target_index)


def _hole_capacity(height: int, width: int) -> int:
    parity_counts = [0, 0]
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            parity_counts[(r + c) % 2] += 1
    return max(parity_counts)


def _candidate_dims(need: int, hole_count: int) -> list[tuple[int, int]]:
    candidates = []
    for height in range(OBJECT_HEIGHT_RANGE[0], OBJECT_HEIGHT_RANGE[1] + 1):
        for width in range(OBJECT_WIDTH_RANGE[0], OBJECT_WIDTH_RANGE[1] + 1):
            if (height - 2) * (width - 2) >= int(need) and _hole_capacity(height, width) >= int(hole_count):
                candidates.append((height, width))
    return candidates


def _sample_dimensions(
    hole_counts: list[int],
    marker_counts: list[int],
    target_index: int,
    want_target_largest: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    needs = [int(hole_counts[idx]) + int(marker_counts[idx]) + 4 for idx in range(len(hole_counts))]
    choices = [_candidate_dims(need, hole_counts[idx]) for idx, need in enumerate(needs)]
    if any(not option_list for option_list in choices):
        return None

    for _ in range(200):
        dims = [rng.choice(option_list) for option_list in choices]
        areas = [int(height * width) for height, width in dims]
        max_area = max(areas)
        if want_target_largest:
            if areas[target_index] == max_area and areas.count(max_area) == 1:
                return dims
        else:
            if areas[target_index] < max_area:
                return dims
    return None


def _choose_holes(height: int, width: int, hole_count: int, rng: random.Random) -> list[tuple[int, int]]:
    parity_cells = [
        [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if (r + c) % 2 == parity]
        for parity in (0, 1)
    ]
    valid_groups = [cells for cells in parity_cells if len(cells) >= int(hole_count)]
    if not valid_groups:
        raise RuntimeError("failed to place non-adjacent holes")
    holes = rng.choice(valid_groups)[:]
    rng.shuffle(holes)
    return holes[:hole_count]


def _make_object(height: int, width: int, body_color: int, hole_count: int, marker_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        sprite = np.full((height, width), int(body_color), dtype=int)

        holes = _choose_holes(height, width, hole_count, rng)
        for r, c in holes:
            sprite[r, c] = BACKGROUND_COLOR

        available = []
        for r in range(height):
            for c in range(width):
                if sprite[r, c] != int(body_color):
                    continue
                if any(
                    0 <= rr < height and 0 <= cc < width and sprite[rr, cc] == int(body_color)
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    available.append((r, c))

        if len(available) < int(marker_count):
            continue

        rng.shuffle(available)
        for r, c in available[:marker_count]:
            sprite[r, c] = MARKER_COLOR
        if len(find_components(sprite, bg=BACKGROUND_COLOR, connectivity=4, monochromatic=False)) == 1:
            return sprite

    raise RuntimeError("failed to place connected gray markers")


def _boxes_conflict(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_objects(sprites: list[np.ndarray], height: int, width: int, rng: random.Random) -> np.ndarray | None:
    order = sorted(range(len(sprites)), key=lambda idx: int(sprites[idx].shape[0] * sprites[idx].shape[1]), reverse=True)
    placements: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        sprite = sprites[idx]
        sprite_h, sprite_w = map(int, sprite.shape)
        if sprite_h > height or sprite_w > width:
            return None

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, height - sprite_h)
            c0 = rng.randint(0, width - sprite_w)
            candidate = (r0, c0, r0 + sprite_h - 1, c0 + sprite_w - 1)
            if all(not _boxes_conflict(candidate, other_box, PLACEMENT_GAP) for other_box in placements.values()):
                placements[idx] = candidate
                break
        else:
            return None

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for idx, sprite in enumerate(sprites):
        r0, c0, _r1, _c1 = placements[idx]
        grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] = sprite
    return grid


def _semantic_summary(grid: np.ndarray) -> dict:
    objects = _analyze_objects(grid)
    holes = [int(obj["hole_count"]) for obj in objects]
    markers = [int(obj["marker_count"]) for obj in objects]
    assert len(set(holes)) == len(holes), "hole counts must be distinct"
    assert len(set(markers)) == len(markers), "marker counts must be distinct"

    match = _matching_object(objects)
    expected = match["restored_crop"]
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must return the matching restored crop"

    return {
        "objects": objects,
        "match": match,
        "areas": [int(obj["area"]) for obj in objects],
        "markers": markers,
        "holes": holes,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
    hole_counts, marker_counts, target_index = _sample_counts(rng, num_objects)
    want_target_largest = bool(rng.getrandbits(1))
    dims = _sample_dimensions(hole_counts, marker_counts, target_index, want_target_largest, rng)
    if dims is None:
        return None

    colors = rng.sample(list(BODY_COLORS), num_objects)
    sprites = [
        _make_object(height, width, colors[idx], hole_counts[idx], marker_counts[idx], rng)
        for idx, (height, width) in enumerate(dims)
    ]

    total_area = sum(int(sprite.shape[0] * sprite.shape[1]) for sprite in sprites)
    for _ in range(80):
        grid_h = rng.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
        grid_w = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
        if grid_h * grid_w < total_area * 2:
            continue

        grid = _place_objects(sprites, grid_h, grid_w, rng)
        if grid is None:
            continue

        summary = _semantic_summary(grid)
        match_area = int(summary["match"]["area"])
        max_area = max(summary["areas"])
        if want_target_largest and match_area != max_area:
            continue
        if not want_target_largest and match_area == max_area:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid task-8 input")
