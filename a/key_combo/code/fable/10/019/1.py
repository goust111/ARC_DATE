import random
import itertools
from collections import Counter
from functools import lru_cache

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
PANEL_WIDTHS = (8, 9, 10)
BAND_HEIGHT = 4
MIN_BANDS = 4
MAX_BANDS = 5
MAX_ATTEMPTS = 500
LEGEND_LENGTHS = (5, 6)
SHAPE_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4))


def _mode(values: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(values).ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.asarray(coords, dtype=int))
    return out


def _crop_coords(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    return tuple(mask.shape), tuple(tuple(map(int, p)) for p in coords.tolist())


def _rotation_class_key(mask: np.ndarray):
    return min(_shape_key(np.rot90(mask, k)) for k in range(4))


def _is_rotation_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(np.rot90(a, k), b) for k in range(4))


def _mirror(mask: np.ndarray, axis: str) -> np.ndarray:
    return np.fliplr(mask) if axis == "lr" else np.flipud(mask)


def _legend_cycle(row: np.ndarray) -> list[int]:
    order: list[int] = []
    for value in map(int, row):
        if not order or value != order[-1]:
            order.append(value)
    assert len(order) >= 4
    assert len(set(order)) == len(order)
    return order


def _objects_by_color(panel: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(panel))) - {bg}):
        for coords in _components(panel == color):
            objects.append({"color": int(color), "coords": coords, "mask": _crop_coords(coords)})
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each left-panel hole with the legend-shifted color of the right object that matches it under counterclockwise quarter-turns.

    Input structure:
    - The last row is a legend strip whose left-to-right color runs define a cyclic color order.
    - Above the legend, the grid splits into two equal-width panels: a left single-color template with several background holes, and a right panel containing one monochrome object per hole on the background.

    Rule:
    - Return only the left panel, excluding the legend row.
    - For each hole, crop its mask to its bounding box and find the unique right-panel object whose cropped mask becomes identical after `k` counterclockwise 90-degree rotations, with `k` in `{0,1,2,3}` and no mirroring allowed.
    - Let the matched object's color appear in the legend cycle; fill the hole with the color reached by advancing `k` steps forward along that cycle.

    Key insight:
    - Shape matching determines which object names each hole, while the object's rotation offset is itself information because it controls how far to move along the visible legend color ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w % 2 == 0

    top = g[:-1]
    legend = g[-1]
    mid = w // 2
    left = top[:, :mid]
    right = top[:, mid:]
    bg = _mode(right)
    cycle = _legend_cycle(legend)
    cycle_index = {color: i for i, color in enumerate(cycle)}

    holes = [_crop_coords(coords) for coords in _components(left == bg)]
    objects = _objects_by_color(right, bg)
    assert holes and len(holes) == len(objects)

    out = left.copy()
    used_objects: set[int] = set()
    for hole_mask, coords in zip(holes, _components(left == bg)):
        candidates: list[tuple[int, int]] = []
        hole_key = _shape_key(hole_mask)
        for obj_idx, obj in enumerate(objects):
            for k in range(4):
                if _shape_key(np.rot90(obj["mask"], k)) == hole_key:
                    candidates.append((obj_idx, k))
        assert len(candidates) == 1
        obj_idx, k = candidates[0]
        assert obj_idx not in used_objects
        used_objects.add(obj_idx)
        obj_color = objects[obj_idx]["color"]
        assert obj_color in cycle_index
        fill_color = cycle[(cycle_index[obj_color] + k) % len(cycle)]
        out[coords[:, 0], coords[:, 1]] = fill_color

    return out


@lru_cache(None)
def _shape_catalog() -> tuple[np.ndarray, ...]:
    representatives: dict[tuple, np.ndarray] = {}
    for height, width in SHAPE_DIMS:
        total = height * width
        min_area = max(5, (total + 2) // 2)
        max_area = total - 2
        for bits in range(1, 1 << total):
            area = bits.bit_count()
            if not (min_area <= area <= max_area):
                continue
            flat = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool)
            mask = flat.reshape(height, width)
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if len(_components(mask)) != 1:
                continue
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            key = _rotation_class_key(mask)
            representatives.setdefault(key, mask.copy())
    return tuple(representatives.values())


def _choose_shapes(count: int) -> tuple[list[np.ndarray], tuple[int, int], str] | None:
    axis = random.choice(("lr", "ud"))
    catalog = list(_shape_catalog())
    chiral = [mask for mask in catalog if not _is_rotation_equivalent(mask, _mirror(mask, axis))]
    if not chiral:
        return None

    base = random.choice(chiral).copy()
    pair = _mirror(base, axis).copy()
    shapes = [base, pair]
    used = {_rotation_class_key(base), _rotation_class_key(pair)}

    remainder = [mask.copy() for mask in catalog if _rotation_class_key(mask) not in used]
    random.shuffle(remainder)
    for mask in remainder:
        shapes.append(mask)
        used.add(_rotation_class_key(mask))
        if len(shapes) == count:
            order = list(range(count))
            random.shuffle(order)
            shuffled = [shapes[i] for i in order]
            pair = (order.index(0), order.index(1))
            return shuffled, pair, axis
    return None


def _choose_steps(count: int, pair: tuple[int, int]) -> list[int]:
    pr0, pr1 = pair
    if count == 5:
        shared = random.randrange(4)
        steps = [None] * count
        steps[pr0] = shared
        steps[pr1] = shared
        others = [k for k in range(4) if k != shared]
        other_slots = [i for i in range(count) if i not in pair]
        random.shuffle(others)
        for slot, step in zip(other_slots, others):
            steps[slot] = step
        return [int(step) for step in steps]

    steps = list(range(4))
    random.shuffle(steps)
    return steps


def _derangement(n: int) -> list[int]:
    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm


def _build_legend_row(width: int, cycle: list[int]) -> np.ndarray:
    min_run = 2
    runs = [min_run] * len(cycle)
    extra = width - min_run * len(cycle)
    assert extra >= 0
    for _ in range(extra):
        runs[random.randrange(len(runs))] += 1
    row = []
    for color, run in zip(cycle, runs):
        row.extend([color] * run)
    return np.asarray(row, dtype=int)


def _assign_object_colors(cycle: list[int], steps: list[int]) -> tuple[list[int], list[int]] | None:
    n = len(steps)
    candidate_perms = []
    for perm in itertools.permutations(range(len(cycle)), n):
        shifted = [((perm[i] + steps[i]) % len(cycle)) for i in range(n)]
        if len(set(shifted)) == n:
            candidate_perms.append(perm)
    if not candidate_perms:
        return None
    chosen = random.choice(candidate_perms)
    object_colors = [cycle[i] for i in chosen]
    fill_colors = [cycle[(chosen[i] + steps[i]) % len(cycle)] for i in range(n)]
    return object_colors, fill_colors


def _render_candidate():
    band_count = 5 if random.random() < 0.65 else 4
    panel_width = random.choice(PANEL_WIDTHS)
    top_height = band_count * (BAND_HEIGHT + 1) + 1
    total_width = 2 * panel_width

    legend_len = 6 if band_count == 5 else random.choice(LEGEND_LENGTHS)
    palette = random.sample(COLOR_POOL, legend_len + 1)
    legend_cycle = palette[:legend_len]
    frame_color = palette[-1]

    chosen = _choose_shapes(band_count)
    if chosen is None:
        return None
    base_shapes, mirror_pair, mirror_axis = chosen

    steps = _choose_steps(band_count, mirror_pair)
    color_assignment = _assign_object_colors(legend_cycle, steps)
    if color_assignment is None:
        return None
    object_colors, fill_colors = color_assignment

    object_rots = [random.randrange(4) for _ in range(band_count)]
    pr0, pr1 = mirror_pair
    shared_rot = random.randrange(4)
    object_rots[pr0] = shared_rot
    object_rots[pr1] = shared_rot

    left_in = np.full((top_height, panel_width), frame_color, dtype=int)
    left_out = left_in.copy()
    right_in = np.full((top_height, panel_width), BG, dtype=int)

    hole_masks: list[np.ndarray] = []
    object_masks: list[np.ndarray] = []
    object_band_slots = _derangement(band_count)

    for idx, base in enumerate(base_shapes):
        obj_mask = np.rot90(base, object_rots[idx])
        hole_mask = np.rot90(obj_mask, steps[idx])
        object_masks.append(obj_mask)
        hole_masks.append(hole_mask)

    for band in range(band_count):
        band_r0 = 1 + band * (BAND_HEIGHT + 1)
        hole_mask = hole_masks[band]
        hole_h, hole_w = hole_mask.shape
        hole_r = band_r0 + random.randint(0, BAND_HEIGHT - hole_h)
        hole_c = random.randint(1, panel_width - hole_w - 1)
        rr, cc = np.where(hole_mask)
        left_in[hole_r + rr, hole_c + cc] = BG
        left_out[hole_r + rr, hole_c + cc] = fill_colors[band]

    for slot, obj_idx in enumerate(object_band_slots):
        band_r0 = 1 + slot * (BAND_HEIGHT + 1)
        obj_mask = object_masks[obj_idx]
        obj_h, obj_w = obj_mask.shape
        obj_r = band_r0 + random.randint(0, BAND_HEIGHT - obj_h)
        obj_c = random.randint(1, panel_width - obj_w - 1)
        rr, cc = np.where(obj_mask)
        right_in[obj_r + rr, obj_c + cc] = object_colors[obj_idx]

    legend_row = _build_legend_row(total_width, legend_cycle)
    grid = np.vstack([np.hstack([left_in, right_in]), legend_row[None, :]])
    meta = {
        "band_count": band_count,
        "steps": steps,
        "mirror_pair": mirror_pair,
        "mirror_axis": mirror_axis,
        "legend_cycle": legend_cycle,
        "object_colors": object_colors,
        "fill_colors": fill_colors,
        "object_band_slots": object_band_slots,
        "hole_masks": hole_masks,
        "object_masks": object_masks,
    }
    return grid, left_out, meta


def _contains_visible_mirror_pair(masks: list[np.ndarray]) -> bool:
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _is_rotation_equivalent(masks[i], masks[j]):
                continue
            if np.array_equal(_mirror(masks[i], "lr"), masks[j]) or np.array_equal(_mirror(masks[i], "ud"), masks[j]):
                return True
    return False


def _validate_candidate(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if sorted(set(meta["steps"])) != [0, 1, 2, 3]:
        return False
    if any(i == slot for i, slot in enumerate(meta["object_band_slots"])):
        return False
    if not _contains_visible_mirror_pair(meta["object_masks"]):
        return False

    top = inp[:-1]
    mid = top.shape[1] // 2
    left = top[:, :mid]
    right = top[:, mid:]
    bg = _mode(right)
    holes = _components(left == bg)
    objects = _objects_by_color(right, bg)
    if len(holes) != meta["band_count"] or len(objects) != meta["band_count"]:
        return False

    cycle = _legend_cycle(inp[-1])
    cycle_index = {color: i for i, color in enumerate(cycle)}
    seen_steps: list[int] = []
    seen_objects: set[int] = set()
    out = left.copy()

    for coords in holes:
        hole_mask = _crop_coords(coords)
        matches: list[tuple[int, int]] = []
        for obj_idx, obj in enumerate(objects):
            for step in range(4):
                if np.array_equal(np.rot90(obj["mask"], step), hole_mask):
                    matches.append((obj_idx, step))
        if len(matches) != 1:
            return False
        obj_idx, step = matches[0]
        if obj_idx in seen_objects:
            return False
        seen_objects.add(obj_idx)
        seen_steps.append(step)
        color = objects[obj_idx]["color"]
        if color not in cycle_index:
            return False
        fill = cycle[(cycle_index[color] + step) % len(cycle)]
        out[coords[:, 0], coords[:, 1]] = fill

    if sorted(set(seen_steps)) != [0, 1, 2, 3]:
        return False
    if np.any(out == bg):
        return False
    return np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _render_candidate()
        if candidate is None:
            continue
        inp, expected, meta = candidate
        if _validate_candidate(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid rotating-key input")
