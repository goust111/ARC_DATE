from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 0
PLACEHOLDER = 5
OBSTACLE = 4
SOURCE_COLORS = (1, 2, 3, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (18, 24)
PLACEHOLDER_COUNT_OPTIONS = (3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    norm = [(int(r - r0), int(c - c0)) for r, c in coords]
    return tuple(sorted(norm))


def _component_patch(grid: np.ndarray, coords: np.ndarray, fill: int | None = None) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG if fill is None else int(fill), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    if fill is None:
        patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    else:
        patch[rr, cc] = int(fill)
    return patch


def _pattern_from_patch(patch: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (int(r), int(c)): int(patch[r, c])
        for r, c in np.argwhere(patch != BG)
    }


def _extract_scene(grid: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(get_bg(g)) == BG, "expected black background"

    source_candidates: list[dict] = []
    placeholders: list[dict] = []
    obstacles: list[dict] = []

    for coords in components_mask(g != BG, connectivity=4):
        colors = {int(g[r, c]) for r, c in coords}
        norm = _normalize_coords(coords)
        r0, c0, r1, c1 = bbox_coords(coords)
        comp = {
            "coords": coords.astype(int),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "colors": colors,
            "mask": norm,
            "patch": _component_patch(g, coords),
        }
        if len(colors) >= 2:
            source_candidates.append(comp)
        elif colors == {PLACEHOLDER}:
            placeholders.append(comp)
        elif colors == {OBSTACLE}:
            obstacles.append(comp)
        else:
            raise AssertionError("unexpected foreground component colors")

    assert len(source_candidates) == 1, "expected exactly one multicolor source sprite"
    source = source_candidates[0]
    assert 3 <= len(source["coords"]) <= 5, "source sprite pixel count out of range"
    assert 2 <= len(source["colors"]) <= 3, "source sprite color count out of range"
    assert PLACEHOLDER not in source["colors"] and OBSTACLE not in source["colors"], "reserved colors in source"

    assert 3 <= len(placeholders) <= 4, "expected 3 or 4 placeholders"
    source_mask = source["mask"]
    for comp in placeholders:
        assert comp["mask"] == source_mask, "placeholder mask must match the source mask"

    placeholder_spans = sorted((comp["bbox"][1], comp["bbox"][3]) for comp in placeholders)
    for (_a0, a1), (b0, _b1) in zip(placeholder_spans, placeholder_spans[1:]):
        assert a1 < b0, "placeholder column spans must be disjoint"

    return source, placeholders, obstacles


def _stamp_placeholders(shape: tuple[int, int], source: dict, placeholders: list[dict], obstacles: list[dict]) -> tuple[np.ndarray, list[dict]]:
    canvas = np.full(shape, BG, dtype=int)
    for obstacle in obstacles:
        r0, c0, _r1, _c1 = obstacle["bbox"]
        paste(canvas, obstacle["patch"], r0, c0, transparent=BG)

    pattern = _pattern_from_patch(source["patch"])
    objects: list[dict] = []
    for placeholder in placeholders:
        r0, c0, _r1, _c1 = placeholder["bbox"]
        patch = np.full(source["patch"].shape, BG, dtype=int)
        for (dr, dc), color in pattern.items():
            patch[dr, dc] = int(color)
        paste(canvas, patch, r0, c0, transparent=BG)
        coords = np.argwhere(patch != BG).astype(int)
        coords[:, 0] += int(r0)
        coords[:, 1] += int(c0)
        values = patch[patch != BG].astype(int)
        objects.append({"coords": coords, "values": values})
    return canvas, objects


def _occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "moving objects must not overlap"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _objects_that_can_shift_down(objects: list[dict], occ: np.ndarray, static_mask: np.ndarray) -> set[int]:
    h, _w = occ.shape
    movable = set(range(len(objects)))
    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            coords = objects[idx]["coords"]
            shifted_rows = coords[:, 0] + 1
            shifted_cols = coords[:, 1]
            if np.any(shifted_rows >= h):
                movable.remove(idx)
                changed = True
                continue

            if np.any(static_mask[shifted_rows, shifted_cols]):
                movable.remove(idx)
                changed = True
                continue

            labels = occ[shifted_rows, shifted_cols]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True
    return movable


def _gravity_settle(
    shape: tuple[int, int],
    objects: list[dict],
    obstacles: list[dict],
    extra_static: np.ndarray | None = None,
) -> np.ndarray:
    static_mask = np.zeros(shape, dtype=bool)
    for obstacle in obstacles:
        static_mask[obstacle["coords"][:, 0], obstacle["coords"][:, 1]] = True
    if extra_static is not None:
        static_mask |= extra_static.astype(bool)

    for _ in range(shape[0]):
        occ = _occupancy(shape, objects)
        movable = _objects_that_can_shift_down(objects, occ, static_mask)
        if not movable:
            break
        for idx in movable:
            objects[idx]["coords"][:, 0] += 1
    else:
        raise AssertionError("downward gravity did not settle")

    out = np.full(shape, BG, dtype=int)
    for obstacle in obstacles:
        out[obstacle["coords"][:, 0], obstacle["coords"][:, 1]] = OBSTACLE

    for obj in objects:
        coords = obj["coords"]
        values = obj["values"]
        assert not np.any(static_mask[coords[:, 0], coords[:, 1]]), "settled object overlaps an obstacle"
        assert np.all(out[coords[:, 0], coords[:, 1]] == BG), "settled objects must not overlap"
        out[coords[:, 0], coords[:, 1]] = values
    return out


def _apply_rule(grid: np.ndarray, *, erase_source: bool, settle: bool) -> np.ndarray:
    source, placeholders, obstacles = _extract_scene(grid)
    stamped, objects = _stamp_placeholders(grid.shape, source, placeholders, obstacles)
    source_static = np.zeros(grid.shape, dtype=bool)
    if not erase_source:
        r0, c0, _r1, _c1 = source["bbox"]
        paste(stamped, source["patch"], r0, c0, transparent=BG)
        source_static[r0 : r0 + source["patch"].shape[0], c0 : c0 + source["patch"].shape[1]] |= source["patch"] != BG
    if not settle:
        return stamped
    return _gravity_settle(grid.shape, objects, obstacles, extra_static=source_static if not erase_source else None)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source sprite onto every matching grey placeholder, erase the source, then let the stamped sprites settle downward under gravity.

    Input structure:
    - A black background contains one 4-connected multicolor source sprite and several 4-connected grey placeholder sprites.
    - Every placeholder has the same normalized mask as the source sprite.
    - Yellow obstacle components may appear near the bottom; they do not match or move.

    Rule:
    - Identify the unique multicolor source sprite and all grey placeholder components whose normalized masks match it.
    - Copy the source sprite's per-cell colors onto each matching placeholder at the corresponding relative coordinates.
    - Erase the original source sprite to black.
    - Treat the newly colored placeholder copies as rigid sprites and apply simultaneous downward gravity until each is blocked by the bottom border, a settled sprite, or a yellow obstacle.

    Key insight:
    - The source sprite is both a color template and a temporary blocker: stamping happens before the source is erased, so understanding the erase-then-gravity order is necessary to reach the correct terminal layout.
    """
    g = np.asarray(grid, dtype=int)
    return _apply_rule(g, erase_source=True, settle=True)


def _is_connected_mask(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _build_mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3, 4):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for count in range(3, 6):
                for combo in itertools.combinations(cells, count):
                    rows = {r for r, _c in combo}
                    cols = {c for _r, c in combo}
                    if rows != set(range(h)) or cols != set(range(w)):
                        continue
                    mask = np.zeros((h, w), dtype=bool)
                    rr, cc = zip(*combo)
                    mask[rr, cc] = True
                    if _is_connected_mask(mask):
                        masks.append(mask)
    unique: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for mask in masks:
        key = tuple(sorted(map(tuple, np.argwhere(mask))))
        unique.setdefault(key, mask)
    return list(unique.values())


MASK_LIBRARY = _build_mask_library()
OBSTACLE_SHAPES = (
    np.array([[OBSTACLE, OBSTACLE]], dtype=int),
    np.array([[OBSTACLE], [OBSTACLE]], dtype=int),
    np.array([[OBSTACLE, OBSTACLE, OBSTACLE]], dtype=int),
    np.array([[OBSTACLE], [OBSTACLE], [OBSTACLE]], dtype=int),
    np.array([[OBSTACLE, OBSTACLE], [OBSTACLE, BG]], dtype=int),
    np.array([[OBSTACLE, BG], [OBSTACLE, OBSTACLE]], dtype=int),
)


def _make_source_patch() -> np.ndarray:
    mask = random.choice(MASK_LIBRARY)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(2, min(3, len(coords)))
    palette = random.sample(SOURCE_COLORS, color_count)
    patch = np.zeros(mask.shape, dtype=int)

    mandatory_indices = random.sample(range(len(coords)), color_count)
    for color, idx in zip(palette, mandatory_indices):
        r, c = coords[idx]
        patch[r, c] = int(color)

    for i, (r, c) in enumerate(coords):
        if patch[r, c] == BG:
            patch[r, c] = int(random.choice(palette))
    return patch


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bbox_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_touch_4(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    expanded = (ar0 - 1, ac0 - 1, ar1 + 1, ac1 + 1)
    return _bbox_overlap(expanded, box_b)


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, occupied: list[tuple[int, int, int, int]]) -> bool:
    h, w = patch.shape
    box = (top, left, top + h - 1, left + w - 1)
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    if any(_bbox_touch_4(box, other) for other in occupied):
        return False
    if np.any(grid[top : top + h, left : left + w] != BG):
        return False
    paste(grid, patch, top, left, transparent=BG)
    occupied.append(box)
    return True


def _pick_placeholder_cols(width: int, sprite_w: int, count: int) -> list[int] | None:
    starts = list(range(width - sprite_w + 1))
    random.shuffle(starts)
    chosen: list[int] = []
    for start in starts:
        end = start + sprite_w - 1
        if any(_spans_overlap(start, end, other, other + sprite_w - 1) for other in chosen):
            continue
        chosen.append(start)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return None
    return sorted(chosen)


def _build_obstacle_patch() -> np.ndarray:
    return random.choice(OBSTACLE_SHAPES).copy()


def _placeholder_drop_distances(grid: np.ndarray) -> list[int]:
    source, placeholders, obstacles = _extract_scene(grid)
    _stamped, objects = _stamp_placeholders(grid.shape, source, placeholders, obstacles)
    starts = [obj["coords"][:, 0].min() for obj in objects]
    settled = _gravity_settle(grid.shape, [{"coords": obj["coords"].copy(), "values": obj["values"].copy()} for obj in objects], obstacles)

    drops: list[int] = []
    for coords in components_mask((settled != BG) & (settled != OBSTACLE), connectivity=4):
        drops.append(int(coords[:, 0].min()))
    drops.sort()
    starts.sort()
    return [end - start for start, end in zip(starts, drops)]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        source, placeholders, obstacles = _extract_scene(grid)
        actual = main(grid)
        stamped_only = _apply_rule(grid, erase_source=True, settle=False)
        no_erase = _apply_rule(grid, erase_source=False, settle=True)
    except AssertionError:
        return False

    if np.array_equal(actual, stamped_only):
        return False
    if np.array_equal(actual, no_erase):
        return False

    source_bottom = int(source["bbox"][2])
    if source_bottom >= grid.shape[0] - 1:
        return False

    source_left, source_right = int(source["bbox"][1]), int(source["bbox"][3])
    aligned_above = [
        ph for ph in placeholders
        if ph["bbox"][0] < source["bbox"][0]
        and _spans_overlap(source_left, source_right, ph["bbox"][1], ph["bbox"][3])
    ]
    if not aligned_above:
        return False

    colored_components = components_mask((actual != BG) & (actual != OBSTACLE), connectivity=4)
    if len(colored_components) != len(placeholders):
        return False

    if not any(drop > 0 for drop in _placeholder_drop_distances(grid)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    source_patch = _make_source_patch()
    sprite_h, sprite_w = source_patch.shape
    placeholder_count = random.choice(PLACEHOLDER_COUNT_OPTIONS)

    cols = _pick_placeholder_cols(width, sprite_w, placeholder_count)
    if cols is None:
        return None

    source_col = random.choice(cols)
    grid = np.zeros((height, width), dtype=int)
    occupied: list[tuple[int, int, int, int]] = []

    source_row_min = max(height // 2, sprite_h + 3)
    source_row_max = height - sprite_h - 2
    if source_row_min > source_row_max:
        return None
    source_row = random.randint(source_row_min, source_row_max)
    if not _place_patch(grid, source_patch, source_row, source_col, occupied):
        return None

    placeholder_patch = np.where(source_patch != BG, PLACEHOLDER, BG).astype(int)
    aligned_row_max = source_row - sprite_h - 1
    if aligned_row_max < 0:
        return None
    aligned_row = random.randint(0, aligned_row_max)

    placeholder_rows = [aligned_row]
    while len(placeholder_rows) < placeholder_count:
        mode = random.choices(("upper", "lower"), weights=(3, 1), k=1)[0]
        if mode == "upper":
            row_max = max(0, source_row - sprite_h - 1)
            row = random.randint(0, row_max)
        else:
            row_min = min(height - sprite_h - 1, source_row + sprite_h + 1)
            row_max = height - sprite_h - 1
            if row_min > row_max:
                row = random.randint(0, max(0, source_row - sprite_h - 1))
            else:
                row = random.randint(row_min, row_max)
        placeholder_rows.append(row)

    combined = list(zip(cols, placeholder_rows))
    random.shuffle(combined)
    combined.sort(key=lambda item: (item[0] != source_col, item[0]))

    placed_placeholder_boxes: list[tuple[int, int, int, int]] = []
    for col, row in combined:
        if col == source_col:
            row = aligned_row
        box = (row, col, row + sprite_h - 1, col + sprite_w - 1)
        if any(_bbox_touch_4(box, other) for other in occupied + placed_placeholder_boxes):
            return None
        paste(grid, placeholder_patch, row, col, transparent=BG)
        placed_placeholder_boxes.append(box)
        occupied.append(box)

    obstacle_target = random.choices((0, 1, 2), weights=(1, 3, 3), k=1)[0]
    for _ in range(obstacle_target):
        patch = _build_obstacle_patch()
        oh, ow = patch.shape
        options = [
            (r, c)
            for r in range(max(source_row + 1, height - 4), height - oh + 1)
            for c in range(width - ow + 1)
        ]
        random.shuffle(options)
        placed = False
        for r, c in options:
            if _place_patch(grid, patch, r, c, occupied):
                placed = True
                break
        if not placed:
            continue

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
