from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
HOLE_COUNTS = (1, 2, 3, 4, 5)
GENERATION_RETRIES = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _monochrome_color(grid: np.ndarray, coords: np.ndarray) -> int:
    vals = {int(grid[r, c]) for r, c in coords}
    assert len(vals) == 1, "expected monochromatic component"
    return next(iter(vals))


def _hole_count_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _hole_count_coords(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _hole_count_mask(sub)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each left-side gray template by the right-side object's color when both enclose the same number of holes, and erase everything else.

    Input structure:
    - Gray template components stand in a vertical stack on the left, while separate colored components are scattered on the right.
    - Every template and every colored object is a single 4-connected monochrome component whose interior may contain enclosed background holes.
    - Hole counts are unique within the gray family and within the colored family, but the two families have different sizes so some counts appear on only one side.

    Rule:
    - Count the number of enclosed 4-connected background holes inside each gray template and each colored object.
    - Match a template to the unique object with the same hole count.
    - Recolor each matched template to that object's color while leaving its holes as background.
    - Erase unmatched templates and erase all right-side colored objects.

    Key insight:
    - Shape outline, size, color, and left-to-right or top-to-bottom order are decoys; the only reliable cross-family invariant is the number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_components = _components(g == TEMPLATE_COLOR)
    object_components = _components((g != BG) & (g != TEMPLATE_COLOR))
    assert template_components, "expected gray templates"
    assert object_components, "expected colored objects"

    object_by_holes: dict[int, int] = {}
    for coords in object_components:
        holes = _hole_count_coords(coords)
        color = _monochrome_color(g, coords)
        assert holes not in object_by_holes, "object hole counts must be unique"
        object_by_holes[holes] = color

    out = np.full_like(g, BG)
    template_holes_seen: set[int] = set()
    for coords in template_components:
        holes = _hole_count_coords(coords)
        assert holes not in template_holes_seen, "template hole counts must be unique"
        template_holes_seen.add(holes)
        if holes in object_by_holes:
            out[coords[:, 0], coords[:, 1]] = object_by_holes[holes]

    return out


def _make_template_sprite(hole_count: int, rng: random.Random) -> np.ndarray:
    cols = 1 if hole_count <= 2 or (hole_count == 3 and rng.random() < 0.4) else 2
    rows = (hole_count + cols - 1) // cols
    base_h = 2 * rows + 1
    base_w = 2 * cols + 1
    sprite = np.zeros((base_h, base_w + 2), dtype=int)
    sprite[:, 1 : 1 + base_w] = 1
    sprite[:, 0] = 1

    slots = [(1 + 2 * rr, 1 + 2 * cc) for rr in range(rows) for cc in range(cols)]
    for rr, cc in rng.sample(slots, hole_count):
        sprite[rr, cc] = 0

    if rows == 1:
        sprite[:, -1] = 1
    else:
        for rr in range(rows):
            a = 2 * rr
            b = min(base_h, a + 2)
            sprite[a:b, -1] = 1
        if rng.random() < 0.5:
            sprite[1:-1, -1] = 1

    if rng.random() < 0.5:
        sprite = np.flip(sprite, axis=0)
    return sprite.astype(int)


def _make_object_sprite(hole_count: int, rng: random.Random) -> np.ndarray:
    rows = 1 if hole_count <= 2 or (hole_count == 3 and rng.random() < 0.45) else 2
    cols = (hole_count + rows - 1) // rows
    base_h = 2 * rows + 1
    base_w = 2 * cols + 1
    sprite = np.zeros((base_h + 2, base_w + 1), dtype=int)
    sprite[1 : 1 + base_h, :base_w] = 1
    sprite[0, 1:base_w] = 1
    sprite[-1, : max(1, base_w - 1)] = 1
    sprite[1:-1, -1] = 1

    slots = [(1 + 2 * rr, 1 + 2 * cc) for rr in range(rows) for cc in range(cols)]
    for rr, cc in rng.sample(slots, hole_count):
        sprite[rr, cc] = 0

    if rng.random() < 0.5:
        sprite[0, 0] = 1
    if rng.random() < 0.5:
        sprite[-1, -1] = 1
    if rng.random() < 0.5:
        sprite = np.flip(sprite, axis=1)
    return sprite.astype(int)


def _paint_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = sprite.shape
    patch = grid[top : top + h, left : left + w]
    patch[sprite.astype(bool)] = color


def _paired_output(
    shape: tuple[int, int],
    templates: list[dict[str, object]],
    object_color_by_holes: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for info in templates:
        holes = int(info["holes"])
        if holes not in object_color_by_holes:
            continue
        sprite = info["sprite"]
        top = int(info["top"])
        left = int(info["left"])
        _paint_sprite(out, sprite, top, left, object_color_by_holes[holes])
    return out


def _scan_order_counts(infos: list[dict[str, object]]) -> list[int]:
    ordered = sorted(infos, key=lambda d: (int(d["top"]), int(d["left"])))
    return [int(d["holes"]) for d in ordered]


def _area_order_counts(infos: list[dict[str, object]]) -> list[int]:
    ordered = sorted(
        infos,
        key=lambda d: (
            int(d["area"]),
            (int(d["bbox"][2]) - int(d["bbox"][0]) + 1) * (int(d["bbox"][3]) - int(d["bbox"][1]) + 1),
            int(d["holes"]),
        ),
    )
    return [int(d["holes"]) for d in ordered]


def _pick_orders(
    template_counts: list[int],
    object_counts: list[int],
    shared: set[int],
    align_mode: bool,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    for _ in range(200):
        temp_order = template_counts[:]
        rng.shuffle(temp_order)
        if align_mode and not any(c in shared for c in temp_order[: min(len(temp_order), len(object_counts))]):
            continue

        obj_order = object_counts[:]
        if align_mode:
            order: list[int | None] = [None] * len(obj_order)
            remaining = set(obj_order)
            for i in range(min(len(temp_order), len(obj_order))):
                count = temp_order[i]
                if count in shared and count in remaining:
                    order[i] = count
                    remaining.remove(count)
            leftover = list(remaining)
            rng.shuffle(leftover)
            for i in range(len(order)):
                if order[i] is None:
                    order[i] = leftover.pop()
            obj_order = [int(v) for v in order]
            if sum(temp_order[i] == obj_order[i] for i in range(min(len(temp_order), len(obj_order)))) == 0:
                continue
        else:
            for _ in range(200):
                rng.shuffle(obj_order)
                if all(temp_order[i] != obj_order[i] for i in range(min(len(temp_order), len(obj_order)))):
                    break
            else:
                continue
        return temp_order, obj_order
    raise RuntimeError("failed to pick decoy orders")


def _candidate_counts(rng: random.Random) -> tuple[list[int], list[int], set[int]]:
    template_n, object_n = rng.choice(((3, 4), (4, 3)))
    pool = rng.sample(list(HOLE_COUNTS), 5)
    shared = set(rng.sample(pool, 2))
    remaining = [count for count in pool if count not in shared]
    rng.shuffle(remaining)
    template_only = remaining[: template_n - 2]
    object_only = remaining[template_n - 2 :]
    template_counts = list(shared) + template_only
    object_counts = list(shared) + object_only
    return template_counts, object_counts, shared


def _build_layout() -> np.ndarray | None:
    rng = random.Random()
    template_counts, object_counts, shared = _candidate_counts(rng)
    align_mode = rng.random() < 0.35
    template_order, object_order = _pick_orders(template_counts, object_counts, shared, align_mode, rng)

    template_sprites = {count: _make_template_sprite(count, rng) for count in template_counts}
    object_sprites = {count: _make_object_sprite(count, rng) for count in object_counts}

    if any(
        template_sprites[count].shape == object_sprites[count].shape
        or template_sprites[count].shape == object_sprites[count].shape[::-1]
        for count in shared
    ):
        return None

    band_w = max(sprite.shape[1] for sprite in template_sprites.values())
    template_gap_choices = [1, 2]
    template_gaps = [rng.choice(template_gap_choices) for _ in range(len(template_order) - 1)]
    template_total_h = sum(template_sprites[count].shape[0] for count in template_order) + sum(template_gaps)

    max_obj_h = max(sprite.shape[0] for sprite in object_sprites.values())
    max_obj_w = max(sprite.shape[1] for sprite in object_sprites.values())
    slot_h = max_obj_h + 1
    slot_w = max_obj_w + 1

    grid_h = max(template_total_h, 2 * slot_h + 1)
    grid_w = band_w + 2 + 2 * slot_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    templates: list[dict[str, object]] = []
    top = 0
    for idx, count in enumerate(template_order):
        sprite = template_sprites[count]
        left = rng.randint(0, band_w - sprite.shape[1])
        _paint_sprite(grid, sprite, top, left, TEMPLATE_COLOR)
        templates.append({"holes": count, "sprite": sprite, "top": top, "left": left})
        top += sprite.shape[0]
        if idx + 1 < len(template_order):
            top += template_gaps[idx]

    right0 = band_w + 2
    slot_origins = [
        (0, right0),
        (0, right0 + slot_w),
        (grid_h - slot_h, right0),
        (grid_h - slot_h, right0 + slot_w),
    ]
    chosen_slots = sorted(rng.sample(slot_origins, len(object_order)))
    object_colors = rng.sample(list(OBJECT_COLORS), len(object_order))
    objects: list[dict[str, object]] = []
    object_color_by_holes: dict[int, int] = {}
    for (slot_top, slot_left), count, color in zip(chosen_slots, object_order, object_colors):
        sprite = object_sprites[count]
        top_slack = slot_h - sprite.shape[0]
        left_slack = slot_w - sprite.shape[1]
        top = slot_top + rng.randint(0, top_slack)
        left = slot_left + rng.randint(0, left_slack)
        _paint_sprite(grid, sprite, top, left, color)
        objects.append({"holes": count, "sprite": sprite, "top": top, "left": left, "color": color})
        object_color_by_holes[count] = color

    actual_templates = _components(grid == TEMPLATE_COLOR)
    actual_objects = _components((grid != BG) & (grid != TEMPLATE_COLOR))
    if len(actual_templates) != len(template_order) or len(actual_objects) != len(object_order):
        return None

    template_infos = []
    for coords in actual_templates:
        template_infos.append(
            {
                "holes": _hole_count_coords(coords),
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    object_infos = []
    for coords in actual_objects:
        object_infos.append(
            {
                "holes": _hole_count_coords(coords),
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
                "color": _monochrome_color(grid, coords),
            }
        )

    if {info["holes"] for info in template_infos} != set(template_counts):
        return None
    if {info["holes"] for info in object_infos} != set(object_counts):
        return None
    if set(template_counts) & set(object_counts) != shared:
        return None

    template_scan = _scan_order_counts(templates)
    object_scan = _scan_order_counts(objects)
    same_index_matches = sum(
        template_scan[i] == object_scan[i] for i in range(min(len(template_scan), len(object_scan)))
    )
    if align_mode:
        if same_index_matches == 0:
            return None
    else:
        if same_index_matches != 0:
            return None

    sorted_template_counts = sorted(template_counts)
    if _area_order_counts(template_infos) in (sorted_template_counts, sorted_template_counts[::-1]):
        return None
    sorted_object_counts = sorted(object_counts)
    if _area_order_counts(object_infos) in (sorted_object_counts, sorted_object_counts[::-1]):
        return None

    expected = _paired_output(
        grid.shape,
        templates,
        {int(info["holes"]): int(info["color"]) for info in object_infos},
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
