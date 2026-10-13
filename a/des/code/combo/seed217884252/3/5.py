from __future__ import annotations

import random
from collections import deque

import numpy as np


_BG = 7  # orange
_MIN_REGION_SIDE = 4
_MAX_SLOT_SIDE = 4
_GRID_EXTRA_RANGE = (8, 14)
_REGION_COUNT_RANGE = (2, 5)
_MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    best = int(colors[order[0]])
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "expected a unique dominant background"
    return best


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(
    box: tuple[int, int, int, int], gap: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, box[0] - gap),
        max(0, box[1] - gap),
        min(h - 1, box[2] + gap),
        min(w - 1, box[3] + gap),
    )


def _is_filled_rectangle(grid: np.ndarray, color: int, box: tuple[int, int, int, int], area: int) -> bool:
    r0, c0, r1, c1 = box
    return area == (r1 - r0 + 1) * (c1 - c0 + 1) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def _mixed_component_info(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    comps = []
    for coords in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "colors": colors,
                "area": len(coords),
            }
        )

    fg_colors = set(map(int, np.unique(grid))) - {bg}
    assert len(fg_colors) >= 2, "expected at least two foreground colors"
    mixed = [comp for comp in comps if comp["colors"] == fg_colors]
    assert len(mixed) == 1, "expected a unique mixed component containing all foreground colors"
    return mixed[0], comps


def _find_slots(grid: np.ndarray, mixed: dict, bg: int) -> dict[int, dict]:
    r0, c0, r1, c1 = mixed["bbox"]
    mixed_mask = np.zeros_like(grid, dtype=bool)
    for r, c in mixed["coords"]:
        mixed_mask[r, c] = True

    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    slots: dict[int, dict] = {}
    sub_h, sub_w = sub_bg.shape
    for comp in _components(sub_bg):
        if any(rr == 0 or cc == 0 or rr == sub_h - 1 or cc == sub_w - 1 for rr, cc in comp):
            continue

        abs_coords = [(r0 + rr, c0 + cc) for rr, cc in comp]
        sr0, sc0, sr1, sc1 = _bbox(abs_coords)
        area = len(abs_coords)
        assert area == (sr1 - sr0 + 1) * (sc1 - sc0 + 1), "slot must be a filled rectangle"

        neighbor_colors = set()
        for r, c in abs_coords:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mixed_mask[nr, nc]:
                    neighbor_colors.add(int(grid[nr, nc]))
        assert len(neighbor_colors) == 1, "slot must belong to exactly one colored region"
        color = next(iter(neighbor_colors))
        assert color not in slots, "each region color should own one unique slot"
        slots[color] = {
            "bbox": (sr0, sc0, sr1, sc1),
            "coords": abs_coords,
            "shape": (sr1 - sr0 + 1, sc1 - sc0 + 1),
        }

    assert set(slots) == set(mixed["colors"]), "every mixed-object color must have one slot"
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique mixed object that contains all foreground colors, then move each outside
    monochrome rectangle into the same-color rectangular slot inside that object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    mixed, comps = _mixed_component_info(g, bg)
    slots = _find_slots(g, mixed, bg)

    out = g.copy()
    matched_colors: set[int] = set()

    for comp in comps:
        if comp is mixed:
            continue
        assert len(comp["colors"]) == 1, "outside objects must be monochrome"
        color = next(iter(comp["colors"]))
        assert color in slots, "outside object color must match a slot color"
        assert color not in matched_colors, "expected one outside rectangle per color"

        box = comp["bbox"]
        assert _is_filled_rectangle(g, color, box, comp["area"]), "outside object must be a solid rectangle"
        slot = slots[color]
        assert slot["shape"] == (box[2] - box[0] + 1, box[3] - box[1] + 1), "rectangle and slot must match in size"

        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

        sr0, sc0, sr1, sc1 = slot["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = color
        matched_colors.add(color)

    assert matched_colors == set(slots), "every slot must be filled by one outside rectangle"
    return out


def _split_rectangles(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, h, w)]
    for _ in range(count - 1):
        choices = []
        for idx, (_, _, rh, rw) in enumerate(rects):
            if rh >= 2 * _MIN_REGION_SIDE:
                choices.append((idx, 0))
            if rw >= 2 * _MIN_REGION_SIDE:
                choices.append((idx, 1))
        if not choices:
            return None

        idx, axis = random.choice(choices)
        r0, c0, rh, rw = rects.pop(idx)
        if axis == 0:
            cut = random.randint(_MIN_REGION_SIDE, rh - _MIN_REGION_SIDE)
            rects.append((r0, c0, cut, rw))
            rects.append((r0 + cut, c0, rh - cut, rw))
        else:
            cut = random.randint(_MIN_REGION_SIDE, rw - _MIN_REGION_SIDE)
            rects.append((r0, c0, rh, cut))
            rects.append((r0, c0 + cut, rh, rw - cut))
    random.shuffle(rects)
    return rects


def _partition_regions(count: int) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    for _ in range(80):
        mix_h = random.randint(max(10, _MIN_REGION_SIDE + 2), 15)
        mix_w = random.randint(max(10, _MIN_REGION_SIDE + 2), 15)
        rects = _split_rectangles(mix_h, mix_w, count)
        if rects is not None:
            return mix_h, mix_w, rects
    return None


def _slot_for_region(region_h: int, region_w: int) -> tuple[int, int, int, int]:
    slot_h = random.randint(1, min(_MAX_SLOT_SIDE, region_h - 2))
    slot_w = random.randint(1, min(_MAX_SLOT_SIDE, region_w - 2))
    top = random.randint(1, region_h - slot_h - 1)
    left = random.randint(1, region_w - slot_w - 1)
    return top, left, slot_h, slot_w


def _place_sources(
    shape: tuple[int, int],
    mixed_box: tuple[int, int, int, int],
    source_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(source_shapes)), key=lambda i: source_shapes[i][0] * source_shapes[i][1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(source_shapes)
    placed_boxes: list[tuple[int, int, int, int] | None] = [None] * len(source_shapes)
    forbidden = _expand_box(mixed_box, 1, shape)

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        sh, sw = source_shapes[idx]
        candidates = []
        for r0 in range(shape[0] - sh + 1):
            for c0 in range(shape[1] - sw + 1):
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if _box_intersects(box, forbidden):
                    continue
                if any(other is not None and _box_intersects(box, _expand_box(other, 1, shape)) for other in placed_boxes):
                    continue
                candidates.append((r0, c0))
        random.shuffle(candidates)
        for r0, c0 in candidates:
            placed[idx] = (r0, c0)
            placed_boxes[idx] = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            if rec(pos + 1):
                return True
            placed[idx] = None
            placed_boxes[idx] = None
        return False

    return placed if rec(0) else None


def _render_case(
    grid_h: int,
    grid_w: int,
    mix_h: int,
    mix_w: int,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    out = np.full((grid_h, grid_w), _BG, dtype=int)
    inp = np.full((grid_h, grid_w), _BG, dtype=int)

    mix_r0 = random.randint(2, grid_h - mix_h - 2)
    mix_c0 = random.randint(2, grid_w - mix_w - 2)
    mixed_box = (mix_r0, mix_c0, mix_r0 + mix_h - 1, mix_c0 + mix_w - 1)

    source_shapes: list[tuple[int, int]] = []
    source_colors: list[int] = []
    slot_records: list[tuple[int, int, int, int, int]] = []

    for (rr0, rc0, rh, rw), color in zip(rects, colors):
        ar0 = mix_r0 + rr0
        ac0 = mix_c0 + rc0
        out[ar0 : ar0 + rh, ac0 : ac0 + rw] = color
        inp[ar0 : ar0 + rh, ac0 : ac0 + rw] = color

        slot_r, slot_c, slot_h, slot_w = _slot_for_region(rh, rw)
        inp[ar0 + slot_r : ar0 + slot_r + slot_h, ac0 + slot_c : ac0 + slot_c + slot_w] = _BG
        source_shapes.append((slot_h, slot_w))
        source_colors.append(color)
        slot_records.append((ar0 + slot_r, ac0 + slot_c, slot_h, slot_w, color))

    placements = _place_sources(inp.shape, mixed_box, source_shapes)
    if placements is None:
        return None

    for (r0, c0), (sh, sw), color in zip(placements, source_shapes, source_colors):
        inp[r0 : r0 + sh, c0 : c0 + sw] = color

    # Structural checks independent of main().
    assert len(slot_records) == len(colors)
    assert np.unique(out).tolist().count(_BG) >= 1
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*_REGION_COUNT_RANGE)
    partition = _partition_regions(count)
    if partition is None:
        return None
    mix_h, mix_w, rects = partition

    grid_h = mix_h + random.randint(*_GRID_EXTRA_RANGE)
    grid_w = mix_w + random.randint(*_GRID_EXTRA_RANGE)

    palette = [c for c in range(10) if c != _BG]
    colors = random.sample(palette, count)
    case = _render_case(grid_h, grid_w, mix_h, mix_w, rects, colors)
    if case is None:
        return None
    inp, expected = case

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
