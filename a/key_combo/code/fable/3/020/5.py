from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
TEAL = 8

MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_count_from_coords(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    obj = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes = 0
    for bg_coords in _components(~obj):
        rr = bg_coords[:, 0]
        cc = bg_coords[:, 1]
        touches_edge = np.any((rr == 0) | (rr == obj.shape[0] - 1) | (cc == 0) | (cc == obj.shape[1] - 1))
        if not touches_edge:
            holes += 1
    return holes


def _diamond_counts(shape: tuple[int, int], centers_and_radii: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    counts = np.zeros((h, w), dtype=np.int16)
    for cr, cc, radius in centers_and_radii:
        for dr in range(-radius, radius + 1):
            rr = cr + dr
            if not (0 <= rr < h):
                continue
            reach = radius - abs(dr)
            c0 = max(0, cc - reach)
            c1 = min(w - 1, cc + reach)
            counts[rr, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each object's enclosed holes, erase the object, and forge a centered diamond whose radius matches that count.

    Input structure:
    - The grid has a black background with several separated foreground objects.
    - Each object is 4-connected and encloses one or more 4-connected black holes inside its own odd-sized bounding box.
    - Object outlines, sizes, and positions vary independently of the number of holes they contain.

    Rule:
    - Split the non-black pixels into 4-connected objects.
    - For each object, count the black regions inside its bounding box that do not touch the bounding-box edge.
    - Remove every original object and draw a solid teal Manhattan diamond centered at that bounding box, using the hole count as the diamond radius; any cell covered by multiple diamonds becomes red.

    Key insight:
    - The output ignores the original shapes and keeps only a topological invariant: how many enclosed holes each object has.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BLACK)
    centers_and_radii: list[tuple[int, int, int]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "object bbox must have odd side lengths"
        radius = _hole_count_from_coords(coords)
        assert radius >= 1, "each object must contain at least one enclosed hole"
        centers_and_radii.append(((r0 + r1) // 2, (c0 + c1) // 2, radius))

    counts = _diamond_counts(g.shape, centers_and_radii)
    out = np.zeros_like(g)
    out[counts == 1] = TEAL
    out[counts >= 2] = RED
    return out


def _hole_sites(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2, h - 1, 2) for c in range(2, w - 1, 2)]


def _sample_dims(hole_count: int) -> tuple[int, int]:
    candidates = []
    for h in (5, 7, 9):
        for w in (5, 7, 9):
            if len(_hole_sites(h, w)) >= hole_count:
                candidates.append((h, w))
    assert candidates
    weights = [1.0 / float(h * w) for h, w in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def _carve_edge_notch(sprite: np.ndarray) -> None:
    h, w = sprite.shape
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in {"top", "bottom"} and w >= 5:
        length = random.randint(1, min(2, w - 3))
        start = random.randint(1, w - 1 - length)
        row = 0 if edge == "top" else h - 1
        sprite[row, start : start + length] = BLACK
    elif edge in {"left", "right"} and h >= 5:
        length = random.randint(1, min(2, h - 3))
        start = random.randint(1, h - 1 - length)
        col = 0 if edge == "left" else w - 1
        sprite[start : start + length, col] = BLACK


def _make_object_sprite(hole_count: int, h: int, w: int) -> np.ndarray:
    sites = _hole_sites(h, w)
    for _ in range(120):
        sprite = np.full((h, w), TEAL, dtype=int)
        for rr, cc in random.sample(sites, hole_count):
            sprite[rr, cc] = BLACK

        for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            if random.random() < 0.45:
                sprite[corner] = BLACK

        for _ in range(random.randint(0, 3)):
            _carve_edge_notch(sprite)

        mask = sprite == TEAL
        comps = _components(mask)
        if len(comps) != 1:
            continue
        coords = comps[0]
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        if _hole_count_from_coords(coords) != hole_count:
            continue
        return sprite
    raise RuntimeError("failed to build a valid object sprite")


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _random_box(
    height: int,
    width: int,
    box_h: int,
    box_w: int,
    existing: list[tuple[int, int, int, int]],
    pad: int,
) -> tuple[int, int, int, int] | None:
    for _ in range(300):
        r0 = random.randint(0, height - box_h)
        c0 = random.randint(0, width - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(_boxes_separated(box, other, pad) for other in existing):
            return box
    return None


def _render_from_specs(shape: tuple[int, int], placed_specs: list[dict]) -> np.ndarray:
    centers_and_radii = [
        (spec["box"][0] + spec["h"] // 2, spec["box"][1] + spec["w"] // 2, spec["holes"]) for spec in placed_specs
    ]
    counts = _diamond_counts(shape, centers_and_radii)
    out = np.zeros(shape, dtype=int)
    out[counts == 1] = TEAL
    out[counts >= 2] = RED
    return out


def _sample_hole_counts(n_objects: int, force_overlap: bool) -> list[int]:
    counts = [random.randint(1, 4) for _ in range(n_objects)]
    if force_overlap:
        counts[0] = 4
        counts[1] = 4
    if len(set(counts)) == 1:
        idx = random.randrange(n_objects)
        options = [k for k in range(1, 5) if k != counts[idx]]
        counts[idx] = random.choice(options)
    return counts


def _place_specs(height: int, width: int, specs: list[dict], force_overlap: bool) -> list[dict] | None:
    placed: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    start_index = 0
    if force_overlap:
        first, second = specs[0], specs[1]
        orientation = random.choice(("horizontal", "vertical"))
        if orientation == "horizontal":
            total_w = first["w"] + 1 + second["w"]
            total_h = max(first["h"], second["h"])
            if total_w > width or total_h > height:
                return None
            r0 = random.randint(0, height - total_h)
            c0 = random.randint(0, width - total_w)
            box1 = (r0, c0, r0 + first["h"] - 1, c0 + first["w"] - 1)
            box2 = (r0, c0 + first["w"] + 1, r0 + second["h"] - 1, c0 + first["w"] + 1 + second["w"] - 1)
        else:
            total_h = first["h"] + 1 + second["h"]
            total_w = max(first["w"], second["w"])
            if total_w > width or total_h > height:
                return None
            r0 = random.randint(0, height - total_h)
            c0 = random.randint(0, width - total_w)
            box1 = (r0, c0, r0 + first["h"] - 1, c0 + first["w"] - 1)
            box2 = (r0 + first["h"] + 1, c0, r0 + first["h"] + 1 + second["h"] - 1, c0 + second["w"] - 1)

        first = dict(first)
        second = dict(second)
        first["box"] = box1
        second["box"] = box2
        placed.extend([first, second])
        boxes.extend([box1, box2])
        start_index = 2

    remaining = sorted(specs[start_index:], key=lambda spec: spec["h"] * spec["w"], reverse=True)
    pad = 2 if force_overlap else 3
    for spec in remaining:
        box = _random_box(height, width, spec["h"], spec["w"], boxes, pad)
        if box is None:
            return None
        placed_spec = dict(spec)
        placed_spec["box"] = box
        placed.append(placed_spec)
        boxes.append(box)

    return placed


def _build_input_grid(height: int, width: int, placed_specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["sprite"]
    return grid


def _validate_generated_scene(grid: np.ndarray, placed_specs: list[dict], force_overlap: bool) -> bool:
    expected_holes = sorted(spec["holes"] for spec in placed_specs)
    comps = _components(grid == TEAL)
    if len(comps) != len(placed_specs):
        return False

    actual_holes = []
    actual_boxes = []
    for coords in comps:
        actual_holes.append(_hole_count_from_coords(coords))
        actual_boxes.append(_bbox(coords))
    if sorted(actual_holes) != expected_holes:
        return False
    if sorted(actual_boxes) != sorted(spec["box"] for spec in placed_specs):
        return False

    expected = _render_from_specs(grid.shape, placed_specs)
    red_pixels = int(np.sum(expected == RED))
    if force_overlap and red_pixels < 1:
        return False
    if (not force_overlap) and red_pixels != 0:
        return False
    if red_pixels > 12:
        return False

    return bool(np.array_equal(main(grid), expected))


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    force_overlap = random.random() < 0.55

    hole_counts = _sample_hole_counts(n_objects, force_overlap)
    specs = []
    for idx, hole_count in enumerate(hole_counts):
        if force_overlap and idx < 2:
            h, w = 7, 7
        else:
            h, w = _sample_dims(hole_count)
        specs.append(
            {
                "holes": hole_count,
                "h": h,
                "w": w,
                "sprite": _make_object_sprite(hole_count, h, w),
            }
        )

    placed_specs = _place_specs(height, width, specs, force_overlap)
    if placed_specs is None:
        return None

    grid = _build_input_grid(height, width, placed_specs)
    if not _validate_generated_scene(grid, placed_specs, force_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
