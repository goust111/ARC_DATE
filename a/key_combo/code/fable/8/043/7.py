from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
D8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
PALETTE = list(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    mask: np.ndarray | None = None,
) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    use = np.ones((h, w), dtype=bool)
    if mask is not None:
        use &= mask.astype(bool)
    if transparent is not None:
        use &= patch != int(transparent)
    view[use] = patch[use]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    nbrs = D4 if int(connectivity) == 4 else D8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    colors: list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    if colors is None:
        colors_iter = sorted(set(map(int, g.ravel())) - {int(bg)})
    else:
        colors_iter = [int(c) for c in colors if int(c) != int(bg)]

    out = []
    for color in colors_iter:
        for coords in components_mask(g == color, connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def shape_area(mask: np.ndarray) -> int:
    return int(np.sum(mask.astype(bool)))


def all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    base = np.asarray(mask, dtype=bool)
    for flip in (False, True):
        cur = np.flipud(base) if flip else base
        for k in range(4):
            turned = np.rot90(cur, k)
            key = (turned.shape, turned.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(turned.copy())
    return variants


def canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((m.shape, m.tobytes()) for m in all_orientations(mask))


def random_orientation(mask: np.ndarray) -> np.ndarray:
    return random.choice(all_orientations(mask))


def parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4)
    assert comps, "expected a non-background frame"
    max_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(frame_candidates) == 1, "expected one unique largest frame"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)
    crop = crop_bbox(g, fb)
    hole_components = components_mask(crop == bg, connectivity=4)
    holes = []
    for coords in hole_components:
        hb = bbox_coords(coords)
        mask = component_mask(coords, hb)
        holes.append(
            {
                "coords": coords,
                "bbox": hb,
                "mask": mask,
                "area": int(coords.shape[0]),
                "key": canonical_key(mask),
            }
        )

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprites = []
    for comp in find_components(outside, bg=bg, connectivity=8):
        mask = component_mask(comp["coords"], comp["bbox"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "coords": comp["coords"],
                "area": int(comp["area"]),
                "mask": mask,
                "key": canonical_key(mask),
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "frame_bbox": fb,
        "crop": crop,
        "holes": holes,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame hole with the outside color whose sprite count equals that hole's area.

    Input structure:
    - The background is the most common color.
    - One large monochrome solid frame contains several background-colored holes.
    - Outside the frame, each non-frame color appears as several separated small sprites.

    Rule:
    - Crop to the frame's bounding box.
    - For each hole inside that crop, count its area in cells.
    - Count outside sprites by color, find the unique color whose sprite count equals the hole area,
      and fill the hole with that color while keeping the frame pixels unchanged.

    Key insight:
    - The matching key is arithmetic: hole area equals sprite count of one color; any shape resemblance
      between holes and outside sprites is only a decoy.
    """
    parsed = parse_instance(np.asarray(grid, dtype=int))
    frame_color = int(parsed["frame_color"])
    out = np.full(parsed["crop"].shape, frame_color, dtype=int)

    color_counts = Counter(int(sprite["color"]) for sprite in parsed["sprites"])
    for hole in parsed["holes"]:
        area = int(hole["area"])
        matches = [color for color, count in color_counts.items() if int(count) == area]
        assert len(matches) == 1, "each hole area must match exactly one sprite color count"
        fill_color = int(matches[0])
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = fill_color
    return out


HOLE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
]


def _build_shapes_by_area() -> dict[int, list[np.ndarray]]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    seen = set()
    for mask in HOLE_LIBRARY:
        key = canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        grouped[shape_area(mask)].append(mask.copy())
    return dict(grouped)


SHAPES_BY_AREA = _build_shapes_by_area()
SPRITE_LIBRARY = [mask.copy() for masks in SHAPES_BY_AREA.values() for mask in masks]

GRID_MIN = 24
GRID_MAX = 30
FRAME_H_RANGE = (9, 14)
FRAME_W_RANGE = (11, 18)
HOLE_COUNT_WEIGHTS = [(3, 4), (4, 4), (5, 1)]
MAX_ATTEMPTS = 400


def _pick_hole_count() -> int:
    values = [v for v, _w in HOLE_COUNT_WEIGHTS]
    weights = [w for _v, w in HOLE_COUNT_WEIGHTS]
    return int(random.choices(values, weights=weights)[0])


def _sample_hole_specs() -> list[dict]:
    hole_count = _pick_hole_count()
    areas = random.sample(list(SHAPES_BY_AREA.keys()), hole_count)
    specs = []
    for area in areas:
        mask = random_orientation(random.choice(SHAPES_BY_AREA[area]))
        specs.append({"area": int(area), "mask": mask})
    random.shuffle(specs)
    return specs


def _place_holes(frame_h: int, frame_w: int, hole_specs: list[dict]) -> list[dict] | None:
    blocked = np.zeros((frame_h, frame_w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    placed = []
    order = hole_specs[:]
    random.shuffle(order)
    order.sort(key=lambda spec: (-int(spec["area"]), -spec["mask"].shape[0] * spec["mask"].shape[1]))

    for spec in order:
        mask = spec["mask"]
        mh, mw = mask.shape
        candidates = []
        for r0 in range(1, frame_h - mh):
            for c0 in range(1, frame_w - mw):
                coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
                if blocked[coords[:, 0], coords[:, 1]].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
        placed.append(
            {
                **spec,
                "top": int(r0),
                "left": int(c0),
                "coords": coords,
                "key": canonical_key(mask),
            }
        )
        for rr, cc in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < frame_h and 0 <= nc < frame_w:
                        blocked[nr, nc] = True
    return placed


def _random_sprite_mask(avoid_keys: set[tuple[tuple[int, int], bytes]] | None = None) -> np.ndarray:
    avoid_keys = set() if avoid_keys is None else set(avoid_keys)
    choices = SPRITE_LIBRARY[:]
    random.shuffle(choices)
    for base in choices:
        oriented = random_orientation(base)
        if canonical_key(oriented) not in avoid_keys:
            return oriented
    raise RuntimeError("sprite library exhausted")


def _build_sprite_groups(hole_specs: list[dict], count_by_color: dict[int, int]) -> dict[int, list[np.ndarray]] | None:
    groups: dict[int, list[np.ndarray]] = {}
    hole_key_by_color = {int(spec["color"]): spec["key"] for spec in hole_specs}

    for color, count in count_by_color.items():
        avoid = {hole_key_by_color[color]} if color in hole_key_by_color else set()
        groups[int(color)] = [_random_sprite_mask(avoid) for _ in range(int(count))]

    hole_order = hole_specs[:]
    random.shuffle(hole_order)
    decoy_count = 0
    for hole in hole_order:
        wrong_choices = [c for c in count_by_color if int(c) != int(hole["color"])]
        if not wrong_choices:
            continue
        wrong_color = int(random.choice(wrong_choices))
        slots = list(range(len(groups[wrong_color])))
        random.shuffle(slots)
        use_n = 1 + int(len(slots) >= 2 and random.random() < 0.5)
        for idx in slots[:use_n]:
            groups[wrong_color][idx] = random_orientation(hole["mask"])
        correct_color = int(hole["color"])
        avoid = {hole["key"]}
        groups[correct_color] = [_random_sprite_mask(avoid) for _ in range(len(groups[correct_color]))]
        decoy_count += 1
        if decoy_count >= 1 and random.random() < 0.45:
            break

    return groups if decoy_count >= 1 else None


def _render_frame_crop(frame_h: int, frame_w: int, frame_color: int, bg: int, hole_specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    input_crop = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    output_crop = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    for hole in hole_specs:
        coords = hole["coords"]
        input_crop[coords[:, 0], coords[:, 1]] = int(bg)
        output_crop[coords[:, 0], coords[:, 1]] = int(hole["color"])
    return input_crop, output_crop


def _place_sprite_patches(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    sprite_groups: dict[int, list[np.ndarray]],
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    rr0 = max(0, fr0 - 1)
    cc0 = max(0, fc0 - 1)
    rr1 = min(h, fr1 + 2)
    cc1 = min(w, fc1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    items = []
    for color, masks in sprite_groups.items():
        for mask in masks:
            items.append((mask.shape[0] * mask.shape[1], int(color), mask))
    items.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for _bbox_area, color, mask in items:
        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[mask] = int(color)
        ph, pw = patch.shape
        candidates = []
        for r0 in range(0, h - ph + 1):
            for c0 in range(0, w - pw + 1):
                x0 = max(0, r0 - 1)
                y0 = max(0, c0 - 1)
                x1 = min(h, r0 + ph + 1)
                y1 = min(w, c0 + pw + 1)
                if blocked[x0:x1, y0:y1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        paste(grid, patch, r0, c0, transparent=bg)
        x0 = max(0, r0 - 1)
        y0 = max(0, c0 - 1)
        x1 = min(h, r0 + ph + 1)
        y1 = min(w, c0 + pw + 1)
        blocked[x0:x1, y0:y1] = True
    return True


def _shape_baseline(grid: np.ndarray) -> np.ndarray:
    parsed = parse_instance(grid)
    out = np.full(parsed["crop"].shape, int(parsed["frame_color"]), dtype=int)
    by_shape: dict[tuple[tuple[int, int], bytes], Counter] = defaultdict(Counter)
    for sprite in parsed["sprites"]:
        by_shape[sprite["key"]][int(sprite["color"])] += 1
    for hole in parsed["holes"]:
        counter = by_shape.get(hole["key"])
        if not counter:
            continue
        color = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _pixel_count_baseline(grid: np.ndarray) -> np.ndarray:
    parsed = parse_instance(grid)
    out = np.full(parsed["crop"].shape, int(parsed["frame_color"]), dtype=int)
    pixel_counts = Counter()
    for sprite in parsed["sprites"]:
        pixel_counts[int(sprite["color"])] += int(sprite["area"])
    for hole in parsed["holes"]:
        area = int(hole["area"])
        matches = [color for color, total in pixel_counts.items() if int(total) == area]
        if len(matches) != 1:
            continue
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(matches[0])
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    hole_specs: list[dict],
    count_by_color: dict[int, int],
    sprite_groups: dict[int, list[np.ndarray]],
) -> bool:
    parsed = parse_instance(inp)
    areas = [int(spec["area"]) for spec in hole_specs]
    if len(set(areas)) != len(areas):
        return False
    if len(set(count_by_color.values())) != len(count_by_color):
        return False
    if len(parsed["holes"]) != len(hole_specs):
        return False

    area_to_color = {int(spec["area"]): int(spec["color"]) for spec in hole_specs}
    for area, color in area_to_color.items():
        matches = [c for c, count in count_by_color.items() if int(count) == int(area)]
        if len(matches) != 1 or int(matches[0]) != int(color):
            return False

    if not any(shape_area(mask) > 1 for masks in sprite_groups.values() for mask in masks):
        return False
    total_pixels = {int(color): sum(shape_area(mask) for mask in masks) for color, masks in sprite_groups.items()}
    used_colors = [int(spec["color"]) for spec in hole_specs]
    if not any(int(total_pixels[color]) != int(count_by_color[color]) for color in used_colors):
        return False

    strong_decoy = False
    for spec in hole_specs:
        key = spec["key"]
        correct = sum(1 for mask in sprite_groups[int(spec["color"])] if canonical_key(mask) == key)
        wrong_max = 0
        for color, masks in sprite_groups.items():
            if int(color) == int(spec["color"]):
                continue
            wrong_max = max(wrong_max, sum(1 for mask in masks if canonical_key(mask) == key))
        if wrong_max > correct:
            strong_decoy = True
            break
    if not strong_decoy:
        return False

    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_shape_baseline(inp), expected):
        return False
    if np.array_equal(_pixel_count_baseline(inp), expected):
        return False

    colors_out = set(map(int, np.unique(expected)))
    if len(colors_out) < 4:
        return False
    if get_bg(inp) not in set(map(int, np.unique(inp))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    hole_specs_seed = _sample_hole_specs()
    for _ in range(30):
        frame_h = random.randint(*FRAME_H_RANGE)
        frame_w = random.randint(*FRAME_W_RANGE)
        placed = _place_holes(frame_h, frame_w, hole_specs_seed)
        if placed is None:
            continue

        grid_h = random.randint(max(GRID_MIN, frame_h + 8), GRID_MAX)
        grid_w = random.randint(max(GRID_MIN, frame_w + 8), GRID_MAX)
        bg = random.randint(0, 9)
        frame_color = random.choice([c for c in PALETTE if c != bg])

        available_colors = [c for c in PALETTE if c not in {bg, frame_color}]
        if len(available_colors) < len(placed):
            continue
        hole_colors = random.sample(available_colors, len(placed))
        for spec, color in zip(placed, hole_colors):
            spec["color"] = int(color)

        remaining_colors = [c for c in available_colors if c not in hole_colors]
        extra_count_max = min(2, len(remaining_colors))
        extra_count = random.choice([0, 1, extra_count_max]) if extra_count_max else 0
        extra_count = min(extra_count, extra_count_max)

        hole_areas = {int(spec["area"]) for spec in placed}
        extra_count_choices = [n for n in range(1, 9) if n not in hole_areas]
        if len(extra_count_choices) < extra_count:
            continue
        extra_counts = random.sample(extra_count_choices, extra_count)

        count_by_color = {int(spec["color"]): int(spec["area"]) for spec in placed}
        for color, count in zip(remaining_colors[:extra_count], extra_counts):
            count_by_color[int(color)] = int(count)

        sprite_groups = _build_sprite_groups(placed, count_by_color)
        if sprite_groups is None:
            continue

        input_crop, expected = _render_frame_crop(frame_h, frame_w, frame_color, bg, placed)
        grid = np.full((grid_h, grid_w), int(bg), dtype=int)

        top = random.randint(1, grid_h - frame_h - 1)
        left = random.randint(1, grid_w - frame_w - 1)
        paste(grid, input_crop, top, left)
        frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)

        if not _place_sprite_patches(grid, bg, frame_bbox, sprite_groups):
            continue
        if np.count_nonzero(grid == bg) <= max(np.count_nonzero(grid == frame_color), frame_h * frame_w):
            continue
        if not _semantic_checks(grid, expected, placed, count_by_color, sprite_groups):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
