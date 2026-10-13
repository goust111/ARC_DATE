from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PLACEHOLDER = 5
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _component_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    box = _bbox(coords)
    patch = _crop(grid, box)
    mask = _component_mask(coords, box)
    colors = sorted({int(grid[r, c]) for r, c in coords})
    return {
        "coords": coords,
        "bbox": box,
        "mask": mask,
        "patch": patch,
        "colors": colors,
        "area": len(coords),
    }


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(out.shape[0], r0 + h)
    cc1 = min(out.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    sub = patch[pr0:pr1, pc0:pc1]
    mask = sub != BG
    out[rr0:rr1, cc0:cc1][mask] = sub[mask]


def _tile_positions(
    start_r: int, start_c: int, shape: tuple[int, int], direction: str, canvas_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    H, W = canvas_shape
    dr, dc = DIR_TO_VEC[direction]
    stride = h if direction in ("up", "down") else w
    positions = []
    step = 0
    while True:
        rr = start_r + dr * stride * step
        cc = start_c + dc * stride * step
        if rr >= H or cc >= W or rr + h <= 0 or cc + w <= 0:
            break
        positions.append((rr, cc))
        step += 1
    return positions


def _rect_from_anchor(r0: int, c0: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _free_copies_available(
    start_r: int, start_c: int, shape: tuple[int, int], direction: str, canvas_shape: tuple[int, int]
) -> int:
    h, w = shape
    H, W = canvas_shape
    if direction == "up":
        return start_r // h
    if direction == "down":
        return (H - (start_r + h)) // h
    if direction == "left":
        return start_c // w
    return (W - (start_c + w)) // w


def _indicator_options(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if direction == "up":
        return [(-1, c) for c in range(w) if mask[0, c]]
    if direction == "down":
        return [(h, c) for c in range(w) if mask[h - 1, c]]
    if direction == "left":
        return [(r, -1) for r in range(h) if mask[r, 0]]
    return [(r, w) for r in range(h) if mask[r, w - 1]]


def _random_connected_mask() -> np.ndarray:
    for _ in range(400):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(10, h * w)
        min_area = max(6, max(h, w) + 1)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        box = _bbox(coords)
        cropped = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if 6 <= int(cropped.sum()) <= 10 and min(cropped.shape) >= 2:
            return cropped
    raise RuntimeError("failed to generate a connected sprite mask")


def _partition_positive(total: int, parts: int, upper_exclusive: int) -> list[int] | None:
    if parts == 1:
        return [total] if 0 < total < upper_exclusive else None
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values if all(0 < v < upper_exclusive for v in values) else None


def _color_sprite(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    area = int(mask.sum())
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    color_count = random.choice([2, 3])
    colors = random.sample(palette, color_count)
    main_color = colors[0]

    for _ in range(200):
        main_count = random.randint(area // 2 + 1, area - (color_count - 1))
        rest = area - main_count
        parts = _partition_positive(rest, color_count - 1, main_count)
        if parts is None:
            continue
        counts = [main_count, *parts]
        sprite = np.zeros(mask.shape, dtype=int)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        random.shuffle(coords)
        idx = 0
        for color, count in zip(colors, counts):
            for _ in range(count):
                r, c = coords[idx]
                sprite[r, c] = color
                idx += 1
        return sprite, main_color, colors
    raise RuntimeError("failed to assign sprite colors")


def _build_scene_output(
    canvas_shape: tuple[int, int],
    sprite: np.ndarray,
    main_color: int,
    placements: list[dict],
) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    h, w = sprite.shape
    for placement in placements:
        r0, c0 = placement["anchor"]
        indicator_color = placement["indicator_color"]
        _paste(out, sprite, r0, c0)
        recolored = sprite.copy()
        recolored[recolored == main_color] = indicator_color
        for rr, cc in _tile_positions(r0, c0, (h, w), placement["direction"], canvas_shape)[1:]:
            _paste(out, recolored, rr, cc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the multicolor source onto gray placeholders, then tile recolored copies outward by each touching indicator.

    Input structure:
    - The grid has a black background, one 4-connected multicolor source sprite, and 2-3 gray placeholder objects.
    - Every placeholder has the same normalized mask as the source sprite and exactly one touching non-gray indicator pixel.
    - Indicator pixels are single cells outside their placeholder bboxes, and each one touches from a cardinal side.

    Rule:
    - Detect the unique multicolor source component and every gray component whose bbox-normalized mask matches it.
    - Replace each matching gray placeholder with the source sprite colors in place.
    - For each stamped placeholder, read the touching indicator: its side gives the tiling direction, and its color replaces the source sprite's unique majority color in every translated copy.
    - Repeat copies along that direction with stride equal to the sprite height or width, then erase the original source sprite and all indicators.

    Key insight:
    - A single touching indicator pixel carries two visible cues at once: where it touches encodes motion direction, while its color encodes the recoloring of the source sprite's dominant color.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    gray_components = [_component_info(g, comp) for comp in _components_from_mask(g == PLACEHOLDER)]
    colored_components = [
        _component_info(g, comp) for comp in _components_from_mask((g != BG) & (g != PLACEHOLDER))
    ]

    source_candidates = [comp for comp in colored_components if comp["area"] > 1 and len(comp["colors"]) >= 2]
    assert len(source_candidates) == 1, "expected a unique multicolor source sprite"
    source = source_candidates[0]
    indicators = [comp for comp in colored_components if comp is not source]
    assert indicators and all(comp["area"] == 1 for comp in indicators), "expected single-pixel indicators"

    source_mask = source["mask"]
    source_patch = source["patch"]
    source_bbox = source["bbox"]
    source_shape = source_patch.shape

    placeholders = [
        comp
        for comp in gray_components
        if comp["mask"].shape == source_mask.shape and np.array_equal(comp["mask"], source_mask)
    ]
    assert 2 <= len(placeholders) <= 3, "expected 2-3 matching placeholders"

    counts = Counter(int(v) for v in source_patch[source_mask])
    main_color, main_count = counts.most_common(1)[0]
    assert sum(count == main_count for count in counts.values()) == 1, "source main color must be unique"

    placeholder_sets = [{tuple(cell) for cell in comp["coords"]} for comp in placeholders]
    assignments = []
    for comp, cell_set in zip(placeholders, placeholder_sets):
        touching = []
        for ind in indicators:
            ir, ic = ind["coords"][0]
            if any((nr, nc) in cell_set for nr, nc in _neighbors4(ir, ic)):
                touching.append(ind)
        assert len(touching) == 1, "each placeholder must have exactly one touching indicator"
        indicator = touching[0]
        ir, ic = indicator["coords"][0]
        r0, c0, r1, c1 = comp["bbox"]
        if ir < r0:
            direction = "up"
        elif ir > r1:
            direction = "down"
        elif ic < c0:
            direction = "left"
        else:
            direction = "right"
        assignments.append((comp, indicator, direction))

    out = g.copy()
    for r, c in source["coords"]:
        out[r, c] = BG
    for indicator in indicators:
        ir, ic = indicator["coords"][0]
        out[ir, ic] = BG

    sprite_h, sprite_w = source_shape
    for placeholder, indicator, direction in assignments:
        pr0, pc0, pr1, pc1 = placeholder["bbox"]
        target = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        target[source_mask] = source_patch[source_mask]

        indicator_color = int(indicator["colors"][0])
        recolored = source_patch.copy()
        recolored[recolored == main_color] = indicator_color

        for rr, cc in _tile_positions(pr0, pc0, (sprite_h, sprite_w), direction, (H, W))[1:]:
            _paste(out, recolored, rr, cc)

    return out


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(18, 26)
    W = random.randint(18, 26)

    mask = _random_connected_mask()
    sprite, main_color, sprite_colors = _color_sprite(mask)
    h, w = sprite.shape

    placeholder_count = random.choice([2, 3])
    directions = random.sample(["up", "down", "left", "right"], placeholder_count)
    indicator_pool = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in sprite_colors]
    random.shuffle(indicator_pool)
    indicator_colors = indicator_pool[:placeholder_count]

    min_copies = 1 if random.random() < 0.35 else 2
    candidates_by_direction: dict[str, list[dict]] = {direction: [] for direction in directions}
    for direction in directions:
        side_options = _indicator_options(mask, direction)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if _free_copies_available(r0, c0, (h, w), direction, (H, W)) < min_copies:
                    continue
                options = side_options[:]
                random.shuffle(options)
                for ir_rel, ic_rel in options:
                    ir = r0 + ir_rel
                    ic = c0 + ic_rel
                    if not (0 <= ir < H and 0 <= ic < W):
                        continue
                    tile_boxes = [
                        _rect_from_anchor(rr, cc, (h, w))
                        for rr, cc in _tile_positions(r0, c0, (h, w), direction, (H, W))
                    ]
                    candidates_by_direction[direction].append(
                        {
                            "anchor": (r0, c0),
                            "indicator": (ir, ic),
                            "tile_boxes": tile_boxes,
                            "bbox": _rect_from_anchor(r0, c0, (h, w)),
                        }
                    )
                    break
        random.shuffle(candidates_by_direction[direction])
        if not candidates_by_direction[direction]:
            return None

    placements: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    used_input_rects: list[tuple[int, int, int, int]] = []
    used_indicator_cells: set[tuple[int, int]] = set()

    ordered_directions = sorted(directions, key=lambda d: len(candidates_by_direction[d]))
    for direction in ordered_directions:
        found = None
        for candidate in candidates_by_direction[direction]:
            bbox = candidate["bbox"]
            indicator_rect = (*candidate["indicator"], *candidate["indicator"])
            if candidate["indicator"] in used_indicator_cells:
                continue
            if any(_rects_overlap(bbox, rect, margin=1) for rect in used_input_rects):
                continue
            if any(_rects_overlap(indicator_rect, rect, margin=1) for rect in used_input_rects):
                continue
            if any(_rects_overlap(box, other_box, margin=0) for box in candidate["tile_boxes"] for other_box in used_boxes):
                continue
            found = candidate
            break
        if found is None:
            return None
        color = indicator_colors[len(placements)]
        placements.append(
            {
                "anchor": found["anchor"],
                "indicator": found["indicator"],
                "direction": direction,
                "indicator_color": color,
            }
        )
        used_boxes.extend(found["tile_boxes"])
        used_input_rects.append(found["bbox"])
        used_input_rects.append((*found["indicator"], *found["indicator"]))
        used_indicator_cells.add(found["indicator"])

    source_candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            box = _rect_from_anchor(r0, c0, (h, w))
            if any(_rects_overlap(box, rect, margin=1) for rect in used_input_rects):
                continue
            if any(_rects_overlap(box, rect, margin=1) for rect in used_boxes):
                continue
            source_candidates.append((r0, c0))
    if not source_candidates:
        return None
    source_anchor = random.choice(source_candidates)

    grid = np.zeros((H, W), dtype=int)
    _paste(grid, sprite, source_anchor[0], source_anchor[1])
    placeholder_patch = np.where(mask, PLACEHOLDER, BG)
    for placement in placements:
        r0, c0 = placement["anchor"]
        ir, ic = placement["indicator"]
        _paste(grid, placeholder_patch, r0, c0)
        grid[ir, ic] = placement["indicator_color"]

    expected = _build_scene_output((H, W), sprite, main_color, placements)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        attempt = _generate_attempt()
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
