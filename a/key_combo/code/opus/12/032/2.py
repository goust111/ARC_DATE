from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
TEMPLATE_SIZE = 5
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(
    grid: np.ndarray,
    *,
    exclude: set[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    blocked = set() if exclude is None else {int(v) for v in exclude}
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    neigh = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color in blocked or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if not _in_bounds((H, W), nr, nc) or (nr, nc) in seen:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return out


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _dihedral_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.copy())
    return variants


def _trimmed_coords(mask: np.ndarray, layer: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        edge_dist = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
        if edge_dist >= layer:
            coords.append((int(r), int(c)))
    return coords


def _render_events(
    shape: tuple[int, int],
    templates: dict[int, np.ndarray],
    seeds: list[dict],
) -> list[dict]:
    events: list[dict] = []
    for seed in seeds:
        color = int(seed["color"])
        mask = templates[color]
        h, w = mask.shape
        center_r = h // 2
        center_c = w // 2
        stride = max(h, w)
        k = 0
        while True:
            layer_coords = _trimmed_coords(mask, k)
            if not layer_coords:
                break
            rr = int(seed["r"]) + k * int(seed["dr"]) * stride
            cc = int(seed["c"]) + k * int(seed["dc"]) * stride
            if not _in_bounds(shape, rr, cc):
                break
            events.append(
                {
                    "k": k,
                    "seed_r": int(seed["r"]),
                    "seed_c": int(seed["c"]),
                    "color": color,
                    "top": rr - center_r,
                    "left": cc - center_c,
                    "coords": layer_coords,
                }
            )
            k += 1
    events.sort(key=lambda item: (item["k"], item["seed_r"], item["seed_c"]))
    return events


def _render_output_from_semantics(
    grid: np.ndarray,
    templates: dict[int, np.ndarray],
    seeds: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == RED] = BG
    for event in _render_events(out.shape, templates, seeds):
        top = int(event["top"])
        left = int(event["left"])
        color = int(event["color"])
        for lr, lc in event["coords"]:
            rr = top + lr
            cc = left + lc
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = color
    return out


def _extract_semantics(grid: np.ndarray) -> tuple[dict[int, np.ndarray], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochrome_components(g, exclude={bg, RED}, connectivity=4)
    assert comps, "expected templates and seeds"

    template_comps = [comp for comp in comps if comp["area"] > 1]
    seed_comps = [comp for comp in comps if comp["area"] == 1]
    assert 2 <= len(template_comps) <= 3, "expected 2-3 templates"
    assert len(seed_comps) >= len(template_comps), "expected each template to be referenced"

    templates: dict[int, np.ndarray] = {}
    for comp in template_comps:
        color = int(comp["color"])
        assert color not in templates, "template colors must be unique"
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (TEMPLATE_SIZE, TEMPLATE_SIZE), "templates must be 5x5"
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        for r, c in comp["coords"]:
            mask[r - r0, c - c0] = True
        templates[color] = mask

    seeds: list[dict] = []
    used_colors = Counter()
    for comp in seed_comps:
        color = int(comp["color"])
        assert color in templates, "every seed must match a template color"
        r, c = comp["coords"][0]
        red_diagonals = []
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds(g.shape, nr, nc):
                continue
            if int(g[nr, nc]) == RED and abs(nr - r) == 1 and abs(nc - c) == 1:
                red_diagonals.append((nr - r, nc - c))
        assert len(red_diagonals) == 1, "each seed needs exactly one diagonal red controller"
        dr, dc = red_diagonals[0]
        seeds.append({"r": int(r), "c": int(c), "color": color, "dr": int(dr), "dc": int(dc)})
        used_colors[color] += 1

    assert set(used_colors) == set(templates), "every template must be referenced by at least one seed"
    return templates, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each color-selected template along a red-controlled diagonal ray while peeling one outer layer per step.

    Input structure:
    - The background is black.
    - The upper-left area contains 2-3 distinct 5x5 monochrome templates, each a 4-connected non-red component.
    - The lower area contains single-pixel seeds whose colors match template colors.
    - Every seed has exactly one red diagonal neighbor, and that red pixel is only a direction controller.

    Rule:
    - Read each template by color and keep its 5x5 local mask.
    - For each seed, use the relative diagonal position of its adjacent red pixel to choose one of the four diagonal ray directions.
    - Stamp the full template centered on the seed at step k=0, then keep moving by one template span along that diagonal; at step k remove k outer bbox layers from the template before stamping.
    - Stop a seed's ray when the peeled template becomes empty or the next stamp center leaves the grid.
    - Remove all red controller pixels from the output, and resolve overlaps by drawing lower k first and, within the same k, smaller seed row first.

    Key insight:
    - Color chooses which template is copied, the red diagonal offset chooses where the ray goes, and progressive shell peeling changes the copied 2D shape at every step.
    """
    templates, seeds = _extract_semantics(np.asarray(grid, dtype=int))
    return _render_output_from_semantics(np.asarray(grid, dtype=int), templates, seeds)


BASE_TEMPLATE_MASKS = [
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
]


def _sample_templates(num_templates: int) -> list[np.ndarray]:
    indices = random.sample(range(len(BASE_TEMPLATE_MASKS)), num_templates)
    templates: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for index in indices:
        variant = random.choice(_dihedral_transforms(BASE_TEMPLATE_MASKS[index]))
        coords = [(int(r), int(c)) for r, c in np.argwhere(variant)]
        key = _normalize_coords(coords)
        if key in seen:
            return []
        seen.add(key)
        templates.append(variant)
    return templates


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}


def _seed_footprint(seed: dict, mask: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for event in _render_events(shape, {int(seed["color"]): mask}, [seed]):
        for lr, lc in event["coords"]:
            rr = int(event["top"]) + lr
            cc = int(event["left"]) + lc
            if _in_bounds(shape, rr, cc):
                cells.add((rr, cc))
    return cells


def _candidate_seed_positions(
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = shape
    dr, dc = direction
    rows = range(H // 2, H - 2)
    cols = range(1, W - 1)
    out: list[tuple[int, int]] = []
    for r in rows:
        if dr == -1 and r < 10:
            continue
        if dr == 1 and r > H - 11:
            continue
        for c in cols:
            if dc == -1 and c < 10:
                continue
            if dc == 1 and c > W - 11:
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _overlap_budget(existing: set[tuple[int, int]], candidate: set[tuple[int, int]]) -> int:
    return len(existing & candidate)


def _build_input_from_plan(
    shape: tuple[int, int],
    colored_templates: list[tuple[int, np.ndarray, int, int]],
    seeds: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, mask, top, left in colored_templates:
        for r, c in np.argwhere(mask):
            grid[top + int(r), left + int(c)] = int(color)
    for seed in seeds:
        grid[int(seed["r"]), int(seed["c"])] = int(seed["color"])
        grid[int(seed["r"]) + int(seed["dr"]), int(seed["c"]) + int(seed["dc"])] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(24, 30)
    W = random.randint(24, 30)
    template_count = random.choice([2, 3])
    seed_count = random.randint(template_count, 4)
    allow_overlap = random.random() < 0.3

    template_masks = _sample_templates(template_count)
    if len(template_masks) != template_count:
        return None
    colors = random.sample(PALETTE, template_count)

    left = random.randint(0, 1)
    top = random.randint(0, 1)
    colored_templates: list[tuple[int, np.ndarray, int, int]] = []
    template_cells: set[tuple[int, int]] = set()
    for color, mask in zip(colors, template_masks):
        if left + TEMPLATE_SIZE > W:
            return None
        cells = _mask_cells(mask, top, left)
        if cells & template_cells:
            return None
        template_cells |= cells
        colored_templates.append((color, mask, top, left))
        left += TEMPLATE_SIZE + random.randint(1, 2)
    if left - 1 >= W:
        return None

    assignments = colors.copy()
    while len(assignments) < seed_count:
        assignments.append(random.choice(colors))
    random.shuffle(assignments)

    directions = random.sample(DIRECTIONS, k=min(len(DIRECTIONS), seed_count))
    while len(directions) < seed_count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)
    if seed_count > 1 and len(set(directions)) < 2:
        return None

    seeds: list[dict] = []
    occupied = set(template_cells)
    stamp_union: set[tuple[int, int]] = set(template_cells)
    color_to_mask = {color: mask for color, mask, _top, _left in colored_templates}

    for idx, color in enumerate(assignments):
        direction = directions[idx]
        placed = False
        for r, c in _candidate_seed_positions((H, W), direction):
            dr, dc = direction
            red = (r + dr, c + dc)
            if red in occupied or (r, c) in occupied:
                continue
            if any(abs(r - other["r"]) <= 1 and abs(c - other["c"]) <= 1 for other in seeds):
                continue
            seed = {"r": r, "c": c, "color": int(color), "dr": dr, "dc": dc}
            footprint = _seed_footprint(seed, color_to_mask[int(color)], (H, W))
            if footprint & template_cells:
                continue
            overlap = _overlap_budget(stamp_union, footprint)
            if not allow_overlap and overlap > 0:
                continue
            if allow_overlap and overlap > 4:
                continue
            if len(footprint) < 8:
                continue
            seeds.append(seed)
            occupied.add((r, c))
            occupied.add(red)
            stamp_union |= footprint
            placed = True
            break
        if not placed:
            return None

    grid = _build_input_from_plan((H, W), colored_templates, seeds)
    templates = {color: mask for color, mask, _top, _left in colored_templates}
    expected = _render_output_from_semantics(grid, templates, seeds)

    try:
        parsed_templates, parsed_seeds = _extract_semantics(grid)
    except AssertionError:
        return None
    if set(parsed_templates) != set(templates):
        return None
    for color in templates:
        if not np.array_equal(parsed_templates[color], templates[color]):
            return None
    if not np.array_equal(main(grid), expected):
        return None
    if seed_count > 1 and len({(seed["dr"], seed["dc"]) for seed in seeds}) < 2:
        return None
    if any(int(grid[r, c]) != RED for seed in seeds for r, c in [(seed["r"] + seed["dr"], seed["c"] + seed["dc"])]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
