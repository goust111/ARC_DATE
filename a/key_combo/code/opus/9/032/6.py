from __future__ import annotations

import itertools
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in _DIR4]


def _bbox(coords: list[Coord]) -> BBox:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_shape(b: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = b
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_distance_to_point(b: BBox, cell: Coord) -> int:
    r0, c0, r1, c1 = b
    r, c = cell
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _find_components(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    comps.sort(key=lambda comp: (int(comp["color"]), comp["bbox"], comp["coords"]))
    return comps


def _mask_in_bbox(coords: list[Coord], bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _mask_connected(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts:
        return False
    seen = {pts[0]}
    stack = [pts[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _draw_mask(grid: np.ndarray, bbox: BBox, mask: np.ndarray, color: int) -> None:
    r0, c0, _r1, _c1 = bbox
    rows, cols = np.nonzero(mask)
    grid[r0 + rows, c0 + cols] = int(color)


def _other_color(colors: list[int], color: int) -> int:
    a, b = colors
    return int(b if int(color) == int(a) else a)


def _pixel_distance_to_mask(mask: np.ndarray, rel_cell: Coord) -> int:
    rr, cc = rel_cell
    pixels = np.argwhere(mask)
    return min(abs(int(r) - int(rr)) + abs(int(c) - int(cc)) for r, c in pixels)


def _relative_marker_positions(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    out: list[dict] = []
    for rr in range(-2, h + 2):
        for cc in range(-2, w + 2):
            if 0 <= rr < h and 0 <= cc < w:
                continue
            bbox_dist = _bbox_distance_to_point((0, 0, h - 1, w - 1), (rr, cc))
            if bbox_dist not in (1, 2):
                continue
            pixel_dist = _pixel_distance_to_mask(mask, (rr, cc))
            if pixel_dist <= 1:
                continue
            out.append(
                {
                    "rel": (rr, cc),
                    "bbox_dist": bbox_dist,
                    "pixel_dist": pixel_dist,
                    "bbox_only": pixel_dist > bbox_dist,
                }
            )
    random.shuffle(out)
    return out


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg)
    sprites = [comp for comp in comps if int(comp["area"]) > 1]
    singles = [comp for comp in comps if int(comp["area"]) == 1]

    colors = sorted({int(v) for v in np.unique(g) if int(v) != int(bg)})
    assert len(colors) == 2, "expected exactly two foreground colors"
    sprite_colors = sorted({int(comp["color"]) for comp in sprites})
    assert sprite_colors == colors, "both foreground colors must form sprites"

    templates: dict[int, np.ndarray] = {}
    by_color: dict[int, list[dict]] = {color: [] for color in colors}
    for sprite in sprites:
        by_color[int(sprite["color"])].append(sprite)

    template_shape = None
    for color in colors:
        assert by_color[color], "each foreground color needs at least one sprite"
        masks = [_mask_in_bbox(sprite["coords"], sprite["bbox"]) for sprite in by_color[color]]
        first = masks[0]
        assert all(mask.shape == first.shape and np.array_equal(mask, first) for mask in masks), (
            "same-color sprites must share one template"
        )
        if template_shape is None:
            template_shape = first.shape
        else:
            assert first.shape == template_shape, "the two sprite templates must use the same bbox size"
        templates[color] = first

    assert not np.array_equal(templates[colors[0]], templates[colors[1]]), "the two templates must differ"

    sprites_sorted = []
    for sprite in sprites:
        sprite = dict(sprite)
        sprite["mask"] = _mask_in_bbox(sprite["coords"], sprite["bbox"])
        sprites_sorted.append(sprite)
    sprites_sorted.sort(key=lambda comp: (comp["bbox"], int(comp["color"])))

    associations: dict[Coord, BBox | None] = {}
    active_boxes: set[BBox] = set()
    for single in singles:
        cell = tuple(single["coords"][0])
        color = int(single["color"])
        nearby = [sprite for sprite in sprites_sorted if int(sprite["color"]) == color and _bbox_distance_to_point(sprite["bbox"], cell) <= 2]
        assert len(nearby) <= 1, "a singleton may not ambiguously activate two same-color sprites"
        if nearby:
            bbox = nearby[0]["bbox"]
            associations[cell] = bbox
            active_boxes.add(bbox)
        else:
            associations[cell] = None

    return {
        "bg": bg,
        "colors": colors,
        "templates": templates,
        "sprites": sprites_sorted,
        "singles": singles,
        "associations": associations,
        "active_boxes": active_boxes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the sprite copies activated by nearby same-color isolated pixels, and recolor those activators too.

    Input structure:
    - A black-background grid contains exactly two foreground colors.
    - Each foreground color forms several repeated 4-connected sprite copies with one shared template and one shared bounding-box size.
    - Some same-color foreground singletons are isolated 1-pixel components placed near one sprite bbox; others may sit farther away.

    Rule:
    - Split all monochromatic 4-connected components into sprites (area > 1) and isolated pixels (area = 1).
    - A sprite is activated iff at least one same-color isolated pixel lies at Manhattan distance at most 2 from that sprite's bounding box, with no ambiguity across same-color sprites.
    - Clear every activated sprite bbox, then simultaneously redraw the other color's template in that bbox using the other color; leave non-activated sprites unchanged.
    - Recolor each activating isolated pixel to that same swapped-to color, while isolated pixels not associated with any sprite keep their original color.

    Key insight:
    - The isolated pixels are selective switches: they do not move sprites globally, they only mark which repeated copies must participate in the synchronized template exchange.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    out = g.copy()
    bg = int(scene["bg"])

    for sprite in scene["sprites"]:
        if sprite["bbox"] in scene["active_boxes"]:
            r0, c0, r1, c1 = sprite["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for sprite in scene["sprites"]:
        if sprite["bbox"] not in scene["active_boxes"]:
            continue
        target_color = _other_color(scene["colors"], int(sprite["color"]))
        target_mask = scene["templates"][target_color]
        _draw_mask(out, sprite["bbox"], target_mask, target_color)

    for cell, bbox in scene["associations"].items():
        if bbox is None:
            continue
        color = next(int(sprite["color"]) for sprite in scene["sprites"] if sprite["bbox"] == bbox)
        out[cell] = _other_color(scene["colors"], color)

    return out


_GRID_MIN = 16
_GRID_MAX = 22
_BBOX_CHOICES = [(2, 3), (3, 2), (3, 3)]
_COLOR_POOL = list(range(1, 10))
_EDGE_GAP = 2
_SPRITE_GAP = 2
_COUNT_RANGE = (2, 4)
_GEN_RETRIES = 500
_TEMPLATE_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _rects_far_enough(a: BBox, b: BBox, gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _template_catalog(shape: tuple[int, int]) -> list[np.ndarray]:
    if shape in _TEMPLATE_CACHE:
        return [mask.copy() for mask in _TEMPLATE_CACHE[shape]]
    h, w = shape
    total = h * w
    masks: list[np.ndarray] = []
    for bits in itertools.product([0, 1], repeat=total):
        area = sum(bits)
        if not (4 <= area <= 8):
            continue
        mask = np.array(bits, dtype=bool).reshape(h, w)
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _mask_connected(mask):
            continue
        candidates = _relative_marker_positions(mask)
        if not candidates or not any(info["bbox_only"] for info in candidates):
            continue
        masks.append(mask)
    _TEMPLATE_CACHE[shape] = [mask.copy() for mask in masks]
    return [mask.copy() for mask in masks]


def _choose_templates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    masks = _template_catalog(shape)
    assert len(masks) >= 2
    for _ in range(200):
        first, second = random.sample(masks, 2)
        if np.count_nonzero(first != second) >= 2:
            return first.copy(), second.copy()
    raise RuntimeError("failed to choose distinct templates")


def _place_sprites(h: int, w: int, shape: tuple[int, int], colors: list[int], counts: dict[int, int]) -> list[dict] | None:
    bh, bw = shape
    positions = [
        (r0, c0)
        for r0 in range(_EDGE_GAP, h - bh - _EDGE_GAP + 1)
        for c0 in range(_EDGE_GAP, w - bw - _EDGE_GAP + 1)
    ]
    order = [color for color in colors for _ in range(counts[color])]

    for _ in range(120):
        random.shuffle(order)
        candidate_positions = positions[:]
        random.shuffle(candidate_positions)
        placed: list[dict] = []
        ok = True
        for color in order:
            options = []
            for r0, c0 in candidate_positions:
                bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
                if all(_rects_far_enough(bbox, item["bbox"], _SPRITE_GAP) for item in placed):
                    options.append((r0, c0, bbox))
            if not options:
                ok = False
                break
            r0, c0, bbox = random.choice(options)
            placed.append({"color": color, "bbox": bbox})
            candidate_positions = [(rr, cc) for rr, cc in candidate_positions if (rr, cc) != (r0, c0)]
        if ok:
            return placed
    return None


def _render_sprites(h: int, w: int, sprites: list[dict], templates: dict[int, np.ndarray]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for sprite in sprites:
        _draw_mask(grid, sprite["bbox"], templates[int(sprite["color"])], int(sprite["color"]))
    return grid


def _cell_is_visually_isolated(grid: np.ndarray, cell: Coord) -> bool:
    h, w = grid.shape
    r, c = cell
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != 0:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != 0:
            return False
    return True


def _activation_candidates(grid: np.ndarray, sprites: list[dict], sprite: dict, rel_infos: list[dict]) -> list[dict]:
    r0, c0, _r1, _c1 = sprite["bbox"]
    color = int(sprite["color"])
    out: list[dict] = []
    for info in rel_infos:
        rr, cc = info["rel"]
        cell = (r0 + rr, c0 + cc)
        if not _cell_is_visually_isolated(grid, cell):
            continue
        same_color_hits = [
            other
            for other in sprites
            if int(other["color"]) == color and _bbox_distance_to_point(other["bbox"], cell) <= 2
        ]
        if len(same_color_hits) != 1 or same_color_hits[0]["bbox"] != sprite["bbox"]:
            continue
        record = dict(info)
        record["cell"] = cell
        out.append(record)
    random.shuffle(out)
    return out


def _far_single_candidates(grid: np.ndarray, sprites: list[dict], color: int) -> list[Coord]:
    h, w = grid.shape
    out: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if not _cell_is_visually_isolated(grid, (r, c)):
                continue
            if any(_bbox_distance_to_point(sprite["bbox"], (r, c)) <= 2 for sprite in sprites if int(sprite["color"]) == color):
                continue
            if any(_bbox_distance_to_point(sprite["bbox"], (r, c)) <= 3 for sprite in sprites):
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _expected_output_from_metadata(inp: np.ndarray, scene: dict) -> np.ndarray:
    out = inp.copy()
    bg = int(scene["bg"])
    for sprite in scene["sprites"]:
        if sprite["bbox"] in scene["active_boxes"]:
            r0, c0, r1, c1 = sprite["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for sprite in scene["sprites"]:
        if sprite["bbox"] not in scene["active_boxes"]:
            continue
        target_color = _other_color(scene["colors"], int(sprite["color"]))
        _draw_mask(out, sprite["bbox"], scene["templates"][target_color], target_color)
    for cell, bbox in scene["associations"].items():
        if bbox is None:
            continue
        color = next(int(sprite["color"]) for sprite in scene["sprites"] if sprite["bbox"] == bbox)
        out[cell] = _other_color(scene["colors"], color)
    return out


def _build_metadata_scene(
    h: int,
    w: int,
    colors: list[int],
    templates: dict[int, np.ndarray],
    sprites: list[dict],
    markers: list[dict],
) -> dict:
    sprite_entries = []
    for sprite in sprites:
        color = int(sprite["color"])
        bbox = sprite["bbox"]
        rows, cols = np.nonzero(templates[color])
        coords = [(bbox[0] + int(r), bbox[1] + int(c)) for r, c in zip(rows, cols)]
        sprite_entries.append({"color": color, "coords": sorted(coords), "bbox": bbox, "area": len(coords), "mask": templates[color]})

    active_boxes = {marker["bbox"] for marker in markers if marker["assoc"]}
    associations = {marker["cell"]: (marker["bbox"] if marker["assoc"] else None) for marker in markers}
    singles = [{"color": int(marker["color"]), "coords": [marker["cell"]], "bbox": (*marker["cell"], *marker["cell"]), "area": 1} for marker in markers]
    sprite_entries.sort(key=lambda comp: (comp["bbox"], int(comp["color"])))

    return {
        "bg": 0,
        "colors": colors,
        "templates": {int(color): mask.copy() for color, mask in templates.items()},
        "sprites": sprite_entries,
        "singles": singles,
        "associations": associations,
        "active_boxes": active_boxes,
        "shape": (h, w),
    }


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    shape = random.choice(_BBOX_CHOICES)
    color_a, color_b = random.sample(_COLOR_POOL, 2)
    colors = sorted([color_a, color_b])
    templates = {colors[0]: None, colors[1]: None}
    first, second = _choose_templates(shape)
    templates[colors[0]] = first
    templates[colors[1]] = second

    counts = {color: random.randint(*_COUNT_RANGE) for color in colors}
    placed = _place_sprites(h, w, shape, colors, counts)
    if placed is None:
        return None

    base_grid = _render_sprites(h, w, placed, templates)
    rel_candidates = {color: _relative_marker_positions(templates[color]) for color in colors}

    sprite_infos = []
    for idx, sprite in enumerate(sorted(placed, key=lambda item: (item["bbox"], int(item["color"])))):
        color = int(sprite["color"])
        candidates = _activation_candidates(base_grid, placed, sprite, rel_candidates[color])
        sprite_infos.append({"id": idx, "color": color, "bbox": sprite["bbox"], "candidates": candidates})

    by_color: dict[int, list[dict]] = {color: [] for color in colors}
    for info in sprite_infos:
        by_color[int(info["color"])].append(info)

    if any(len(items) < 2 for items in by_color.values()):
        return None
    if any(not any(item["candidates"] for item in items) for items in by_color.values()):
        return None

    bbox_only_options = [item for item in sprite_infos if any(cand["bbox_only"] for cand in item["candidates"])]
    if not bbox_only_options:
        return None
    special = random.choice(bbox_only_options)

    active_boxes: set[BBox] = set()
    for color in colors:
        items = by_color[color]
        available = [item for item in items if item["candidates"]]
        max_active = min(len(items) - 1, len(available))
        if max_active < 1:
            return None
        need = random.randint(1, max_active)
        if int(color) == int(special["color"]):
            others = [item for item in available if item["bbox"] != special["bbox"]]
            if need == 1:
                chosen = [special]
            else:
                if len(others) < need - 1:
                    return None
                chosen = [special] + random.sample(others, need - 1)
        else:
            chosen = random.sample(available, need)
        active_boxes.update(item["bbox"] for item in chosen)

    grid = base_grid.copy()
    markers: list[dict] = []
    active_order = [item for item in sprite_infos if item["bbox"] in active_boxes]
    random.shuffle(active_order)
    for item in active_order:
        bbox_only_first = item["bbox"] == special["bbox"]
        target_count = 1 + int(random.random() < 0.35)
        for mark_index in range(target_count):
            candidates = _activation_candidates(grid, placed, item, rel_candidates[int(item["color"])])
            if bbox_only_first and mark_index == 0:
                candidates = [cand for cand in candidates if cand["bbox_only"]]
            if not candidates:
                if mark_index == 0:
                    return None
                break
            chosen = random.choice(candidates)
            cell = chosen["cell"]
            grid[cell] = int(item["color"])
            markers.append({"cell": cell, "color": int(item["color"]), "bbox": item["bbox"], "assoc": True, "bbox_only": bool(chosen["bbox_only"])})

    if not any(bool(marker["bbox_only"]) for marker in markers if marker["assoc"]):
        return None

    far_colors = [random.choice(colors)]
    if random.random() < 0.35:
        far_colors.append(_other_color(colors, far_colors[0]))
    for color in far_colors:
        candidates = _far_single_candidates(grid, placed, color)
        if not candidates:
            return None
        cell = candidates[0]
        grid[cell] = int(color)
        markers.append({"cell": cell, "color": int(color), "bbox": None, "assoc": False, "bbox_only": False})

    metadata = _build_metadata_scene(h, w, colors, templates, placed, markers)
    expected = _expected_output_from_metadata(grid, metadata)

    parsed = _analyze_scene(grid)
    if parsed["active_boxes"] != metadata["active_boxes"]:
        return None
    if parsed["associations"] != metadata["associations"]:
        return None
    if any(np.array_equal(parsed["templates"][color], metadata["templates"][color]) is False for color in colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    global_swap = grid.copy()
    for sprite in parsed["sprites"]:
        r0, c0, r1, c1 = sprite["bbox"]
        global_swap[r0 : r1 + 1, c0 : c1 + 1] = 0
    for sprite in parsed["sprites"]:
        target_color = _other_color(parsed["colors"], int(sprite["color"]))
        _draw_mask(global_swap, sprite["bbox"], parsed["templates"][target_color], target_color)

    if np.array_equal(expected, grid) or np.array_equal(expected, global_swap):
        return None

    for color in colors:
        items = [sprite for sprite in parsed["sprites"] if int(sprite["color"]) == color]
        active_count = sum(sprite["bbox"] in parsed["active_boxes"] for sprite in items)
        if active_count < 1 or active_count >= len(items):
            return None

    if not any(bbox is None for bbox in parsed["associations"].values()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-activation sprite-swap task")
