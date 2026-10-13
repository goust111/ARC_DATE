from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SPRITE_COLORS = (BLUE, GREEN)
LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2))
SPRITE_BOX_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask.any() and mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _connected_mask(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _template_candidates(shape: tuple[int, int]) -> list[np.ndarray]:
    cache = getattr(_template_candidates, "_cache", {})
    if shape in cache:
        return cache[shape]

    h, w = shape
    total = h * w
    out: list[np.ndarray] = []
    for area in range(3, min(6, total - 1) + 1):
        for combo in itertools.combinations(range(total), area):
            mask = np.zeros((h, w), dtype=bool)
            for idx in combo:
                mask[idx // w, idx % w] = True
            if _mask_spans_bbox(mask) and _connected_mask(mask):
                out.append(mask)

    cache[shape] = out
    setattr(_template_candidates, "_cache", cache)
    return out


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}, "input must only contain black, blue, red, and green"

    regions = _component_coords(g != RED)
    assert len(regions) >= 4, "expected at least four regions"

    parsed_regions: list[dict] = []
    templates_by_color: dict[int, list[np.ndarray]] = {BLUE: [], GREEN: []}

    for coords in regions:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        region_area = int(coords.shape[0])
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        assert region_area == region_h * region_w, "red walls must partition the board into solid rectangles"
        assert not np.any(sub == RED), "regions may not contain red pixels"

        colors = {int(v) for v in np.unique(sub)} - {BLACK}
        assert colors <= set(SPRITE_COLORS), "regions may only contain one blue or green sprite"
        assert len(colors) == 1, "each region must contain exactly one sprite color"
        color = next(iter(colors))

        sprite_mask = sub == color
        assert 3 <= int(sprite_mask.sum()) <= 6, "sprite area must be between 3 and 6"
        sprite_comps = _component_coords(sprite_mask)
        assert len(sprite_comps) == 1, "each region must contain exactly one 4-connected sprite"

        local_coords = sprite_comps[0]
        assert np.all(local_coords[:, 0] > 0) and np.all(local_coords[:, 0] < region_h - 1), "sprite must not touch red walls"
        assert np.all(local_coords[:, 1] > 0) and np.all(local_coords[:, 1] < region_w - 1), "sprite must not touch red walls"

        local_box = _bbox(local_coords)
        sprite_template = _crop_bbox(sprite_mask, local_box)
        templates_by_color[color].append(sprite_template)

        global_box = (
            r0 + local_box[0],
            c0 + local_box[1],
            r0 + local_box[2],
            c0 + local_box[3],
        )
        parsed_regions.append(
            {
                "bbox": box,
                "area": region_area,
                "color": color,
                "sprite_bbox": global_box,
                "sprite_mask": sprite_template,
            }
        )

    assert len(templates_by_color[BLUE]) >= 2 and len(templates_by_color[GREEN]) >= 2, "need at least two sprites of each color"

    templates: dict[int, np.ndarray] = {}
    for color in SPRITE_COLORS:
        masks = templates_by_color[color]
        shape = masks[0].shape
        assert all(mask.shape == shape for mask in masks), "same-color sprites must share one bbox size"
        first = masks[0]
        assert all(np.array_equal(mask, first) for mask in masks), "same-color sprites must share one template"
        templates[color] = first

    assert templates[BLUE].shape == templates[GREEN].shape, "blue and green templates must have the same bbox size"
    assert not np.array_equal(templates[BLUE], templates[GREEN]), "the two templates must differ"

    areas = [region["area"] for region in parsed_regions]
    assert len(set(areas)) == len(areas), "region areas must be unique"

    return {"grid": g, "regions": parsed_regions, "templates": templates}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the blue and green sprite templates inside red-walled regions, then yellow-fill odd-ranked region backgrounds.

    Input structure:
    - The grid has black background plus red divider walls that partition the board into solid rectangular non-red regions.
    - Each region contains exactly one 4-connected blue or green sprite, and sprites do not touch the red walls.
    - All blue sprites share one template, all green sprites share another template, and the two template bounding boxes are the same size.

    Rule:
    - Treat red as an immovable wall and compute every rectangular region separated by the red lines.
    - Rank the regions by non-red-region area in ascending order, using unique areas so the ranking is unambiguous.
    - Simultaneously swap sprite templates: clear every original sprite to black, then redraw a blue sprite with the green template and a green sprite with the blue template in the original sprite bbox.
    - After the swap, fill every remaining black background cell in odd-ranked regions with yellow, while even-ranked regions stay black and all red wall pixels remain unchanged.

    Key insight:
    - The red wall layout drives both effects at once: it determines which pixels belong to each sprite's home region and it also determines the area ranking that controls which region backgrounds become yellow.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    g = parsed["grid"]
    regions = parsed["regions"]
    templates = parsed["templates"]
    out = g.copy()

    for region in regions:
        r0, c0, r1, c1 = region["sprite_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][region["sprite_mask"]] = BLACK

    other_color = {BLUE: GREEN, GREEN: BLUE}
    for region in regions:
        target_color = other_color[region["color"]]
        target_mask = templates[target_color]
        r0, c0, r1, c1 = region["sprite_bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert target_mask.shape == (h, w), "template swap must preserve bbox center"
        patch = np.zeros((h, w), dtype=int)
        patch[target_mask] = target_color
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    ranked = sorted(regions, key=lambda region: region["area"])
    for rank, region in enumerate(ranked):
        if rank % 2 == 1:
            r0, c0, r1, c1 = region["bbox"]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[patch == BLACK] = YELLOW

    return out


def _sample_partition_sizes(n_parts: int, minimum: int, maximum: int) -> list[int]:
    values = list(range(minimum, maximum + 1))
    return random.sample(values, n_parts)


def _region_boxes(row_heights: list[int], col_widths: list[int]) -> list[tuple[int, int, int, int]]:
    row_starts = []
    cur = 0
    for idx, height in enumerate(row_heights):
        row_starts.append(cur)
        cur += height
        if idx < len(row_heights) - 1:
            cur += 1

    col_starts = []
    cur = 0
    for idx, width in enumerate(col_widths):
        col_starts.append(cur)
        cur += width
        if idx < len(col_widths) - 1:
            cur += 1

    boxes = []
    for r0, height in zip(row_starts, row_heights):
        for c0, width in zip(col_starts, col_widths):
            boxes.append((r0, c0, r0 + height - 1, c0 + width - 1))
    return boxes


def _layout_is_nontrivial(row_heights: list[int], col_widths: list[int], n_rows: int, n_cols: int) -> bool:
    areas = [h * w for h in row_heights for w in col_widths]
    if len(set(areas)) != len(areas):
        return False

    rank_order = sorted(range(len(areas)), key=lambda idx: areas[idx])
    row_major = list(range(len(areas)))
    if rank_order == row_major or rank_order == row_major[::-1]:
        return False

    odd_cells = [divmod(idx, n_cols) for rank, idx in enumerate(rank_order) if rank % 2 == 1]
    if len({r for r, _ in odd_cells}) < min(2, n_rows):
        return False
    if len({c for _, c in odd_cells}) < min(2, n_cols):
        return False
    return True


def _choose_partition(shape: tuple[int, int], layout: tuple[int, int]):
    box_h, box_w = shape
    n_rows, n_cols = layout
    min_h = box_h + 2
    min_w = box_w + 2
    max_h = min(9, min_h + 4)
    max_w = min(9, min_w + 4)
    if max_h - min_h + 1 < n_rows or max_w - min_w + 1 < n_cols:
        return None

    for _ in range(200):
        row_heights = _sample_partition_sizes(n_rows, min_h, max_h)
        col_widths = _sample_partition_sizes(n_cols, min_w, max_w)
        grid_h = sum(row_heights) + (n_rows - 1)
        grid_w = sum(col_widths) + (n_cols - 1)
        if grid_h > 30 or grid_w > 30:
            continue
        if not _layout_is_nontrivial(row_heights, col_widths, n_rows, n_cols):
            continue
        return row_heights, col_widths
    return None


def _choose_templates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = _template_candidates(shape)
    by_area: dict[int, list[np.ndarray]] = {}
    for mask in candidates:
        by_area.setdefault(int(mask.sum()), []).append(mask)

    valid_areas = [area for area, masks in by_area.items() if len(masks) >= 2]
    if not valid_areas:
        return None

    area = random.choice(valid_areas)
    a, b = random.sample(by_area[area], 2)
    return a.copy(), b.copy()


def _assign_colors_by_region(n_regions: int, area_rank_order: list[int]) -> list[int] | None:
    for _ in range(200):
        blue_count = random.randint(2, n_regions - 2)
        colors = [BLUE] * blue_count + [GREEN] * (n_regions - blue_count)
        random.shuffle(colors)

        parity_sets = {0: set(), 1: set()}
        for rank, idx in enumerate(area_rank_order):
            parity_sets[rank % 2].add(colors[idx])
        if parity_sets[0] != set(SPRITE_COLORS) or parity_sets[1] != set(SPRITE_COLORS):
            continue

        return colors
    return None


def _draw_walls(grid: np.ndarray, row_heights: list[int], col_widths: list[int]) -> None:
    cur = row_heights[0]
    for height in row_heights[1:]:
        grid[cur, :] = RED
        cur += 1 + height

    cur = col_widths[0]
    for width in col_widths[1:]:
        grid[:, cur] = RED
        cur += 1 + width


def _paste_sprite(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _build_input(
    row_heights: list[int],
    col_widths: list[int],
    blue_template: np.ndarray,
    green_template: np.ndarray,
    colors_by_region: list[int],
) -> np.ndarray | None:
    grid_h = sum(row_heights) + len(row_heights) - 1
    grid_w = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_walls(grid, row_heights, col_widths)

    boxes = _region_boxes(row_heights, col_widths)
    templates = {BLUE: blue_template, GREEN: green_template}

    for box, color in zip(boxes, colors_by_region):
        r0, c0, r1, c1 = box
        mask = templates[color]
        h, w = mask.shape
        top_min = r0 + 1
        left_min = c0 + 1
        top_max = r1 - h
        left_max = c1 - w
        if top_min > top_max or left_min > left_max:
            return None
        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)
        _paste_sprite(grid, mask, color, top, left)

    return grid


def _validate_generated_instance(inp: np.ndarray) -> bool:
    parsed = _parse_input(inp)
    out = main(inp)
    regions = parsed["regions"]
    templates = parsed["templates"]

    rank_order = sorted(range(len(regions)), key=lambda idx: regions[idx]["area"])
    parity_sets = {0: set(), 1: set()}
    for rank, idx in enumerate(rank_order):
        parity_sets[rank % 2].add(regions[idx]["color"])
    if parity_sets[0] != set(SPRITE_COLORS) or parity_sets[1] != set(SPRITE_COLORS):
        return False

    area_rank_coords = []
    n_cols = len({region["bbox"][1] for region in regions})
    for idx in rank_order:
        box = regions[idx]["bbox"]
        area_rank_coords.append((box[0], box[1]))
    if area_rank_coords == sorted(area_rank_coords) or area_rank_coords == sorted(area_rank_coords, reverse=True):
        return False

    if np.any(inp == YELLOW):
        return False
    if not np.array_equal(inp == RED, out == RED):
        return False

    other_color = {BLUE: GREEN, GREEN: BLUE}
    found_bbox_yellow = False
    for rank, region_idx in enumerate(rank_order):
        region = regions[region_idx]
        r0, c0, r1, c1 = region["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]

        target_color = other_color[region["color"]]
        target_template = templates[target_color]
        sr0, sc0, sr1, sc1 = region["sprite_bbox"]
        sprite_patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if not np.array_equal(sprite_patch == target_color, target_template):
            return False

        allowed_colors = {target_color, YELLOW} if rank % 2 == 1 else {target_color, BLACK}
        if set(map(int, np.unique(patch))) - allowed_colors:
            return False

        if rank % 2 == 1:
            if np.any(patch == BLACK):
                return False
            old_mask = region["sprite_mask"]
            if not np.array_equal(old_mask, target_template):
                bbox_patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
                if np.any(bbox_patch == YELLOW):
                    found_bbox_yellow = True
        else:
            if np.any(patch == YELLOW):
                return False

    if not found_bbox_yellow:
        return False

    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = random.choice(LAYOUT_CHOICES)
        shape = random.choice(SPRITE_BOX_CHOICES)

        partition = _choose_partition(shape, layout)
        if partition is None:
            continue
        row_heights, col_widths = partition
        boxes = _region_boxes(row_heights, col_widths)
        area_rank_order = sorted(
            range(len(boxes)),
            key=lambda idx: (boxes[idx][2] - boxes[idx][0] + 1) * (boxes[idx][3] - boxes[idx][1] + 1),
        )

        templates = _choose_templates(shape)
        if templates is None:
            continue
        blue_template, green_template = templates

        colors_by_region = _assign_colors_by_region(len(boxes), area_rank_order)
        if colors_by_region is None:
            continue

        inp = _build_input(row_heights, col_widths, blue_template, green_template, colors_by_region)
        if inp is None:
            continue
        try:
            if _validate_generated_instance(inp):
                return inp
        except Exception:
            continue

    raise RuntimeError("failed to generate a valid input for puzzle 4")
