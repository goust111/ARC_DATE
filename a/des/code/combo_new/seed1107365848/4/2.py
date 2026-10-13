from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(1, 10))
TEMP_CANVAS = 30
GEN_RETRIES = 400
PAIR_RETRIES = 120
PLACE_RETRIES = 200


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _tight_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                pixels.append((rr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.asarray(pixels, dtype=int)
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            bbox = (r0, c0, r1, c1)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": _component_mask(coords, bbox),
                }
            )

    return components


def _reconstruct_pair(pair: list[dict]) -> np.ndarray:
    comp_a, comp_b = pair
    mask_a = comp_a["mask"]
    mask_b = comp_b["mask"]
    bbox_a = comp_a["bbox"]
    bbox_b = comp_b["bbox"]

    same_rows = bbox_a[0] == bbox_b[0] and bbox_a[2] == bbox_b[2]
    same_cols = bbox_a[1] == bbox_b[1] and bbox_a[3] == bbox_b[3]
    same_shape = mask_a.shape == mask_b.shape

    vertical_ok = same_rows and same_shape and (
        np.array_equal(mask_a, np.fliplr(mask_b)) or np.array_equal(np.fliplr(mask_a), mask_b)
    )
    horizontal_ok = same_cols and same_shape and (
        np.array_equal(mask_a, np.flipud(mask_b)) or np.array_equal(np.flipud(mask_a), mask_b)
    )

    if vertical_ok and not horizontal_ok:
        mode = "vertical"
    elif horizontal_ok and not vertical_ok:
        mode = "horizontal"
    elif vertical_ok and horizontal_ok:
        col_gap = abs((bbox_a[1] + bbox_a[3]) - (bbox_b[1] + bbox_b[3]))
        row_gap = abs((bbox_a[0] + bbox_a[2]) - (bbox_b[0] + bbox_b[2]))
        mode = "vertical" if col_gap >= row_gap else "horizontal"
    else:
        raise AssertionError("same-color pieces do not form a valid mirrored pair")

    if mode == "vertical":
        left, right = sorted(pair, key=lambda comp: int(comp["bbox"][1]))
        left_mask = left["mask"]
        right_mask = right["mask"]
        h, w = left_mask.shape
        full = np.zeros((h, w * 2), dtype=bool)
        full[:, :w] = left_mask
        full[:, w:] = right_mask
        return full

    top, bottom = sorted(pair, key=lambda comp: int(comp["bbox"][0]))
    top_mask = top["mask"]
    bottom_mask = bottom["mask"]
    h, w = top_mask.shape
    full = np.zeros((h * 2, w), dtype=bool)
    full[:h, :] = top_mask
    full[h:, :] = bottom_mask
    return full


def _shape_sort_key(item: dict) -> tuple:
    mask = item["mask"]
    return (-int(item["area"]), -int(mask.shape[0]), -int(mask.shape[1]), mask.tobytes())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join each same-colored mirror pair into a full symmetric shape, then stack the completed shapes by area.

    Input structure:
    - The background is black.
    - Every non-background color appears in exactly two monochrome 4-connected fragments.
    - Within each color pair, the two fragments are mirror halves of one symmetric shape across a horizontal or vertical axis.

    Rule:
    - Pair the two fragments of each color.
    - Detect whether the pair is split left/right or top/bottom, then translate the halves toward each other until their cut edges meet and the full symmetric shape is restored.
    - Compute the area of every restored shape.
    - Build a new black output grid whose width is the largest restored width and whose height is the sum of restored heights, then stack the restored shapes from bottom to top in descending area order, centering each shape horizontally and leaving no vertical gaps.

    Key insight:
    - The scattered pieces are not independent objects: each color encodes two mirrored halves of one hidden figure, and the final task is a second-stage size-based repacking of those reconstructed figures.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    components = _find_components(g, bg=bg)
    assert components

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    shapes: list[dict] = []
    for color, pair in by_color.items():
        assert len(pair) == 2, "each color must appear in exactly two fragments"
        full_mask = _reconstruct_pair(pair)
        shapes.append({"color": int(color), "mask": full_mask, "area": int(full_mask.sum())})

    shapes.sort(key=_shape_sort_key)
    out_h = sum(int(shape["mask"].shape[0]) for shape in shapes)
    out_w = max(int(shape["mask"].shape[1]) for shape in shapes)
    out = np.full((out_h, out_w), bg, dtype=int)

    cursor = out_h
    for shape in shapes:
        mask = shape["mask"]
        h, w = map(int, mask.shape)
        cursor -= h
        left = (out_w - w) // 2
        patch = out[cursor : cursor + h, left : left + w]
        patch[mask] = int(shape["color"])

    return out


def _grow_connected_mask(height: int, width: int, area: int, anchor: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if anchor == "right":
        seed = (random.randrange(height), width - 1)
    else:
        seed = (height - 1, random.randrange(width))
    mask[seed] = True
    cells = {seed}

    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in FOUR_NEIGHBORS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = random.choice(tuple(frontier))
        mask[pick] = True
        cells.add(pick)

    assert int(mask.sum()) == area
    return _crop_mask(mask)


def _make_pair_spec(color: int, used_full_areas: set[int]) -> dict | None:
    for _ in range(PAIR_RETRIES):
        orientation = random.choice(("vertical", "horizontal"))
        if orientation == "vertical":
            height = random.randint(2, 6)
            width = random.randint(2, 5)
            min_area = max(2, (height + width) // 2)
            area = random.randint(min_area, height * width)
            half = _grow_connected_mask(height, width, area, anchor="right")
            other = np.fliplr(half)
            full = np.concatenate([half, other], axis=1)
            gap = random.randint(2, 7)
            pair_h = int(half.shape[0])
            pair_w = int(half.shape[1] * 2 + gap)
        else:
            height = random.randint(2, 5)
            width = random.randint(2, 6)
            min_area = max(2, (height + width) // 2)
            area = random.randint(min_area, height * width)
            half = _grow_connected_mask(height, width, area, anchor="bottom")
            other = np.flipud(half)
            full = np.concatenate([half, other], axis=0)
            gap = random.randint(2, 7)
            pair_h = int(half.shape[0] * 2 + gap)
            pair_w = int(half.shape[1])

        full_area = int(full.sum())
        if full_area in used_full_areas:
            continue

        return {
            "color": int(color),
            "orientation": orientation,
            "half": half,
            "other": other,
            "full": full,
            "full_area": full_area,
            "pair_h": pair_h,
            "pair_w": pair_w,
            "gap": gap,
        }
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[rows + r0, cols + c0] = int(color)


def _place_pairs(specs: list[dict]) -> np.ndarray | None:
    reserved = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(specs)), key=lambda i: specs[i]["pair_h"] * specs[i]["pair_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        pair_h = int(spec["pair_h"])
        pair_w = int(spec["pair_w"])
        placed = None

        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, TEMP_CANVAS - pair_h)
            c0 = random.randint(0, TEMP_CANVAS - pair_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(TEMP_CANVAS, r0 + pair_h + 1)
            cc1 = min(TEMP_CANVAS, c0 + pair_w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            reserved[rr0:rr1, cc0:cc1] = True
            placed = (r0, c0)
            break

        if placed is None:
            return None
        placements.append((idx, placed[0], placed[1]))

    grid = np.full((TEMP_CANVAS, TEMP_CANVAS), BG, dtype=int)
    for idx, r0, c0 in placements:
        spec = specs[idx]
        half = spec["half"]
        other = spec["other"]
        color = int(spec["color"])
        gap = int(spec["gap"])
        h, w = map(int, half.shape)

        if spec["orientation"] == "vertical":
            _paint_mask(grid, half, r0, c0, color)
            _paint_mask(grid, other, r0, c0 + w + gap, color)
        else:
            _paint_mask(grid, half, r0, c0, color)
            _paint_mask(grid, other, r0 + h + gap, c0, color)

    return grid


def _random_crop(grid: np.ndarray) -> np.ndarray:
    fg = grid != BG
    r0, c0, r1, c1 = _tight_bbox(fg)

    top_room = r0
    left_room = c0
    bottom_room = grid.shape[0] - 1 - r1
    right_room = grid.shape[1] - 1 - c1

    top_margin = random.randint(1, min(4, top_room)) if top_room else 0
    left_margin = random.randint(1, min(4, left_room)) if left_room else 0
    bottom_margin = random.randint(1, min(4, bottom_room)) if bottom_room else 0
    right_margin = random.randint(1, min(4, right_room)) if right_room else 0

    return grid[
        r0 - top_margin : r1 + bottom_margin + 1,
        c0 - left_margin : c1 + right_margin + 1,
    ]


def _sample_specs() -> list[dict] | None:
    count = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, count)
    used_full_areas: set[int] = set()
    specs: list[dict] = []

    for color in colors:
        spec = _make_pair_spec(color, used_full_areas)
        if spec is None:
            return None
        used_full_areas.add(int(spec["full_area"]))
        specs.append(spec)

    total_height = sum(int(spec["full"].shape[0]) for spec in specs)
    max_width = max(int(spec["full"].shape[1]) for spec in specs)
    if total_height > 30 or max_width > 30:
        return None
    return specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        specs = _sample_specs()
        if specs is None:
            continue

        grid = _place_pairs(specs)
        if grid is None:
            continue
        grid = _random_crop(grid)

        if grid.ndim != 2 or max(grid.shape) > 30:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        expected_masks = [spec["full"] for spec in specs]
        expected_h = sum(mask.shape[0] for mask in expected_masks)
        expected_w = max(mask.shape[1] for mask in expected_masks)
        if out.shape != (expected_h, expected_w):
            continue
        if not np.any(grid == BG):
            continue
        return grid

    raise RuntimeError("failed to generate a valid mirrored-pair puzzle input")
