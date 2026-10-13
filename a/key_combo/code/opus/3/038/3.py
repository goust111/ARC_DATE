from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_ATTEMPTS = 400

BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    H, W = shape
    h, w = tile.shape
    rows = np.arange(H) % h
    cols = np.arange(W) % w
    return tile[rows[:, None], cols[None, :]]


def _cropped_color_mask(grid: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, BBox]:
    mask = grid == int(color)
    box = bbox(mask)
    assert box is not None, "expected each non-background color to be present"
    r0, c0, r1, c1 = box
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    crop[mask[r0 : r1 + 1, c0 : c1 + 1]] = int(color)
    return crop, box


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    rr, cc = end
    if random.random() < 0.5:
        while c != cc:
            mask[r, c] = True
            c += 1 if cc > c else -1
        while r != rr:
            mask[r, c] = True
            r += 1 if rr > r else -1
    else:
        while r != rr:
            mask[r, c] = True
            r += 1 if rr > r else -1
        while c != cc:
            mask[r, c] = True
            c += 1 if cc > c else -1
    mask[r, c] = True


def _mirror_lr(half_mask: np.ndarray, full_w: int) -> np.ndarray:
    h, half_w = half_mask.shape
    out = np.zeros((h, full_w), dtype=bool)
    for r, c in np.argwhere(half_mask):
        r = int(r)
        c = int(c)
        out[r, c] = True
        out[r, full_w - 1 - c] = True
    return out


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    crop = np.zeros(mask.shape, dtype=int)
    crop[mask] = int(color)
    return crop


def _gen_symmetric_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        half_w = (w + 1) // 2
        half = np.zeros((h, half_w), dtype=bool)

        seed = (random.randrange(h), random.randrange(half_w))
        edge_points = [
            (0, random.randrange(half_w)),
            (h - 1, random.randrange(half_w)),
            (random.randrange(h), 0),
        ]
        random.shuffle(edge_points)
        for point in edge_points:
            _draw_path(half, seed, point)
            if random.random() < 0.6:
                seed = point

        for _ in range(random.randint(1, 2)):
            occupied = [tuple(map(int, rc)) for rc in np.argwhere(half)]
            src = random.choice(occupied)
            dst = (random.randrange(h), random.randrange(half_w))
            _draw_path(half, src, dst)

        mask = _mirror_lr(half, w)
        area = h * w
        filled = int(mask.sum())
        if not _touches_all_sides(mask):
            continue
        if not np.array_equal(mask, np.fliplr(mask)):
            continue
        if filled < max(4, area // 3):
            continue
        if filled > (4 * area) // 5:
            continue
        if len({tuple(row.tolist()) for row in mask}) < 2:
            continue
        return mask
    return None


def _gen_asymmetric_mask(min_h: int, min_w: int) -> np.ndarray | None:
    max_h = min(10, min_h + 3)
    max_w = min(10, min_w + 3)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(250):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        mask = np.zeros((h, w), dtype=bool)

        seed = (random.randrange(h), random.randrange(w))
        edge_points = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(edge_points)
        for point in edge_points:
            _draw_path(mask, seed, point)
            if random.random() < 0.7:
                seed = point

        for _ in range(random.randint(2, 4)):
            occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
            src = random.choice(occupied)
            dst = (random.randrange(h), random.randrange(w))
            _draw_path(mask, src, dst)

        for _ in range(random.randint(0, 2)):
            occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
            r, c = random.choice(occupied)
            if random.random() < 0.5 and c + 1 < w:
                mask[r, c : c + 2] = True
            elif r + 1 < h:
                mask[r : r + 2, c] = True

        area = h * w
        filled = int(mask.sum())
        if not _touches_all_sides(mask):
            continue
        if filled < max(6, area // 4):
            continue
        if filled > (3 * area) // 4:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if int(np.count_nonzero(mask != np.fliplr(mask))) < 4:
            continue
        return mask
    return None


def _layout_items(items: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    assert len(items) in {3, 4}
    order = items.copy()
    random.shuffle(order)
    if len(order) == 3:
        split = random.choice([1, 2])
        rows = [order[:split], order[split:]]
    else:
        rows = [order[:2], order[2:]]

    outer_margin = random.randint(1, 2)
    col_gap = random.randint(2, 3)
    row_gap = random.randint(2, 3)

    row_heights = [max(item["crop"].shape[0] for item in row) for row in rows]
    row_widths = [
        sum(item["crop"].shape[1] for item in row) + col_gap * max(0, len(row) - 1)
        for row in rows
    ]

    base_h = 2 * outer_margin + sum(row_heights) + row_gap
    base_w = 2 * outer_margin + max(row_widths)
    if base_h > 30 or base_w > 30:
        return None

    H = base_h + random.randint(0, min(4, 30 - base_h))
    W = base_w + random.randint(0, min(4, 30 - base_w))

    vertical_extra = _composition(H - base_h, 3)
    row_starts = [
        outer_margin + vertical_extra[0],
        outer_margin + vertical_extra[0] + row_heights[0] + row_gap + vertical_extra[1],
    ]

    grid = np.zeros((H, W), dtype=int)
    placements: list[dict] = []
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        horizontal_extra = _composition(W - 2 * outer_margin - row_width, len(row) + 1)
        c = outer_margin + horizontal_extra[0]
        for item_idx, item in enumerate(row):
            crop = item["crop"]
            h, w = crop.shape
            row_start = row_starts[row_idx]
            r = row_start + random.randint(0, row_heights[row_idx] - h)
            grid[r : r + h, c : c + w] = crop
            placements.append(
                {
                    "template": bool(item["template"]),
                    "crop": crop,
                    "r": r,
                    "c": c,
                }
            )
            c += w
            if item_idx < len(row) - 1:
                c += col_gap + horizontal_extra[item_idx + 1]
    return grid, placements


def _expected_from_placements(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    template_crop = next(entry["crop"] for entry in placements if entry["template"])
    expected = grid.copy()
    for entry in placements:
        if entry["template"]:
            continue
        crop = entry["crop"]
        r = int(entry["r"])
        c = int(entry["c"])
        h, w = crop.shape
        expected[r : r + h, c : c + w] = tile_fill((h, w), template_crop)
    return expected


def _generate_attempt() -> np.ndarray | None:
    tile_mask = _gen_symmetric_mask()
    if tile_mask is None:
        return None

    tile_h, tile_w = tile_mask.shape
    n_colors = random.randint(3, 4)
    colors = random.sample(COLOR_POOL, n_colors)

    items = [{"template": True, "crop": _colorize(tile_mask, colors[0])}]
    for color in colors[1:]:
        mask = _gen_asymmetric_mask(2 * tile_h, 2 * tile_w)
        if mask is None:
            return None
        items.append({"template": False, "crop": _colorize(mask, color)})

    if sum(np.array_equal(item["crop"], np.fliplr(item["crop"])) for item in items) != 1:
        return None

    laid_out = _layout_items(items)
    if laid_out is None:
        return None
    grid, placements = laid_out

    expected = _expected_from_placements(grid, placements)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the unique left-right symmetric color crop into a tile and periodically fill every non-symmetric bounding box with it.

    Input structure:
    - The grid has a black-dominant background and 3-4 non-black colors.
    - Each color defines one mask, possibly split into several 4-connected pieces, and each mask is interpreted through its overall bounding box crop.
    - Exactly one color crop is invariant under left-right reflection; every other color crop is not.

    Rule:
    - Crop each non-background color to its minimal bounding box while keeping only that color and background inside the crop.
    - Select the unique crop equal to its left-right mirror; this crop is the tile template.
    - For every non-symmetric color, erase its pixels and remember its original bounding box.
    - Fill each remembered bounding box by repeating the symmetric tile from that box's top-left corner, clipping partial tiles at the box edges.
    - Keep the original symmetric source crop in place.

    Key insight:
    - Vertical symmetry identifies the reusable stencil rather than the answer region; the asymmetric colors contribute only their locations, which become tiled target rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    template_color: int | None = None
    template_crop: np.ndarray | None = None
    asymmetric_boxes: list[BBox] = []

    for color in colors_present(g, exclude={bg}):
        crop, box = _cropped_color_mask(g, color, bg)
        if np.array_equal(crop, np.fliplr(crop)):
            assert template_crop is None, "expected a unique symmetric crop"
            template_crop = crop
            template_color = int(color)
        else:
            asymmetric_boxes.append(box)

    assert template_crop is not None and template_color is not None, "missing symmetric template"
    assert asymmetric_boxes, "expected at least one asymmetric target box"

    out = g.copy()
    for color in colors_present(g, exclude={bg}):
        if int(color) != template_color:
            out[out == int(color)] = bg

    for box in asymmetric_boxes:
        r0, c0, r1, c1 = box
        region_shape = (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = tile_fill(region_shape, template_crop)

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
