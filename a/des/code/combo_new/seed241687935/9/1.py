import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
SPRITE_RETRIES = 120


def _mode_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    if set(coords) != border:
        return None
    return r0, c0, r1, c1


def _find_frame_color_and_boxes(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for comp in comps:
            box = _frame_bbox(comp)
            if box is None:
                ok = False
                break
            boxes.append(box)
        if ok:
            candidates.append((color, sorted(boxes)))
    assert len(candidates) == 1, "expected exactly one color used only for rectangular frames"
    return candidates[0]


def _mask_from_component(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _scale_mask(mask: np.ndarray, sy: int, sx: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, sy, axis=0), sx, axis=1)


def _extract_outside_sprite(
    grid: np.ndarray,
    color: int,
    frame_occupancy: np.ndarray,
) -> np.ndarray:
    comps = _components(grid == color)
    outside = []
    for comp in comps:
        in_frame = [bool(frame_occupancy[r, c]) for r, c in comp]
        assert all(in_frame) or not any(in_frame), "same-color component must be fully inside a frame or fully outside"
        if not any(in_frame):
            outside.append(comp)
    assert len(outside) == 1, f"expected one outside sprite for color {color}"
    return _mask_from_component(outside[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each region's markers by the matching outside sprite scaled to the region interior.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames of the same grey color enclose empty regions of different sizes.
    - Each frame interior contains a few scattered marker pixels of exactly one non-grey color, and the same colors appear again as monochrome sprite templates outside all frames.

    Rule:
    - Identify every rectangular closed frame and read the unique marker color inside its interior.
    - Find the outside sprite component with the same color.
    - Compute the integer vertical and horizontal scale factors whose repeated-pixel enlargement makes the sprite bbox exactly match the frame interior size.
    - Clear the interior and stamp the scaled sprite aligned to the interior's top-left corner, while keeping the grey frames and outside templates unchanged.

    Key insight:
    - The interior colored pixels are only color keys; they select which outside template to enlarge, and the frame interior dimensions reveal the required integer scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    frame_color, frames = _find_frame_color_and_boxes(g, bg)

    frame_occupancy = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in frames:
        frame_occupancy[r0 : r1 + 1, c0 : c1 + 1] = True

    region_colors: list[int] = []
    jobs: list[tuple[int, int, int, int, int]] = []
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior))) - {bg, frame_color}
        assert len(colors) == 1, "each frame interior must contain one unique marker color"
        color = colors.pop()
        region_colors.append(color)
        jobs.append((r0, c0, r1, c1, color))
    assert len(set(region_colors)) == len(region_colors), "frame marker colors must be unique"

    sprite_masks = {
        color: _extract_outside_sprite(g, color, frame_occupancy)
        for color in region_colors
    }

    out = g.copy()
    for r0, c0, r1, c1, color in jobs:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        sprite = sprite_masks[color]
        sh, sw = sprite.shape
        assert inner_h % sh == 0 and inner_w % sw == 0, "frame interior must be an integer scale of the sprite bbox"
        sy = inner_h // sh
        sx = inner_w // sw
        scaled = _scale_mask(sprite, sy, sx)
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[r0 + 1 : r1, c0 + 1 : c1][scaled] = color
    return out


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    for _ in range(SPRITE_RETRIES):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        total = h * w
        min_area = max(3, (total + 2) // 3)
        max_area = total - 1
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            choice = rng.choice(tuple(frontier))
            frontier.remove(choice)
            cells.add(choice)
            for nr, nc in _neighbors4(*choice):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crop_h = r1 - r0 + 1
        crop_w = c1 - c0 + 1
        if crop_h < 2 or crop_w < 2:
            continue

        mask = np.zeros((crop_h, crop_w), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if np.all(mask):
            continue
        return mask
    raise RuntimeError("failed to build a nontrivial connected sprite")


def _choose_scales(rng: random.Random, upper: int) -> tuple[int, int]:
    if upper <= 1:
        return 1, 1
    if rng.random() < 0.5:
        s = rng.randint(1, upper)
        return s, s
    sy = rng.randint(1, upper)
    sx = rng.randint(1, upper)
    while sy == sx:
        sx = rng.randint(1, upper)
    return sy, sx


def _layout_items(
    rng: random.Random,
    sizes: list[tuple[int, int]],
    rows: int,
    cols: int,
    gap_max: int = 2,
) -> tuple[int, int, list[tuple[int, int]]]:
    n = len(sizes)
    assert rows * cols >= n
    order = list(range(n))
    rng.shuffle(order)

    slot_for_item: list[tuple[int, int]] = [(-1, -1)] * n
    row_heights = [0] * rows
    col_widths = [0] * cols
    for k, idx in enumerate(order):
        rr, cc = divmod(k, cols)
        slot_for_item[idx] = (rr, cc)
        h, w = sizes[idx]
        row_heights[rr] = max(row_heights[rr], h)
        col_widths[cc] = max(col_widths[cc], w)

    row_offsets = [0] * rows
    col_offsets = [0] * cols
    for r in range(1, rows):
        row_offsets[r] = row_offsets[r - 1] + row_heights[r - 1] + rng.randint(1, gap_max)
    for c in range(1, cols):
        col_offsets[c] = col_offsets[c - 1] + col_widths[c - 1] + rng.randint(1, gap_max)

    positions = [(-1, -1)] * n
    for idx, (rr, cc) in enumerate(slot_for_item):
        h, w = sizes[idx]
        top = row_offsets[rr] + rng.randint(0, row_heights[rr] - h)
        left = col_offsets[cc] + rng.randint(0, col_widths[cc] - w)
        positions[idx] = (top, left)

    zone_h = 0 if not sizes else max(r + h for (r, _), (h, _) in zip(positions, sizes))
    zone_w = 0 if not sizes else max(c + w for (_, c), (_, w) in zip(positions, sizes))
    return zone_h, zone_w, positions


def _frame_layout_options(count: int) -> list[tuple[int, int]]:
    if count == 2:
        return [(1, 2), (2, 1)]
    if count == 3:
        return [(1, 3), (2, 2), (3, 1)]
    return [(2, 2)]


def _sample_markers(rng: random.Random, scaled: np.ndarray) -> list[tuple[int, int]]:
    h, w = scaled.shape
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    marker_count = rng.randint(1, min(4, len(all_cells)))
    output_cells = {(r, c) for r in range(h) for c in range(w) if scaled[r, c]}
    background_cells = [cell for cell in all_cells if cell not in output_cells]

    for _ in range(40):
        chosen: list[tuple[int, int]] = []
        pool = list(all_cells)
        if background_cells:
            first = rng.choice(background_cells)
            chosen.append(first)
            pool.remove(first)
        rng.shuffle(pool)
        chosen.extend(pool[: marker_count - len(chosen)])
        if set(chosen) != output_cells:
            return chosen
    return rng.sample(all_cells, marker_count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout_mode = rng.choice(("right", "bottom", "mixed"))
    region_count = rng.randint(2, 4)
    colors = rng.sample(PALETTE, region_count)
    scale_upper = 2 if region_count == 4 or layout_mode == "mixed" else 3

    regions = []
    any_scaled = False
    for color in colors:
        sprite = _random_sprite_mask(rng)
        sy, sx = _choose_scales(rng, scale_upper)
        any_scaled |= (sy > 1 or sx > 1)
        scaled = _scale_mask(sprite, sy, sx)
        markers = _sample_markers(rng, scaled)
        regions.append(
            {
                "color": color,
                "sprite": sprite,
                "sy": sy,
                "sx": sx,
                "scaled": scaled,
                "markers": markers,
                "frame_h": scaled.shape[0] + 2,
                "frame_w": scaled.shape[1] + 2,
            }
        )
    if not any_scaled:
        return None

    frame_sizes = [(r["frame_h"], r["frame_w"]) for r in regions]
    frame_rows, frame_cols = rng.choice(_frame_layout_options(region_count))
    frame_zone_h, frame_zone_w, frame_positions = _layout_items(rng, frame_sizes, frame_rows, frame_cols)

    indices = list(range(region_count))
    rng.shuffle(indices)
    if layout_mode == "right":
        right_ids = indices
        bottom_ids: list[int] = []
    elif layout_mode == "bottom":
        right_ids = []
        bottom_ids = indices
    else:
        split = rng.randint(1, region_count - 1)
        right_ids = indices[:split]
        bottom_ids = indices[split:]

    right_zone_h = right_zone_w = 0
    right_positions_local: dict[int, tuple[int, int]] = {}
    if right_ids:
        sizes = [regions[idx]["sprite"].shape for idx in right_ids]
        zh, zw, pos = _layout_items(rng, sizes, len(right_ids), 1)
        right_zone_h, right_zone_w = zh, zw
        right_positions_local = {idx: pos[k] for k, idx in enumerate(right_ids)}

    bottom_zone_h = bottom_zone_w = 0
    bottom_positions_local: dict[int, tuple[int, int]] = {}
    if bottom_ids:
        sizes = [regions[idx]["sprite"].shape for idx in bottom_ids]
        zh, zw, pos = _layout_items(rng, sizes, 1, len(bottom_ids))
        bottom_zone_h, bottom_zone_w = zh, zw
        bottom_positions_local = {idx: pos[k] for k, idx in enumerate(bottom_ids)}

    left_w = max(frame_zone_w, bottom_zone_w)
    top_h = max(frame_zone_h, right_zone_h)
    horiz_gap = rng.randint(1, 2) if right_ids else 0
    vert_gap = rng.randint(1, 2) if bottom_ids else 0

    inner_h = top_h + (vert_gap + bottom_zone_h if bottom_ids else 0)
    inner_w = left_w + (horiz_gap + right_zone_w if right_ids else 0)

    margin_top = rng.randint(0, 2)
    margin_left = rng.randint(0, 2)
    margin_bottom = rng.randint(0, 2)
    margin_right = rng.randint(0, 2)
    h = margin_top + inner_h + margin_bottom
    w = margin_left + inner_w + margin_right
    if h > 30 or w > 30:
        return None

    frame_origin = (
        margin_top + rng.randint(0, top_h - frame_zone_h),
        margin_left + rng.randint(0, left_w - frame_zone_w),
    )
    right_origin = None
    if right_ids:
        right_origin = (
            margin_top + rng.randint(0, top_h - right_zone_h),
            margin_left + left_w + horiz_gap,
        )
    bottom_origin = None
    if bottom_ids:
        bottom_origin = (
            margin_top + top_h + vert_gap,
            margin_left + rng.randint(0, left_w - bottom_zone_w),
        )

    grid = np.full((h, w), BACKGROUND, dtype=int)
    expected = grid.copy()

    for idx, region in enumerate(regions):
        fr = frame_origin[0] + frame_positions[idx][0]
        fc = frame_origin[1] + frame_positions[idx][1]
        fh = region["frame_h"]
        fw = region["frame_w"]
        color = region["color"]
        scaled = region["scaled"]

        grid[fr, fc : fc + fw] = FRAME_COLOR
        grid[fr + fh - 1, fc : fc + fw] = FRAME_COLOR
        grid[fr : fr + fh, fc] = FRAME_COLOR
        grid[fr : fr + fh, fc + fw - 1] = FRAME_COLOR

        expected[fr, fc : fc + fw] = FRAME_COLOR
        expected[fr + fh - 1, fc : fc + fw] = FRAME_COLOR
        expected[fr : fr + fh, fc] = FRAME_COLOR
        expected[fr : fr + fh, fc + fw - 1] = FRAME_COLOR

        for mr, mc in region["markers"]:
            grid[fr + 1 + mr, fc + 1 + mc] = color
        expected[fr + 1 : fr + fh - 1, fc + 1 : fc + fw - 1][scaled] = color

    for idx in right_ids:
        sr, sc = right_positions_local[idx]
        origin_r = right_origin[0] + sr
        origin_c = right_origin[1] + sc
        sprite = regions[idx]["sprite"]
        color = regions[idx]["color"]
        grid[origin_r : origin_r + sprite.shape[0], origin_c : origin_c + sprite.shape[1]][sprite] = color
        expected[origin_r : origin_r + sprite.shape[0], origin_c : origin_c + sprite.shape[1]][sprite] = color

    for idx in bottom_ids:
        sr, sc = bottom_positions_local[idx]
        origin_r = bottom_origin[0] + sr
        origin_c = bottom_origin[1] + sc
        sprite = regions[idx]["sprite"]
        color = regions[idx]["color"]
        grid[origin_r : origin_r + sprite.shape[0], origin_c : origin_c + sprite.shape[1]][sprite] = color
        expected[origin_r : origin_r + sprite.shape[0], origin_c : origin_c + sprite.shape[1]][sprite] = color

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, solved):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
