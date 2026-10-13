from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_filled_rect(g: np.ndarray, points: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(points)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch == BG):
        return None
    if len(points) != patch.size:
        return None
    colors = np.unique(patch)
    if colors.size < 2:
        return None
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(patch.size),
    }


def _parse_hollow_frame(g: np.ndarray, points: list[tuple[int, int]]) -> dict | None:
    values = {int(g[r, c]) for r, c in points}
    if len(values) != 1:
        return None

    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True

    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None

    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
        "color": int(next(iter(values))),
        "inner_h": int(r1 - r0 - 1),
        "inner_w": int(c1 - c0 - 1),
    }


def _find_structure(g: np.ndarray) -> tuple[dict, list[dict]]:
    comps = _components(g != BG)
    mosaic_candidates: list[dict] = []
    frames: list[dict] = []
    for pts in comps:
        frame = _parse_hollow_frame(g, pts)
        if frame is not None:
            frames.append(frame)
            continue
        mosaic = _parse_filled_rect(g, pts)
        if mosaic is not None:
            mosaic_candidates.append(mosaic)

    assert mosaic_candidates, "expected a filled multicolor mosaic"
    mosaic = max(mosaic_candidates, key=lambda item: item["area"])
    mr0, mc0, mr1, _mc1 = mosaic["bbox"]

    usable_frames = []
    for frame in frames:
        ir0, _ic0, ir1, _ic1 = frame["inner_bbox"]
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        if fc0 <= mosaic["bbox"][3]:
            continue
        if mr0 <= ir0 <= ir1 <= mr1 and fr0 >= 0:
            usable_frames.append(frame)

    assert usable_frames, "expected at least one filter frame to the right of the mosaic"
    usable_frames.sort(key=lambda item: (item["inner_bbox"][0], item["bbox"][1]))
    return mosaic, usable_frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project left-mosaic slices into colored filter frames, keeping only pixels that match each frame color.

    Input structure:
    - The background is black.
    - One large filled multicolor rectangle on the left is the source mosaic.
    - Several small one-pixel-thick hollow rectangular frames sit to the right; each frame border is a single non-black color and each frame interior is black.

    Rule:
    - Use each frame's interior row span and interior width to select a same-size source region from the left edge of the mosaic at the same absolute rows.
    - Copy that source region into the frame interior, but keep only pixels whose color matches the frame border color; replace all other copied pixels with black.
    - Leave the mosaic, the frame borders, and the rest of the background unchanged.

    Key insight:
    - The frame border color is a color filter, and the frame's vertical placement tells you which horizontal slice of the left mosaic to sample.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mosaic, frames = _find_structure(g)
    _mr0, mc0, _mr1, mc1 = mosaic["bbox"]
    mosaic_w = int(mc1 - mc0 + 1)

    out = g.copy()
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        height = int(ir1 - ir0 + 1)
        width = int(ic1 - ic0 + 1)
        assert width <= mosaic_w, "frame window must fit inside the mosaic width"
        source = g[ir0 : ir0 + height, mc0 : mc0 + width]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = np.where(source == frame["color"], source, BG)
    return out


def _random_partition(total: int, parts: int, min_size: int = 1) -> list[int] | None:
    if parts * min_size > total:
        return None
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _build_block_mosaic(height: int, width: int, palette: list[int]) -> np.ndarray:
    row_parts = random.randint(3, min(6, height))
    col_parts = random.randint(4, min(8, width))
    row_sizes = _random_partition(height, row_parts)
    col_sizes = _random_partition(width, col_parts)
    assert row_sizes is not None and col_sizes is not None

    logical = np.zeros((row_parts, col_parts), dtype=int)
    for r in range(row_parts):
        for c in range(col_parts):
            choices = palette[:]
            banned = set()
            if r > 0 and random.random() < 0.8:
                banned.add(int(logical[r - 1, c]))
            if c > 0 and random.random() < 0.8:
                banned.add(int(logical[r, c - 1]))
            if len(banned) < len(choices):
                choices = [color for color in choices if color not in banned]
            logical[r, c] = random.choice(choices)

    patch = np.zeros((height, width), dtype=int)
    rr = 0
    for r, rh in enumerate(row_sizes):
        cc = 0
        for c, cw in enumerate(col_sizes):
            patch[rr : rr + rh, cc : cc + cw] = int(logical[r, c])
            cc += cw
        rr += rh
    return patch


def _frame_row_plan(count: int) -> tuple[int, list[tuple[int, int]]]:
    inner_heights = [random.randint(2, 4) for _ in range(count)]
    gaps = [random.randint(3, 5) for _ in range(count - 1)]
    top_slack = random.randint(0, 3)
    bottom_slack = random.randint(0, 3)

    starts: list[int] = []
    cursor = top_slack
    for i, height in enumerate(inner_heights):
        starts.append(cursor)
        cursor += height
        if i < len(gaps):
            cursor += gaps[i]
    mosaic_h = cursor + bottom_slack
    return mosaic_h, list(zip(starts, inner_heights, strict=False))


def _choose_canvas(
    mosaic_h: int, mosaic_w: int, max_outer_w: int
) -> tuple[int, int, int, int, int, int, int] | None:
    height_choices = [
        (top, bottom)
        for top in range(1, 4)
        for bottom in range(1, 4)
        if mosaic_h + top + bottom <= 30
    ]
    if not height_choices:
        return None
    top_margin, bottom_margin = random.choice(height_choices)

    width_choices = []
    for left in range(1, 4):
        for gap in range(2, 5):
            for jitter in range(0, 3):
                zone_w = max_outer_w + jitter
                for right in range(1, 4):
                    total_w = left + mosaic_w + gap + zone_w + right
                    if total_w <= 30:
                        width_choices.append((left, gap, jitter, right, total_w))
    if not width_choices:
        return None
    left_margin, gap, jitter, right_margin, total_w = random.choice(width_choices)
    total_h = mosaic_h + top_margin + bottom_margin
    return total_h, total_w, top_margin, left_margin, gap, jitter, right_margin


def _pick_filter_color(region: np.ndarray) -> int | None:
    values, counts = np.unique(region, return_counts=True)
    if values.size < 2:
        return None
    candidates = [int(v) for v, cnt in zip(values.tolist(), counts.tolist(), strict=False) if 0 < int(cnt) < region.size]
    if not candidates:
        return None
    return random.choice(candidates)


def _generate_candidate() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    mosaic_h, row_plan = _frame_row_plan(frame_count)
    if not (8 <= mosaic_h <= 24):
        return None

    inner_widths = [random.randint(2, 6) for _ in range(frame_count)]
    max_inner_w = max(inner_widths)
    min_mosaic_w = max(7, max_inner_w)
    max_mosaic_w = min(15, max_inner_w + random.randint(2, 7))
    if min_mosaic_w > max_mosaic_w:
        return None
    mosaic_w = random.randint(min_mosaic_w, max_mosaic_w)
    max_outer_w = max_inner_w + 2

    canvas = _choose_canvas(mosaic_h, mosaic_w, max_outer_w)
    if canvas is None:
        return None
    grid_h, grid_w, top_margin, left_margin, gap, jitter, _right_margin = canvas

    palette_size = random.randint(4, 6)
    palette = random.sample(list(COLOR_POOL), palette_size)
    mosaic = _build_block_mosaic(mosaic_h, mosaic_w, palette)
    if np.unique(mosaic).size < 3:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    mosaic_r0 = top_margin
    mosaic_c0 = left_margin
    grid[mosaic_r0 : mosaic_r0 + mosaic_h, mosaic_c0 : mosaic_c0 + mosaic_w] = mosaic

    frame_zone_left = mosaic_c0 + mosaic_w + gap
    zone_w = max_outer_w + jitter

    frame_specs = []
    for (rel_top, inner_h), inner_w in zip(row_plan, inner_widths, strict=False):
        src_r0 = mosaic_r0 + rel_top
        src_r1 = src_r0 + inner_h - 1
        source = grid[src_r0 : src_r1 + 1, mosaic_c0 : mosaic_c0 + inner_w]
        color = _pick_filter_color(source)
        if color is None:
            return None
        max_shift = zone_w - (inner_w + 2)
        outer_c0 = frame_zone_left + random.randint(0, max_shift)
        outer_r0 = src_r0 - 1
        outer_r1 = src_r1 + 1
        outer_c1 = outer_c0 + inner_w + 1
        frame_specs.append((outer_r0, outer_c0, outer_r1, outer_c1, color))

    for i, (r0, c0, r1, c1, color) in enumerate(frame_specs):
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return None
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        filtered = np.where(grid[ir0 : ir1 + 1, mosaic_c0 : mosaic_c0 + (ic1 - ic0 + 1)] == color, color, BG)
        if not np.any(filtered == color):
            return None
        if i > 0:
            prev_r1 = frame_specs[i - 1][2]
            if r0 <= prev_r1:
                return None

    mosaic_info, frames = _find_structure(grid)
    if len(frames) != frame_count:
        return None
    if mosaic_info["bbox"] != (mosaic_r0, mosaic_c0, mosaic_r0 + mosaic_h - 1, mosaic_c0 + mosaic_w - 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid filter-frame puzzle")
