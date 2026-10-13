import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = [1, 2, 3, 7, 9]
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_components(grid, colors=None, row_start=0):
    h, w = grid.shape
    allow = None if colors is None else set(colors)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(max(0, row_start), h):
        for c in range(w):
            v = int(grid[r, c])
            if seen[r, c] or v == BG or (allow is not None and v not in allow):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == v:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": v, "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _region_intervals(cut_cols, width):
    cuts = sorted(int(c) for c in cut_cols)
    regions = []
    start = 0
    for cut in cuts:
        if cut <= start or cut >= width:
            raise ValueError("invalid partition cut")
        regions.append((start, cut - 1))
        start = cut + 1
    if start > width - 1:
        raise ValueError("empty final region")
    regions.append((start, width - 1))
    if any(c1 < c0 for c0, c1 in regions):
        raise ValueError("empty region")
    return regions


def _parse_palette(grid, regions):
    colors = []
    for c0, c1 in regions:
        vals = sorted(set(int(v) for v in grid[0, c0 : c1 + 1]) - {BG})
        if len(vals) != 1:
            raise ValueError("palette row must encode one color per region")
        colors.append(vals[0])
    return colors


def _sprite_region(sprite, regions):
    _r0, c0, _r1, c1 = sprite["bbox"]
    found = []
    for idx, (x0, x1) in enumerate(regions):
        if x0 <= c0 and c1 <= x1:
            found.append(idx)
    if len(found) != 1:
        raise ValueError("sprite must belong to exactly one region")
    return found[0]


def _drop_region_sprites(sprites, region, region_color, grid_shape):
    h, _w = grid_shape
    c0, c1 = region
    blocked = {(h - 1, c) for c in range(c0, c1 + 1)}
    ordered = sorted(sprites, key=lambda s: (s["bbox"][2], s["bbox"][0], s["bbox"][1]), reverse=True)
    settled = []
    for sprite in ordered:
        coords = list(sprite["coords"])
        drop = 0
        while True:
            shifted = [(r + drop + 1, c) for r, c in coords]
            if any(c < c0 or c > c1 or r >= h or (r, c) in blocked for r, c in shifted):
                break
            drop += 1
        final_coords = [(r + drop, c) for r, c in coords]
        settled.append(
            {
                "orig_color": sprite["color"],
                "region_color": region_color,
                "region": region,
                "drop": drop,
                "coords": final_coords,
                "bbox": _bbox(final_coords),
            }
        )
        for r, c in final_coords:
            blocked.add((r, c))
            if r - 1 >= 0:
                blocked.add((r - 1, c))
    return settled


def _solve(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    if h < 4 or w < 5:
        raise ValueError("grid too small")

    bottom_grey = [c for c in range(w) if int(g[h - 1, c]) == GREY]
    if not (2 <= len(bottom_grey) <= 3):
        raise ValueError("expected 2-3 bottom control pixels")
    if np.any(g[1 : h - 1, :] == GREY):
        raise ValueError("grey controls must be on the bottom edge")

    regions = _region_intervals(bottom_grey, w)
    palette = _parse_palette(g, regions)

    sprite_grid = g.copy()
    sprite_grid[0, :] = BG
    sprite_grid[h - 1, bottom_grey] = BG
    sprites = _find_components(sprite_grid, colors=SPRITE_COLORS, row_start=1)
    if not sprites:
        raise ValueError("expected at least one sprite")

    by_region = [[] for _ in regions]
    for sprite in sprites:
        ridx = _sprite_region(sprite, regions)
        by_region[ridx].append(sprite)

    out = np.zeros_like(g)
    final_sprites = []
    for ridx, region in enumerate(regions):
        if not by_region[ridx]:
            raise ValueError("each region must contain a sprite")
        settled = _drop_region_sprites(by_region[ridx], region, palette[ridx], g.shape)
        final_sprites.extend(settled)
        for sprite in settled:
            for r, c in sprite["coords"]:
                out[r, c] = sprite["region_color"]
    return out, {
        "regions": regions,
        "palette": palette,
        "input_sprites": sprites,
        "final_sprites": final_sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each sprite inside its ray-defined vertical region, then recolor it by that region's palette block.

    Input structure:
    - Black background with a 1-cell-high palette row at the top and 2-3 grey control pixels on the bottom edge.
    - Bottom control pixels define invisible vertical rays, so the space below the palette is partitioned into 3 or 4 vertical rectangular regions.
    - Each region contains one or more small 4-connected monochrome sprites in the upper part of the grid.

    Rule:
    - Treat each grey bottom-edge control pixel as an invisible vertical boundary column and partition the grid into the vertical strips between those boundaries.
    - Read the unique non-black palette color spanning each strip in the top row; that strip color is the target recolor for that region.
    - In each region independently, apply downward gravity with a one-row cushion above the grid bottom and above every already-settled lower sprite, preserving each sprite's shape and horizontal position.
    - Erase the palette row and control pixels; output only the dropped sprites recolored to their region's palette color.

    Key insight:
    - The same invisible ray partition serves both as the physical constraint for region-local gravity and as the semantic lookup that decides each fallen sprite's final color.
    """
    out, _meta = _solve(grid)
    return out


def _random_polyomino(area, max_h, max_w):
    for _ in range(80):
        cells = [(0, 0)]
        while len(cells) < area:
            r, c = random.choice(cells)
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            random.shuffle(neighbors)
            placed = False
            for nr, nc in neighbors:
                if (nr, nc) in cells:
                    continue
                trial = cells + [(nr, nc)]
                rs = [x for x, _ in trial]
                cs = [y for _, y in trial]
                if max(rs) - min(rs) + 1 <= max_h and max(cs) - min(cs) + 1 <= max_w:
                    cells.append((nr, nc))
                    placed = True
                    break
            if not placed:
                break
        if len(cells) != area:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        norm = sorted((r - r0, c - c0) for r, c in cells)
        if (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2:
            return norm
    return None


def _sample_region_widths(region_count):
    min_width = 4
    max_extra = 8
    widths = [min_width] * region_count
    extra = random.randint(region_count, region_count + max_extra)
    for _ in range(extra):
        idx = random.randrange(region_count)
        if widths[idx] < 7:
            widths[idx] += 1
    return widths


def _build_regions(widths):
    cuts = []
    start = 0
    regions = []
    for i, width in enumerate(widths):
        regions.append((start, start + width - 1))
        start += width
        if i < len(widths) - 1:
            cuts.append(start)
            start += 1
    return regions, cuts, start


def _place_sprite(grid, cells, top, left, color):
    for r, c in cells:
        grid[top + r, left + c] = color


def _make_input():
    region_count = random.choice([3, 4])
    sprite_count = 5 if region_count == 4 else random.choice([4, 5])
    widths = _sample_region_widths(region_count)
    regions, cuts, width = _build_regions(widths)
    height = random.randint(17, 23)
    if width > 30 or height > 30:
        return None, None

    grid = np.zeros((height, width), dtype=int)
    palette = random.sample(PALETTE_COLORS, region_count)
    for ridx, (c0, c1) in enumerate(regions):
        grid[0, c0 : c1 + 1] = palette[ridx]
    for cut in cuts:
        grid[height - 1, cut] = GREY

    counts = [1] * region_count
    extra = sprite_count - region_count
    extra_regions = random.sample(range(region_count), extra)
    for ridx in extra_regions:
        counts[ridx] += 1

    colors = random.sample(SPRITE_COLORS, sprite_count)
    color_idx = 0
    placements = []
    stacked_region = None

    for ridx, count in enumerate(counts):
        c0, c1 = regions[ridx]
        region_w = c1 - c0 + 1
        shapes = []
        for _ in range(count):
            for _attempt in range(60):
                area = random.randint(3, 6)
                max_h = random.choice([2, 3])
                max_w = min(region_w, random.choice([2, 3]))
                shape = _random_polyomino(area, max_h=max_h, max_w=max_w)
                if shape is not None:
                    shapes.append(shape)
                    break
            else:
                return None, None

        bboxes = [_bbox(shape) for shape in shapes]
        widths_s = [bb[3] - bb[1] + 1 for bb in bboxes]
        heights_s = [bb[2] - bb[0] + 1 for bb in bboxes]

        if count >= 2:
            stacked_region = ridx if stacked_region is None else stacked_region
            common_c = random.randint(c0 + 1, c1 - 1)
            tops = []
            cursor = random.randint(2, 4)
            for hgt in heights_s:
                tops.append(cursor)
                cursor += hgt + random.randint(2, 3)
            if cursor >= height - 4:
                return None, None
            lefts = []
            for width_s in widths_s:
                lo = max(c0, common_c - width_s + 1)
                hi = min(common_c, c1 - width_s + 1)
                if lo > hi:
                    return None, None
                lefts.append(random.randint(lo, hi))
        else:
            shape_h = heights_s[0]
            shape_w = widths_s[0]
            top = random.randint(2, min(7, height - shape_h - 5))
            left = random.randint(c0, c1 - shape_w + 1)
            tops = [top]
            lefts = [left]

        region_occ = set()
        for shape, top, left in zip(shapes, tops, lefts):
            abs_cells = {(top + r, left + c) for r, c in shape}
            if any(r <= 0 or r >= height - 2 or c < c0 or c > c1 for r, c in abs_cells):
                return None, None
            if abs_cells & region_occ:
                return None, None
            for r, c in abs_cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in region_occ:
                        return None, None
            region_occ |= abs_cells
            color = colors[color_idx]
            color_idx += 1
            _place_sprite(grid, shape, top, left, color)
            placements.append(
                {
                    "region": ridx,
                    "orig_color": color,
                    "coords": sorted(abs_cells),
                    "bbox": _bbox(abs_cells),
                }
            )

    if stacked_region is None:
        return None, None

    meta = {
        "regions": regions,
        "cuts": cuts,
        "palette": palette,
        "placements": placements,
        "stacked_region": stacked_region,
    }
    return grid, meta


def _floor_drop(coords, height):
    drop = 0
    while True:
        shifted = [(r + drop + 1, c) for r, c in coords]
        if any(r >= height - 1 for r, _c in shifted):
            break
        drop += 1
    return drop


def _semantic_ok(grid, meta):
    out, solved = _solve(grid)
    if np.array_equal(grid, out):
        return False

    regions = meta["regions"]
    placements = meta["placements"]
    stacked_region = meta["stacked_region"]

    if len(regions) != len(meta["palette"]):
        return False
    if len(placements) < len(regions):
        return False

    region_counts = Counter(p["region"] for p in placements)
    if any(region_counts[i] == 0 for i in range(len(regions))):
        return False
    if max(region_counts.values()) < 2:
        return False

    changed_colors = 0
    for sprite in solved["final_sprites"]:
        if sprite["orig_color"] != sprite["region_color"]:
            changed_colors += 1
    if changed_colors == 0:
        return False

    stacked_inputs = [p for p in placements if p["region"] == stacked_region]
    if len(stacked_inputs) < 2:
        return False
    overlap = False
    for i in range(len(stacked_inputs)):
        cols_i = {c for _r, c in stacked_inputs[i]["coords"]}
        for j in range(i + 1, len(stacked_inputs)):
            cols_j = {c for _r, c in stacked_inputs[j]["coords"]}
            if cols_i & cols_j:
                overlap = True
    if not overlap:
        return False

    by_color = {sprite["orig_color"]: sprite for sprite in solved["final_sprites"]}
    stack_dependent = False
    for placed in stacked_inputs:
        actual = by_color[placed["orig_color"]]["drop"]
        floor_only = _floor_drop(placed["coords"], grid.shape[0])
        if actual < floor_only:
            stack_dependent = True
            break
    if not stack_dependent:
        return False

    occupied_output_colors = sorted(set(int(v) for v in out.flatten()) - {BG})
    if len(occupied_output_colors) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, meta = _make_input()
        if grid is None:
            continue
        if _semantic_ok(grid, meta):
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
