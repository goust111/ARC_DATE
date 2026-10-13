import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
LEGEND_ROWS = 2
PAINT_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _u_mask(shape, direction):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif direction == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(direction)
    return mask


def _parse_legend(grid):
    top = grid[:LEGEND_ROWS]
    h, w = top.shape
    assert h == 2
    mapping = {}
    c = 0
    while c < w:
        if np.all(top[:, c] == GREY):
            c += 1
            continue
        if (
            c + 1 < w
            and top[0, c] == top[1, c]
            and top[0, c + 1] == top[1, c + 1]
            and int(top[0, c]) not in (BG, GREY, ORANGE)
            and int(top[0, c + 1]) not in (BG, GREY, ORANGE)
        ):
            mapping[int(top[0, c])] = int(top[0, c + 1])
            c += 2
            continue
        c += 1
    assert mapping, "expected at least one legend entry"
    return mapping


def _opening_cells(bbox, direction):
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if direction == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if direction == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if direction == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(direction)


def _extract_regions(grid):
    wall_mask = np.zeros_like(grid, dtype=bool)
    wall_mask[LEGEND_ROWS:, :] = grid[LEGEND_ROWS:, :] == GREY
    regions = []
    for comp in _components4(wall_mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = wall_mask[r0 : r1 + 1, c0 : c1 + 1]
        direction = None
        for candidate in ("up", "down", "left", "right"):
            if np.array_equal(patch, _u_mask(patch.shape, candidate)):
                direction = candidate
                break
        if direction is None:
            continue
        interior = ~patch
        seed_cells = []
        interior_coords = []
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if not interior[rr - r0, cc - c0]:
                    continue
                interior_coords.append((rr, cc))
                color = int(grid[rr, cc])
                if color not in (BG, GREY, ORANGE):
                    seed_cells.append((rr, cc))
        assert len(seed_cells) == 1, "expected exactly one seed inside each wall"
        seed = seed_cells[0]
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "seed": seed,
                "seed_color": int(grid[seed]),
                "interior_coords": interior_coords,
                "opening": _opening_cells((r0, c0, r1, c1), direction),
            }
        )
    regions.sort(key=lambda item: item["bbox"])
    return regions


def _flood_interior(grid, region):
    allowed = set(region["interior_coords"])
    start = region["seed"]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in allowed and (nr, nc) not in seen and int(grid[nr, nc]) != ORANGE:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _lane_cells(grid, opening_cell, direction, filled_mask):
    h, w = grid.shape
    dr, dc = DIRS[direction]
    rr = opening_cell[0] + dr
    cc = opening_cell[1] + dc
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        if rr < LEGEND_ROWS:
            break
        if filled_mask[rr, cc]:
            break
        color = int(grid[rr, cc])
        if color == GREY:
            break
        if color == ORANGE:
            break
        if color != BG:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _solve(grid, return_regions=False):
    g = np.asarray(grid, dtype=int)
    mapping = _parse_legend(g)
    regions = _extract_regions(g)
    out = g.copy()
    filled_mask = np.zeros_like(g, dtype=bool)
    solved = []

    for region in regions:
        seed_color = region["seed_color"]
        assert seed_color in mapping, "seed color must appear in the legend"
        fill_coords = set(_flood_interior(g, region))
        beam_coords = []
        for opening_cell in region["opening"]:
            beam_coords.extend(_lane_cells(g, opening_cell, region["direction"], filled_mask))
        fill_coords.update(beam_coords)
        target_color = mapping[seed_color]
        for rr, cc in fill_coords:
            out[rr, cc] = target_color
            filled_mask[rr, cc] = True
        solved.append(
            {
                "direction": region["direction"],
                "seed_color": seed_color,
                "target_color": target_color,
                "bbox": region["bbox"],
                "opening": list(region["opening"]),
                "interior_coords": list(region["interior_coords"]),
                "filled_coords": list(fill_coords),
                "beam_coords": beam_coords,
            }
        )

    if return_regions:
        return out, solved
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each half-enclosed cavity, project its color through the opening, then repaint it via the legend.

    Input structure:
    - A black background grid has a two-row legend band at the top, where each legend entry is a 2-row by 2-column pair
      separated by grey columns: the left column is a seed color and the right column is its output color.
    - Below the legend sit several grey one-pixel-thick U-shaped walls, each containing exactly one colored seed pixel.
    - Orange singleton obstacles may lie on the straight radiation path outside some openings.

    Rule:
    - For each U-shaped wall, flood-fill the interior background cells starting from the seed, including the cells at the open mouth.
    - Continue the fill straight outward through each opening lane until the grid edge, a grey wall, or a previously filled region stops it.
      Orange obstacles are never filled and block all further fill behind them on that lane.
    - After a region is fully determined, recolor every filled pixel of that region from its seed color to the corresponding legend output color.
    - Keep the legend, grey walls, orange obstacles, and untouched background unchanged.

    Key insight:
    - The seed color is only a lookup key: geometry decides which cells belong to a region, but the final color must be read from the top legend.
    """
    return _solve(grid)


def _legend_width(region_count):
    return 1 + 3 * region_count


def _intervals_ok(start, span, intervals, gap=2):
    end = start + span - 1
    for other_start, other_end in intervals:
        if not (end + gap < other_start or other_end + gap < start):
            return False
    return True


def _sample_grid_shape(rng, axis_mode, region_count):
    legend_w = _legend_width(region_count)
    if axis_mode == "vertical":
        w_min = max(15, legend_w + 2, 5 * region_count + 3)
        w = rng.randint(w_min, min(21, w_min + 4))
        h = rng.randint(14, 19)
    else:
        h_min = max(15, 5 * region_count + 5)
        h = rng.randint(h_min, min(21, h_min + 4))
        w_min = max(15, legend_w + 2)
        w = rng.randint(w_min, min(20, w_min + 4))
    return h, w


def _sample_directions(rng, axis_mode, region_count):
    if axis_mode == "vertical":
        dirs = ["up", "down"]
    else:
        dirs = ["left", "right"]
    chosen = [dirs[0], dirs[1]]
    while len(chosen) < region_count:
        chosen.append(rng.choice(dirs))
    rng.shuffle(chosen)
    return chosen


def _build_region_specs(rng, h, w, axis_mode, directions):
    specs = []
    intervals = []
    if axis_mode == "vertical":
        widths = [rng.randint(1, 2) + 2 for _ in directions]
        order = sorted(range(len(directions)), key=lambda i: widths[i], reverse=True)
        cols = [None] * len(directions)
        for i in order:
            span = widths[i]
            for _ in range(100):
                c0 = rng.randint(1, w - span - 1)
                if _intervals_ok(c0, span, intervals):
                    intervals.append((c0, c0 + span - 1))
                    cols[i] = c0
                    break
            else:
                return None
        for i, direction in enumerate(directions):
            opening_width = widths[i] - 2
            depth = rng.randint(3, 5)
            box_h = depth + 1
            if direction == "up":
                low, high = 4, h - box_h - 1
            else:
                low, high = 3, h - box_h - 2
            if low > high:
                return None
            r0 = rng.randint(low, high)
            specs.append(
                {
                    "direction": direction,
                    "r0": r0,
                    "c0": cols[i],
                    "h": box_h,
                    "w": widths[i],
                    "opening_width": opening_width,
                }
            )
    else:
        heights = [rng.randint(1, 2) + 2 for _ in directions]
        order = sorted(range(len(directions)), key=lambda i: heights[i], reverse=True)
        rows = [None] * len(directions)
        for i in order:
            span = heights[i]
            for _ in range(100):
                r0 = rng.randint(LEGEND_ROWS + 1, h - span - 1)
                if _intervals_ok(r0, span, intervals):
                    intervals.append((r0, r0 + span - 1))
                    rows[i] = r0
                    break
            else:
                return None
        for i, direction in enumerate(directions):
            opening_width = heights[i] - 2
            depth = rng.randint(3, 5)
            box_w = depth + 1
            if direction == "left":
                low, high = 3, w - box_w - 1
            else:
                low, high = 2, w - box_w - 2
            if low > high:
                return None
            c0 = rng.randint(low, high)
            specs.append(
                {
                    "direction": direction,
                    "r0": rows[i],
                    "c0": c0,
                    "h": heights[i],
                    "w": box_w,
                    "opening_width": opening_width,
                }
            )
    return specs


def _draw_u_wall(grid, spec, color):
    r0, c0, h, w = spec["r0"], spec["c0"], spec["h"], spec["w"]
    mask = _u_mask((h, w), spec["direction"])
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = color
    return mask


def _seed_choices(spec):
    r0, c0, h, w = spec["r0"], spec["c0"], spec["h"], spec["w"]
    direction = spec["direction"]
    mask = ~_u_mask((h, w), direction)
    all_choices = [(r0 + rr, c0 + cc) for rr in range(h) for cc in range(w) if mask[rr, cc]]
    if direction == "up":
        preferred = [(rr, cc) for rr, cc in all_choices if rr >= r0 + h - 3]
    elif direction == "down":
        preferred = [(rr, cc) for rr, cc in all_choices if rr <= r0 + 2]
    elif direction == "left":
        preferred = [(rr, cc) for rr, cc in all_choices if cc >= c0 + w - 3]
    else:
        preferred = [(rr, cc) for rr, cc in all_choices if cc <= c0 + 2]
    return preferred or all_choices


def _place_legend(grid, entries, rng):
    legend_w = _legend_width(len(entries))
    start_c = rng.randint(0, grid.shape[1] - legend_w)
    c = start_c
    grid[:LEGEND_ROWS, c] = GREY
    c += 1
    for seed_color, out_color in entries:
        grid[:LEGEND_ROWS, c] = seed_color
        grid[:LEGEND_ROWS, c + 1] = out_color
        c += 2
        grid[:LEGEND_ROWS, c] = GREY
        c += 1


def _candidate_obstacle_lanes(grid, regions):
    lanes = []
    blank_mask = np.zeros_like(grid, dtype=bool)
    for idx, region in enumerate(regions):
        lane_lists = []
        for opening in region["opening"]:
            cells = _lane_cells(grid, opening, region["direction"], blank_mask)
            if len(cells) >= 3:
                lane_lists.append(cells)
        if lane_lists:
            lanes.append((idx, lane_lists))
    return lanes


def _shadow_visible(region):
    opening = set(region["opening"])
    beam = [coord for coord in region["beam_coords"] if coord not in opening]
    return len(beam) > 0


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        region_count = rng.choices([2, 3], weights=[3, 2])[0]
        axis_mode = rng.choice(["vertical", "horizontal"])
        h, w = _sample_grid_shape(rng, axis_mode, region_count)
        directions = _sample_directions(rng, axis_mode, region_count)
        specs = _build_region_specs(rng, h, w, axis_mode, directions)
        if specs is None:
            continue

        palette = rng.sample(PAINT_COLORS, 2 * region_count)
        seed_colors = palette[:region_count]
        out_colors = palette[region_count:]
        entries = list(zip(seed_colors, out_colors))
        rng.shuffle(entries)

        grid = np.zeros((h, w), dtype=int)
        _place_legend(grid, entries, rng)
        color_map = dict(entries)

        for spec in specs:
            _draw_u_wall(grid, spec, GREY)

        spec_order = specs[:]
        rng.shuffle(spec_order)
        assigned_seed_colors = seed_colors[:]
        rng.shuffle(assigned_seed_colors)
        for spec, seed_color in zip(spec_order, assigned_seed_colors):
            choices = _seed_choices(spec)
            if not choices:
                break
            rr, cc = rng.choice(choices)
            grid[rr, cc] = seed_color
        else:
            regions = _extract_regions(grid)
            if len(regions) != region_count:
                continue
            if sorted(region["seed_color"] for region in regions) != sorted(seed_colors):
                continue

            lanes = _candidate_obstacle_lanes(grid, regions)
            if not lanes:
                continue
            obstacle_cells = []
            rng.shuffle(lanes)
            lane_pool = []
            for _idx, lane_lists in lanes:
                rng.shuffle(lane_lists)
                lane_pool.extend(lane_lists)
            if not lane_pool:
                continue
            first_lane = lane_pool[0]
            first_pos = rng.randint(1, len(first_lane) - 2)
            obstacle_cells.append(first_lane[first_pos])
            if len(lane_pool) > 1 and rng.random() < 0.6:
                second_lane = lane_pool[1]
                second_pos = rng.randint(1, len(second_lane) - 2)
                obstacle_cells.append(second_lane[second_pos])
            for rr, cc in obstacle_cells:
                grid[rr, cc] = ORANGE

            out, solved = _solve(grid, return_regions=True)
            if len(solved) != region_count:
                continue
            if np.count_nonzero(out != grid) < 14:
                continue
            if np.count_nonzero(out != grid) > (h * w) // 3 + 8:
                continue
            if not all(_shadow_visible(region) for region in solved):
                continue
            if not all(region["target_color"] == color_map[region["seed_color"]] for region in solved):
                continue
            if len({region["direction"] for region in solved}) < 2:
                continue
            return grid

    raise RuntimeError("failed to generate a valid input")
