from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    return np.pad(arr, ((1, 0), (1, 0)), constant_values=0).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    if top > bottom or left > right:
        return 0
    return int(
        ps[bottom + 1, right + 1]
        - ps[top, right + 1]
        - ps[bottom + 1, left]
        + ps[top, left]
    )


def _boundary_cells(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    cells = [(top, c) for c in range(left, right + 1)]
    if bottom != top:
        cells.extend((bottom, c) for c in range(left, right + 1))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        if right != left:
            cells.append((r, right))
    return cells


def _classify_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in grid.ravel())
    bg = max(counts, key=counts.get)
    other_colors = [color for color in counts if color != bg]
    assert other_colors, "expected wall and seed colors"
    wall = max(other_colors, key=lambda color: counts[color])
    assert counts[wall] > 1, "wall color must be repeated"
    return int(bg), int(wall)


def _find_gap(
    grid: np.ndarray,
    bg: int,
    wall: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> tuple[int, int] | None:
    gap_cells = []
    for r, c in _boundary_cells(top, left, bottom, right):
        val = int(grid[r, c])
        if val == wall:
            continue
        if val != bg:
            return None
        gap_cells.append((r, c))
    if not gap_cells:
        return None
    if len(gap_cells) != 1:
        return None
    gr, gc = gap_cells[0]
    if gr in (top, bottom) and gc in (left, right):
        return None
    return int(gr), int(gc)


def _find_region_for_seed(
    grid: np.ndarray,
    bg: int,
    wall: int,
    wall_ps: np.ndarray,
    seed_ps: np.ndarray,
    seed_r: int,
    seed_c: int,
) -> dict[str, object]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []

    for top in range(seed_r):
        for bottom in range(seed_r + 1, h):
            for left in range(seed_c):
                for right in range(seed_c + 1, w):
                    if not (top < seed_r < bottom and left < seed_c < right):
                        continue

                    interior_seed_count = _rect_sum(seed_ps, top + 1, left + 1, bottom - 1, right - 1)
                    if interior_seed_count != 1:
                        continue

                    interior_wall_count = _rect_sum(wall_ps, top + 1, left + 1, bottom - 1, right - 1)
                    if interior_wall_count != 0:
                        continue

                    height = bottom - top + 1
                    width = right - left + 1
                    perimeter = 2 * (height + width) - 4

                    total_seed_count = _rect_sum(seed_ps, top, left, bottom, right)
                    boundary_seed_count = total_seed_count - interior_seed_count
                    if boundary_seed_count != 0:
                        continue

                    total_wall_count = _rect_sum(wall_ps, top, left, bottom, right)
                    boundary_wall_count = total_wall_count - interior_wall_count
                    boundary_bg_count = perimeter - boundary_wall_count
                    if boundary_bg_count < 0 or boundary_bg_count > 1:
                        continue

                    gap = _find_gap(grid, bg, wall, top, left, bottom, right)
                    if boundary_bg_count == 1 and gap is None:
                        continue
                    if boundary_bg_count == 0 and gap is not None:
                        continue

                    candidates.append(
                        {
                            "top": int(top),
                            "left": int(left),
                            "bottom": int(bottom),
                            "right": int(right),
                            "seed": (int(seed_r), int(seed_c)),
                            "seed_color": int(grid[seed_r, seed_c]),
                            "gap": gap,
                        }
                    )

    assert candidates, f"no valid rectangle found for seed {(seed_r, seed_c)}"
    keys = {(c["top"], c["left"], c["bottom"], c["right"]) for c in candidates}
    assert len(keys) == 1, f"seed {(seed_r, seed_c)} matched multiple regions"
    return candidates[0]


def _extract_regions(grid: np.ndarray, bg: int, wall: int) -> list[dict[str, object]]:
    seed_mask = (grid != bg) & (grid != wall)
    seeds = [tuple(map(int, coord)) for coord in np.argwhere(seed_mask)]
    assert seeds, "expected at least one seed"
    seed_colors = [int(grid[r, c]) for r, c in seeds]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"

    wall_ps = _prefix_sum(grid == wall)
    seed_ps = _prefix_sum(seed_mask)

    regions: list[dict[str, object]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for seed_r, seed_c in seeds:
        region = _find_region_for_seed(grid, bg, wall, wall_ps, seed_ps, seed_r, seed_c)
        box = (int(region["top"]), int(region["left"]), int(region["bottom"]), int(region["right"]))
        assert box not in seen_boxes, "multiple seeds cannot belong to the same region"
        seen_boxes.add(box)
        regions.append(region)

    owner = np.full(grid.shape, -1, dtype=int)
    for idx, region in enumerate(regions):
        top = int(region["top"])
        left = int(region["left"])
        bottom = int(region["bottom"])
        right = int(region["right"])
        patch = owner[top + 1 : bottom, left + 1 : right]
        assert np.all(patch == -1), "region interiors must not overlap"
        patch[:, :] = idx

    for idx, region in enumerate(regions):
        gap = region["gap"]
        if gap is None:
            region["neighbor"] = None
            continue
        gr, gc = gap
        neighbors = []
        for nr, nc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                owner_id = int(owner[nr, nc])
                if owner_id != -1 and owner_id != idx:
                    neighbors.append(owner_id)
        neighbor_ids = sorted(set(neighbors))
        assert len(neighbor_ids) == 1, "each gap must connect exactly two regions"
        region["neighbor"] = neighbor_ids[0]

    for idx, region in enumerate(regions):
        neighbor = region["neighbor"]
        if neighbor is not None:
            assert regions[int(neighbor)]["neighbor"] == idx, "gap borrowing must be reciprocal"

    return regions


def _pattern_type(region: dict[str, object]) -> str:
    top = int(region["top"])
    left = int(region["left"])
    bottom = int(region["bottom"])
    right = int(region["right"])
    seed_r, seed_c = region["seed"]

    inner_h = bottom - top - 1
    inner_w = right - left - 1
    local_r = seed_r - (top + 1)
    local_c = seed_c - (left + 1)

    corners = {
        (0, 0),
        (0, inner_w - 1),
        (inner_h - 1, 0),
        (inner_h - 1, inner_w - 1),
    }
    if (local_r, local_c) in corners:
        return "checker"

    edge_midpoints = set()
    if inner_w % 2 == 1:
        mid_c = inner_w // 2
        edge_midpoints.add((0, mid_c))
        edge_midpoints.add((inner_h - 1, mid_c))
    if inner_h % 2 == 1:
        mid_r = inner_h // 2
        edge_midpoints.add((mid_r, 0))
        edge_midpoints.add((mid_r, inner_w - 1))
    if (local_r, local_c) in edge_midpoints:
        return "horizontal"

    return "vertical"


def _paint_region(
    out: np.ndarray,
    region: dict[str, object],
    fill_color: int,
    bg: int,
) -> None:
    top = int(region["top"])
    left = int(region["left"])
    bottom = int(region["bottom"])
    right = int(region["right"])
    seed_r, seed_c = region["seed"]

    inner_h = bottom - top - 1
    inner_w = right - left - 1
    seed_rr = seed_r - (top + 1)
    seed_cc = seed_c - (left + 1)
    rr, cc = np.indices((inner_h, inner_w))

    pattern = _pattern_type(region)
    if pattern == "checker":
        paint_mask = ((rr + cc) % 2) == ((seed_rr + seed_cc) % 2)
    elif pattern == "horizontal":
        paint_mask = (rr % 2) == (seed_rr % 2)
    else:
        paint_mask = (cc % 2) == (seed_cc % 2)

    patch = out[top + 1 : bottom, left + 1 : right]
    patch[:, :] = bg
    patch[paint_mask] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded rectangle with a seed-coded pattern whose paint color may be borrowed through a wall gap.

    Input structure:
    - Black is the dominant background.
    - Grey one-pixel walls form several rectangular regions that may share walls.
    - Each region interior contains exactly one unique colored seed pixel.
    - Some shared walls contain a single black gap connecting two neighboring regions.

    Rule:
    - Recover each rectangular region bounded by grey walls, allowing at most one black gap on its boundary.
    - Interpret a seed in an interior corner as a checkerboard cue, a seed on an edge midpoint as a horizontal-stripe cue, and any other interior seed as a vertical-stripe cue.
    - If a region has a boundary gap into exactly one neighboring region, paint the pattern with that neighbor's seed color; otherwise paint with the region's own seed color.
    - Fill the whole interior with the chosen phase-aligned pattern so the original seed cell stays colored; keep walls grey and gap cells black.

    Key insight:
    - The seed encodes the pattern family through geometry, while the single wall gap independently reroutes the paint color to the adjacent region's seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, wall = _classify_colors(g)
    regions = _extract_regions(g, bg, wall)

    out = g.copy()
    for region in regions:
        neighbor = region["neighbor"]
        if neighbor is None:
            fill_color = int(region["seed_color"])
        else:
            fill_color = int(regions[int(neighbor)]["seed_color"])
        _paint_region(out, region, fill_color, bg)
    return out


_BG = 0
_WALL = 5
_SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_LAYOUTS = ((1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
_DIM_CHOICES = (3, 4, 5)
_PATTERNS = ("checker", "horizontal", "vertical")
_GEN_ATTEMPTS = 400


def _balanced_pattern_counts(n_cells: int) -> dict[str, int]:
    counts = {pattern: n_cells // 3 for pattern in _PATTERNS}
    extras = list(_PATTERNS)
    random.shuffle(extras)
    for pattern in extras[: n_cells - sum(counts.values())]:
        counts[pattern] += 1
    return counts


def _grid_lines(sizes: list[int], margin: int) -> list[int]:
    lines = [margin]
    for size in sizes:
        lines.append(lines[-1] + size + 1)
    return lines


def _choose_matching(
    edges: list[tuple[tuple[int, int], tuple[int, int], str]],
    pair_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int], str]] | None:
    for _ in range(80):
        used: set[tuple[int, int]] = set()
        picked = []
        shuffled = edges[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            a, b, _axis = edge
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            picked.append(edge)
            if len(picked) == pair_count:
                return picked
    return None


def _available_seed_positions(pattern: str, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    if pattern == "checker":
        return [
            (0, 0),
            (0, inner_w - 1),
            (inner_h - 1, 0),
            (inner_h - 1, inner_w - 1),
        ]

    if pattern == "horizontal":
        positions = []
        if inner_w % 2 == 1:
            mid_c = inner_w // 2
            positions.extend([(0, mid_c), (inner_h - 1, mid_c)])
        if inner_h % 2 == 1:
            mid_r = inner_h // 2
            positions.extend([(mid_r, 0), (mid_r, inner_w - 1)])
        return positions

    positions = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    if len(positions) > 1 and inner_h % 2 == 1 and inner_w % 2 == 1:
        center = (inner_h // 2, inner_w // 2)
        off_center = [pos for pos in positions if pos != center]
        if off_center:
            return off_center
    return positions


def _build_candidate() -> np.ndarray | None:
    rows, cols = random.choice(_LAYOUTS)
    n_cells = rows * cols
    margin = random.randint(1, 2)
    row_sizes = [random.choice(_DIM_CHOICES) for _ in range(rows)]
    col_sizes = [random.choice(_DIM_CHOICES) for _ in range(cols)]

    row_lines = _grid_lines(row_sizes, margin)
    col_lines = _grid_lines(col_sizes, margin)
    height = row_lines[-1] + margin + 1
    width = col_lines[-1] + margin + 1
    if height > 30 or width > 30:
        return None

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    pattern_counts = _balanced_pattern_counts(n_cells)
    horizontal_eligible = [
        cell
        for cell in cells
        if row_sizes[cell[0]] % 2 == 1 or col_sizes[cell[1]] % 2 == 1
    ]
    if len(horizontal_eligible) < pattern_counts["horizontal"]:
        return None

    horizontal_cells = set(random.sample(horizontal_eligible, pattern_counts["horizontal"]))
    remaining = [cell for cell in cells if cell not in horizontal_cells]
    checker_cells = set(random.sample(remaining, pattern_counts["checker"]))

    cell_info: dict[tuple[int, int], dict[str, object]] = {}
    palette = list(_SEED_COLORS)
    random.shuffle(palette)
    for idx, cell in enumerate(cells):
        r, c = cell
        pattern = "vertical"
        if cell in horizontal_cells:
            pattern = "horizontal"
        elif cell in checker_cells:
            pattern = "checker"
        inner_h = row_sizes[r]
        inner_w = col_sizes[c]
        choices = _available_seed_positions(pattern, inner_h, inner_w)
        if not choices:
            return None
        seed_rr, seed_cc = random.choice(choices)
        top = row_lines[r]
        left = col_lines[c]
        cell_info[cell] = {
            "top": top,
            "left": left,
            "bottom": row_lines[r + 1],
            "right": col_lines[c + 1],
            "pattern": pattern,
            "seed_color": palette[idx],
            "seed_offset": (seed_rr, seed_cc),
            "neighbor": None,
            "gap": None,
        }

    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1), "vertical"))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c), "horizontal"))

    pair_count = random.randint(1, max(1, (n_cells - 1) // 2))
    matching = _choose_matching(edges, pair_count)
    if matching is None:
        return None

    grid = np.full((height, width), _BG, dtype=int)
    for r in row_lines:
        grid[r, col_lines[0] : col_lines[-1] + 1] = _WALL
    for c in col_lines:
        grid[row_lines[0] : row_lines[-1] + 1, c] = _WALL

    for cell_a, cell_b, axis in matching:
        info_a = cell_info[cell_a]
        info_b = cell_info[cell_b]
        if axis == "vertical":
            gap_r = random.randint(int(info_a["top"]) + 1, int(info_a["bottom"]) - 1)
            gap_c = int(info_a["right"])
        else:
            gap_r = int(info_a["bottom"])
            gap_c = random.randint(int(info_a["left"]) + 1, int(info_a["right"]) - 1)
        grid[gap_r, gap_c] = _BG
        info_a["neighbor"] = cell_b
        info_b["neighbor"] = cell_a
        info_a["gap"] = (gap_r, gap_c)
        info_b["gap"] = (gap_r, gap_c)

    for info in cell_info.values():
        top = int(info["top"])
        left = int(info["left"])
        seed_rr, seed_cc = info["seed_offset"]
        grid[top + 1 + seed_rr, left + 1 + seed_cc] = int(info["seed_color"])

    closed_count = sum(1 for info in cell_info.values() if info["neighbor"] is None)
    if closed_count == 0:
        return None

    patterns_present = {str(info["pattern"]) for info in cell_info.values()}
    if patterns_present != set(_PATTERNS):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
