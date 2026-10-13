import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_COLOR_POOL = (1, 2, 3, 4)
SEED_COLOR_POOL = (5, 6, 7, 8, 9)

GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
FRAME_MARGIN_RANGE = (1, 2)
TARGET_SPLIT_RANGE = (3, 6)
MIN_CHILD_SPAN = 4
MAX_GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components[next_id] = coords
    return labels, components


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    counts = Counter(int(v) for v in g.ravel() if int(v) != 0)
    seed_colors = {color for color, count in counts.items() if count == 1}
    if seed_colors:
        seed_mask = np.isin(g, list(seed_colors))
    else:
        seed_mask = np.zeros_like(g, dtype=bool)
    wall_mask = (g != 0) & (~seed_mask)
    passable = (g == 0) | seed_mask

    labels, components = _label_components(passable)
    regions: dict[int, dict] = {}
    for comp_id, coords in components.items():
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)
        seeds = [(r, c) for r, c in coords if bool(seed_mask[r, c])]
        seed_color = int(g[seeds[0]]) if len(seeds) == 1 else None
        regions[comp_id] = {
            "coords": coords,
            "touches_border": touches_border,
            "is_closed": not touches_border,
            "seed_positions": seeds,
            "seed_color": seed_color,
            "area": len(coords),
        }
    return {
        "grid": g,
        "seed_mask": seed_mask,
        "wall_mask": wall_mask,
        "labels": labels,
        "regions": regions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seeded closed regions and recolor the partition lines by the largest filled region they touch.

    Input structure:
    - The background is black.
    - One-pixel-wide orthogonal line segments partition the grid into closed compartments.
    - Some closed compartments contain exactly one uniquely colored seed pixel; other compartments contain no seed.
    - Seed colors never appear on the line network.

    Rule:
    - Treat the line network as walls and the seed pixels as part of the enclosed empty space.
    - For each closed region, if it contains a seed pixel, fill the whole region with that seed color; unseeded closed regions stay black.
    - Then recolor each wall pixel to the fill color of the largest filled closed region adjacent to it; if a wall pixel touches no filled region, erase it to black.

    Key insight:
    - The task has two separate inferences: first identify enclosed regions using the line network as barriers, then decide each line color from neighboring filled-region areas rather than from its original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    analysis = _analyze_grid(g)
    labels = analysis["labels"]
    wall_mask = analysis["wall_mask"]
    regions = analysis["regions"]

    out = g.copy()
    for comp_id, info in regions.items():
        if not info["is_closed"] or info["seed_color"] is None:
            continue
        for r, c in info["coords"]:
            out[r, c] = int(info["seed_color"])

    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not bool(wall_mask[r, c]):
                continue
            adjacent_filled = set()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    comp_id = int(labels[nr, nc])
                    if comp_id == 0:
                        continue
                    info = regions[comp_id]
                    if info["is_closed"] and info["seed_color"] is not None:
                        adjacent_filled.add(comp_id)
            if not adjacent_filled:
                out[r, c] = 0
                continue
            best_id = max(adjacent_filled, key=lambda comp_id: (regions[comp_id]["area"], -comp_id))
            out[r, c] = int(regions[best_id]["seed_color"])

    return out


def _iter_segment_pixels(segment: tuple[str, int, int, int]):
    orient, fixed, start, end = segment
    if orient == "h":
        for c in range(start, end + 1):
            yield fixed, c
    else:
        for r in range(start, end + 1):
            yield r, fixed


def _cell_area(cell: tuple[int, int, int, int]) -> int:
    top, bottom, left, right = cell
    return (bottom - top - 1) * (right - left - 1)


def _share_boundary(cell_a: tuple[int, int, int, int], cell_b: tuple[int, int, int, int]) -> bool:
    ta, ba, la, ra = cell_a
    tb, bb, lb, rb = cell_b
    if ra == lb or rb == la:
        overlap = min(ba, bb) - max(ta, tb)
        return overlap >= 2
    if ba == tb or bb == ta:
        overlap = min(ra, rb) - max(la, lb)
        return overlap >= 2
    return False


def _split_cell(
    rng: random.Random, cell: tuple[int, int, int, int]
) -> tuple[tuple[str, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    top, bottom, left, right = cell
    choices = []
    if right - left >= 2 * MIN_CHILD_SPAN:
        choices.append("v")
    if bottom - top >= 2 * MIN_CHILD_SPAN:
        choices.append("h")
    if not choices:
        return None

    orient = rng.choice(choices)
    if orient == "v":
        options = list(range(left + MIN_CHILD_SPAN, right - MIN_CHILD_SPAN + 1))
        fixed = rng.choice(options)
        segment = ("v", fixed, top, bottom)
        left_child = (top, bottom, left, fixed)
        right_child = (top, bottom, fixed, right)
        return segment, left_child, right_child

    options = list(range(top + MIN_CHILD_SPAN, bottom - MIN_CHILD_SPAN + 1))
    fixed = rng.choice(options)
    segment = ("h", fixed, left, right)
    top_child = (top, fixed, left, right)
    bottom_child = (fixed, bottom, left, right)
    return segment, top_child, bottom_child


def _cell_can_split(cell: tuple[int, int, int, int]) -> bool:
    top, bottom, left, right = cell
    return (right - left >= 2 * MIN_CHILD_SPAN) or (bottom - top >= 2 * MIN_CHILD_SPAN)


def _build_partition(
    rng: random.Random, h: int, w: int
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]] | None:
    top = rng.randint(*FRAME_MARGIN_RANGE)
    left = rng.randint(*FRAME_MARGIN_RANGE)
    bottom = h - 1 - rng.randint(*FRAME_MARGIN_RANGE)
    right = w - 1 - rng.randint(*FRAME_MARGIN_RANGE)
    if bottom - top < 10 or right - left < 10:
        return None

    cells = [(top, bottom, left, right)]
    segments = [
        ("h", top, left, right),
        ("h", bottom, left, right),
        ("v", left, top, bottom),
        ("v", right, top, bottom),
    ]

    target_splits = rng.randint(*TARGET_SPLIT_RANGE)
    for _ in range(target_splits):
        candidates = [idx for idx, cell in enumerate(cells) if _cell_can_split(cell)]
        if not candidates:
            break
        cell_idx = rng.choice(candidates)
        cell = cells.pop(cell_idx)
        split = _split_cell(rng, cell)
        if split is None:
            cells.append(cell)
            continue
        segment, child_a, child_b = split
        segments.append(segment)
        cells.extend([child_a, child_b])

    if len(cells) < 4:
        return None
    return cells, segments


def _seeded_cells(
    rng: random.Random, cells: list[tuple[int, int, int, int]]
) -> tuple[list[int], list[int]] | None:
    if len(cells) < 3:
        return None
    adj_pairs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if _share_boundary(cells[i], cells[j]) and _cell_area(cells[i]) != _cell_area(cells[j]):
                adj_pairs.append((i, j))
    if not adj_pairs:
        return None

    max_seed_count = min(len(SEED_COLOR_POOL), len(cells) - 1)
    if max_seed_count < 2:
        return None
    seed_count = rng.randint(2, min(5, max_seed_count))
    first_pair = rng.choice(adj_pairs)
    seeded = {first_pair[0], first_pair[1]}
    remaining = [idx for idx in range(len(cells)) if idx not in seeded]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(seeded) >= seed_count:
            break
        seeded.add(idx)

    unseeded = [idx for idx in range(len(cells)) if idx not in seeded]
    if not unseeded:
        return None
    return sorted(seeded), unseeded


def _draw_grid(
    rng: random.Random,
    h: int,
    w: int,
    cells: list[tuple[int, int, int, int]],
    segments: list[tuple[str, int, int, int]],
) -> np.ndarray | None:
    seeded_info = _seeded_cells(rng, cells)
    if seeded_info is None:
        return None
    seeded_idxs, _ = seeded_info

    grid = np.zeros((h, w), dtype=int)
    wall_palette = list(rng.sample(WALL_COLOR_POOL, rng.randint(2, len(WALL_COLOR_POOL))))
    junction_color = rng.choice(wall_palette)
    segment_colors = [rng.choice(wall_palette) for _ in segments]

    cover: dict[tuple[int, int], list[int]] = {}
    for seg_idx, segment in enumerate(segments):
        for point in _iter_segment_pixels(segment):
            cover.setdefault(point, []).append(seg_idx)

    for (r, c), seg_ids in cover.items():
        if len(seg_ids) >= 2:
            grid[r, c] = junction_color
        else:
            grid[r, c] = segment_colors[seg_ids[0]]

    seed_colors = rng.sample(SEED_COLOR_POOL, len(seeded_idxs))
    for cell_idx, seed_color in zip(seeded_idxs, seed_colors):
        top, bottom, left, right = cells[cell_idx]
        seed_r = rng.randint(top + 2, bottom - 2)
        seed_c = rng.randint(left + 2, right - 2)
        grid[seed_r, seed_c] = seed_color

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    analysis = _analyze_grid(grid)
    regions = analysis["regions"]
    wall_mask = analysis["wall_mask"]
    labels = analysis["labels"]
    g = analysis["grid"]
    h, w = g.shape

    closed_regions = [info for info in regions.values() if info["is_closed"]]
    seeded_regions = [info for info in closed_regions if info["seed_color"] is not None]
    unseeded_regions = [info for info in closed_regions if info["seed_color"] is None]
    if len(seeded_regions) < 2 or len(unseeded_regions) < 1:
        return False
    if any(len(info["seed_positions"]) > 1 for info in closed_regions):
        return False

    have_compare_branch = False
    have_erase_branch = False
    for r in range(h):
        for c in range(w):
            if not bool(wall_mask[r, c]):
                continue
            adjacent = set()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    comp_id = int(labels[nr, nc])
                    if comp_id == 0:
                        continue
                    info = regions[comp_id]
                    if info["is_closed"] and info["seed_color"] is not None:
                        adjacent.add(comp_id)
            if not adjacent:
                have_erase_branch = True
                continue
            areas = sorted((regions[comp_id]["area"] for comp_id in adjacent), reverse=True)
            if len(areas) >= 2 and areas[0] == areas[1]:
                return False
            if len({regions[comp_id]["area"] for comp_id in adjacent}) >= 2:
                have_compare_branch = True

    if not have_compare_branch or not have_erase_branch:
        return False

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    partition = _build_partition(rng, h, w)
    if partition is None:
        return None
    cells, segments = partition
    grid = _draw_grid(rng, h, w, cells, segments)
    if grid is None:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
