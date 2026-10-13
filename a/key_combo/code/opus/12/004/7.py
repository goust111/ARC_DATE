from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from itertools import product

import numpy as np

try:
    from arc_utils import bbox, crop_bbox, scale_sprite
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    bbox = _ARC_UTILS.bbox
    crop_bbox = _ARC_UTILS.crop_bbox
    scale_sprite = _ARC_UTILS.scale_sprite


BG = 0
BLUE = 1
RED = 2
GREEN = 3
SEP = 5
PALETTE = (BLUE, RED, GREEN)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

WIDTH_RANGE = (16, 24)
HEIGHT_OPTIONS = (
    (2, 3, 4),
    (2, 4, 3),
    (3, 2, 4),
    (3, 4, 2),
    (4, 2, 3),
    (4, 3, 2),
    (3, 3, 4),
    (3, 4, 3),
    (4, 3, 3),
)
MAX_SHAPE_PIXELS = 13
GEN_ATTEMPTS = 600


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]


def _region_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    bounds = [-1] + _separator_rows(grid) + [grid.shape[0]]
    regions = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            regions.append((a + 1, b))
    return regions


def _component_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    seen = set()
    q = [tuple(map(int, coords[0]))]
    seen.add(q[0])
    out = []
    while q:
        r, c = q.pop()
        out.append((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
                and mask[nr, nc]
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _select_region_and_crop(grid: np.ndarray) -> tuple[np.ndarray, int]:
    regions = _region_bounds(grid)
    assert len(regions) == 3, f"expected 3 regions, found {len(regions)}"

    densities = []
    crops = []
    for r0, r1 in regions:
        region = grid[r0:r1, :]
        mask = (region != BG) & (region != SEP)
        count = int(mask.sum())
        assert count > 0, "each region must contain one motif"
        b = bbox(mask)
        assert b is not None
        densities.append(count / float(region.size))
        crops.append(crop_bbox(region, b))

    winner = int(np.argmax(densities))
    return crops[winner], winner


def _stack_wave(unit: np.ndarray) -> np.ndarray:
    h, w = unit.shape
    out = np.full((4 * h - 3, w), BG, dtype=int)
    flipped = np.flipud(unit)
    for k in range(4):
        patch = unit if k % 2 == 0 else flipped
        r0 = k * (h - 1)
        view = out[r0 : r0 + h, :]
        mask = patch != BG
        view[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the densest partition motif, enlarge it, and stack alternating vertical flips into a wave.

    Input structure:
    - The background is black, and two full grey separator rows divide the grid into three horizontal regions.
    - Each region contains exactly one 4-connected monochrome motif in red, blue, or green on black.
    - Motifs sit on the bottom edge of their own region, and each region may have a different height.

    Rule:
    - Split the grid at the grey separator rows and compute each region's density as non-black, non-grey pixels divided by the full region area.
    - Choose the unique densest region, crop its motif to the tight bounding box while preserving black gaps inside the box, and scale that crop by 2x in both directions.
    - Build the output from four overlapping copies of the scaled crop: original, vertically flipped, original, vertically flipped, with consecutive copies overlapping by exactly one row and black treated as transparent.

    Key insight:
    - The separator rows define the candidate source regions, but the winner is chosen by density over the whole partition, not by raw pixel count; that winning motif then determines both the scaled unit and the final wave height.
    """
    g = np.asarray(grid, dtype=int)
    crop, _winner = _select_region_and_crop(g)
    scaled = scale_sprite(crop, 2)
    return _stack_wave(scaled)


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    opts = set()
    for r, c in cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                opts.add((nr, nc))
    return list(opts)


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_coords(mask)) == int(mask.sum())


def _path_seed(h: int, w: int) -> set[tuple[int, int]]:
    left_to_right = random.choice([False, True])
    if left_to_right:
        r, c = 0, 0
        horizontal_step = 1
    else:
        r, c = 0, w - 1
        horizontal_step = -1
    cells = {(r, c)}
    steps = ["V"] * (h - 1) + ["H"] * (w - 1)
    random.shuffle(steps)
    for step in steps:
        if step == "V":
            r += 1
        else:
            c += horizontal_step
        cells.add((r, c))
    return cells


def _gap_template(h: int, w: int, count: int) -> np.ndarray | None:
    if h != 3 or w < 3:
        return None

    if count == 2 * w + 2:
        mask = np.ones((3, w), dtype=bool)
        mask[1, 1:-1] = False
        return mask

    if count == 2 * w + 1:
        mask = np.zeros((3, w), dtype=bool)
        mask[0, :] = True
        mask[2, :] = True
        mid = random.choice([w // 2, (w - 1) // 2])
        mask[1, mid] = True
        return mask

    return None


def _generate_mask(h: int, w: int, count: int) -> np.ndarray:
    template = _gap_template(h, w, count)
    if template is not None and random.random() < 0.45:
        return template

    for _ in range(200):
        cells = _path_seed(h, w)
        while len(cells) < count:
            frontier = _frontier(cells, h, w)
            if not frontier:
                break
            weights = []
            for r, c in frontier:
                edge_bonus = int(r in (0, h - 1)) + int(c in (0, w - 1))
                degree_bonus = sum((r + dr, c + dc) in cells for dr, dc in DIRS4)
                weights.append(1 + 2 * edge_bonus + degree_bonus)
            cells.add(random.choices(frontier, weights=weights, k=1)[0])

        if len(cells) != count:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _touches_all_edges(mask) and _mask_connected(mask):
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            return mask

    raise RuntimeError(f"failed to generate mask {h}x{w} with {count} pixels")


def _find_count_tuple(
    region_heights: tuple[int, int, int],
    boxes: list[tuple[int, int]],
    desired_winner: int,
    require_counterexample: bool,
) -> tuple[int, int, int] | None:
    ranges = []
    for h, w in boxes:
        lo = h + w - 1
        hi = min(h * w, MAX_SHAPE_PIXELS)
        if lo > hi:
            return None
        ranges.append(range(lo, hi + 1))

    candidates = []
    for counts in product(*ranges):
        if len(set(counts)) != 3:
            continue
        density_scores = [counts[i] / region_heights[i] for i in range(3)]
        max_density = max(density_scores)
        if density_scores.count(max_density) != 1:
            continue
        winner = int(np.argmax(density_scores))
        if winner != desired_winner:
            continue
        count_winner = int(np.argmax(counts))
        if require_counterexample and count_winner == winner:
            continue
        gap = max_density - sorted(density_scores, reverse=True)[1]
        if gap < 0.14:
            continue
        candidates.append(tuple(int(v) for v in counts))

    if not candidates:
        return None
    return random.choice(candidates)


def _sample_specs() -> tuple[int, tuple[int, int, int], list[tuple[int, int]], tuple[int, int, int], int, bool]:
    for _ in range(300):
        width = random.randint(*WIDTH_RANGE)
        region_heights = random.choice(HEIGHT_OPTIONS)
        boxes = []
        for region_h in region_heights:
            box_h = random.randint(2, min(3, region_h))
            box_w = random.randint(2, 6)
            boxes.append((box_h, box_w))

        desired_winner = random.randrange(3)
        require_counterexample = random.random() < 0.6
        counts = _find_count_tuple(region_heights, boxes, desired_winner, require_counterexample)
        if counts is None and require_counterexample:
            require_counterexample = False
            counts = _find_count_tuple(region_heights, boxes, desired_winner, require_counterexample)
        if counts is None:
            continue
        return width, region_heights, boxes, counts, desired_winner, require_counterexample
    raise RuntimeError("failed to sample valid region specs")


def _place_regions(
    width: int,
    region_heights: tuple[int, int, int],
    boxes: list[tuple[int, int]],
    counts: tuple[int, int, int],
) -> np.ndarray:
    total_h = sum(region_heights) + 2
    grid = np.full((total_h, width), BG, dtype=int)

    sep_row = region_heights[0]
    grid[sep_row, :] = SEP
    grid[sep_row + 1 + region_heights[1], :] = SEP

    colors = list(PALETTE)
    random.shuffle(colors)

    region_starts = [0, region_heights[0] + 1, region_heights[0] + region_heights[1] + 2]
    for idx in range(3):
        region_h = region_heights[idx]
        box_h, box_w = boxes[idx]
        count = counts[idx]
        color = colors[idx]
        mask = _generate_mask(box_h, box_w, count)

        r0 = region_starts[idx] + (region_h - box_h)
        c0 = random.randint(1, width - box_w - 1) if width - box_w >= 2 else 0
        patch = np.where(mask, color, BG)
        grid[r0 : r0 + box_h, c0 : c0 + box_w] = patch

    return grid


def _semantic_ok(grid: np.ndarray, require_counterexample: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, BLUE, RED, GREEN, SEP}:
        return False

    sep_rows = _separator_rows(grid)
    if len(sep_rows) != 2:
        return False

    regions = _region_bounds(grid)
    if len(regions) != 3:
        return False

    colors_seen = []
    counts = []
    heights = []
    density_scores = []
    count_winner = None

    for r0, r1 in regions:
        region = grid[r0:r1, :]
        heights.append(r1 - r0)
        mask = (region != BG) & (region != SEP)
        if int(mask.sum()) == 0:
            return False
        coords = _component_coords(mask)
        if len(coords) != int(mask.sum()):
            return False
        palette = sorted(int(v) for v in np.unique(region[mask]))
        if len(palette) != 1:
            return False
        colors_seen.append(palette[0])
        counts.append(int(mask.sum()))
        density_scores.append(int(mask.sum()) / float(region.size))

        bb = bbox(mask)
        if bb is None:
            return False
        br0, _bc0, br1, bc1 = bb
        if br1 != region.shape[0] - 1:
            return False
        if br1 - br0 + 1 > 3 or bc1 - _bc0 + 1 > 6:
            return False

    if set(colors_seen) != set(PALETTE):
        return False
    if len(set(counts)) != 3:
        return False
    if density_scores.count(max(density_scores)) != 1:
        return False
    if require_counterexample and int(np.argmax(counts)) == int(np.argmax(density_scores)):
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    crop, winner = _select_region_and_crop(grid)
    scaled = scale_sprite(crop, 2)
    expected = _stack_wave(scaled)
    if not np.array_equal(out, expected):
        return False

    count_winner = int(np.argmax(counts))
    if require_counterexample and count_winner == winner:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        width, region_heights, boxes, counts, _winner, require_counterexample = _sample_specs()
        grid = _place_regions(width, region_heights, boxes, counts)
        if _semantic_ok(grid, require_counterexample):
            return grid
    raise RuntimeError("failed to generate a valid density-filter wave input")
