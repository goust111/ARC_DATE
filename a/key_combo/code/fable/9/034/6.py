from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, paste


BG = 0
MARKER = 8
PAYLOAD_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
MARKER_SIZE = 2
MAX_ATTEMPTS = 400


def _paired_marker_blocks(
    grid: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = int(get_bg(grid))
    candidates: list[list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]] = []

    for color in colors_present(grid, exclude={bg}):
        blocks = []
        for comp in find_components(grid, bg=bg, colors=[int(color)], connectivity=4):
            box = comp["bbox"]
            if int(comp["area"]) != 4 or bbox_h(box) != 2 or bbox_w(box) != 2:
                blocks = []
                break
            blocks.append(tuple(map(int, box)))

        if not blocks or len(blocks) % 2 != 0:
            continue

        by_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for box in blocks:
            by_span.setdefault((int(box[0]), int(box[2])), []).append(box)

        pairs = []
        gap_widths = set()
        for span in sorted(by_span):
            row_blocks = sorted(by_span[span], key=lambda box: int(box[1]))
            if len(row_blocks) != 2:
                pairs = []
                break
            left, right = row_blocks
            gap_width = int(right[1] - left[3] - 1)
            if gap_width <= 0:
                pairs = []
                break
            pairs.append((left, right))
            gap_widths.add(gap_width)

        if pairs and len(gap_widths) == 1:
            candidates.append(pairs)

    assert len(candidates) == 1, "expected exactly one valid marker color"
    return candidates[0]


def _overlay_regions(regions: list[np.ndarray], order: tuple[int, ...] | list[int], bg: int) -> np.ndarray:
    out = np.full(regions[0].shape, int(bg), dtype=int)
    for idx in order:
        paste(out, regions[int(idx)], 0, 0, transparent=int(bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the marker-bracketed regions and overlay them by increasing foreground density.

    Input structure:
    - The grid has a dominant background, sparse noise, and exactly one marker color whose
      connected components are all solid 2x2 blocks.
    - These marker blocks occur as left/right pairs on shared two-row spans; the rectangle
      strictly between the blocks is the same size for every pair.

    Rule:
    - Crop the rectangle between each paired marker block, excluding the markers themselves.
    - Count each cropped region's non-background pixels, sort the regions from least dense to
      most dense, and overlay them onto one canvas of that common size.
    - During overlay, a later region writes only its non-background pixels and therefore
      replaces earlier painted pixels at those cells.

    Key insight:
    - The marker pairs only tell you which equal-size regions to extract; the final stacking
      order is not spatial but must be inferred from each region's own foreground density.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    pairs = _paired_marker_blocks(g)

    regions = []
    shapes = set()
    counts = []
    for left, right in pairs:
        region = g[int(left[0]) : int(left[2]) + 1, int(left[3]) + 1 : int(right[1])].copy()
        assert region.shape[0] > 0 and region.shape[1] > 0
        regions.append(region)
        shapes.add(tuple(map(int, region.shape)))
        counts.append(int(np.count_nonzero(region != bg)))

    assert len(regions) >= 1 and len(shapes) == 1
    assert len(counts) == len(set(counts)), "region foreground densities must be unique"
    order = tuple(int(i) for i in np.argsort(np.array(counts, dtype=int), kind="stable"))
    return _overlay_regions(regions, order, bg)


def _gen_row_starts(rng: np.random.Generator, height: int, num_pairs: int) -> list[int] | None:
    if num_pairs == 1:
        return [int(rng.integers(0, height - MARKER_SIZE + 1))]

    gaps = rng.integers(1, 4, size=num_pairs - 1)
    used = MARKER_SIZE * num_pairs + int(gaps.sum())
    spare = height - used
    if spare < 0:
        return None

    row = int(rng.integers(0, spare + 1))
    starts = [row]
    for gap in gaps:
        row += MARKER_SIZE + int(gap)
        starts.append(int(row))
    return starts


def _choose_layout(
    rng: np.random.Generator,
) -> tuple[int, int, int, list[int], list[int]] | None:
    num_pairs = int(rng.integers(2, 5))
    min_gap_width = 6 if num_pairs == 4 else 4
    gap_width = int(rng.integers(min_gap_width, 8))
    height = int(rng.integers(10, 22))
    width_min = max(13, gap_width + 7)
    width = int(rng.integers(width_min, 25))

    row_starts = _gen_row_starts(rng, height, num_pairs)
    if row_starts is None:
        return None

    max_left = width - (gap_width + 4)
    if max_left < 0:
        return None
    left_cols = [int(v) for v in rng.choice(np.arange(max_left + 1), size=num_pairs, replace=False)]
    return gap_width, height, width, sorted(row_starts), left_cols


def _sample_counts(rng: np.random.Generator, area: int, mandatory_sizes: list[int]) -> list[int] | None:
    counts = []
    prev = 0
    total = len(mandatory_sizes)
    for idx, need in enumerate(mandatory_sizes):
        remaining = total - idx - 1
        lo = max(int(need), prev + 1)
        hi = area - remaining
        if lo > hi:
            return None
        counts.append(int(rng.integers(lo, hi + 1)))
        prev = counts[-1]
    if len(set(counts)) != len(counts):
        return None
    return counts


def _coords_from_indices(width: int, indices: list[int]) -> list[tuple[int, int]]:
    return [(int(idx) // width, int(idx) % width) for idx in indices]


def _build_density_regions(
    rng: np.random.Generator, num_pairs: int, gap_width: int
) -> tuple[list[np.ndarray], list[int]] | None:
    area = 2 * gap_width
    mandatory_sizes = [2] + [3] * max(0, num_pairs - 2) + [2]
    counts = _sample_counts(rng, area, mandatory_sizes)
    if counts is None:
        return None

    base_indices = list(range(area))
    rng.shuffle(base_indices)
    unique_cells = _coords_from_indices(gap_width, base_indices[:num_pairs])
    overlap_cells = [None]
    overlap_cells.extend(_coords_from_indices(gap_width, base_indices[num_pairs : 2 * num_pairs - 1]))

    regions: list[np.ndarray] = []
    overlap_colors: dict[tuple[int, int], int] = {}
    for rank, target_count in enumerate(counts):
        mandatory = {unique_cells[rank]}
        if rank > 0:
            mandatory.add(overlap_cells[rank])
        if rank < num_pairs - 1:
            mandatory.add(overlap_cells[rank + 1])

        forbidden = set(unique_cells[:rank])
        if mandatory & forbidden:
            return None
        available = [(r, c) for r in range(2) for c in range(gap_width) if (r, c) not in forbidden]
        if target_count > len(available) or len(mandatory) > target_count:
            return None

        extras_pool = [coord for coord in available if coord not in mandatory]
        extra_count = target_count - len(mandatory)
        extra_indices = rng.choice(len(extras_pool), size=extra_count, replace=False) if extra_count else []
        chosen = set(mandatory)
        for idx in np.atleast_1d(extra_indices):
            if extra_count == 0:
                break
            chosen.add(extras_pool[int(idx)])

        region = np.full((2, gap_width), BG, dtype=int)
        chosen_list = list(chosen)
        rng.shuffle(chosen_list)
        for coord in chosen_list:
            color = int(rng.choice(PAYLOAD_COLORS))
            previous = overlap_colors.get(coord)
            while previous is not None and color == previous:
                color = int(rng.choice(PAYLOAD_COLORS))
            region[int(coord[0]), int(coord[1])] = color
        for coord in mandatory:
            overlap_colors[coord] = int(region[int(coord[0]), int(coord[1])])
        regions.append(region)

    return regions, counts


def _simulate_overlay(
    regions: list[np.ndarray], order: tuple[int, ...], bg: int
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    out = np.full(regions[0].shape, int(bg), dtype=int)
    owner = np.full(regions[0].shape, -1, dtype=int)
    overwrites: list[bool] = []
    for idx in order:
        idx = int(idx)
        region = regions[idx]
        mask = region != bg
        before = out.copy()
        overwrite_mask = mask & (before != bg) & (before != region)
        overwrites.append(bool(np.any(overwrite_mask)))
        out[mask] = region[mask]
        owner[mask] = idx
    return out, owner, overwrites


def _semantic_ok(regions: list[np.ndarray], bg: int) -> bool:
    counts = [int(np.count_nonzero(region != bg)) for region in regions]
    if len(counts) != len(set(counts)):
        return False

    density_order = tuple(int(i) for i in np.argsort(np.array(counts, dtype=int), kind="stable"))
    top_down = tuple(range(len(regions)))
    if density_order == top_down:
        return False

    final_out, owner, overwrites = _simulate_overlay(regions, density_order, bg)
    if not all(np.any(owner == idx) for idx in range(len(regions))):
        return False
    if len(regions) > 1 and not all(overwrites[i] for i in range(1, len(overwrites))):
        return False
    if np.array_equal(final_out, _overlay_regions(regions, top_down, bg)):
        return False

    matches = 0
    for perm in permutations(range(len(regions))):
        if np.array_equal(_overlay_regions(regions, perm, bg), final_out):
            matches += 1
            if perm != density_order:
                return False
    return matches == 1


def _scatter_noise(
    rng: np.random.Generator, grid: np.ndarray, protected: np.ndarray, used_colors: set[int]
) -> bool:
    free = [tuple(map(int, coord)) for coord in np.argwhere(~protected)]
    rng.shuffle(free)
    if len(free) < len(PAYLOAD_COLORS):
        return False

    occupied_noise: dict[tuple[int, int], int] = {}

    def safe_for_color(coord: tuple[int, int], color: int) -> bool:
        r, c = coord
        for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
                if occupied_noise.get((rr, cc)) == color:
                    return False
        return True

    placed = 0
    for color in PAYLOAD_COLORS:
        for coord in free:
            if grid[coord] != BG or not safe_for_color(coord, int(color)):
                continue
            grid[coord] = int(color)
            occupied_noise[coord] = int(color)
            placed += 1
            break
        else:
            return False

    extra_target = int(rng.integers(0, 7))
    candidate_colors = sorted(set(map(int, used_colors)) | set(map(int, PAYLOAD_COLORS)))
    for _ in range(extra_target):
        rng.shuffle(free)
        for coord in free:
            if grid[coord] != BG:
                continue
            color = int(rng.choice(candidate_colors))
            if not safe_for_color(coord, color):
                continue
            grid[coord] = color
            occupied_noise[coord] = color
            break

    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[BG]) > int(max(counts[1:]))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    layout = _choose_layout(rng)
    if layout is None:
        return None
    gap_width, height, width, row_starts, left_cols = layout
    num_pairs = len(row_starts)

    built = _build_density_regions(rng, num_pairs, gap_width)
    if built is None:
        return None
    density_regions, _density_counts = built

    perm = tuple(int(v) for v in rng.permutation(num_pairs))
    if perm == tuple(range(num_pairs)):
        return None
    regions = [density_regions[int(rank)] for rank in perm]
    if not _semantic_ok(regions, BG):
        return None

    grid = np.full((height, width), BG, dtype=int)
    protected = np.zeros((height, width), dtype=bool)
    used_colors = {int(v) for region in regions for v in region.ravel() if int(v) != BG}

    for row, left, region in zip(row_starts, left_cols, regions):
        right = int(left) + MARKER_SIZE + gap_width
        grid[int(row) : int(row) + MARKER_SIZE, int(left) : int(left) + MARKER_SIZE] = MARKER
        grid[int(row) : int(row) + MARKER_SIZE, int(left) + MARKER_SIZE : right] = region
        grid[int(row) : int(row) + MARKER_SIZE, right : right + MARKER_SIZE] = MARKER
        protected[int(row) : int(row) + MARKER_SIZE, int(left) : right + MARKER_SIZE] = True

    if not _scatter_noise(rng, grid, protected, used_colors):
        return None

    expected = main(grid)
    if expected.shape != (2, gap_width):
        return None
    if not _semantic_ok(
        [
            grid[int(row) : int(row) + MARKER_SIZE, int(left) + MARKER_SIZE : int(left) + MARKER_SIZE + gap_width].copy()
            for row, left in zip(row_starts, left_cols)
        ],
        BG,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid density-overlay ARC input")
