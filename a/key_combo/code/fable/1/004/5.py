from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox_coords as _arc_bbox_coords
except Exception:
    _arc_bbox_coords = None


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    if _arc_bbox_coords is not None:
        r0, c0, r1, c1 = _arc_bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _partition_spans(divider_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_divider in enumerate(divider_mask.tolist()):
        if not is_divider and start is None:
            start = idx
        elif is_divider and start is not None:
            spans.append((int(start), int(idx)))
            start = None
    if start is not None:
        spans.append((int(start), int(len(divider_mask))))
    return spans


def _majority_color(region: np.ndarray) -> int:
    colors, counts = np.unique(region, return_counts=True)
    idx = int(np.argmax(counts))
    assert int(np.count_nonzero(counts == counts[idx])) == 1
    return int(colors[idx])


def _find_divider_partition(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in np.unique(grid):
        row_mask = np.all(grid == int(color), axis=1)
        col_mask = np.all(grid == int(color), axis=0)
        if not row_mask.any() or not col_mask.any():
            continue
        occurrences = grid == int(color)
        expected = row_mask[:, None] | col_mask[None, :]
        if not np.array_equal(occurrences, expected):
            continue
        row_spans = _partition_spans(row_mask)
        col_spans = _partition_spans(col_mask)
        if row_spans and col_spans:
            candidates.append((int(color), row_spans, col_spans))
    assert len(candidates) == 1
    return candidates[0]


def _analyze_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider, row_spans, col_spans = _find_divider_partition(g)
    R, C = len(row_spans), len(col_spans)
    assert R >= 2 and C >= 2

    region_map = np.zeros((R, C), dtype=int)
    nonuniform_regions: list[tuple[int, int, np.ndarray, int]] = []

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            region = g[r0:r1, c0:c1]
            assert region.size > 0
            assert not np.any(region == divider)
            major = _majority_color(region)
            region_map[rr, cc] = int(major)
            if np.unique(region).size > 1:
                nonuniform_regions.append((int(rr), int(cc), region.copy(), int(major)))

    assert len(nonuniform_regions) == 1
    host_r, host_c, host_region, host_color = nonuniform_regions[0]
    host_h, host_w = host_region.shape

    non_main = np.argwhere(host_region != host_color)
    assert non_main.size > 0
    nr0, nc0, nr1, nc1 = bbox_coords(non_main)

    pr_lo = max(0, nr1 - R + 1)
    pr_hi = min(nr0, host_h - R)
    pc_lo = max(0, nc1 - C + 1)
    pc_hi = min(nc0, host_w - C)
    assert pr_lo <= pr_hi and pc_lo <= pc_hi

    patch_candidates: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    outside_mask = np.ones((host_h, host_w), dtype=bool)
    for pr in range(pr_lo, pr_hi + 1):
        for pc in range(pc_lo, pc_hi + 1):
            outside_mask.fill(True)
            outside_mask[pr : pr + R, pc : pc + C] = False
            if np.any(host_region[outside_mask] != host_color):
                continue
            patch = host_region[pr : pr + R, pc : pc + C]
            diff_mask = patch != region_map
            if int(diff_mask.sum()) != 1:
                continue
            dr, dc = map(int, np.argwhere(diff_mask)[0])
            patch_candidates.append((int(pr), int(pc), patch.copy(), (int(dr), int(dc))))

    assert len(patch_candidates) == 1
    patch_r, patch_c, patch, diff_pos = patch_candidates[0]

    host_r0, host_r1 = row_spans[host_r]
    host_c0, host_c1 = col_spans[host_c]
    return {
        "divider": int(divider),
        "row_spans": row_spans,
        "col_spans": col_spans,
        "region_map": region_map,
        "host": (int(host_r), int(host_c)),
        "host_color": int(host_color),
        "host_region": host_region,
        "host_bbox": (int(host_r0), int(host_c0), int(host_r1), int(host_c1)),
        "patch_top_left": (int(host_r0 + patch_r), int(host_c0 + patch_c)),
        "patch_offset": (int(patch_r), int(patch_c)),
        "patch": patch,
        "diff_pos": (int(diff_pos[0]), int(diff_pos[1])),
    }


def _apply_revision(grid: np.ndarray, analysis: dict[str, Any]) -> np.ndarray:
    out = np.array(grid, copy=True)
    dr, dc = analysis["diff_pos"]
    row_spans = analysis["row_spans"]
    col_spans = analysis["col_spans"]
    patch = analysis["patch"]

    r0, r1 = row_spans[dr]
    c0, c1 = col_spans[dc]
    out[r0:r1, c0:c1] = int(patch[dr, dc])

    pr0, pc0 = analysis["patch_top_left"]
    out[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the divider-partitioned scene into a coarse map, compare it with the hidden miniature, then write the lone correction back to the matching large region.

    Input structure:
    - Full-row and full-column divider lines partition the canvas into an R×C grid of rectangular regions.
    - Every region is uniformly filled by a main color except one host region that also contains an R×C miniature patch.
    - The miniature patch uses the same color vocabulary as the region map and sits entirely inside its host region.

    Rule:
    - Compress each large region to one cell by taking its region main color; the host region contributes its dominant base color, not the overlaid patch colors.
    - Find the unique R×C patch inside the non-uniform host region whose cellwise comparison against the compressed region map has exactly one mismatch.
    - Recolor the corresponding large target region to the patch's mismatching color, while keeping the miniature patch itself unchanged.

    Key insight:
    - The small patch is not the output image; it is a self-referential edited version of the folded region-color map, so the only useful move is to detect the one edited coarse cell and project that change back onto the matching large region.
    """
    analysis = _analyze_grid(grid)
    return _apply_revision(np.asarray(grid, dtype=int), analysis)


def _sample_sizes(count: int, min_size: int) -> list[int]:
    max_size = max(min_size, min(8, 30 - (count - 1) - min_size * (count - 1)))
    for _ in range(300):
        sizes = [random.randint(min_size, max_size) for _ in range(count)]
        if sum(sizes) + (count - 1) <= 30:
            return sizes
    return [min_size] * count


def _spans_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], int]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        spans.append((int(cursor), int(cursor + size)))
        cursor += int(size)
        if idx != len(sizes) - 1:
            cursor += 1
    return spans, int(cursor)


def _boundary_cells(R: int, C: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(R):
        for c in range(C):
            if r in (0, R - 1) or c in (0, C - 1):
                cells.append((int(r), int(c)))
    return cells


def _make_region_map(R: int, C: int, palette: list[int], host: tuple[int, int]) -> np.ndarray:
    host_r, host_c = host
    for _ in range(300):
        pool_size = random.randint(3, min(6, len(palette)))
        pool = random.sample(palette, pool_size)
        host_color = random.choice(pool)
        out = np.zeros((R, C), dtype=int)
        for r in range(R):
            for c in range(C):
                if (r, c) == host:
                    out[r, c] = int(host_color)
                    continue
                choices = pool
                if r in (0, R - 1) or c in (0, C - 1):
                    choices = [color for color in pool if color != host_color]
                out[r, c] = int(random.choice(choices))
        unique, counts = np.unique(out, return_counts=True)
        if len(unique) < 3:
            continue
        if int(counts.max()) < 2:
            continue
        boundary = [int(out[r, c]) for r, c in _boundary_cells(R, C)]
        if len(set(boundary)) < 2:
            continue
        if any(int(out[r, c]) == int(host_color) for r, c in _boundary_cells(R, C)):
            continue
        return out
    raise RuntimeError("failed to build a non-degenerate region map")


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    R = random.randint(3, 4)
    C = random.randint(3, 4)
    row_sizes = _sample_sizes(R, R + 2)
    col_sizes = _sample_sizes(C, C + 2)
    row_spans, H = _spans_from_sizes(row_sizes)
    col_spans, W = _spans_from_sizes(col_sizes)

    divider = random.choice(PALETTE)
    palette = [color for color in PALETTE if color != divider]

    host = (random.randint(1, R - 2), random.randint(1, C - 2))
    region_map = _make_region_map(R, C, palette, host)
    patch = region_map.copy()

    all_cells = [(r, c) for r in range(R) for c in range(C)]
    if random.random() < 0.4:
        diff_pos = host
    else:
        diff_pos = random.choice([cell for cell in all_cells if cell != host])

    old_color = int(region_map[diff_pos])
    new_color = random.choice([color for color in sorted(set(map(int, region_map.flat))) if color != old_color])
    patch[diff_pos] = int(new_color)

    grid = np.full((H, W), divider, dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            grid[r0:r1, c0:c1] = int(region_map[rr, cc])

    host_r0, host_r1 = row_spans[host[0]]
    host_c0, host_c1 = col_spans[host[1]]
    host_h = host_r1 - host_r0
    host_w = host_c1 - host_c0
    patch_r = random.randint(0, host_h - R)
    patch_c = random.randint(0, host_w - C)
    grid[host_r0 + patch_r : host_r0 + patch_r + R, host_c0 + patch_c : host_c0 + patch_c + C] = patch

    expected = grid.copy()
    target_r0, target_r1 = row_spans[diff_pos[0]]
    target_c0, target_c1 = col_spans[diff_pos[1]]
    expected[target_r0:target_r1, target_c0:target_c1] = int(patch[diff_pos])
    expected[host_r0 + patch_r : host_r0 + patch_r + R, host_c0 + patch_c : host_c0 + patch_c + C] = patch

    metadata = {
        "divider": int(divider),
        "row_spans": row_spans,
        "col_spans": col_spans,
        "region_map": region_map,
        "host": (int(host[0]), int(host[1])),
        "patch_top_left": (int(host_r0 + patch_r), int(host_c0 + patch_c)),
        "patch_offset": (int(patch_r), int(patch_c)),
        "patch": patch,
        "diff_pos": (int(diff_pos[0]), int(diff_pos[1])),
    }
    return grid, expected, metadata


def _validate_case(grid: np.ndarray, expected: np.ndarray, metadata: dict[str, Any]) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except Exception:
        return False
    if analysis["divider"] != metadata["divider"]:
        return False
    if analysis["row_spans"] != metadata["row_spans"] or analysis["col_spans"] != metadata["col_spans"]:
        return False
    if analysis["host"] != metadata["host"]:
        return False
    if analysis["patch_top_left"] != metadata["patch_top_left"]:
        return False
    if analysis["patch_offset"] != metadata["patch_offset"]:
        return False
    if analysis["diff_pos"] != metadata["diff_pos"]:
        return False
    if not np.array_equal(analysis["region_map"], metadata["region_map"]):
        return False
    if not np.array_equal(analysis["patch"], metadata["patch"]):
        return False
    if int((analysis["patch"] != analysis["region_map"]).sum()) != 1:
        return False
    if not set(map(int, np.unique(analysis["patch"]))).issubset(set(map(int, np.unique(analysis["region_map"])))):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if not np.array_equal(_apply_revision(grid, analysis), expected):
        return False
    if np.array_equal(grid, expected):
        return False

    dr, dc = metadata["diff_pos"]
    row_spans = metadata["row_spans"]
    col_spans = metadata["col_spans"]
    target_mask = np.zeros(grid.shape, dtype=bool)
    tr0, tr1 = row_spans[dr]
    tc0, tc1 = col_spans[dc]
    target_mask[tr0:tr1, tc0:tc1] = True
    changed = grid != expected
    if metadata["diff_pos"] == metadata["host"]:
        pr0, pc0 = metadata["patch_top_left"]
        patch_mask = np.zeros(grid.shape, dtype=bool)
        patch_mask[pr0 : pr0 + analysis["patch"].shape[0], pc0 : pc0 + analysis["patch"].shape[1]] = True
        if not np.any(changed & target_mask & ~patch_mask):
            return False
        if np.any(changed & ~target_mask):
            return False
    else:
        if not np.any(changed & target_mask):
            return False
        if np.any(changed & ~target_mask):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected, metadata = _build_case()
        if _validate_case(grid, expected, metadata):
            return grid
    raise RuntimeError("failed to generate a valid map-inside-map input")
