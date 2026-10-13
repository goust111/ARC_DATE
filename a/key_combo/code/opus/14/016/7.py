from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEAL = 8
MARKER_COLORS = (3, 4, 5, 6, 7, 9)

LAYOUTS = ((2, 2), (2, 3), (3, 2))
SEGMENT_MIN = 5
SEGMENT_MAX = 8
GEN_ATTEMPTS = 400
ASSIGNMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_even_cells(height: int, width: int, mr: int, mc: int) -> int:
    count = 0
    for r in range(height):
        for c in range(width):
            if max(abs(r - mr), abs(c - mc)) % 2 == 0:
                count += 1
    return count


def _region_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for coords in _components(g != RED):
        colors = sorted({int(g[r, c]) for r, c in coords})
        marker_colors = [color for color in colors if color not in (BLACK, RED)]
        assert len(marker_colors) == 1, "each region must contain exactly one marker color"
        marker_color = int(marker_colors[0])
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == marker_color]
        assert len(marker_cells) == 1, "each region must contain exactly one marker pixel"
        mr, mc = marker_cells[0]
        filled = [
            (int(r), int(c))
            for r, c in coords
            if max(abs(int(r) - mr), abs(int(c) - mc)) % 2 == 0
        ]
        r0, c0, r1, c1 = _bbox(coords)
        infos.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
                "marker_color": marker_color,
                "marker": (mr, mc),
                "filled": filled,
                "filled_count": len(filled),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Draw clipped concentric square rings in each divider-bounded region, then recolor the regions with the most and fewest ring pixels.

    Input structure:
    - Black background is partitioned into rectangular regions by full red horizontal and vertical divider lines.
    - Each non-red region contains exactly one non-black, non-red marker pixel, and the marker is not on that region's boundary.
    - Marker colors are distinct across regions, while divider lines remain single-pixel wide.

    Rule:
    - For each region independently, compute every region cell's Chebyshev distance to that region's marker.
    - Color the cells at even distances, including the marker itself, with the marker color; leave odd-distance region cells black.
    - Count the colored cells in each region, keep red divider lines unchanged, recolor the unique maximum-count region's colored cells blue, and recolor the unique minimum-count region's colored cells teal.

    Key insight:
    - The decisive statistic is the number of even-distance cells after the square-ring pattern is clipped by the enclosing region, so marker eccentricity can outweigh raw region area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    out[g == RED] = RED
    infos = _region_infos(g)

    counts = [info["filled_count"] for info in infos]
    assert len(counts) >= 4
    assert len(set(counts)) == len(counts), "ring-pixel counts must be unique"

    max_idx = int(np.argmax(counts))
    min_idx = int(np.argmin(counts))

    for idx, info in enumerate(infos):
        if idx == max_idx:
            color = BLUE
        elif idx == min_idx:
            color = TEAL
        else:
            color = int(info["marker_color"])
        for r, c in info["filled"]:
            out[r, c] = color
    return out


def _sample_segments(count: int) -> list[int] | None:
    sizes = [random.randint(SEGMENT_MIN, SEGMENT_MAX) for _ in range(count)]
    total = sum(sizes) + count - 1
    if total > 30:
        return None
    return sizes


def _region_shapes(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int]]:
    return [(int(h), int(w)) for h in row_sizes for w in col_sizes]


def _marker_options(height: int, width: int) -> list[dict]:
    options = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            options.append(
                {
                    "pos": (int(r), int(c)),
                    "count": _count_even_cells(height, width, r, c),
                    "bias": min(r, c, height - 1 - r, width - 1 - c),
                }
            )
    options.sort(key=lambda item: (int(item["count"]), -int(item["bias"]), item["pos"]))
    return options


def _choice_pool(options: list[dict], mode: str) -> list[dict]:
    n = len(options)
    third = max(1, n // 3)
    half = max(1, n // 2)
    if mode == "top":
        return options[-third:]
    if mode == "bottom":
        return options[:third]
    if mode == "upper_half":
        return options[-half:]
    if mode == "lower_half":
        return options[:half]
    return options


def _choose_markers(areas: list[int], options_by_region: list[list[dict]]) -> list[tuple[int, int]] | None:
    n = len(areas)
    largest_area_idx = int(np.argmax(areas))
    smallest_area_idx = int(np.argmin(areas))

    high_target = max(
        (idx for idx in range(n) if idx != largest_area_idx),
        key=lambda idx: max(int(opt["count"]) for opt in options_by_region[idx]),
    )
    low_candidates = sorted(
        (idx for idx in range(n) if idx != smallest_area_idx),
        key=lambda idx: min(int(opt["count"]) for opt in options_by_region[idx]),
    )
    low_target = int(low_candidates[0])
    if low_target == high_target and len(low_candidates) > 1:
        low_target = int(low_candidates[1])

    for _ in range(ASSIGNMENT_ATTEMPTS):
        chosen: list[dict] = []
        counts: list[int] = []
        for idx, options in enumerate(options_by_region):
            mode = "any"
            if idx == high_target:
                mode = "top"
            elif idx == low_target:
                mode = "bottom"
            elif idx == largest_area_idx:
                mode = "lower_half"
            elif idx == smallest_area_idx:
                mode = "upper_half"
            pool = _choice_pool(options, mode)
            if random.random() < 0.2:
                pool = options
            pick = random.choice(pool)
            chosen.append(pick)
            counts.append(int(pick["count"]))

        if len(set(counts)) != n:
            continue

        max_idx = int(np.argmax(counts))
        min_idx = int(np.argmin(counts))
        if max_idx == largest_area_idx or min_idx == smallest_area_idx:
            continue
        if max_idx == min_idx:
            continue
        if max(counts) - min(counts) < 3:
            continue
        if not any(areas[i] > areas[j] and counts[i] < counts[j] for i in range(n) for j in range(n)):
            continue
        if not any(int(pick["bias"]) == 1 for pick in chosen):
            continue
        if not any(int(pick["bias"]) >= 2 for pick in chosen):
            continue
        return [tuple(map(int, pick["pos"])) for pick in chosen]
    return None


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    markers: list[tuple[int, int]],
    marker_colors: list[int],
) -> np.ndarray:
    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    grid = np.zeros((h, w), dtype=int)

    split_r = 0
    for size in row_sizes[:-1]:
        split_r += size
        grid[split_r, :] = RED
        split_r += 1

    split_c = 0
    for size in col_sizes[:-1]:
        split_c += size
        grid[:, split_c] = RED
        split_c += 1

    region_idx = 0
    top = 0
    for rh in row_sizes:
        left = 0
        for cw in col_sizes:
            local_r, local_c = markers[region_idx]
            color = marker_colors[region_idx]
            grid[top + local_r, left + local_c] = int(color)
            left += cw + 1
            region_idx += 1
        top += rh + 1
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        infos = _region_infos(grid)
        if not (4 <= len(infos) <= 6):
            return False
        if any(info["area"] != (info["bbox"][2] - info["bbox"][0] + 1) * (info["bbox"][3] - info["bbox"][1] + 1) for info in infos):
            return False
        if len({int(info["marker_color"]) for info in infos}) != len(infos):
            return False
        counts = [int(info["filled_count"]) for info in infos]
        areas = [int(info["area"]) for info in infos]
        if len(set(counts)) != len(counts):
            return False
        if int(np.argmax(counts)) == int(np.argmax(areas)):
            return False
        if int(np.argmin(counts)) == int(np.argmin(areas)):
            return False
        if not any(areas[i] > areas[j] and counts[i] < counts[j] for i in range(len(infos)) for j in range(len(infos))):
            return False
        out = main(grid)
        if out.shape != grid.shape or not np.array_equal(out[grid == RED], np.full(np.count_nonzero(grid == RED), RED, dtype=int)):
            return False
    except AssertionError:
        return False
    return True


def _generate_once() -> np.ndarray | None:
    row_count, col_count = random.choice(LAYOUTS)
    row_sizes = _sample_segments(row_count)
    col_sizes = _sample_segments(col_count)
    if row_sizes is None or col_sizes is None:
        return None

    shapes = _region_shapes(row_sizes, col_sizes)
    areas = [h * w for h, w in shapes]
    if len(set(areas)) != len(areas):
        return None

    options_by_region = [_marker_options(h, w) for h, w in shapes]
    if any(not options for options in options_by_region):
        return None

    markers = _choose_markers(areas, options_by_region)
    if markers is None:
        return None

    region_count = len(shapes)
    marker_colors = random.sample(MARKER_COLORS, region_count)
    grid = _build_grid(row_sizes, col_sizes, markers, marker_colors)
    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring region puzzle")
