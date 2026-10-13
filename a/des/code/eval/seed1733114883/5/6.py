from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

BUCKET_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR4 = tuple(DIR_VECTORS.values())

ROW_COUNT_OPTIONS = (3, 4)
COL_COUNT_OPTIONS = (3, 4)
CELL_SIZE_RANGE = (5, 7)
GRID_SIZE_RANGE = (18, 30)
GEN_RETRIES = 400
_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_arrow(points: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(points) != 3:
        return None
    r0, c0, r1, c1 = _bbox(points)
    norm = {(r - r0, c - c0) for r, c in points}
    patterns: dict[frozenset[tuple[int, int]], tuple[str, tuple[int, int]]] = {
        frozenset({(0, 0), (2, 0), (1, 1)}): ("right", (r0 + 1, c0 + 1)),
        frozenset({(0, 1), (2, 1), (1, 0)}): ("left", (r0 + 1, c0 + 0)),
        frozenset({(0, 0), (0, 2), (1, 1)}): ("down", (r0 + 1, c0 + 1)),
        frozenset({(1, 0), (1, 2), (0, 1)}): ("up", (r0 + 0, c0 + 1)),
    }
    return patterns.get(frozenset(norm))


def _classify_region(grid: np.ndarray, points: list[tuple[int, int]]) -> dict:
    values = [int(grid[r, c]) for r, c in points]
    non_black_points = [(int(r), int(c)) for r, c in points if int(grid[r, c]) != BLACK]
    non_black_values = [int(grid[r, c]) for r, c in non_black_points]

    if not non_black_points:
        return {"kind": "empty"}

    orange_points = [(int(r), int(c)) for r, c in points if int(grid[r, c]) == ORANGE]
    if len(orange_points) == 3 and len(non_black_points) == 3:
        arrow = _detect_arrow(orange_points)
        if arrow is not None:
            direction, tip = arrow
            return {"kind": "arrow", "direction": direction, "tip": tip}

    color_set = sorted(set(non_black_values))
    if len(color_set) == 1 and color_set[0] in BUCKET_COLORS and len(non_black_points) == 4:
        r0, c0, r1, c1 = _bbox(non_black_points)
        if r1 - r0 == 1 and c1 - c0 == 1:
            block = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if set(non_black_points) == block:
                return {"kind": "bucket", "color": int(color_set[0])}

    return {
        "kind": "unknown",
        "colors": tuple(sorted(set(values))),
        "non_black_count": len(non_black_points),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    region_id = -np.ones((h, w), dtype=int)
    regions: list[dict] = []
    for idx, points in enumerate(_components(g != GREY)):
        for r, c in points:
            region_id[r, c] = int(idx)
        info = _classify_region(g, points)
        info["id"] = int(idx)
        info["cells"] = points
        regions.append(info)

    buckets = {int(region["id"]): region for region in regions if region["kind"] == "bucket"}
    arrows = [region for region in regions if region["kind"] == "arrow"]
    unknown = [region for region in regions if region["kind"] == "unknown"]
    assert not unknown, "all regions must be empty, arrow, or bucket"
    assert arrows, "expected at least one arrow region"
    assert buckets, "expected at least one bucket region"

    arrow_links: list[tuple[int, int]] = []
    bucket_target_counts: Counter[int] = Counter()
    for arrow in arrows:
        tip_r, tip_c = map(int, arrow["tip"])
        dr, dc = DIR_VECTORS[str(arrow["direction"])]
        r = int(tip_r + dr)
        c = int(tip_c + dc)

        while 0 <= r < h and 0 <= c < w and int(g[r, c]) != GREY:
            r += int(dr)
            c += int(dc)
        while 0 <= r < h and 0 <= c < w and int(g[r, c]) == GREY:
            r += int(dr)
            c += int(dc)

        assert 0 <= r < h and 0 <= c < w, "arrow must cross into a neighboring region"
        target_region = int(region_id[r, c])
        assert target_region in buckets, "arrow must point to a bucket region"
        arrow_links.append((int(arrow["id"]), target_region))
        bucket_target_counts[target_region] += 1

    return {
        "regions": regions,
        "arrows": arrows,
        "buckets": buckets,
        "arrow_links": arrow_links,
        "bucket_target_counts": bucket_target_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill arrow regions from the adjacent bucket they point to, with an even-count
    override to azure.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    out = g.copy()

    for arrow_region_id, bucket_region_id in scene["arrow_links"]:
        bucket_color = int(scene["buckets"][bucket_region_id]["color"])
        count = int(scene["bucket_target_counts"][bucket_region_id])
        fill_color = bucket_color if count % 2 == 1 else AZURE
        for r, c in scene["regions"][arrow_region_id]["cells"]:
            out[r, c] = int(fill_color)

    return out


def _sample_axis(rng: random.Random, count: int) -> list[int]:
    low, high = CELL_SIZE_RANGE
    total_low, total_high = GRID_SIZE_RANGE
    for _ in range(100):
        sizes = [rng.randint(low, high) for _ in range(count)]
        total = sum(sizes) + count + 1
        if total_low <= total <= total_high:
            return sizes
    raise RuntimeError("failed to sample axis sizes")


def _axis_layout(sizes: list[int]) -> tuple[list[int], list[int], int]:
    starts: list[int] = []
    boundaries = [0]
    pos = 1
    for size in sizes:
        starts.append(int(pos))
        pos += int(size)
        boundaries.append(int(pos))
        pos += 1
    return starts, boundaries, int(boundaries[-1] + 1)


def _build_partition(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]]:
    row_starts, row_boundaries, height = _axis_layout(row_sizes)
    col_starts, col_boundaries, width = _axis_layout(col_sizes)

    grid = np.full((height, width), BLACK, dtype=int)
    for r in row_boundaries:
        grid[int(r), :] = GREY
    for c in col_boundaries:
        grid[:, int(c)] = GREY

    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i, row_start in enumerate(row_starts):
        for j, col_start in enumerate(col_starts):
            boxes[(int(i), int(j))] = (
                int(row_start),
                int(col_start),
                int(row_start + row_sizes[i] - 1),
                int(col_start + col_sizes[j] - 1),
            )
    return grid, boxes


def _candidate_sources(bucket: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], str]]:
    br, bc = bucket
    out: list[tuple[tuple[int, int], str]] = []
    for direction, (dr, dc) in DIR_VECTORS.items():
        source = (int(br - dr), int(bc - dc))
        if 0 <= source[0] < n_rows and 0 <= source[1] < n_cols:
            out.append((source, direction))
    return out


def _choose_primary_groups(
    rng: random.Random, n_rows: int, n_cols: int
) -> list[dict] | None:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    source_map = {cell: _candidate_sources(cell, n_rows, n_cols) for cell in cells}

    odd_buckets = cells[:]
    even_buckets = cells[:]
    rng.shuffle(odd_buckets)
    rng.shuffle(even_buckets)

    for odd_bucket in odd_buckets:
        odd_counts = [k for k in (1, 3) if k <= len(source_map[odd_bucket])]
        rng.shuffle(odd_counts)
        if not odd_counts:
            continue

        for even_bucket in even_buckets:
            if even_bucket == odd_bucket:
                continue
            even_counts = [k for k in (2, 4) if k <= len(source_map[even_bucket])]
            rng.shuffle(even_counts)
            if not even_counts:
                continue

            for odd_count in odd_counts:
                odd_choices = list(combinations(source_map[odd_bucket], odd_count))
                rng.shuffle(odd_choices)
                for odd_choice in odd_choices:
                    odd_sources = {src for src, _ in odd_choice}
                    if even_bucket in odd_sources:
                        continue

                    available_even = [
                        item
                        for item in source_map[even_bucket]
                        if item[0] not in odd_sources and item[0] != odd_bucket
                    ]
                    for even_count in even_counts:
                        if len(available_even) < even_count:
                            continue
                        even_choices = list(combinations(available_even, even_count))
                        rng.shuffle(even_choices)
                        for even_choice in even_choices:
                            used_sources = odd_sources | {src for src, _ in even_choice}
                            groups = [
                                {
                                    "bucket": odd_bucket,
                                    "sources": list(odd_choice),
                                    "parity": "odd",
                                },
                                {
                                    "bucket": even_bucket,
                                    "sources": list(even_choice),
                                    "parity": "even",
                                },
                            ]
                            free_cells = [
                                cell
                                for cell in cells
                                if cell not in used_sources
                                and cell != odd_bucket
                                and cell != even_bucket
                            ]
                            if free_cells:
                                return groups
    return None


def _draw_bucket(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    br = int(r0 + (height - 2) // 2)
    bc = int(c0 + (width - 2) // 2)
    grid[br : br + 2, bc : bc + 2] = int(color)


def _draw_arrow(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    direction: str,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = box
    if direction == "right":
        mid = rng.randint(r0 + 1, r1 - 1)
        tip_c = c1 - rng.randint(0, 1)
        base_c = tip_c - 1
        cells = [(mid - 1, base_c), (mid + 1, base_c), (mid, tip_c)]
    elif direction == "left":
        mid = rng.randint(r0 + 1, r1 - 1)
        tip_c = c0 + rng.randint(0, 1)
        base_c = tip_c + 1
        cells = [(mid - 1, base_c), (mid + 1, base_c), (mid, tip_c)]
    elif direction == "down":
        mid = rng.randint(c0 + 1, c1 - 1)
        tip_r = r1 - rng.randint(0, 1)
        base_r = tip_r - 1
        cells = [(base_r, mid - 1), (base_r, mid + 1), (tip_r, mid)]
    else:
        mid = rng.randint(c0 + 1, c1 - 1)
        tip_r = r0 + rng.randint(0, 1)
        base_r = tip_r + 1
        cells = [(base_r, mid - 1), (base_r, mid + 1), (tip_r, mid)]

    for r, c in cells:
        grid[int(r), int(c)] = ORANGE


def _validate_generated(
    grid: np.ndarray,
    expected_arrow_count: int,
    expected_group_sizes: list[int],
) -> bool:
    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False
    if len(scene["arrows"]) != expected_arrow_count:
        return False

    counts = sorted(int(v) for v in scene["bucket_target_counts"].values())
    if sorted(expected_group_sizes) != counts:
        return False
    if not any(count % 2 == 1 for count in counts):
        return False
    if not any(count % 2 == 0 for count in counts):
        return False

    for arrow_region_id, bucket_region_id in scene["arrow_links"]:
        bucket_color = int(scene["buckets"][bucket_region_id]["color"])
        count = int(scene["bucket_target_counts"][bucket_region_id])
        fill_color = bucket_color if count % 2 == 1 else AZURE
        for r, c in scene["regions"][arrow_region_id]["cells"]:
            if int(out[r, c]) != fill_color:
                return False

    for bucket_region_id, bucket_info in scene["buckets"].items():
        for r, c in scene["regions"][bucket_region_id]["cells"]:
            if int(out[r, c]) != int(grid[r, c]):
                return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.choice(ROW_COUNT_OPTIONS)
    n_cols = rng.choice(COL_COUNT_OPTIONS)
    row_sizes = _sample_axis(rng, n_rows)
    col_sizes = _sample_axis(rng, n_cols)

    grid, boxes = _build_partition(row_sizes, col_sizes)
    groups = _choose_primary_groups(rng, n_rows, n_cols)
    if groups is None:
        return None

    used_arrow_cells = {src for group in groups for src, _ in group["sources"]}
    bucket_cells = [group["bucket"] for group in groups]

    remaining_cells = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if (r, c) not in used_arrow_cells and (r, c) not in bucket_cells
    ]
    rng.shuffle(remaining_cells)
    extra_bucket_count = rng.randint(0, min(2, len(remaining_cells)))
    extra_buckets = remaining_cells[:extra_bucket_count]
    bucket_cells.extend(extra_buckets)

    bucket_colors = {cell: rng.choice(BUCKET_COLORS) for cell in bucket_cells}
    if len(bucket_cells) >= 2 and rng.random() < 0.5:
        a, b = rng.sample(bucket_cells, 2)
        bucket_colors[b] = bucket_colors[a]

    for cell in bucket_cells:
        _draw_bucket(grid, boxes[cell], bucket_colors[cell])

    for group in groups:
        for source, direction in group["sources"]:
            _draw_arrow(grid, boxes[source], direction, rng)

    expected_sizes = [len(group["sources"]) for group in groups]
    expected_arrow_count = sum(expected_sizes)
    if not _validate_generated(grid, expected_arrow_count, expected_sizes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
